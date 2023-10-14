import torch
import torch.nn as nn
import torch.nn.functional as F

from .resnet import resnet50
from .matcher import Matcher


class SetSim(nn.Module):
    """
    Build a MoCo model with: a query encoder, a key encoder, and a queue
    https://arxiv.org/abs/1911.05722
    """

    def __init__(
        self,
        base_encoder=resnet50,
        dim=128,
        K=65536,
        m=0.999,
        T=0.2,
        attention=False,
        attention_threshold=0.7,
        neg=0.2,
        geometry=False,
        geo_thres=0.7,
        nearest=False,
        bs=64,
    ):
        """
        dim: feature dimension (default: 128)
        K: queue size; number of negative keys (default: 65536)
        m: moco momentum of updating key encoder (default: 0.999)
        T: softmax temperature (default: 0.07)
        """
        super(SetSim, self).__init__()

        self.K = K
        self.m = m
        self.T = T

        # create the encoders
        # num_classes is the output fc dimension
        encoder_q = base_encoder(num_classes=dim)
        encoder_k = base_encoder(num_classes=dim)
        # create the global projector
        dim_mlp = encoder_q.fc.weight.shape[1]
        encoder_q.fc = nn.Sequential(
            nn.Linear(dim_mlp, dim_mlp), nn.ReLU(), encoder_q.fc
        )
        encoder_k.fc = nn.Sequential(
            nn.Linear(dim_mlp, dim_mlp), nn.ReLU(), encoder_k.fc
        )
        # create the dense projector
        fc_pix_q = nn.Sequential(
            nn.Conv2d(dim_mlp, dim_mlp, 1), nn.ReLU(), nn.Conv2d(dim_mlp, dim, 1)
        )
        fc_pix_k = nn.Sequential(
            nn.Conv2d(dim_mlp, dim_mlp, 1), nn.ReLU(), nn.Conv2d(dim_mlp, dim, 1)
        )
        self.encoder_q = nn.Sequential(encoder_q, fc_pix_q)
        self.encoder_k = nn.Sequential(encoder_k, fc_pix_k)
        # matcher
        self.matcher = Matcher(
            attention=attention,
            attention_threshold=attention_threshold,
            neg=neg,
            geometry=geometry,
            geo_threshold=geo_thres,
            nn_match=nearest,
        )

        for param_q, param_k in zip(
            self.encoder_q.parameters(), self.encoder_k.parameters()
        ):
            param_k.data.copy_(param_q.data)  # initialize
            param_k.requires_grad = False  # not update by gradient

        # create logits mask & negative mask
        self.register_buffer("logits_mask", torch.ones(bs, 49, self.K))
        self.register_buffer("negative_mask", torch.zeros(bs, 49, self.K))
        # create the queue
        self.register_buffer("queue", torch.randn(dim, K))
        self.queue = F.normalize(self.queue, dim=0)
        self.register_buffer("queue_ptr", torch.zeros(1, dtype=torch.long))
        # create the second queue for dense output
        self.register_buffer("queue2", torch.randn(dim, K))
        self.queue2 = F.normalize(self.queue2, dim=0)
        self.register_buffer("queue2_ptr", torch.zeros(1, dtype=torch.long))

    @torch.no_grad()
    def _momentum_update_key_encoder(self):
        """
        Momentum update of the key encoder
        """
        for param_q, param_k in zip(
            self.encoder_q.parameters(), self.encoder_k.parameters()
        ):
            param_k.data = param_k.data * self.m + param_q.data * (1.0 - self.m)

    @torch.no_grad()
    def _dequeue_and_enqueue(self, keys):
        # gather keys before updating queue
        keys = concat_all_gather(keys)

        batch_size = keys.shape[0]

        ptr = int(self.queue_ptr)
        assert self.K % batch_size == 0  # for simplicity

        # replace the keys at ptr (dequeue and enqueue)
        self.queue[:, ptr : ptr + batch_size] = keys.T
        ptr = (ptr + batch_size) % self.K  # move pointer

        self.queue_ptr[0] = ptr

    @torch.no_grad()
    def _dequeue_and_enqueue2(self, keys):
        # gather keys before updating queue
        keys = concat_all_gather(keys)

        batch_size = keys.shape[0]

        ptr = int(self.queue2_ptr)
        assert self.K % batch_size == 0  # for simplicity

        # replace the keys at ptr (dequeue and enqueue)
        self.queue2[:, ptr : ptr + batch_size] = keys.transpose(0, 1)
        ptr = (ptr + batch_size) % self.K  # move pointer

        self.queue2_ptr[0] = ptr

    @torch.no_grad()
    def _batch_shuffle_ddp(self, x):
        """
        Batch shuffle, for making use of BatchNorm.
        *** Only support DistributedDataParallel (DDP) model. ***
        """
        # gather from all gpus
        batch_size_this = x.shape[0]
        x_gather = concat_all_gather(x)
        batch_size_all = x_gather.shape[0]

        num_gpus = batch_size_all // batch_size_this

        # random shuffle index
        idx_shuffle = torch.randperm(batch_size_all).cuda()

        # broadcast to all gpus
        torch.distributed.broadcast(idx_shuffle, src=0)

        # index for restoring
        idx_unshuffle = torch.argsort(idx_shuffle)

        # shuffled index for this gpu
        gpu_idx = torch.distributed.get_rank()
        idx_this = idx_shuffle.view(num_gpus, -1)[gpu_idx]

        return x_gather[idx_this], idx_unshuffle

    @torch.no_grad()
    def _batch_unshuffle_ddp(self, x, idx_unshuffle):
        """
        Undo batch shuffle.
        *** Only support DistributedDataParallel (DDP) model. ***
        """
        # gather from all gpus
        batch_size_this = x.shape[0]
        x_gather = concat_all_gather(x)
        batch_size_all = x_gather.shape[0]

        num_gpus = batch_size_all // batch_size_this

        # restored index for this gpu
        gpu_idx = torch.distributed.get_rank()
        idx_this = idx_unshuffle.view(num_gpus, -1)[gpu_idx]

        return x_gather[idx_this]

    def _cal_glb_loss(self, q, k):
        """
        Input:
            q: NxC
            k: NxC
        """
        # positive logits: Nx1
        l_pos = torch.einsum("nc,nc->n", [q, k]).unsqueeze(-1)
        # negative logits: NxK
        l_neg = torch.einsum("nc,ck->nk", [q, self.queue.clone().detach()])
        # logits
        logits = torch.cat([l_pos, l_neg], dim=1)  # Nx(1+K)
        # apply temperature
        logits /= self.T
        # labels: positive key indicators
        labels = torch.zeros(logits.shape[0], dtype=torch.long).cuda()

        return (logits, labels)

    def _cal_pix_loss(self, q, k, q_b, k_b, coord_q, coord_k):
        """
        Input:
            q: Nx128xHxW
            k: Nx128xHxW
            q_b: NxCxHxW
            k_b: NxCxHxW
        """
        b, c = q.size()[:2]
        q = q.reshape(b, c, -1)  # NxCxHW
        k = k.reshape(b, c, -1)  # NxCxHW
        # match
        valid_mask, unstable, geometry, neighbour = self.matcher(
            [q_b, coord_q], [k_b, coord_k]
        )  # NxHWxHW, a number
        # compute pixel logits
        l_pos = torch.matmul(q.transpose(-2, -1), k)  # NxHWxHW
        l_neg = torch.einsum("nci,cj->nij", [q, self.queue2.clone().detach()])  # NxHWxK
        # logits
        logits = torch.cat([l_pos, l_neg], dim=-1)  # NxHWx(HW+K)
        # apply temperature
        logits /= self.T
        # compute log_prob
        logits_mask = torch.cat([valid_mask, self.logits_mask], dim=-1)  # NxHWx(HW+K)
        p_log_prob = logits - torch.log(
            (torch.exp(logits) * logits_mask).sum(dim=-1, keepdim=True)
        )
        # compute mean of positive log-likehood
        valid = torch.cat([valid_mask, self.negative_mask], dim=-1)  # NxHWx(HW+K)
        p_mean_log_prob = (
            -(valid * p_log_prob).sum(-1) / (valid.sum(-1) + 1e-8).detach()
        )  # NxHW
        count = (valid_mask.sum(-1).bool().sum(-1) + 1e-8).detach()
        loss = (p_mean_log_prob.sum(-1) / count).mean()

        return loss, unstable, geometry, neighbour

    def forward(self, im_q, im_k):
        """
        Input:
            im_q: a batch of query images and coord_q
            im_k: a batch of key images and coord_k
        Output:
            logits, targets
        """

        # compute query features
        q_glb, q = self.encoder_q[0](im_q[0])  # q: NxCxHxW, q_glb: Nxdim
        q_pix = self.encoder_q[1](q)  # q_pix: NxdimxHxW

        # q = F.normalize(q, dim=1) # NxCxHxW
        q_glb = F.normalize(q_glb, p=2, dim=1)  # Nxdim
        q_pix = F.normalize(q_pix, p=2, dim=1)  # NxdimxHxW

        # compute key features
        with torch.no_grad():  # no gradient to keys
            self._momentum_update_key_encoder()  # update the key encoder

            # shuffle for making use of BN
            im_k_s, idx_unshuffle = self._batch_shuffle_ddp(im_k[0])

            k_glb, k = self.encoder_k[0](im_k_s)  # k: NxCxHxW, k_glb: Nxdim
            k_pix = self.encoder_k[1](k)  # k_pix: NxdimxHxW

            k_glb = F.normalize(k_glb, p=2, dim=1)  # Nxdim
            k_pix_q = F.normalize(
                F.adaptive_avg_pool2d(k_pix, (1, 1)).flatten(1, -1), p=2, dim=1
            )  # NxdimxHxW
            k_pix = F.normalize(k_pix, p=2, dim=1)  # NxdimxHxW

            # undo shuffle
            k = self._batch_unshuffle_ddp(k, idx_unshuffle)
            k_glb = self._batch_unshuffle_ddp(k_glb, idx_unshuffle)
            k_pix = self._batch_unshuffle_ddp(k_pix, idx_unshuffle)
            k_pix_q = self._batch_unshuffle_ddp(k_pix_q, idx_unshuffle)

        # compute global logits
        logits_glb, labels_glb = self._cal_glb_loss(q_glb, k_glb)

        # compute pixel logits
        ploss, unstable, geometry, neighbour = self._cal_pix_loss(
            q_pix, k_pix, q, k, im_q[1], im_k[1]
        )

        # dequeue and enqueue
        self._dequeue_and_enqueue(k_glb)
        self._dequeue_and_enqueue2(k_pix_q)

        return [logits_glb, labels_glb], ploss, unstable, geometry, neighbour


# utils
@torch.no_grad()
def concat_all_gather(tensor):
    """
    Performs all_gather operation on the provided tensors.
    *** Warning ***: torch.distributed.all_gather has no gradient.
    """
    tensors_gather = [
        torch.ones_like(tensor) for _ in range(torch.distributed.get_world_size())
    ]
    torch.distributed.all_gather(tensors_gather, tensor, async_op=False)

    output = torch.cat(tensors_gather, dim=0)
    return output
