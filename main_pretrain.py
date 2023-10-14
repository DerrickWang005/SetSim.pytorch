#!/usr/bin/env python
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
import argparse
import builtins
import math
import os
import random
import shutil
import time
from utils.transform import Compose, RandomHorizontalFlipCoord, RandomResizedCropCoord
from utils.imagenet import ImageNetLMDB
import warnings
import numpy as np

import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.distributed as dist
import torch.optim
import torch.multiprocessing as mp
import torch.utils.data
import torch.utils.data.distributed
import torch.cuda.amp as amp
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import torchvision.models as models
from tensorboardX import SummaryWriter

from utils.setlogger import get_logger
from utils.loader import TwoCropsTransform, GaussianBlur
from setstim.moco import MoCo
from setstim.framework import SetSim

saved_path = os.path.join(
    "exp/SemanticFocus_attention0p7_negtive0p2_geometry0p5_lr0p03_bs256_ep200/"
)
if not os.path.exists(saved_path):
    os.makedirs(saved_path)
logger = get_logger(os.path.join(saved_path, "train.log"))

model_names = sorted(
    name
    for name in models.__dict__
    if name.islower() and not name.startswith("__") and callable(models.__dict__[name])
)

parser = argparse.ArgumentParser(description="PyTorch ImageNet Training")
parser.add_argument("data", metavar="DIR", help="path to dataset")
parser.add_argument(
    "-a",
    "--arch",
    metavar="ARCH",
    default="resnet50",
    choices=model_names,
    help="model architecture: " + " | ".join(model_names) + " (default: resnet50)",
)
parser.add_argument(
    "-j",
    "--workers",
    default=32,
    type=int,
    metavar="N",
    help="number of data loading workers (default: 32)",
)
parser.add_argument(
    "--epochs", default=200, type=int, metavar="N", help="number of total epochs to run"
)
parser.add_argument(
    "--start-epoch",
    default=0,
    type=int,
    metavar="N",
    help="manual epoch number (useful on restarts)",
)
parser.add_argument(
    "-b",
    "--batch-size",
    default=256,
    type=int,
    metavar="N",
    help="mini-batch size (default: 256), this is the total "
    "batch size of all GPUs on the current node when "
    "using Data Parallel or Distributed Data Parallel",
)
parser.add_argument(
    "--lr",
    "--learning-rate",
    default=0.03,
    type=float,
    metavar="LR",
    help="initial learning rate",
    dest="lr",
)
parser.add_argument(
    "--schedule",
    default=[120, 160],
    nargs="*",
    type=int,
    help="learning rate schedule (when to drop lr by 10x)",
)
parser.add_argument(
    "--momentum", default=0.9, type=float, metavar="M", help="momentum of SGD solver"
)
parser.add_argument(
    "--wd",
    "--weight-decay",
    default=1e-4,
    type=float,
    metavar="W",
    help="weight decay (default: 1e-4)",
    dest="weight_decay",
)
parser.add_argument(
    "-p",
    "--print-freq",
    default=100,
    type=int,
    metavar="N",
    help="print frequency (default: 10)",
)
parser.add_argument(
    "--resume",
    default="",
    type=str,
    metavar="PATH",
    help="path to latest checkpoint (default: none)",
)
parser.add_argument(
    "--world-size",
    default=-1,
    type=int,
    help="number of nodes for distributed training",
)
parser.add_argument(
    "--rank", default=-1, type=int, help="node rank for distributed training"
)
parser.add_argument(
    "--dist-url",
    default="tcp://224.66.41.62:23456",
    type=str,
    help="url used to set up distributed training",
)
parser.add_argument(
    "--dist-backend", default="nccl", type=str, help="distributed backend"
)
parser.add_argument(
    "--seed", default=None, type=int, help="seed for initializing training. "
)
parser.add_argument("--gpu", default=None, type=int, help="GPU id to use.")
parser.add_argument(
    "--multiprocessing-distributed",
    action="store_true",
    help="Use multi-processing distributed training to launch "
    "N processes per node, which has N GPUs. This is the "
    "fastest way to use PyTorch for either single node or "
    "multi node data parallel training",
)

# moco specific configs:
parser.add_argument(
    "--moco-dim", default=128, type=int, help="feature dimension (default: 128)"
)
parser.add_argument(
    "--moco-k",
    default=65536,
    type=int,
    help="queue size; number of negative keys (default: 65536)",
)
parser.add_argument(
    "--moco-m",
    default=0.999,
    type=float,
    help="moco momentum of updating key encoder (default: 0.999)",
)
parser.add_argument(
    "--moco-t", default=0.2, type=float, help="softmax temperature (default: 0.07)"
)
parser.add_argument(
    "--p-weight", default=0.5, type=float, help="pixel loss weight (default: 0.5)"
)
parser.add_argument(
    "--attention", action="store_true", help="use attention correspondence"
)
parser.add_argument(
    "--att-threshold", default=0.7, type=float, help="attention threhold (default: 0.6)"
)
parser.add_argument(
    "--neg", default=0.0, type=float, help="bottom similarity (default: 0.)"
)
parser.add_argument("--cosine", action="store_true", help="use cosine correspondence")
parser.add_argument(
    "--geometry", action="store_true", help="use geometry correspondence"
)
parser.add_argument(
    "--geo-threshold", default=0.7, type=float, help="geometry threhold (default: 0.7)"
)
parser.add_argument(
    "--nearest", action="store_true", help="use nearest neighbour correspondence"
)
parser.add_argument("--mlp", action="store_true", help="use mlp head")
parser.add_argument(
    "--aug-plus", action="store_true", help="use moco v2 data augmentation"
)
parser.add_argument("--cos", action="store_true", help="use cosine lr schedule")


def main():
    args = parser.parse_args()

    if args.seed is not None:
        random.seed(args.seed)
        np.random.seed(args.seed)
        torch.manual_seed(args.seed)
        torch.cuda.manual_seed_all(args.seed)
        cudnn.deterministic = True
        warnings.warn(
            "You have chosen to seed training. "
            "This will turn on the CUDNN deterministic setting, "
            "which can slow down your training considerably! "
            "You may see unexpected behavior when restarting "
            "from checkpoints."
        )

    if args.gpu is not None:
        warnings.warn(
            "You have chosen a specific GPU. This will completely "
            "disable data parallelism."
        )

    args.lr *= args.batch_size / 256
    if args.dist_url == "env://" and args.world_size == -1:
        args.world_size = int(os.environ["WORLD_SIZE"])

    args.distributed = args.world_size > 1 or args.multiprocessing_distributed

    ngpus_per_node = torch.cuda.device_count()
    if args.multiprocessing_distributed:
        # Since we have ngpus_per_node processes per node, the total world_size
        # needs to be adjusted accordingly
        args.world_size = ngpus_per_node * args.world_size
        # Use torch.multiprocessing.spawn to launch distributed processes: the
        # main_worker process function
        mp.spawn(main_worker, nprocs=ngpus_per_node, args=(ngpus_per_node, args))
    else:
        # Simply call main_worker function
        main_worker(args.gpu, ngpus_per_node, args)


def main_worker(gpu, ngpus_per_node, args):
    args.gpu = gpu
    args.ngpus_per_node = ngpus_per_node

    if args.gpu is not None:
        print("Use GPU: {} for training".format(args.gpu))

    # suppress printing if not master
    if args.multiprocessing_distributed and args.gpu != 0:

        def print_pass(*args):
            pass

        builtins.print = print_pass

    if args.gpu == 0:
        global writer
        writer = SummaryWriter(logdir=saved_path, comment="vis_log")

    if args.distributed:
        if args.dist_url == "env://" and args.rank == -1:
            args.rank = int(os.environ["RANK"])
        if args.multiprocessing_distributed:
            # For multiprocessing distributed training, rank needs to be the
            # global rank among all the processes
            args.rank = args.rank * ngpus_per_node + gpu
        dist.init_process_group(
            backend=args.dist_backend,
            init_method=args.dist_url,
            world_size=args.world_size,
            rank=args.rank,
        )
    # create model
    print("=> creating model '{}'".format(args.arch))
    model = SetSim(
        dim=args.moco_dim,
        K=args.moco_k,
        m=args.moco_m,
        T=args.moco_t,
        attention=args.attention,
        attention_threshold=args.att_threshold,
        neg=args.neg,
        geometry=args.geometry,
        geo_thres=args.geo_threshold,
        nearest=args.nearest,
        bs=int(args.batch_size / ngpus_per_node),
    )
    print(model)

    if args.distributed:
        # For multiprocessing distributed, DistributedDataParallel constructor
        # should always set the single device scope, otherwise,
        # DistributedDataParallel will use all available devices.
        if args.gpu is not None:
            torch.cuda.set_device(args.gpu)
            model.cuda(args.gpu)
            # When using a single GPU per process and per
            # DistributedDataParallel, we need to divide the batch size
            # ourselves based on the total number of GPUs we have
            args.batch_size = int(args.batch_size / ngpus_per_node)
            args.workers = int(args.workers / ngpus_per_node)
            model = torch.nn.parallel.DistributedDataParallel(
                model, device_ids=[args.gpu]
            )
        else:
            model.cuda()
            # DistributedDataParallel will divide and allocate batch_size to all
            # available GPUs if device_ids are not set
            model = torch.nn.parallel.DistributedDataParallel(model)
    elif args.gpu is not None:
        torch.cuda.set_device(args.gpu)
        model = model.cuda(args.gpu)
        # comment out the following line for debugging
        raise NotImplementedError("Only DistributedDataParallel is supported.")
    else:
        # AllGather implementation (batch shuffle, queue update, etc.) in
        # this code only supports DistributedDataParallel.
        raise NotImplementedError("Only DistributedDataParallel is supported.")

    # define loss function (criterion) and optimizer
    criterion = nn.CrossEntropyLoss().cuda(args.gpu)

    optimizer = torch.optim.SGD(
        model.parameters(),
        args.lr,
        momentum=args.momentum,
        weight_decay=args.weight_decay,
    )

    # optionally resume from a checkpoint
    if args.resume:
        if os.path.isfile(args.resume):
            print("=> loading checkpoint '{}'".format(args.resume))
            if args.gpu is None:
                checkpoint = torch.load(args.resume)
            else:
                # Map model to be loaded to specified single gpu.
                loc = "cuda:{}".format(args.gpu)
                checkpoint = torch.load(args.resume, map_location=loc)
            args.start_epoch = checkpoint["epoch"]
            model.load_state_dict(checkpoint["state_dict"])
            optimizer.load_state_dict(checkpoint["optimizer"])
            print(
                "=> loaded checkpoint '{}' (epoch {})".format(
                    args.resume, checkpoint["epoch"]
                )
            )
        else:
            print("=> no checkpoint found at '{}'".format(args.resume))

    cudnn.benchmark = True

    # Data loading code
    traindir = os.path.join(args.data, "train.lmdb")
    normalize = transforms.Normalize(
        mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
    )
    if args.aug_plus:
        # MoCo v2's aug: similar to SimCLR https://arxiv.org/abs/2002.05709
        augmentation = [
            RandomResizedCropCoord(224, scale=(0.2, 1.0)),
            RandomHorizontalFlipCoord(),
            transforms.RandomApply(
                [transforms.ColorJitter(0.4, 0.4, 0.4, 0.1)], p=0.8  # not strengthened
            ),
            transforms.RandomGrayscale(p=0.2),
            transforms.RandomApply([GaussianBlur([0.1, 2.0])], p=0.5),
            transforms.ToTensor(),
            normalize,
        ]
    else:
        # MoCo v1's aug: the same as InstDisc https://arxiv.org/abs/1805.01978
        augmentation = [
            RandomResizedCropCoord(224, scale=(0.2, 1.0)),
            RandomHorizontalFlipCoord(),
            transforms.RandomGrayscale(p=0.2),
            transforms.ColorJitter(0.4, 0.4, 0.4, 0.4),
            transforms.ToTensor(),
            normalize,
        ]

    train_dataset = ImageNetLMDB(traindir, Compose(augmentation))

    if args.distributed:
        train_sampler = torch.utils.data.distributed.DistributedSampler(train_dataset)
    else:
        train_sampler = None

    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=(train_sampler is None),
        num_workers=args.workers,
        pin_memory=True,
        sampler=train_sampler,
        drop_last=True,
    )

    for epoch in range(args.start_epoch, args.epochs):
        if args.distributed:
            train_sampler.set_epoch(epoch)
        adjust_learning_rate(optimizer, epoch, args)

        # train for one epoch
        train(train_loader, model, criterion, optimizer, epoch, args)

        if not args.multiprocessing_distributed or (
            args.multiprocessing_distributed and args.rank % ngpus_per_node == 0
        ):
            state_dict = model.state_dict()
            for key in [
                "module.queue",
                "module.queue_ptr",
                "module.queue2",
                "module.queue2_ptr",
                "module.logits_mask",
                "module.negative_mask",
            ]:
                del state_dict[key]
            save_checkpoint(
                {
                    "epoch": epoch + 1,
                    "arch": args.arch,
                    "state_dict": state_dict,
                    "optimizer": optimizer.state_dict(),
                },
                is_best=False,
                filename=saved_path + "checkpoint_{}.pth.tar".format(epoch),
            )


def train(train_loader, model, criterion, optimizer, epoch, args):
    batch_time = AverageMeter("Time", ":2.3f")
    data_time = AverageMeter("Data", ":2.3f")
    glosses = AverageMeter("GlbLoss", ":2.5f")
    plosses = AverageMeter("PixLoss", ":2.5f")
    losses = AverageMeter("Loss", ":2.5f")
    gtop1 = AverageMeter("Global Acc@1", ":3.2f")
    unstables = AverageMeter("Unstable pixels", ":2.2f")
    geometries = AverageMeter("Geometry pixels", ":2.2f")
    neighbours = AverageMeter("Neighbour pixels", ":2.2f")
    progress = ProgressMeter(
        len(train_loader),
        [
            batch_time,
            data_time,
            glosses,
            plosses,
            losses,
            gtop1,
            unstables,
            geometries,
            neighbours,
        ],
        prefix="Epoch: [{}]".format(epoch),
    )

    # switch to train mode
    scaler = amp.GradScaler()
    model.train()

    end = time.time()
    for i, images in enumerate(train_loader):
        i = i + 1
        # p_weight = _get_warmup_factor_at_iter('linear', i, epoch, 500, 1e-2, args.p_weight)
        # measure data loading time
        data_time.update(time.time() - end)

        if args.gpu is not None:
            images[0][0] = images[0][0].cuda(args.gpu, non_blocking=True)
            images[0][1] = images[0][1].cuda(args.gpu, non_blocking=True)
            images[1][0] = images[1][0].cuda(args.gpu, non_blocking=True)
            images[1][1] = images[1][1].cuda(args.gpu, non_blocking=True)

        # compute output
        with amp.autocast():
            glb_pair, ploss, unstable, geometry, neighbour = model(
                im_q=images[0], im_k=images[1]
            )
            gloss = criterion(glb_pair[0], glb_pair[1])
            loss = (1 - args.p_weight) * gloss + args.p_weight * ploss
            # loss = (1 - p_weight) * gloss + p_weight * ploss

        # compute gradient and do SGD step
        optimizer.zero_grad()
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        # measure accuracy and record loss
        dist.all_reduce(loss), dist.all_reduce(gloss), dist.all_reduce(ploss)
        loss /= args.ngpus_per_node
        gloss /= args.ngpus_per_node
        ploss /= args.ngpus_per_node
        losses.update(loss.item(), images[0][0].size(0))
        glosses.update(gloss.item(), images[0][0].size(0))
        plosses.update(ploss.item(), images[0][0].size(0))

        gacc1 = accuracy(glb_pair[0], glb_pair[1], topk=(1,))
        gtop1.update(gacc1[0].item(), images[0][0].size(0))
        unstables.update(unstable.item(), images[0][0].size(0))
        geometries.update(geometry.item(), images[0][0].size(0))
        neighbours.update(neighbour.item(), images[0][0].size(0))

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if i % args.print_freq == 0 and args.gpu == 0:
            progress.display(i)
            cur_iter = i + epoch * len(train_loader)
            writer.add_scalars(
                "train_loss",
                {
                    "total_loss": losses.val,
                    "global_loss": glosses.val,
                    "pixel_loss": plosses.val,
                },
                cur_iter,
            )
            writer.add_scalars("train_acc", {"global acc@1": gtop1.val}, cur_iter)
            writer.add_scalar("lr", optimizer.param_groups[0]["lr"], cur_iter)
            writer.add_scalar("unstable_pixels", unstables.val, cur_iter)
            writer.add_scalar("geometry_pixels", geometries.val, cur_iter)
            writer.add_scalar("neighbour_pixels", neighbours.val, cur_iter)


def _get_warmup_factor_at_iter(
    method: str,
    iter: int,
    epoch: int,
    warmup_iters: int,
    warmup_factor: float,
    base_factor: float,
) -> float:
    """
    Return the learning rate warmup factor at a specific iteration.
    See :paper:`ImageNet in 1h` for more details.

    Args:
        method (str): warmup method; either "constant" or "linear".
        iter (int): iteration at which to calculate the warmup factor.
        warmup_iters (int): the number of warmup iterations.
        warmup_factor (float): the base warmup factor (the meaning changes according
            to the method used).

    Returns:
        float: the effective warmup factor at the given iteration.
    """
    if epoch > 0 or iter >= warmup_iters:
        return base_factor

    if method == "constant":
        return warmup_factor * base_factor
    elif method == "linear":
        alpha = iter / warmup_iters
        return (warmup_factor * (1 - alpha) + alpha) * base_factor
    else:
        raise ValueError("Unknown warmup method: {}".format(method))


def save_checkpoint(state, is_best, filename="checkpoint.pth.tar"):
    torch.save(state, filename)
    if is_best:
        shutil.copyfile(filename, "model_best.pth.tar")


class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self, name, fmt=":f"):
        self.name = name
        self.fmt = fmt
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

    def __str__(self):
        fmtstr = "{name} {val" + self.fmt + "} ({avg" + self.fmt + "})"
        return fmtstr.format(**self.__dict__)


class ProgressMeter(object):
    def __init__(self, num_batches, meters, prefix=""):
        self.batch_fmtstr = self._get_batch_fmtstr(num_batches)
        self.meters = meters
        self.prefix = prefix

    def display(self, batch):
        entries = [self.prefix + self.batch_fmtstr.format(batch)]
        entries += [str(meter) for meter in self.meters]
        # print('\t'.join(entries))
        logger.info(" ".join(entries))

    def _get_batch_fmtstr(self, num_batches):
        num_digits = len(str(num_batches // 1))
        fmt = "{:" + str(num_digits) + "d}"
        return "[" + fmt + "/" + fmt.format(num_batches) + "]"


def adjust_learning_rate(optimizer, epoch, args):
    """Decay the learning rate based on schedule"""
    lr = args.lr
    if args.cos:  # cosine lr schedule
        lr *= 0.5 * (1.0 + math.cos(math.pi * epoch / args.epochs))
    else:  # stepwise lr schedule
        for milestone in args.schedule:
            lr *= 0.1 if epoch >= milestone else 1.0
    for param_group in optimizer.param_groups:
        param_group["lr"] = lr


def accuracy(output, target, topk=(1,)):
    """Computes the accuracy over the k top predictions for the specified values of k"""
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)

        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.reshape(1, -1).expand_as(pred))

        res = []
        for k in topk:
            correct_k = correct[:k].reshape(-1).float().sum(0, keepdim=True)
            res.append(correct_k.mul_(100.0 / batch_size))
        return res


if __name__ == "__main__":
    main()
