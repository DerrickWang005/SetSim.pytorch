from math import cos
import torch
import torch.nn as nn
import torch.nn.functional as F


class Propagation(nn.Module):
    def __init__(self, sharpen, clamp_value, in_channel, out_channel):
        super(Propagation, self).__init__()
        self.sharpen = sharpen
        self.clamp_value = clamp_value
        self.transform = nn.Conv2d(in_channel, out_channel, kernel_size=1, stride=1, padding=0, bias=True)
    
    def forward(self, x):
        b, c, h, w = x.size()

        value = self.transform(x)
        value = F.normalize(value, p=2, dim=1).reshape(b, c, -1)

        x = F.normalize(x, p=2, dim=1).reshape(b, c, -1)
        cosine = torch.matmul(x.transpose(-2, -1), x).clamp(min=self.clamp_value)
        cosine = cosine ** self.sharpen

        out = torch.matmul(value, cosine.transpose(-2, -1)).reshape(b, c, h, w)

        return out
