import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Function
from .module_util import default_conv


def default_conv(in_channels, out_channels, kernel_size, stride=1, bias=True):
    return nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding=(kernel_size // 2), bias=bias, padding_mode='zeros')


class PreFilter(nn.Module):
    def __init__(self, n_feats=32, kernel_size=3, stride=1) -> None:
        super(PreFilter, self).__init__()
        self.conv = nn.Sequential(
            default_conv(3, n_feats, kernel_size),
            nn.ReLU(True),
            default_conv(n_feats, n_feats, kernel_size),
            nn.ReLU(True),
            default_conv(n_feats, 3, kernel_size),
            nn.Tanh(),
        )

    def forward(self, x):
        x_skip = x
        x = self.conv(x)
        x = x + x_skip
        return x


class PostFilter(nn.Module):
    def __init__(self, n_feats=32, kernel_size=3, stride=1) -> None:
        super(PostFilter, self).__init__()
        self.conv = nn.Sequential(default_conv(3, n_feats, kernel_size), nn.ReLU(True), default_conv(n_feats, n_feats, kernel_size), nn.ReLU(True),
                                  default_conv(n_feats, 3, kernel_size), nn.Tanh())

    def forward(self, x):
        x_skip = x
        x = self.conv(x)
        x = x + x_skip
        return x


class ResBlock(nn.Module):
    def __init__(self, conv, cin, n_feats, kernel_size=3, s=2, bias=True, act=nn.ReLU(True)) -> None:

        super(ResBlock, self).__init__()
        self.conv1 = nn.Sequential(
            conv(cin, n_feats, kernel_size, bias=bias),
            nn.BatchNorm2d(n_feats),
            act,
        )
        self.conv2 = nn.Sequential(
            conv(n_feats, n_feats, kernel_size, bias=bias),
            nn.BatchNorm2d(n_feats),
            act,
        )
        self.conv3 = conv(n_feats, n_feats, kernel_size, bias=bias)
        self.pool = nn.MaxPool2d(s, s)

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        _ = self.conv3(x)
        x = x + _
        x = self.pool(x)
        x = F.relu(x)
        return x


class ResizeParamNet(nn.Module):
    def __init__(self, *args, **kwargs) -> None:
        super(ResizeParamNet, self).__init__(*args, **kwargs)
        self.res1 = ResBlock(default_conv, 3, 16)
        self.res2 = ResBlock(default_conv, 16, 32)
        self.res3 = ResBlock(default_conv, 32, 64)

        self.pre = PreFilter()
        self.post = PostFilter()

    def forward(self, x):
        B, _, _, _ = x.shape
        x = self.res1(x)
        x = self.res2(x)
        x = self.res3(x)
        x = torch.mean(x, dim=(1, 2, 3)).view(-1, 1)
        x = F.relu(x)
        return x


class ResizeNet(nn.Module):
    def __init__(self) -> None:
        super(ResizeNet, self).__init__()

        self.pre = PreFilter()
        self.post = PostFilter()
        self.resizenet = ResizeParamNet()

    # Spatial transformer network forward function
    def stn(self, x, theta, reverse=False):
        """Warp an image or feature map with resize parameter
        Args:
            x (Tensor): size (N, C, H, W)
            theta (Tensor): size (N, 1), resize parameter

        Returns:
            Tensor: warped image or feature map
        """
        B, _, _, _ = x.shape
        theta0 = torch.zeros_like(theta, dtype=torch.float32)
        if reverse:
            stn_matrix = torch.cat((theta, theta0, theta0, theta0, theta, theta0), axis=1)
        else:
            stn_matrix = torch.cat((1 / theta, theta0, theta0, theta0, 1 / theta, theta0), axis=1)
        grid = F.affine_grid(stn_matrix.view(B, 2, 3), x.size())
        x = F.grid_sample(x, grid, mode='bicubic')
        return x

    def forward(self, x, theta=None, reverse=False):
        B, _, _, _ = x.shape
        if reverse:
            assert theta != None, 'theta not exist'
            x = self.stn(x, theta.view(B, 1))
            out = self.post(x)
        else:
            theta = self.resizenet(x)
            x = self.pre(x)
            out = self.stn(x, theta.view(B, 1), reverse=False)
        return out, theta
