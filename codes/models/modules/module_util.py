import torch
import torch.nn as nn
import torch.nn.init as init
import torch.nn.functional as F


def initialize_weights(net_l, scale=1):
    if not isinstance(net_l, list):
        net_l = [net_l]
    for net in net_l:
        for m in net.modules():
            if isinstance(m, nn.Conv2d):
                init.kaiming_normal_(m.weight, a=0, mode='fan_in')
                m.weight.data *= scale  # for residual block
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                init.kaiming_normal_(m.weight, a=0, mode='fan_in')
                m.weight.data *= scale
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                init.constant_(m.weight, 1)
                init.constant_(m.bias.data, 0.0)


def initialize_weights_xavier(net_l, scale=1):
    if not isinstance(net_l, list):
        net_l = [net_l]
    for net in net_l:
        for m in net.modules():
            if isinstance(m, nn.Conv2d):
                init.xavier_normal_(m.weight)
                m.weight.data *= scale  # for residual block
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                init.xavier_normal_(m.weight)
                m.weight.data *= scale
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                init.constant_(m.weight, 1)
                init.constant_(m.bias.data, 0.0)


def make_layer(block, n_layers):
    layers = []
    for _ in range(n_layers):
        layers.append(block())
    return nn.Sequential(*layers)


class ResidualBlock_noBN(nn.Module):
    '''Residual block w/o BN
    ---Conv-ReLU-Conv-+-
     |________________|
    '''
    def __init__(self, nf=64):
        super(ResidualBlock_noBN, self).__init__()
        self.conv1 = nn.Conv2d(nf, nf, 3, 1, 1, bias=True)
        self.conv2 = nn.Conv2d(nf, nf, 3, 1, 1, bias=True)

        # initialization
        initialize_weights([self.conv1, self.conv2], 0.1)

    def forward(self, x):
        identity = x
        out = F.relu(self.conv1(x), inplace=True)
        out = self.conv2(out)
        return identity + out


def flow_warp(x, flow, interp_mode='bilinear', padding_mode='zeros'):
    """Warp an image or feature map with optical flow
    Args:
        x (Tensor): size (N, C, H, W)
        flow (Tensor): size (N, H, W, 2), normal value
        interp_mode (str): 'nearest' or 'bilinear'
        padding_mode (str): 'zeros' or 'border' or 'reflection'

    Returns:
        Tensor: warped image or feature map
    """
    assert x.size()[-2:] == flow.size()[1:3]
    B, C, H, W = x.size()
    # mesh grid
    grid_y, grid_x = torch.meshgrid(torch.arange(0, H), torch.arange(0, W))
    grid = torch.stack((grid_x, grid_y), 2).float()  # W(x), H(y), 2
    grid.requires_grad = False
    grid = grid.type_as(x)
    vgrid = grid + flow
    # scale grid to [-1,1]
    vgrid_x = 2.0 * vgrid[:, :, :, 0] / max(W - 1, 1) - 1.0
    vgrid_y = 2.0 * vgrid[:, :, :, 1] / max(H - 1, 1) - 1.0
    vgrid_scaled = torch.stack((vgrid_x, vgrid_y), dim=3)
    output = F.grid_sample(x, vgrid_scaled, mode=interp_mode, padding_mode=padding_mode)
    return output


def default_conv(in_channels, out_channels, kernel_size, bias=True):
    return nn.Conv2d(in_channels, out_channels, kernel_size, padding=(kernel_size // 2), bias=bias)


class ResBlock(nn.Module):
    def __init__(self, conv, n_feats, kernel_size, bias=True, bn=False, act=nn.ReLU(True), res_scale=1):

        super(ResBlock, self).__init__()
        m = []
        for i in range(2):
            m.append(conv(n_feats, n_feats, kernel_size, bias=bias))
            if bn:
                m.append(nn.BatchNorm2d(n_feats))
            if i == 0:
                m.append(act)

        self.body = nn.Sequential(*m)
        self.res_scale = res_scale

    def forward(self, x):
        res = self.body(x).mul(self.res_scale)
        res += x

        return res


class RateEstNet(nn.Module):
    def __init__(self, channel_in):
        super(RateEstNet, self).__init__()
        self.h1 = nn.Parameter(torch.randn((1, channel_in, 1, 1), requires_grad=True))
        self.b1 = nn.Parameter(torch.randn((1, channel_in, 1, 1), requires_grad=True))
        self.h2 = nn.Parameter(torch.randn((1, channel_in, 1, 1), requires_grad=True))
        self.b2 = nn.Parameter(torch.randn((1, channel_in, 1, 1), requires_grad=True))
        self.h3 = nn.Parameter(torch.randn((1, channel_in, 1, 1), requires_grad=True))
        self.b3 = nn.Parameter(torch.randn((1, channel_in, 1, 1), requires_grad=True))
        self.h4 = nn.Parameter(torch.randn((1, channel_in, 1, 1), requires_grad=True))
        self.b4 = nn.Parameter(torch.randn((1, channel_in, 1, 1), requires_grad=True))
        self.a1 = nn.Parameter(torch.randn((1, channel_in, 1, 1), requires_grad=True))
        self.a2 = nn.Parameter(torch.randn((1, channel_in, 1, 1), requires_grad=True))
        self.a3 = nn.Parameter(torch.randn((1, channel_in, 1, 1), requires_grad=True))
        self.sp = torch.nn.Softplus()

    def forward(self, x):
        b, c, h, w = x.shape
        self.h1.repeat((b, 1, h, w))
        self.h2.repeat((b, 1, h, w))
        self.h3.repeat((b, 1, h, w))
        self.h4.repeat((b, 1, h, w))
        self.b1.repeat((b, 1, h, w))
        self.b2.repeat((b, 1, h, w))
        self.b3.repeat((b, 1, h, w))
        self.b4.repeat((b, 1, h, w))
        self.a1.repeat((b, 1, h, w))
        self.a2.repeat((b, 1, h, w))
        self.a3.repeat((b, 1, h, w))
        x = self.sp(self.h1).mul(x) + self.b1
        x = torch.tanh(self.a1).mul(torch.tanh(x)) + x
        x = self.sp(self.h2).mul(x) + self.b2
        x = torch.tanh(self.a2).mul(torch.tanh(x)) + x
        x = self.sp(self.h3).mul(x) + self.b3
        x = torch.tanh(self.a3).mul(torch.tanh(x)) + x
        x = self.sp(self.h4).mul(x) + self.b4
        x = torch.sigmoid(x)
        return x


class Quant(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input):
        output = (input).round()
        return output

    @staticmethod
    def backward(ctx, grad_output):
        return grad_output


class Quantization(nn.Module):
    def __init__(self):
        super(Quantization, self).__init__()

    def forward(self, input):
        return Quant.apply(input)


class PixelShuffle(nn.Module):
    def __init__(self, scale):
        super(PixelShuffle, self).__init__()
        self.hscale = scale
        self.wscale = scale

    def forward(self, input):
        b, c, h, w = input.shape
        oc = c // (self.hscale * self.wscale)

        input = torch.reshape(input, (b, self.hscale, self.wscale, oc, h, w))
        input = input.permute((0, 3, 4, 1, 5, 2))
        input = torch.reshape(input, (b, oc, h * self.hscale, w * self.wscale))
        return input


class PixelInvShuffle(nn.Module):
    def __init__(self, scale):
        super(PixelInvShuffle, self).__init__()
        self.hscale = scale
        self.wscale = scale

    def forward(self, input):
        b, c, h, w = input.shape
        oh = h // self.hscale
        ow = w // self.wscale

        input = torch.reshape(input, (b, c, oh, self.hscale, ow, self.wscale))
        input = input.permute((0, 3, 5, 1, 2, 4))
        input = torch.reshape(input, (b, c * self.hscale * self.wscale, oh, ow))
        return input


def get_grid(x):
    torchHorizontal = torch.linspace(-1.0, 1.0, x.size(3)).view(1, 1, 1, x.size(3)).expand(x.size(0), 1, x.size(2), x.size(3))
    torchVertical = torch.linspace(-1.0, 1.0, x.size(2)).view(1, 1, x.size(2), 1).expand(x.size(0), 1, x.size(2), x.size(3))
    grid = torch.cat([torchHorizontal, torchVertical], 1)

    return grid


class WarpingLayer(nn.Module):
    def __init__(self):
        super(WarpingLayer, self).__init__()

    def forward(self, x, flow, align):
        grid = torch.clip((get_grid(x).cuda() + flow).permute(0, 2, 3, 1), -1.0, 1.0)

        x_warp = F.grid_sample(x, grid, align_corners=align)

        return x_warp


class InvResBlock(nn.Module):
    def __init__(self, channel_in, channel_out, bias=True):
        super(InvResBlock, self).__init__()
        self.conv1 = nn.Conv2d(channel_in, channel_out, 1, 1, 0, bias=bias)

        self.conv2 = nn.Conv2d(channel_out, channel_out, 3, 1, 1, bias=bias)
        self.conv3 = nn.Conv2d(channel_out, channel_out, 1, 1, 0, bias=bias)

        self.conv4 = nn.Conv2d(channel_out, channel_out, 3, 1, 1, bias=bias)

        self.conv5 = nn.Conv2d(channel_in, channel_out, 1, 1, 0, bias=bias)

        self.relu = nn.ReLU()

    def forward(self, x):
        x1 = self.relu(self.conv2(self.relu(self.conv1(x))))
        x2 = self.relu(self.conv4(self.relu(self.conv3(x1))))
        x3 = self.conv5(x) + x2
        return x3


class InvBlockPredTran(nn.Module):
    def __init__(self, channel_num):
        super(InvBlockPredTran, self).__init__()

        self.channel_num = channel_num

        self.P1 = InvResBlock(self.channel_num, self.channel_num)
        self.P2 = InvResBlock(self.channel_num * 2, self.channel_num)
        self.P3 = InvResBlock(self.channel_num * 3, self.channel_num)

        self.C = InvResBlock(self.channel_num * 3, self.channel_num)

    def forward(self, x, rev=False):
        if not rev:
            b, c, h, w = x.shape
            oh = h // 2
            ow = w // 2
            oc = c * 4
            x = torch.reshape(x, (b, c, oh, 2, ow, 2))
            x = x.permute((0, 3, 5, 1, 2, 4))
            x = torch.reshape(x, (b, oc, oh, ow))

            x1, x2, x3, x4 = (x.narrow(1, 0, self.channel_num), x.narrow(1, self.channel_num, self.channel_num),
                              x.narrow(1, 2 * self.channel_num, self.channel_num), x.narrow(1, 3 * self.channel_num, self.channel_num))
            y2 = x2 - self.P1(x1)
            y3 = x3 - self.P2(torch.cat((x1, x2), 1))
            y4 = x4 - self.P3(torch.cat((x1, x2, x3), 1))
            y1 = x1 + self.C(torch.cat((y2, y3, y4), 1))

            out = torch.cat((y1, y2, y3, y4), 1)

        else:
            x1, x2, x3, x4 = (x.narrow(1, 0, self.channel_num), x.narrow(1, self.channel_num, self.channel_num),
                              x.narrow(1, 2 * self.channel_num, self.channel_num), x.narrow(1, 3 * self.channel_num, self.channel_num))
            y1 = x1 - self.C(torch.cat((x2, x3, x4), 1))
            y2 = x2 + self.P1(y1)
            y3 = x3 + self.P2(torch.cat((y1, y2), 1))
            y4 = x4 + self.P3(torch.cat((y1, y2, y3), 1))
            out = torch.cat((y1, y2, y3, y4), 1)

            b, c, h, w = out.shape

            oh = h * 2
            ow = w * 2
            oc = c // 4
            out = torch.reshape(out, (b, 2, 2, oc, h, w))
            out = out.permute((0, 3, 4, 1, 5, 2))
            out = torch.reshape(out, (b, oc, oh, ow))

        return out