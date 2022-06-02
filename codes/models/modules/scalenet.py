import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch.autograd import Function
from .GDN import GDNEncoder, GDNDecoder
# from .analysis_prior import *
# from compressai.models import CompressionModel
# from compressai.entropy_models import EntropyBottleneck, GaussianConditional
# from compressai.models.utils import update_registered_buffers
from .layers import Win_noShift_Attention
from .module_util import *


class scalenet(nn.Module):
    def __init__(self):
        super(scalenet, self).__init__()
        n_resblocks = 16
        n_feats = 64
        kernel_size = 3
        scale = 2
        n_colors = 6
        act = nn.ReLU(True)
        self.intraframe_downscaler = nn.Sequential(
            PixelInvShuffle(2),
            nn.Conv2d(4 * n_colors, n_feats, kernel_size, padding=(kernel_size // 2)),
        )
        self.local_atten = nn.Sequential(
            nn.Conv2d(n_feats, n_feats, kernel_size, padding=(kernel_size // 2)),
            Win_noShift_Attention(dim=n_feats, num_heads=8, window_size=8, shift_size=4),
            nn.Conv2d(n_feats, n_feats, kernel_size, padding=(kernel_size // 2)),
            Win_noShift_Attention(dim=n_feats, num_heads=8, window_size=8, shift_size=4),
            nn.Conv2d(n_feats, n_feats, kernel_size, padding=(kernel_size // 2)),
        )
        self.res_net = nn.Sequential(
            ResBlock(default_conv, n_feats + 16, kernel_size, act=act, res_scale=1),
            ResBlock(default_conv, n_feats + 16, kernel_size, act=act, res_scale=1),
            default_conv(n_feats + 16, n_colors, 1),
        )
        # self.refine_net = nn.Sequential(nn.Conv2d(n_colors, n_feats, kernel_size, padding=(kernel_size // 2)),
        #                                 ResBlock(default_conv, n_feats, kernel_size, act=act, res_scale=1),
        #                                 ResBlock(default_conv, n_feats, kernel_size, act=act, res_scale=1),
        #                                 ResBlock(default_conv, n_feats, kernel_size, act=act, res_scale=1),
        #                                 ResBlock(default_conv, n_feats, kernel_size, act=act, res_scale=1),
        #                                 default_conv(n_feats, n_colors, 1))
        self.GDN = GDNEncoder(192, 192, n_feats)
        self.IGDN = GDNDecoder(192, 192, 16)

    def forward(self, x, up=True, flow=None, I=None):
        if up:
            if (I):
                '''x_I = self.I_head(x[:, :3, :, :])
                res_I = self.I_body(x_I)
                res_I += x_I
                x_warp = self.I_tail(res_I)'''
                x_warp = torch.cat((self.intra_head(x[:, :3, :, :]), self.inter_head(x[:, :3, :, :])), dim=1)
                res = self.body(x_warp)
                res += x_warp
                x_warp = self.tail(res)
            else:
                flow = flow.repeat_interleave(16, dim=1)
                flow = self.upshuffle(flow) / 4
                flow = flow[:, :, :x.size(2), :x.size(3)]
                flow[:, 0, :, :] = flow[:, 0, :, :] / ((flow.size(3) - 1) / 2.0)
                flow[:, 1, :, :] = flow[:, 1, :, :] / ((flow.size(2) - 1) / 2.0)

                predict_flow = self.refine_flow(
                    torch.cat(
                        (self.predict_flow(torch.cat(
                            (self.warp(self.flow_head(x[:, :3, :, :]), flow, True), self.flow_head(x[:, 3:6, :, :])), dim=1)), flow), 1))

                x_warp = torch.cat((self.warp(self.intra_head(x[:, :3, :, :]), predict_flow, False), self.inter_head(x[:, 3:6, :, :])), dim=1)
                res = self.body(x_warp)
                res += x_warp
                x_warp = self.tail(res)
            return x_warp
        else:
            b, c, h, w = x.shape
            down = self.intraframe_downscaler(x)
            cdf = self.GDN(down)
            cdf = self.IGDN(cdf)
            down = self.local_atten(down)
            out = torch.cat((down, cdf), axis=1)
            out = self.res_net(out)
            return out


class analysis_arch(nn.Module):
    def __init__(self):
        super(analysis_arch, self).__init__()
        z_feats = 64
        kernel_size = 3
        scale = 2
        n_colors = 6
        act = nn.ReLU(True)

        operations = []
        current_channel = n_colors * 4
        for j in range(3):
            b = InvBlockPredTran(current_channel)
            operations.append(b)
            current_channel *= 4
        self.enc_operations = nn.ModuleList(operations)

        operations = []
        current_channel = n_colors * 4
        for j in range(3):
            b = InvBlockPredTran(current_channel)
            operations.append(b)
            current_channel *= 4
        self.dec_operations = nn.ModuleList(operations)

        # self.rate_est = RateEstNet(current_channel)
        # self.rate_est_p = RateEstNet(current_channel)

        # self.Quant = Quantization()
        self.PixelInvShuffle = PixelInvShuffle(2)
        self.upshuffle = PixelShuffle(2)
        # self.deblock_net = nn.Sequential(nn.Conv2d(2 * n_colors, z_feats, kernel_size, padding=(kernel_size // 2)),
        #                                  ResBlock(default_conv, z_feats, kernel_size, act=act, res_scale=1),
        #                                  ResBlock(default_conv, z_feats, kernel_size, act=act, res_scale=1),
        #                                  ResBlock(default_conv, z_feats, kernel_size, act=act, res_scale=1), default_conv(z_feats, 2 * n_colors, 1))
        # self.warp = WarpingLayer()

        self.deblock_net_i = nn.Sequential(nn.Conv2d(n_colors, z_feats, kernel_size, padding=(kernel_size // 2)),
                                           ResBlock(default_conv, z_feats, kernel_size, act=act, res_scale=1),
                                           Win_noShift_Attention(dim=z_feats, num_heads=8, window_size=8, shift_size=4),
                                           ResBlock(default_conv, z_feats, kernel_size, act=act, res_scale=1),
                                           Win_noShift_Attention(dim=z_feats, num_heads=8, window_size=8, shift_size=4),
                                           ResBlock(default_conv, z_feats, kernel_size, act=act, res_scale=1), default_conv(z_feats, n_colors, 1))

        self.scale_net_i = nn.Sequential(nn.Conv2d(n_colors, z_feats, kernel_size, padding=(kernel_size // 2)),
                                         ResBlock(default_conv, z_feats, kernel_size, act=act, res_scale=1),
                                         Win_noShift_Attention(dim=z_feats, num_heads=8, window_size=8, shift_size=4),
                                         ResBlock(default_conv, z_feats, kernel_size, act=act, res_scale=1),
                                         Win_noShift_Attention(dim=z_feats, num_heads=8, window_size=8, shift_size=4),
                                         ResBlock(default_conv, z_feats, kernel_size, act=act, res_scale=1), default_conv(z_feats, n_colors, 1))

    def forward(self, x, mv, I):
        b, c, h, w = x.shape
        num_pixels = b * h * w
        if (I):
            x = self.scale_net_i(x)
            x = self.PixelInvShuffle(x)
            for op in self.enc_operations:
                x = op(x, False)
            # x_rec = self.Quant(x)
            # x_enc = x + (torch.rand(x.size()).cuda() - 0.5)
            # prob = self.rate_est(x_enc + 0.5) - self.rate_est(x_enc - 0.5)
            # rates = torch.sum(torch.clamp(-1.0 * torch.log(prob + 1e-5) / math.log(2.0), 0, 50))
            # bpp = rates / num_pixels
            # x = x_enc
            bpp = None
            for op in reversed(self.dec_operations):
                x = op(x, True)
            x = self.upshuffle(x)
            x = self.deblock_net_i(x)

        else:
            x = self.PixelInvShuffle(x)

            mv = mv[:, 2:, :x.size(2), :x.size(3)]
            mv[:, 0, :, :] = mv[:, 0, :, :] / 16 / (mv.size(3) / 2.0)
            mv[:, 1, :, :] = mv[:, 1, :, :] / 16 / (mv.size(2) / 2.0)

            xI = x[:, :48, :, :]
            xP = x[:, 48:, :, :]

            xP = torch.cat((xP, self.warp(xI, mv)), dim=1)
            for op in self.p_enc_operations:
                xP = op(xP, False)
            xP = torch.cat((self.Quant(torch.clamp(xP[:, :768, :, :], -16, 16) * 255) / 255, xP[:, 768:, :, :]), 1)
            prob = self.rate_est_p(xP[:, :768, :, :] * 255 + 0.5) - self.rate_est_p(xP[:, :768, :, :] * 255 - 0.5)
            rates = torch.sum(torch.clamp(-1.0 * torch.log(prob + 1e-5) / math.log(2.0), 0, 50))
            bpp = rates / num_pixels
            for op in reversed(self.p_dec_operations):
                xP = op(xP, True)
            xP = xP[:, :48, :, :]

            x = self.upshuffle(xP)
            x = self.deblock_net_p(x)
        return x, bpp

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
