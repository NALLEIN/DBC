import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Function
from .utils import default_conv, PixelInvShuffle, PixelShuffle, ResBlock, InvBlockPredTran, RateEstNet, Quantization
from .layers import Win_noShift_Attention, ResidualBlockNoBN
from .GDN import GDNEncoder, GDNDecoder
from .CRM import CRM


class resize_arch(nn.Module):
    def __init__(self, scale):
        super(resize_arch, self).__init__()
        self.scale = scale

        n_feats = 64
        kernel_size = 3
        n_colors = 6
        act = nn.ReLU(True)

        ## Down scale part
        self.fea_encoder = nn.Sequential(
            nn.Conv2d(n_colors, n_feats // 2, kernel_size, padding=(kernel_size // 2)),
            ResBlock(default_conv, n_feats // 2, kernel_size, act=act, res_scale=1),
            nn.Conv2d(n_feats // 2, n_feats, kernel_size, padding=(kernel_size // 2)),
        )
        self.local_atten = nn.Sequential(
            nn.Conv2d(n_feats, n_feats, kernel_size, padding=(kernel_size // 2)),
            Win_noShift_Attention(dim=n_feats, num_heads=8, window_size=8, shift_size=4),
            nn.Conv2d(n_feats, n_feats // 2, kernel_size, padding=(kernel_size // 2)),
        )
        self.GDN = GDNEncoder(192, 192, n_feats)
        self.IGDN = GDNDecoder(192, 192, n_feats // 2)
        self.crm = CRM(n_feats)
        self.res1 = nn.Sequential(
            default_conv(n_feats, n_feats, kernel_size),
            ResBlock(default_conv, n_feats, kernel_size, act=act, res_scale=1),
            ResBlock(default_conv, n_feats, kernel_size, act=act, res_scale=1),
            default_conv(n_feats, n_colors, 1),
        )

        # SR part
        m_body = [ResidualBlockNoBN(num_feat=n_feats, res_scale=1, pytorch_init=True) for _ in range(16)]
        m_body.append(nn.Conv2d(n_feats, n_feats, 3, 1, 1))

        self.conv_first = nn.Conv2d(n_colors, n_feats, 3, 1, 1)
        self.conv_body = nn.Sequential(*m_body)
        self.icrm = CRM(n_feats)
        self.res2 = nn.Sequential(
            ResBlock(default_conv, n_feats, kernel_size, act=act, res_scale=1),
            ResBlock(default_conv, n_feats, kernel_size, act=act, res_scale=1),
            default_conv(n_feats, n_colors, 1),
        )

    def forward(self, x, up=True, I=True):
        if up:
            x = self.conv_first(x)
            res = self.conv_body(x)
            res += x
            x = self.icrm(x, self.scale)
            x = self.res2(x)
            return x
        else:
            b, c, h, w = x.shape
            feat = self.fea_encoder(x)
            cdf = self.GDN(feat)
            cdf = self.IGDN(cdf)
            down = self.local_atten(feat)
            out = torch.cat((down, cdf), axis=1)
            out = self.crm(out, 1 / self.scale)
            out = self.res1(out)
            return out


class VCN(nn.Module):
    def __init__(self, z_feats=64):
        super(VCN, self).__init__()
        self.sacle = 2

        kernel_size = 3
        n_colors = 6
        act = nn.ReLU(True)

        self.scale_net_i = nn.Sequential(
            nn.Conv2d(n_colors, z_feats, kernel_size, padding=(kernel_size // 2)),
            ResBlock(default_conv, z_feats, kernel_size, act=act, res_scale=1),
            Win_noShift_Attention(dim=z_feats, num_heads=8, window_size=8, shift_size=4),
            ResBlock(default_conv, z_feats, kernel_size, act=act, res_scale=1),
            Win_noShift_Attention(dim=z_feats, num_heads=8, window_size=8, shift_size=4),
            ResBlock(default_conv, z_feats, kernel_size, act=act, res_scale=1),
            default_conv(z_feats, n_colors, 1),
        )

        self.deblock_net_i = nn.Sequential(
            nn.Conv2d(n_colors, z_feats, kernel_size, padding=(kernel_size // 2)),
            ResBlock(default_conv, z_feats, kernel_size, act=act, res_scale=1),
            Win_noShift_Attention(dim=z_feats, num_heads=8, window_size=8, shift_size=4),
            ResBlock(default_conv, z_feats, kernel_size, act=act, res_scale=1),
            Win_noShift_Attention(dim=z_feats, num_heads=8, window_size=8, shift_size=4),
            ResBlock(default_conv, z_feats, kernel_size, act=act, res_scale=1),
            default_conv(z_feats, n_colors, 1),
        )

        operations = []
        current_channel = n_colors * 4
        for j in range(2):
            b = InvBlockPredTran(current_channel)
            operations.append(b)
            current_channel *= 4
        self.enc_operations = nn.ModuleList(operations)

        operations = []
        current_channel = n_colors * 4
        for j in range(2):
            b = InvBlockPredTran(current_channel)
            operations.append(b)
            current_channel *= 4
        self.dec_operations = nn.ModuleList(operations)
        self.rate_est = RateEstNet(current_channel)
        # self.rate_est_p = RateEstNet(current_channel)
        self.Quant = Quantization()
        self.invshuffle = PixelInvShuffle(2)
        self.upshuffle = PixelShuffle(2)

    def forward(self, x, mv=None, I=True):
        b, c, h, w = x.shape
        num_pixels = b * h * w
        if (I):
            x = self.scale_net_i(x)
            x = self.invshuffle(x)
            for op in self.enc_operations:
                x = op(x, False)
            # x_rec = self.Quant(x)
            x_enc = x + (torch.rand(x.size()).cuda() - 0.5)
            prob = self.rate_est(x_enc + 0.5) - self.rate_est(x_enc - 0.5)
            rates = torch.sum(torch.clamp(-1.0 * torch.log(prob + 1e-5) / math.log(2.0), 0, 50))
            bpp = rates / num_pixels
            x = x_enc
            for op in reversed(self.dec_operations):
                x = op(x, True)
            x = self.upshuffle(x)
            x = self.deblock_net_i(x)

        else:
            x = self.invshuffle(x)

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
