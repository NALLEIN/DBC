import torch
import torch.nn as nn
import torch.nn.functional as F

from utils import make_coord
from EDSR_arch import make_edsr_baseline
from mlp import MLP


class CRM(nn.Module):
    def __init__(self, n_feats=32, hidden_list=[256, 256]) -> None:
        super(CRM, self).__init__()
        self.n_feats = n_feats
        # self.n_colors = 6
        self.hidden_list = hidden_list
        # self.fea_encoder = make_edsr_baseline(n_colors=self.n_colors, n_feats=self.n_feats)

        mlp_in_dim = self.n_feats + 4
        self.mlp = MLP(in_dim=mlp_in_dim, out_dim=self.n_feats, hidden_list=self.hidden_list)
        self.routing = MLP(in_dim=self.n_feats, out_dim=self.n_feats, hidden_list=[256])
        self.offset = MLP(in_dim=self.n_feats, out_dim=2, hidden_list=[256])

    def resample(self, feat, coord, cell):
        H, W = feat.shape[-2:]
        feat_coord = make_coord(feat.shape[-2:], flatten=False).cuda() \
            .permute(2, 0, 1) \
            .unsqueeze(0).expand(feat.shape[0], 2, *feat.shape[-2:])

        coord_ = coord.clone()
        coord_.clamp_(-1 + 1e-6, 1 - 1e-6)
        q_feat = F.grid_sample(
            feat, coord_.flip(-1).unsqueeze(1),
            mode='bicubic', align_corners=False)[:, :, 0, :] \
            .permute(0, 2, 1)
        q_coord = F.grid_sample(
            feat_coord, coord_.flip(-1).unsqueeze(1),
            mode='bicubic', align_corners=False)[:, :, 0, :] \
            .permute(0, 2, 1)
        rel_coord = coord - q_coord
        rel_coord[:, :, 0] *= feat.shape[-2]
        rel_coord[:, :, 1] *= feat.shape[-1]
        inp = torch.cat([q_feat, rel_coord], dim=-1)

        rel_cell = cell.clone()
        rel_cell[:, :, 0] *= feat.shape[-2]
        rel_cell[:, :, 1] *= feat.shape[-1]
        inp = torch.cat([inp, rel_cell], dim=-1)

        bs, q = coord.shape[:2]
        pred = self.mlp(inp.view(bs * q, -1)).view(bs, q, -1)

        routing_weights = self.routing(pred.view(bs * q, -1)).view(bs, q, -1)
        rel_offset = self.offset(pred.view(bs * q, -1)).view(bs, q, -1)

        coord_ = coord_.flip(-1) + rel_offset
        q_feat1 = F.grid_sample(
            feat, coord_.unsqueeze(1),
            mode='bicubic', align_corners=False)[:, :, 0, :] \
            .permute(0, 2, 1) #(bs, oh * ow, c)
        # res = F.conv2d(q_feat1, routing_weights, padding='same')
        res = torch.mul(q_feat1, routing_weights)
        out = q_feat1 + res
        return out

    def forward(self, x, scale):

        B, C, H, W = x.shape
        H = int(H * scale)
        W = int(W * scale)
        out_coords = make_coord([H, W]).cuda().unsqueeze(0).expand(B, H * W, 2)
        cell = torch.ones_like(out_coords)
        cell[:, 0] *= 2 / H
        cell[:, 1] *= 2 / W

        # feat = self.fea_encoder(x)
        out = self.resample(x, out_coords, cell)  #(bs, oh * ow, c)
        out = out.permute(0, 2, 1)
        out = out.reshape([B, C, H, W])

        return out


if __name__ == "__main__":
    model = CRM().cuda()
    x = torch.zeros([4, 32, 128, 128]).cuda()
    out = model(x, scale=0.5)
    print(out.size())
