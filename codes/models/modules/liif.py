import torch
import torch.nn as nn
import torch.nn.functional as F

from utils import make_coord
from EDSR_arch import make_edsr_baseline
from codes.models.modules.mlp import MLP


class LIIF(nn.Module):
    def __init__(self, use_mlp=True, feat_unfold=True, hidden_list=[256, 256, 256, 256]) -> None:
        super(LIIF, self).__init__()
        self.n_feats = 64
        self.n_colors = 6
        self.feat_unfold = feat_unfold
        self.hidden_list = hidden_list
        self.fea_encoder = make_edsr_baseline(n_colors=self.n_colors, n_feats=self.n_feats)
        # self.mlp = MLP(in_dim=self.n_feats + 4, out_dim=self.n_colors, hidden_list=self.hidden_list)

        if use_mlp:
            mlp_in_dim = self.n_feats
            if self.feat_unfold:
                mlp_in_dim *= 9
            mlp_in_dim += 4  # attach coord and scale
            self.mlp = MLP(in_dim=mlp_in_dim, out_dim=self.n_colors, hidden_list=self.hidden_list)
        else:
            self.mlp = None

    def resample(self, feat, coord, cell):

        if self.mlp is None:
            ret = F.grid_sample(feat, coord.flip(-1).unsqueeze(1),
                                mode='bicubic', align_corners=False)[:, :, 0, :] \
                                .permute(0, 2, 1)
            return ret

        if self.feat_unfold:
            feat = F.unfold(feat, 3, padding=1).view(feat.shape[0], feat.shape[1] * 9, feat.shape[2], feat.shape[3])

        vx_lst = [-1, 1]
        vy_lst = [-1, 1]
        eps_shift = 1e-6
        # field radius (global: [-1, 1])
        rx = 2 / feat.shape[-2] / 2
        ry = 2 / feat.shape[-1] / 2
        feat_coord = make_coord(feat.shape[-2:], flatten=False).cuda() \
            .permute(2, 0, 1) \
            .unsqueeze(0).expand(feat.shape[0], 2, *feat.shape[-2:])
        preds = []
        areas = []
        for vx in vx_lst:
            for vy in vy_lst:
                coord_ = coord.clone()
                coord_[:, :, 0] += vx * rx + eps_shift
                coord_[:, :, 1] += vy * ry + eps_shift
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

                print(inp.shape)

                pred = self.mlp(inp.view(bs * q, -1)).view(bs, q, -1)
                preds.append(pred)

                area = torch.abs(rel_coord[:, :, 0] * rel_coord[:, :, 1])
                areas.append(area + 1e-9)

        tot_area = torch.stack(areas).sum(dim=0)

        t = areas[0]
        areas[0] = areas[3]
        areas[3] = t
        t = areas[1]
        areas[1] = areas[2]
        areas[2] = t
        ret = 0
        for pred, area in zip(preds, areas):
            ret = ret + pred * (area / tot_area).unsqueeze(-1)
        return ret

    def forward(self, x, scale):

        B, C, H, W = x.shape
        H = int(H * scale)
        W = int(W * scale)
        out_coords = make_coord([H, W]).cuda().unsqueeze(0).expand(B, H * W, 2)
        cell = torch.ones_like(out_coords)
        cell[:, 0] *= 2 / H
        cell[:, 1] *= 2 / W

        feat = self.fea_encoder(x)
        out = self.resample(feat, out_coords, cell)
        out = out.permute(0, 2, 1).view(B, C, H, W)
        return out


if __name__ == "__main__":
    model = LIIF().cuda()
    x = torch.zeros([4, 6, 128, 128]).cuda()
    out = model(x, scale=0.5)
    print(out.size())
