import torch.nn as nn


class MLP(nn.Module):
    def __init__(self, in_dim, out_dim, hidden_list):
        super().__init__()
        layers = []
        lastv = in_dim
        for hidden in hidden_list:
            layers.append(nn.Linear(lastv, hidden))
            layers.append(nn.ReLU())
            lastv = hidden
        layers.append(nn.Linear(lastv, out_dim))
        self.layers = nn.Sequential(*layers)

    def forward(self, x):
        shape = x.shape[:-1]
        x = self.layers(x.view(-1, x.shape[-1]))
        return x.view(*shape, -1)


# import torch
# if __name__ == '__main__':
#     x = torch.zeros([4, 6, 16, 16]).cuda()
#     B, C, H, W = x.shape
#     x = x.permute(0, 2, 3, 1)
#     model = MLP(6, 6, [128, 128]).cuda()
#     out = model(x.reshape(B * H * W, C)).view(B, H, W, C).permute(0, 3, 1, 2)
#     print(out.size())
