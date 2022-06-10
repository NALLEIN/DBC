# from codes.models.modules.resize import ResizeParamNet
import torch
import torch.nn as nn


device = torch.device('cuda')
if __name__ == '__main__':
    x = torch.randn(8, 3, 64, 64)
    y = x.reshape(8, 3, 64, 8, 8)
    print(y.shape)
    # x.to(device)
    # net = ResizeParamNet()
    # y = net(x)
    # print(y)
