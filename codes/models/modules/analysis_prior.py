import torch
import torch.nn as nn
import math
out_channel_N = 64
out_channel_M = 96
# out_channel_N = 128
# out_channel_M = 192
out_channel_mv = 384

class Analysis_prior_net(nn.Module):
    '''
    Compress residual prior
    '''
    def __init__(self):
        super(Analysis_prior_net, self).__init__()
        self.conv1 = nn.Conv2d(out_channel_mv, out_channel_N, 3, stride=1, padding=1)
        torch.nn.init.xavier_normal_(self.conv1.weight.data,
                                     (math.sqrt(2 * (out_channel_M + out_channel_N) / (out_channel_M + out_channel_M))))
        torch.nn.init.constant_(self.conv1.bias.data, 0.01)
        self.relu1 = nn.ReLU()
        self.conv2 = nn.Conv2d(out_channel_N, out_channel_N, 5, stride=2, padding=2)
        torch.nn.init.xavier_normal_(self.conv2.weight.data, math.sqrt(2))
        torch.nn.init.constant_(self.conv2.bias.data, 0.01)
        self.relu2 = nn.ReLU()
        self.conv3 = nn.Conv2d(out_channel_N, out_channel_N, 5, stride=2, padding=2)
        torch.nn.init.xavier_normal_(self.conv3.weight.data, math.sqrt(2))
        torch.nn.init.constant_(self.conv3.bias.data, 0.01)
        # self.priorencoder = nn.Sequential(
        #     nn.Conv2d(out_channel_M, out_channel_N, 3, stride=1, padding=1),
        #     nn.ReLU(),
        #     nn.Conv2d(out_channel_N, out_channel_N, 5, stride=2, padding=2),
        #     nn.ReLU(),
        #     nn.Conv2d(out_channel_N, out_channel_N, 5, stride=2, padding=2)
        # )

    def forward(self, x):
        x = torch.abs(x)
        x = self.relu1(self.conv1(x))
        x = self.relu2(self.conv2(x))
        return self.conv3(x)

class Synthesis_prior_net(nn.Module):
    '''
    Decode residual prior
    '''
    def __init__(self):
        super(Synthesis_prior_net, self).__init__()
        self.deconv1 = nn.ConvTranspose2d(out_channel_N, out_channel_N, 5, stride=2, padding=2, output_padding=1)
        torch.nn.init.xavier_normal_(self.deconv1.weight.data, math.sqrt(2 * 1))
        torch.nn.init.constant_(self.deconv1.bias.data, 0.01)
        self.relu1 = nn.ReLU()
        self.deconv2 = nn.ConvTranspose2d(out_channel_N, out_channel_N, 5, stride=2, padding=2, output_padding=1)
        torch.nn.init.xavier_normal_(self.deconv2.weight.data, math.sqrt(2 * 1))
        torch.nn.init.constant_(self.deconv2.bias.data, 0.01)
        self.relu2 = nn.ReLU()
        self.deconv3 = nn.ConvTranspose2d(out_channel_N, out_channel_mv, 3, stride=1, padding=1)
        torch.nn.init.xavier_normal_(self.deconv3.weight.data,
                                     (math.sqrt(2 * 1 * (out_channel_mv + out_channel_N) / (out_channel_N + out_channel_N))))
        torch.nn.init.constant_(self.deconv3.bias.data, 0.01)
        # self.priordecoder = nn.Sequential(
        #     nn.ConvTranspose2d(out_channel_N, out_channel_N, 5, stride=2, padding=2, output_padding=1),
        #     nn.ReLU(),
        #     nn.ConvTranspose2d(out_channel_N, out_channel_N, 5, stride=2, padding=2, output_padding=1),
        #     nn.ReLU(),
        #     nn.ConvTranspose2d(out_channel_N, out_channel_M, 3, stride=1, padding=1)
        # )

    def forward(self, x):
        x = self.relu1(self.deconv1(x))
        x = self.relu2(self.deconv2(x))
        return torch.exp(self.deconv3(x))
