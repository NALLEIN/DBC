import numpy as np
import torch
import torch.utils.data as data
import data.util as util
import random

class GTDataset(data.Dataset):
    '''Read LR images only in the test phase.'''

    def __init__(self, opt):
        super(GTDataset, self).__init__()
        self.opt = opt
        self.data_type = self.opt['data_type']
        self.paths_LQ, self.paths_GT = None, None
        self.sizes_LQ, self.sizes_GT = None, None
        self.LQ_env, self.GT_env = None, None  # environment for lmdb

        self.paths_GT, self.sizes_GT = util.get_image_paths(self.data_type, opt['dataroot_GT'])
        self.paths_LQ, self.sizes_LQ = util.get_image_paths(self.data_type, opt['dataroot_LQ'])
        assert self.paths_GT, 'Error: GT path is empty.'
        if self.paths_LQ and self.paths_GT:
            assert len(self.paths_LQ) == len(
                self.paths_GT
            ), 'GT and LQ datasets have different number of images - {}, {}.'.format(
                len(self.paths_LQ), len(self.paths_GT))
        self.random_scale_list = [1]



    def __getitem__(self, index):
        GT_path = self.paths_GT[index]
        scale = self.opt['scale']
        GT_size = self.opt['GT_size']

        resolution = None

        img_GT = util.read_img(self.GT_env, GT_path, resolution)

        if self.opt['color']:
            img_GT = util.channel_convert(img_GT.shape[2], self.opt['color'], [img_GT])[0]


        if img_GT.shape[2] == 3:
            img_GT = img_GT[:, :, [2, 1, 0]]
        if (GT_size):
            H_s, W_s, _ = img_GT.shape
            hs = random.randint(0,H_s-GT_size-1)
            ws = random.randint(0,W_s-GT_size-1)
            img_GT = img_GT[hs:hs+GT_size,ws:ws+GT_size,:]

        img_GT = torch.from_numpy(np.ascontiguousarray(np.transpose(img_GT, (2, 0, 1)))).float()
        return {'GT': img_GT, 'GT_path': GT_path}

    def __len__(self):
        return len(self.paths_GT)

