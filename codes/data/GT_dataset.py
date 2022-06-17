import os.path as osp
import random
import pickle
import logging
import numpy as np
import cv2
import lmdb
import torch
import torch.utils.data as data
import data.util as util

from pathlib import Path


class GTDataset(data.Dataset):
    '''
    Reading the training Flicker dataset
    key example: 000/00001.png
    GT: Ground-Truth;
    LQ: Low-Quality, e.g., low-resolution frames
    support reading N HR frames, N = 3, 5, 7
    '''
    def __init__(self, opt):
        super(GTDataset, self).__init__()
        self.opt = opt
        self.gt_root = Path(opt['dataroot_GT'])
        self.keys = []
        with open(opt['meta_info_file'], 'r') as fin:
            self.keys = [line.split('\n')[0] for line in fin]
        # begin = int(self.opt['begin'])
        # end = int(self.opt['end']) + 1
        # self.keys.extend([f'{i:06d}' for i in range(begin, end, 1)])

    def __getitem__(self, index):
        key = self.keys[index]
        # center_frame_idx = int(key)
        # self.neighbor_list = [center_frame_idx]
        GT_path = str(self.gt_root / key)
        img_list_GT = []

        crop_border = self.opt['crop_border']  # used for test
        img_GT = util.read_img(None, GT_path, None)
        # BGR => RGB
        img_GT = img_GT[:, :, [2, 1, 0]]
        if crop_border is None:
            img_GT = util.random_crop(img_GT, 256, 256)
        else:
            h, w, _ = img_GT.shape
            h = h // 64 * 64
            w = w // 64 * 64
            img_GT = img_GT[:h, :w, :]
        img_list_GT.append(img_GT)
        if crop_border is None:
            img_list_GT = util.augment(img_list_GT)
        img_list_GT = torch.from_numpy(np.ascontiguousarray(np.transpose(np.concatenate(img_list_GT, axis=2), (2, 0, 1)))).float()
        return {'GT': img_list_GT, 'LQ_path': GT_path, 'GT_path': GT_path}

    def __len__(self):
        return len(self.keys)