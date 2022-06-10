from logging import Logger
import logging
from pathlib import Path
import random
import numpy as np
import cv2
import lmdb
import torch
import torch.utils.data as data
import data.util as util


def yuv420_to_yuv444(yuv):
    total_h, w = yuv.shape
    h = int(total_h / 1.5)
    uv_h = h // 4
    _w = int(w / 2)
    _h = int(h / 2)
    y = yuv[0:h, :]

    u = cv2.resize(np.reshape(yuv[h:h + uv_h, :], (_h, _w)), (w, h), interpolation=cv2.INTER_NEAREST)
    v = cv2.resize(np.reshape(yuv[h + uv_h:, :], (_h, _w)), (w, h), interpolation=cv2.INTER_NEAREST)
    return np.stack((y, u, v), axis=2)


class Vimeo90KDataset(data.Dataset):
    """Vimeo90K dataset for training.

    The keys are generated from a meta info txt file.
    basicsr/data/meta_info/meta_info_Vimeo90K_train_GT.txt

    Each line contains:
    1. clip name; 2. frame number; 3. image shape, seperated by a white space.
    Examples:
        00001/0001 7 (256,448,3)
        00001/0002 7 (256,448,3)

    Key examples: "00001/0001"
    GT (gt): Ground-Truth;
    LQ (lq): Low-Quality, e.g., low-resolution/blurry/noisy/compressed frames.

    The neighboring frame list for different num_frame:
    num_frame | frame list
             1 | 4
             3 | 3,4,5
             5 | 2,3,4,5,6
             7 | 1,2,3,4,5,6,7

    Args:
        opt (dict): Config for train dataset. It contains the following keys:
            dataroot_gt (str): Data root path for gt.
            dataroot_lq (str): Data root path for lq.
            meta_info_file (str): Path for meta information file.
            io_backend (dict): IO backend type and other kwarg.

            num_frame (int): Window size for input frames.
            gt_size (int): Cropped patched size for gt patches.
            random_reverse (bool): Random reverse input frames.
            use_flip (bool): Use horizontal flips.
            use_rot (bool): Use rotation (use vertical flip and transposing h
                and w for implementation).

            scale (bool): Scale, which will be added automatically.
    """
    def __init__(self, opt):
        super(Vimeo90KDataset, self).__init__()
        self.opt = opt

        self.gt_root = Path(opt['dataroot_GT'])
        self.random_reverse = opt['random_reverse']
        with open(opt['meta_info_file'], 'r') as fin:
            self.keys = [line.split(' ')[0] for line in fin]
        # indices of input images
        self.neighbor_list = [i + (9 - opt['num_frame']) // 2 for i in range(opt['num_frame'])]

    def __getitem__(self, index):
        # random reverse
        if self.random_reverse and random.random() < 0.5:
            self.neighbor_list.reverse()

        scale = self.opt['scale']
        GT_size = self.opt['GT_size']

        key = self.keys[index]
        clip, seq = key.split('/')  # key example: 00001/0001

        # get the neighboring LQ frames
        img_list_GT = []
        img_list_LQ = []
        for neighbor in self.neighbor_list:
            num = "im" + str(neighbor) + ".png"
            GT_path = str(self.gt_root / clip / seq / num)
            img_GT = util.read_img(None, GT_path, None)
            # if (GT_size == None):
            #     GT_size = 2

            # img_GT = util.modcrop(img_GT, GT_size)
            # img_GT = util.bgr2ycbcr(img_GT,False)
            # img_LQ = util.imresize_np(img_GT, 1 / scale, True)
            crop_border = self.opt['crop_border']
            if crop_border:
                img_GT = img_GT[crop_border:-crop_border, crop_border:-crop_border, :]
            # BGR => RGB
            img_GT = img_GT[:, :, [2, 1, 0]]
            img_list_GT.append(img_GT)

            # img_list_LQ.append(img_LQ)

        # img_list_LQ = [img_LQ[:, :, [2, 1, 0]] for img_LQ in img_list_LQ]
        # img_list_GT = [img_GT[:, :, [2, 1, 0]] for img_GT in img_list_GT]
        img_list_GT = torch.from_numpy(np.ascontiguousarray(np.transpose(np.concatenate(img_list_GT, axis=2), (2, 0, 1)))).float()
        # img_list_LQ = torch.from_numpy(np.ascontiguousarray(np.transpose(np.concatenate(img_list_LQ,axis=2), (2, 0, 1)))).float()
        GT_path = str(self.gt_root / (clip + '_' + seq))

        # img_list_GT.size() : [3, 252, 444]
        return {'GT': img_list_GT, 'LQ_path': GT_path, 'GT_path': GT_path}

    def __len__(self):
        return len(self.keys)
