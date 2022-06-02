import os.path as osp
import logging
import time
import argparse
from collections import OrderedDict

import numpy as np
from torch.nn.modules.upsampling import Upsample
import options.options as option
import utils.util as util
from data.util import bgr2ycbcr
from data import create_dataset, create_dataloader
from models import create_model
import torch
import torch.nn.functional as F
import cv2
import os
import av


def rgb2yuv420(rgb):
    b, c, height, width = rgb.shape
    rgb = (torch.clamp(rgb, 0, 1) * 255).clone().detach().cpu().numpy()
    rgb = np.round(np.transpose(rgb, (0, 2, 3, 1))).astype(np.uint8)
    yuv420p = (av.VideoFrame.from_ndarray(rgb[0, :, :, :], format='rgb24')).reformat(format='yuv420p').to_ndarray()
    img_cuda = np.array(yuv420p).astype(np.float32) / 255.
    img_cuda = torch.tensor(img_cuda).cuda()
    y_tensor = torch.nn.functional.pixel_unshuffle(img_cuda[:height, :].view(1, 1, height, width), 2)
    u_tensor = img_cuda[height:height + height // 4, :].view(1, 1, height // 2, width // 2)
    v_tensor = img_cuda[height + height // 4:, :].view(1, 1, height // 2, width // 2)
    out = torch.cat((y_tensor, u_tensor, v_tensor), 1)
    return out


def yuv4202rgb(yuv):
    b, c, height, width = yuv.shape
    img = yuv[0, :, :, :].view(1, 6, height, width).clone().detach()
    y_tensor = torch.nn.functional.pixel_shuffle(img[:, :4, :, :], 2)[0, 0, :, :].cpu().numpy()
    u_tensor = np.reshape(img[0, 4, :, :].cpu().numpy(), (height // 2, width * 2))
    v_tensor = np.reshape(img[0, 5, :, :].cpu().numpy(), (height // 2, width * 2))
    img = np.round(np.clip(np.concatenate((y_tensor, u_tensor, v_tensor), axis=0), 0, 1) * 255.).astype(np.uint8)
    rgb = (av.VideoFrame.from_ndarray(img, format='yuv420p')).reformat(format='rgb24').to_ndarray()
    img_cuda = np.array(rgb).astype(np.float32) / 255.
    img_cuda = torch.tensor(img_cuda).cuda().view(1, height * 2, width * 2, 3).permute(0, 3, 1, 2)
    out = img_cuda.view(1, 3, height * 2, width * 2)
    return out


def to_yuv420(yuv):
    total_h, w = yuv.shape
    h = int(total_h / 1.5)
    uv_h = h // 4
    _w = int(w / 2)
    _h = int(h / 2)
    y = yuv[0:h, :]
    y = np.reshape(y, (_h, 2, _w, 2))
    y = np.transpose(y, (0, 2, 1, 3))
    y = np.reshape(y, (_h, _w, 4))
    u = np.reshape(yuv[h:h + uv_h, :], (_h, _w, 1))
    v = np.reshape(yuv[h + uv_h:, :], (_h, _w, 1))
    return np.concatenate((y, u, v), axis=2)


def pixel_shuffle(tensor, scale_factor):
    """
    Implementation of pixel shuffle using numpy

    Parameters:
    -----------
    tensor: input tensor, shape is [N, C, H, W]
    scale_factor: scale factor to up-sample tensor

    Returns:
    --------
    tensor: tensor after pixel shuffle, shape is [N, C/(s*s), s*H, s*W],
        where s refers to scale factor
    """
    num, ch, height, width = tensor.shape
    if ch % (scale_factor * scale_factor) != 0:
        raise ValueError('channel of tensor must be divisible by ' '(scale_factor * scale_factor).')

    new_ch = ch // (scale_factor * scale_factor)
    new_height = height * scale_factor
    new_width = width * scale_factor

    tensor = tensor.reshape([num, new_ch, scale_factor, scale_factor, height, width])
    # new axis: [num, new_ch, height, scale_factor, width, scale_factor]
    tensor = tensor.transpose([0, 1, 4, 2, 5, 3])
    tensor = tensor.reshape([num, new_ch, new_height, new_width])
    return tensor


def pixel_shuffle_inv(tensor, scale_factor):
    """
    Implementation of inverted pixel shuffle using numpy

    Parameters:
    -----------
    tensor: input tensor, shape is [N, C, H, W]
    scale_factor: scale factor to down-sample tensor

    Returns:
    --------
    tensor: tensor after pixel shuffle, shape is [N, (s*s)*C, H/s, W/s],
        where s refers to scale factor
    """
    num, ch, height, width = tensor.shape
    if height % scale_factor != 0 or width % scale_factor != 0:
        raise ValueError('height and widht of tensor must be divisible by ' 'scale_factor.')

    new_ch = ch * (scale_factor * scale_factor)
    new_height = height // scale_factor
    new_width = width // scale_factor

    tensor = tensor.reshape([num, ch, new_height, scale_factor, new_width, scale_factor])
    # new axis: [num, ch, scale_factor, scale_factor, new_height, new_width]
    tensor = tensor.transpose([0, 1, 3, 5, 2, 4])
    tensor = tensor.reshape([num, new_ch, new_height, new_width])
    return tensor



class YUVWriter(object):
    def __init__(self, video_out, video_size):
        self.size = video_size
        self.out = video_out
        self.down_sampler = torch.nn.AvgPool2d(kernel_size=2, stride=2).cuda()
        self.writer = open(video_out, 'wb')

    def write(self, _tensor):
        y_tensor = np.round(
            np.clip(
                torch.nn.functional.pixel_shuffle(_tensor[:, :4, :, :], 2).squeeze().cpu().detach().numpy() * 255.0, 0, 255))
        uv_tensor = np.round(np.clip(_tensor[:, 4:, :, :].squeeze(0).cpu().detach().numpy() * 255.0, 0, 255))

        u_tensor = np.reshape(uv_tensor[0, :, :], (self.size[0] // 4, self.size[1]))
        v_tensor = np.reshape(uv_tensor[1, :, :], (self.size[0] // 4, self.size[1]))
        yuv_tensor = np.concatenate((y_tensor, u_tensor, v_tensor), axis=0).astype(np.uint8)

        yuv_frame = yuv_tensor.reshape((yuv_tensor.shape[0] * yuv_tensor.shape[1], ))
        binary_frame = yuv_frame.tostring()
        self.writer.write(binary_frame)

    def writenparray(self, _nparray):
        y = np.squeeze(pixel_shuffle(_nparray[:, :4, :, :], 2))
        uv = np.squeeze(_nparray[:, 4:, :, :], axis=0)

        u = np.reshape(uv[0, :, :], (self.size[0] // 4, self.size[1]))
        v = np.reshape(uv[1:, :, :], (self.size[0] // 4, self.size[1]))
        yuv = np.concatenate((y, u, v), axis=0).astype(np.uint8)

        yuv_frame = yuv.reshape((yuv.shape[0] * yuv.shape[1], ))
        binary_frame = yuv_frame.tostring()
        self.writer.write(binary_frame)

    def close(self):
        self.writer.close()


class DownSample():
    def __init__(self, video_path, opt) -> None:
        self.opt = opt
        self.frame_height = opt['SH']
        self.frame_width = opt['SW']
        self.yuv_path = video_path
        self.video_name = video_path.split('/')[-1].split('.')[0]
        self.frame_size = self.frame_height * self.frame_width
        self.frame_length = int((self.frame_size * 3) / 2)
        self.video_size = os.stat(self.yuv_path)[6]
        self.num_frames = int(self.video_size / self.frame_length)
        self.f = open(self.yuv_path, 'rb')
        self.vid_arr = []
        self.yuvwriter = YUVWriter(osp.join(opt['basepath'], self.video_name + '_down.yuv'), (opt['LH'], opt['LW']))

    def read(self):
        raw = self.f.read(self.frame_length)
        yuv = np.frombuffer(raw, dtype=np.uint8)
        yuv = yuv.reshape((int(self.frame_height * 1.5), self.frame_width))
        yuv = to_yuv420(yuv)
        return yuv

    def yuvResize(self, yuv, ow, oh):
        b, c, h, w = yuv.shape
        y = pixel_shuffle(yuv[:, :4, :, :], 2).squeeze()
        u = yuv[:, 4, :, :].reshape(h // 2, w * 2)
        v = yuv[:, 5, :, :].reshape(h // 2, w * 2)
        img = np.concatenate((y, u, v), axis=0)
        w = ow
        h = oh

        yuv = (av.VideoFrame.from_ndarray(img, format='yuv420p')).reformat(width=w, height=h,
                                                                           interpolation='LANCZOS').to_ndarray()
        # print('img shape:', img.shape)
        # print('oh ,ow', oh, ow)
        # print('yuv shape:', yuv.shape)
        # img shape: (1620, 1920)
        # oh ,ow 360 640
        # yuv shape: (540, 640)
        y = pixel_shuffle_inv(yuv[:h, :].reshape(1, 1, h, w), 2)
        u = yuv[h:h + h // 4, :].reshape(1, 1, h // 2, w // 2)
        v = yuv[h + h // 4:, :].reshape(1, 1, h // 2, w // 2)
        out = np.concatenate((y, u, v), 1)
        # NCHW
        return out

    def sample(self):
        for i in range(self.num_frames):
            frame = self.read()
            frame = np.transpose(frame, (2, 0, 1))
            frame = np.expand_dims(frame, axis=0)
            frame = self.yuvResize(frame, self.opt['LW'], self.opt['LH'])
            self.yuvwriter.writenparray(frame)
        self.yuvwriter.close()


class UpSample():
    def __init__(self, video_path, opt) -> None:
        self.opt = opt
        self.frame_height = opt['LH']
        self.frame_width = opt['LW']
        self.yuv_path = video_path
        self.video_name = video_path.split('/')[-1].split('.')[0]
        self.frame_size = self.frame_height * self.frame_width
        self.frame_length = int((self.frame_size * 3) / 2)
        self.video_size = os.stat(self.yuv_path)[6]
        self.num_frames = int(self.video_size / self.frame_length)
        self.f = open(self.yuv_path, 'rb')
        self.vid_arr = []
        self.yuvwriter = YUVWriter(osp.join(opt['basepath'], self.video_name + '_up.yuv'), (opt['SH'], opt['SW']))

    def read(self):
        raw = self.f.read(self.frame_length)
        yuv = np.frombuffer(raw, dtype=np.uint8)
        yuv = yuv.reshape((int(self.frame_height * 1.5), self.frame_width))
        frame = to_yuv420(yuv)
        return frame

    def yuvUpsample(self, x, h, w):
        y_tensor = torch.nn.functional.pixel_unshuffle(
            F.interpolate(torch.nn.functional.pixel_shuffle(x[:, :4, :, :], 2), size=[2 * h, 2 * w], mode='bicubic'), 2)
        uv_tensor = F.interpolate(x[:, 4:, :, :], size=[h, w], mode='bicubic')
        sr = torch.cat((y_tensor, uv_tensor), 1)
        return sr

    def sample(self):
        for i in range(self.num_frames):
            frame = self.read()
            frame = torch.from_numpy(np.ascontiguousarray(np.transpose(frame, (2, 0, 1)))).float() / 255.0
            frame = frame.unsqueeze(0)
            frame = self.yuvUpsample(frame, self.opt['SH'] // 2, self.opt['SW'] // 2)
            self.yuvwriter.write(frame)
        self.yuvwriter.close()


#### options
parser = argparse.ArgumentParser()
parser.add_argument('-opt',
                    type=str,
                    default='/home/jianghao/Code/Graduation/4k1/codes/options/test/scaler.yml',
                    help='Path to options YMAL file.')
parser.add_argument('-video_path',
                    type=str,
                    default='/media/jianghao/Elements/Code_archive/PreProcess/test/decode.yuv',
                    help='Path to yuv video.')
parser.add_argument('-mode', choices=['down', 'up'], help='down or up scale')
args = parser.parse_args()
opt = option.parse(args.opt, is_train=False)
opt = option.dict_to_nonedict(opt)

util.setup_logger('base', opt['path']['log'], 'test_' + opt['name'], level=logging.INFO, screen=True, tofile=False)
logger = logging.getLogger('base')
logger.info(option.dict2str(opt))

if args.mode == 'up':
    sample = UpSample(args.video_path, opt)
    sample.sample()
elif args.mode == 'down':
    sample = DownSample(args.video_path, opt)
    sample.sample()