import os
import sys
import time
import math
from datetime import datetime
import random
import logging
from collections import OrderedDict
from unicodedata import name
import numpy as np
import cv2
import torch
from torch.nn.parallel import DistributedDataParallel
import torch.nn as nn


def get_timestamp():
    return datetime.now().strftime('%y%m%d-%H%M%S')


def setup_logger(logger_name, root, phase, level=logging.INFO, screen=False, tofile=False):
    '''set up logger'''
    lg = logging.getLogger(logger_name)
    formatter = logging.Formatter('%(asctime)s.%(msecs)03d - %(levelname)s: %(message)s', datefmt='%y-%m-%d %H:%M:%S')
    lg.setLevel(level)
    if tofile:
        log_file = os.path.join(root, phase + '_{}.log'.format(get_timestamp()))
        fh = logging.FileHandler(log_file, mode='w')
        fh.setFormatter(formatter)
        lg.addHandler(fh)
    if screen:
        sh = logging.StreamHandler()
        sh.setFormatter(formatter)
        lg.addHandler(sh)


def load_resume_state(ckpt):
    device_id = torch.cuda.current_device()
    resume_state = torch.load(ckpt, map_location=lambda storage, loc: storage.cuda(device_id))


def load_network(load_path, network, strict=True):
    if isinstance(network, nn.DataParallel) or isinstance(network, DistributedDataParallel):
        network = network.module
    load_net = torch.load(load_path)
    load_net_clean = OrderedDict()  # remove unnecessary 'module.'
    for k, v in load_net.items():
        if k.startswith('module.'):
            load_net_clean[k[7:]] = v
        else:
            load_net_clean[k] = v
    network.load_state_dict(load_net_clean)
    name_dict = {}


def load_satet_dict(ckpt):
    load_net = torch.load(ckpt)
    name_list = {}
    for k, v in load_net.items():
        name_list.update({k: v.size()})
    return name_list


if __name__ == '__main__':
    logger = setup_logger('base', '/home/jianghao/Code/Graduation/4k1/test', 'dict_parse', level=logging.INFO, screen=True, tofile=True)
    name_list = load_satet_dict('/home/jianghao/Code/Graduation/PreProcess/experiments/models/1000_G.pth')
    logger = logging.getLogger('base')
    for k, v in name_list.items():
        logger.info('%s : % s' % (k, v))
        # logger.info(v)

# a   10519.1400

# import numpy as np
# import scipy.interpolate
# import re
# import argparse
# import matplotlib.pyplot as plt
# import matplotlib

# rate = []
# f = open('/home/jianghao/Code/Graduation/4k1/videos/460813306/transcode_lanczos_37', mode='r')
# lines = f.readlines()
# line = lines[-4]
# # print(line)
# # re.search("(\d+(\.\d+)) kb", line)
# # print(re.search("a   (\d+(\.\d+))", line).group(1))
# print(float(re.search("a\s+(\d+(\.\d+))", line).group(1)))