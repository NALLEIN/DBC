import os.path as osp
import logging
import time
import argparse
from collections import OrderedDict

import numpy as np
import options.options as option
import utils.util as util
from data.util import bgr2ycbcr
from data import create_dataset, create_dataloader
from models import create_model
import torch


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

    def close(self):
        self.writer.close()


#### options
parser = argparse.ArgumentParser()
parser.add_argument('-opt', type=str, required=True, help='Path to options YMAL file.')
opt = option.parse(parser.parse_args().opt, is_train=False)
opt = option.dict_to_nonedict(opt)

util.mkdirs((path for key, path in opt['path'].items()
             if not key == 'experiments_root' and 'pretrain_model' not in key and 'resume' not in key))
util.setup_logger('base', opt['path']['log'], 'test_' + opt['name'], level=logging.INFO, screen=True, tofile=False)
logger = logging.getLogger('base')
logger.info(option.dict2str(opt))

#### Create test dataset and dataloader
test_loaders = []
for phase, dataset_opt in sorted(opt['datasets'].items()):
    test_set = create_dataset(dataset_opt)
    test_loader = create_dataloader(test_set, dataset_opt)
    logger.info('Number of test images in [{:s}]: {:d}'.format(dataset_opt['name'], len(test_set)))
    test_loaders.append(test_loader)

model = create_model(opt)
for test_loader in test_loaders:
    test_set_name = test_loader.dataset.opt['name']
    logger.info('\nTesting [{:s}]...'.format(test_set_name))
    dataset_dir = osp.join(opt['path']['results_root'], test_set_name)
    util.mkdir(dataset_dir)

    yuvwriter = YUVWriter(osp.join(dataset_dir, 'decode_net.yuv'), (dataset_opt['height'] // 2, dataset_opt['width'] // 2))
    num = test_loader.__len__()
    begin = time.time()
    for data in test_loader:
        sr_img = model.downscale(data['GT'].cuda())
        yuvwriter.write(sr_img)
    yuvwriter.close()
    end = time.time()
    print("test % d frames cost time %.3f s" % (num, end - begin))
    fps = num / (end - begin)
    print("inference speed: %.3f " % fps)
