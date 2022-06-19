import os.path as osp
import logging
import time
import argparse
from collections import OrderedDict
from bitarray import test
import cv2
import numpy as np
import options.options as option
import utils.util as util
from data import create_dataset, create_dataloader
from models import create_model

#### options
parser = argparse.ArgumentParser()
parser.add_argument('-opt', type=str, required=True, help='Path to options YMAL file.')
opt = option.parse(parser.parse_args().opt, is_train=False)
opt = option.dict_to_nonedict(opt)

util.mkdirs((path for key, path in opt['path'].items() if not key == 'experiments_root' and 'pretrain_model' not in key and 'resume' not in key))
util.setup_logger('base', opt['path']['log'], 'test_' + opt['name'], level=logging.INFO, screen=True, tofile=False)
logger = logging.getLogger('base')
# logger.info(option.dict2str(opt))
logger.info('PSNR; SSIM; BPP; VALPSNR; VALSSIM; VALBPP')
#### Create test dataset and dataloader
test_loaders = []
for phase, dataset_opt in sorted(opt['datasets'].items()):
    test_set = create_dataset(dataset_opt)
    test_loader = create_dataloader(test_set, dataset_opt)
    # logger.info('Number of test images in [{:s}]: {:d}'.format(dataset_opt['name'], len(test_set)))
    test_loaders.append(test_loader)

model = create_model(opt)
for test_loader in test_loaders:
    test_set_name = test_loader.dataset.opt['name']
    # logger.info('\nTesting [{:s}]...'.format(test_set_name))
    test_start_time = time.time()
    dataset_dir = osp.join(opt['path']['results_root'], test_set_name)
    util.mkdir(dataset_dir)

    test_results = OrderedDict()
    test_results['psnr'] = []
    test_results['ssim'] = []
    test_results['psnr_val'] = []
    test_results['ssim_val'] = []
    test_results['bpp_net'] = []
    test_results['bpp_val'] = []

    for data in test_loader:
        model.feed_data(data)
        img_path = data['GT_path'][0]
        img_name = osp.splitext(osp.basename(img_path))[0]

        model.test()
        visuals = model.get_current_visuals()

        hr_rec = util.tensor2img(visuals['hr_rec'])  # uint8
        val_img = util.tensor2img(visuals['valnet'])  # uint8
        gt_img = util.tensor2img(visuals['gt'])
        test_results['bpp_net'].append(visuals['bpp_net'])
        test_results['bpp_val'].append(visuals['bpp_fix'])
        test_results['psnr'].append(visuals['PSNR_net'])
        test_results['psnr_val'].append(visuals['PSNR_fix'])

        # save images
        save_img_path = osp.join(dataset_dir, img_name + '_GT.png')
        util.save_img(gt_img, save_img_path)
        save_img_path = osp.join(dataset_dir, img_name + '_rec.png')
        util.save_img(hr_rec, save_img_path)
        save_img_path = osp.join(dataset_dir, img_name + '_val.png')
        util.save_img(val_img, save_img_path)

        # calculate PSNR and SSIM
        ssim = util.calculate_ssim(gt_img, hr_rec)
        test_results['ssim'].append(ssim)
        ssim_val = util.calculate_ssim(gt_img, val_img)
        test_results['ssim_val'].append(ssim_val)
        logger.info('{:10s}'.format(img_name))
        logger.info('{:.6f};{:.6f};{:.6f};{:.6f};{:.6f};{:.6f}.'.format(visuals['PSNR_net'], ssim, visuals['bpp_net'], visuals['PSNR_fix'],
                                                                        ssim_val, visuals['bpp_fix']))

        # print(visuals['PSNR_fix'])
        # print(ssim_val)
        # print(visuals['bpp_fix'])

    # Average PSNR/SSIM results
    ave_psnr = sum(test_results['psnr']) / len(test_results['psnr'])
    ave_ssim = sum(test_results['ssim']) / len(test_results['ssim'])

    ave_psnr_val = sum(test_results['psnr_val']) / len(test_results['psnr_val'])
    ave_ssim_val = sum(test_results['ssim_val']) / len(test_results['ssim_val'])

    avg_bpp = sum(test_results['bpp_net']) / len(test_results['bpp_net'])
    avg_bpp_val = sum(test_results['bpp_val']) / len(test_results['bpp_val'])

    logger.info(
        '----Average PSNR/SSIM results for {}----\n\tpsnr: {:.6f} db; ssim: {:.6f}. psnr_val: {:.6f} db; ssim_val: {:.6f}; bpp: {:.3f}; bpp_val: {:.3f}..\n'
        .format(test_set_name, ave_psnr, ave_ssim, ave_psnr_val, ave_ssim_val, avg_bpp, avg_bpp_val))

    # print(test_results['psnr'])
    # print(test_results['ssim'])
    # print(test_results['bpp_net'])
    print(ave_psnr_val)
    print(ave_ssim_val)
    print(avg_bpp_val)
