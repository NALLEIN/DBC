import os
import math
import argparse
import random
import logging

import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from data.data_sampler import DistIterSampler
from data.util import bgr2ycbcr

import options.options as option
from utils import util
from data import create_dataloader, create_dataset
from models import create_model


def init_dist(backend='nccl', **kwargs):
    ''' initialization for distributed training'''
    # if mp.get_start_method(allow_none=True) is None:
    if mp.get_start_method(allow_none=True) != 'spawn':
        mp.set_start_method('spawn')
    rank = int(os.environ['RANK'])
    num_gpus = torch.cuda.device_count()
    torch.cuda.set_device(rank % num_gpus)
    dist.init_process_group(backend=backend, **kwargs)


def main():
    #### options
    parser = argparse.ArgumentParser()
    parser.add_argument('-opt',
                        type=str,
                        default='codes/options/train/train_resize.yml',
                        help='Path to option YMAL file.')
    parser.add_argument('--launcher', choices=['none', 'pytorch'], default='none', help='job launcher')
    parser.add_argument('--local_rank', type=int, default=0)
    args = parser.parse_args()
    opt = option.parse(args.opt, is_train=True)
    #### distributed training settings
    if args.launcher == 'none':  # disabled distributed training
        opt['dist'] = False
        rank = -1
        print('Disabled distributed training.')
    else:
        opt['dist'] = True
        init_dist()
        world_size = torch.distributed.get_world_size()
        rank = torch.distributed.get_rank()

    #### loading resume state if exists
    if opt['path'].get('resume_state', None):
        # distributed resuming: all load into default GPU
        device_id = torch.cuda.current_device()
        resume_state = torch.load(opt['path']['resume_state'], map_location=lambda storage, loc: storage.cuda(device_id))
        option.check_resume(opt, resume_state['iter'])  # check resume options
    else:
        resume_state = None

    #### mkdir and loggers
    if rank <= 0:  # normal training (rank -1) OR distributed training (rank 0)
        if resume_state is None:
            util.mkdir_and_rename(opt['path']['experiments_root'])  # rename experiment folder if exists
            util.mkdirs((path for key, path in opt['path'].items()
                         if not key == 'experiments_root' and 'pretrain_model' not in key and 'resume' not in key))

        # config loggers. Before it, the log will not work
        util.setup_logger('base', opt['path']['log'], 'train_' + opt['name'], level=logging.INFO, screen=True, tofile=True)
        util.setup_logger('val', opt['path']['log'], 'val_' + opt['name'], level=logging.INFO, screen=True, tofile=True)
        logger = logging.getLogger('base')
        logger.info(option.dict2str(opt))
        # tensorboard logger

    else:
        util.setup_logger('base', opt['path']['log'], 'train', level=logging.INFO, screen=True)
        logger = logging.getLogger('base')

    # convert to NoneDict, which returns None for missing keys
    opt = option.dict_to_nonedict(opt)

    #### random seed
    seed = opt['train']['manual_seed']
    if seed is None:
        seed = random.randint(1, 10000)
    if rank <= 0:
        logger.info('Random seed: {}'.format(seed))
    util.set_random_seed(seed)

    torch.backends.cudnn.benchmark = True
    # torch.backends.cudnn.deterministic = True

    #### create train and val dataloader
    dataset_ratio = 100  # enlarge the size of each epoch
    for phase, dataset_opt in opt['datasets'].items():
        if phase == 'train':
            train_set = create_dataset(dataset_opt)
            train_size = int(math.ceil(len(train_set) / dataset_opt['batch_size']))
            total_iters = int(opt['train']['niter'])
            total_epochs = int(math.ceil(total_iters / train_size))
            if opt['dist']:
                train_sampler = DistIterSampler(train_set, world_size, rank, dataset_ratio)
                total_epochs = int(math.ceil(total_iters / (train_size * dataset_ratio)))
            else:
                train_sampler = None
            train_loader = create_dataloader(train_set, dataset_opt, opt, train_sampler)
            if rank <= 0:
                logger.info('Number of train images: {:,d}, iters: {:,d}'.format(len(train_set), train_size))
                logger.info('Total epochs needed: {:d} for iters {:,d}'.format(total_epochs, total_iters))
        elif phase == 'val':
            val_set = create_dataset(dataset_opt)
            val_loader = create_dataloader(val_set, dataset_opt, opt, None)
            if rank <= 0:
                logger.info('Number of val images in [{:s}]: {:d}'.format(dataset_opt['name'], len(val_set)))
        else:
            raise NotImplementedError('Phase [{:s}] is not recognized.'.format(phase))
    assert train_loader is not None

    #### create model
    model = create_model(opt)

    #### resume training
    if resume_state:
        logger.info('Resuming training from epoch: {}, iter: {}.'.format(resume_state['epoch'], resume_state['iter']))

        start_epoch = resume_state['epoch']
        current_step = resume_state['iter']
        model.resume_training(resume_state)  # handle optimizers and schedulers
    else:
        current_step = 0
        start_epoch = 0

    #### training
    logger.info('Start training from epoch: {:d}, iter: {:d}'.format(start_epoch, current_step))
    for epoch in range(start_epoch, total_epochs + 1):
        if opt['dist']:
            train_sampler.set_epoch(epoch)
        for _, train_data in enumerate(train_loader):
            current_step += 1
            if current_step > total_iters:
                break
            #### training
            model.feed_data(train_data)
            model.optimize_parameters(current_step)

            #### update learning rate
            model.update_learning_rate(current_step, warmup_iter=opt['train']['warmup_iter'])

            #### log
            if current_step % opt['logger']['print_freq'] == 0:
                logs = model.get_current_log()
                message = '<epoch:{:3d}, iter:{:8,d}, lr:{:.3e}> '.format(epoch, current_step,
                                                                          model.get_current_learning_rate())
                for k, v in logs.items():
                    message += '{:s}: {:.4e} '.format(k, v)
                    # tensorboard logger

                if rank <= 0:
                    logger.info(message)

            # validation
            if current_step % opt['train']['val_freq'] == 0 and rank <= 0:
                avg_resize_param = 0.0
                avg_psnr = 0.0
                avg_psnr_val = 0.0
                avg_bpp = 0.0
                avg_bpp_val = 0.0
                idx = 0
                logger_val = logging.getLogger('val')  # validation logger
                '''model_dict = model.netG.state_dict()
                for k, v in model_dict.items():
                    if ('weight' not in k):
                        print(k, v)'''
                for val_data in val_loader:
                    idx += 1
                    img_name = os.path.splitext(os.path.basename(val_data['GT_path'][0]))[0]
                    img_dir = os.path.join(opt['path']['val_images'], img_name)
                    util.mkdir(img_dir)

                    model.feed_data(val_data)
                    model.test()
                    visuals = model.get_current_visuals()

                    lr = util.tensor2img(visuals['lr'])  # uint8
                    lr_codec = util.tensor2img(visuals['lr_codec'])  # uint8
                    hr_rec = util.tensor2img(visuals['hr_rec'])
                    gt = util.tensor2img(visuals['gt'])
                    valnet = util.tensor2img(visuals['valnet'])

                    # # Save images for visualization
                    if current_step % (opt['train']['val_freq'] * 10) == 0:
                        save_img_path = os.path.join(img_dir, '{:s}_lr_{:d}.png'.format(img_name, current_step))
                        util.save_img(lr, save_img_path)
                        save_img_path_L = os.path.join(img_dir, '{:s}_lr_codec_{:d}.png'.format(img_name, current_step))
                        util.save_img(lr_codec, save_img_path_L)
                        save_img_path_L = os.path.join(img_dir, '{:s}_hr_rec_{:d}.png'.format(img_name, current_step))
                        util.save_img(hr_rec, save_img_path_L)

                    # Save ground truth
                    if current_step == opt['train']['val_freq']:

                        save_img_path_gt = os.path.join(img_dir, '{:s}_GT_{:d}.png'.format(img_name, current_step))
                        util.save_img(gt, save_img_path_gt)
                        save_img_path_gt = os.path.join(img_dir, '{:s}_valnet_{:d}.png'.format(img_name, current_step))
                        util.save_img(valnet, save_img_path_gt)
                    # calculate PSNR

                    avg_resize_param += visuals['theta']
                    logger.info('# Validation {:s} # resize parameter: {:.4e}.'.format(img_name, visuals['theta']))
                    avg_psnr += visuals['PSNR_net']
                    avg_psnr_val += visuals['PSNR_fix']
                    logger_val.info('# Validation {:s} # net PSNR: {:.4e}.'.format(img_name, visuals['PSNR_net']))
                    logger_val.info('# Validation {:s} # fix PSNR: {:.4e}.'.format(img_name, visuals['PSNR_fix']))

                    avg_bpp += visuals['bpp_net']
                    avg_bpp_val += visuals['bpp_fix']
                    logger_val.info('# Validation {:s} # net bpp: {:.4e}.'.format(img_name, visuals['bpp_net']))
                    logger_val.info('# Validation {:s} # fix bpp: {:.4e}.'.format(img_name, visuals['bpp_fix']))

                avg_resize_param = avg_resize_param / idx
                avg_psnr = avg_psnr / idx
                avg_psnr_val = avg_psnr_val / idx
                avg_bpp = avg_bpp / idx
                avg_bpp_val = avg_bpp_val / idx
                # log
                logger.info('# Validation # avg resize parameter: {:.4e}.'.format(avg_resize_param))
                logger.info('# Validation # net PSNR: {:.4e}.'.format(avg_psnr))
                logger.info('# Validation # fix PSNR: {:.4e}.'.format(avg_psnr_val))
                logger.info('# Validation # net bpp: {:.4e}.'.format(avg_bpp))
                logger.info('# Validation # fix bpp: {:.4e}.'.format(avg_bpp_val))
                logger_val.info('<epoch:{:3d}, iter:{:8,d}> avg resize parameter: {:.4e}.'.format(epoch, current_step, avg_resize_param))
                logger_val.info('<epoch:{:3d}, iter:{:8,d}> net psnr: {:.4e}.'.format(epoch, current_step, avg_psnr))
                logger_val.info('<epoch:{:3d}, iter:{:8,d}> fix psnr: {:.4e}.'.format(epoch, current_step, avg_psnr_val))
                logger_val.info('<epoch:{:3d}, iter:{:8,d}> net bpp: {:.4e}.'.format(epoch, current_step, avg_bpp))
                logger_val.info('<epoch:{:3d}, iter:{:8,d}> fix bpp: {:.4e}.'.format(epoch, current_step, avg_bpp_val))
                # tensorboard logger

            #### save models and training states
            if current_step % opt['logger']['save_checkpoint_freq'] == 0:
                if rank <= 0:
                    logger.info('Saving models and training states.')
                    model.save(current_step)
                    model.save_training_state(epoch, current_step)

    if rank <= 0:
        logger.info('Saving the final model.')
        model.save('latest')
        logger.info('End of training.')


if __name__ == '__main__':
    main()
