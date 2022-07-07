import logging
from collections import OrderedDict
import numpy as np
import torch
import cv2
import math
import random
import torch.nn as nn
from torch.nn.parallel import DataParallel, DistributedDataParallel
import torch.nn.functional as F
from pytorch_msssim import ms_ssim
import models.lr_scheduler as lr_scheduler
from .base_model import BaseModel
from models.modules.loss import ReconstructionLoss
from models.modules.quantization import DecodeEncode, DownSample, UpSample, rgb2yuv420, yuv4202rgb, rgb2yuv420_down, rgbdown

# import data.util as util
import sys

from models.modules.resize_arch import resize_arch, VCN
from models.modules.loss import RateDistortionLoss

sys.path.append("..")
logger = logging.getLogger('base')
val_logger = logging.getLogger('val')


class ResizeModel(BaseModel):
    def __init__(self, opt):
        super(ResizeModel, self).__init__(opt)

        if opt['dist']:
            self.rank = torch.distributed.get_rank()
        else:
            self.rank = -1  # non dist training
        train_opt = opt['train']
        test_opt = opt['test']
        self.train_opt = train_opt
        self.test_opt = test_opt
        self.scale = opt['scale']

        # self.criterion = RateDistortionLoss(lmbda=self.lamada, metrics=self.metrics)
        self.netG = resize_arch(self.scale).to(self.device)
        self.netA = VCN().to(self.device)
        self.DecodeEncode = DecodeEncode()

        if opt['dist']:
            self.netG = DistributedDataParallel(self.netG, device_ids=[torch.cuda.current_device()])
            self.netA = DistributedDataParallel(self.netA, device_ids=[torch.cuda.current_device()])
        else:
            self.netG = DataParallel(self.netG)
            self.netA = DataParallel(self.netA)

        # print network
        self.print_network()
        self.load()

        if self.is_train:
            self.netG.train()
            self.netA.train()

            # loss
            self.Reconstruction_forw = ReconstructionLoss(losstype=self.train_opt['pixel_criterion_forw'])
            self.Reconstruction_back = ReconstructionLoss(losstype=self.train_opt['pixel_criterion_back'])

            # optimizers
            wd_G = train_opt['weight_decay_G'] if train_opt['weight_decay_G'] else 0
            optim_params = []
            for k, v in self.netG.named_parameters():
                if v.requires_grad:
                    optim_params.append(v)
                else:
                    if self.rank <= 0:
                        logger.warning('Params [{:s}] will not optimize.'.format(k))
            self.optimizer_G = torch.optim.Adam(optim_params, lr=train_opt['lr_G'], weight_decay=wd_G, betas=(train_opt['beta1'], train_opt['beta2']))
            self.optimizers.append(self.optimizer_G)

            wd_G = train_opt['weight_decay_G'] if train_opt['weight_decay_G'] else 0
            optim_params = []
            optim_rates = []
            for k, v in self.netA.named_parameters():
                if v.requires_grad:
                    if ('rate' not in k):
                        optim_params.append(v)
                    else:
                        optim_rates.append(v)
                else:
                    if self.rank <= 0:
                        logger.warning('Params [{:s}] will not optimize.'.format(k))
            self.optimizer_A = torch.optim.Adam(optim_params, lr=train_opt['lr_G'], weight_decay=wd_G, betas=(train_opt['beta1'], train_opt['beta2']))
            self.optimizers.append(self.optimizer_A)
            self.optimizer_C = torch.optim.Adam(optim_rates, lr=train_opt['lr_G'], weight_decay=wd_G, betas=(train_opt['beta1'], train_opt['beta2']))
            self.optimizers.append(self.optimizer_C)

            # schedulers
            if train_opt['lr_scheme'] == 'MultiStepLR':
                for optimizer in self.optimizers:
                    self.schedulers.append(
                        lr_scheduler.MultiStepLR_Restart(optimizer,
                                                         train_opt['lr_steps'],
                                                         restarts=train_opt['restarts'],
                                                         weights=train_opt['restart_weights'],
                                                         gamma=train_opt['lr_gamma'],
                                                         clear_state=train_opt['clear_state']))
            elif train_opt['lr_scheme'] == 'CosineAnnealingLR_Restart':
                for optimizer in self.optimizers:
                    self.schedulers.append(
                        lr_scheduler.CosineAnnealingLR_Restart(optimizer,
                                                               train_opt['T_period'],
                                                               eta_min=train_opt['eta_min'],
                                                               restarts=train_opt['restarts'],
                                                               weights=train_opt['restart_weights']))
            else:
                raise NotImplementedError('MultiStepLR learning rate scheme is enough.')

            self.log_dict = OrderedDict()

    def feed_data(self, data):
        self.real_H = data['GT'].to(self.device)  # GT

    def gaussian_batch(self, dims):
        return torch.randn(tuple(dims)).to(self.device)

    def loss_forward(self, out, y):
        l_forw_fit = self.train_opt['lambda_fit_forw'] * self.Reconstruction_forw(out, y)
        return l_forw_fit

    def loss_backward(self, out, y):
        l_back_rec = self.train_opt['lambda_rec_back'] * self.Reconstruction_back(out, y)
        return l_back_rec

    def yuvUpsample(self, x, h, w):
        y_tensor = torch.nn.functional.pixel_unshuffle(
            F.interpolate(torch.nn.functional.pixel_shuffle(x[:, :4, :, :], 2), size=[2 * h, 2 * w], mode='bicubic'), 2)
        uv_tensor = F.interpolate(x[:, 4:, :, :], size=[h, w], mode='bicubic')
        sr = torch.cat((y_tensor, uv_tensor), 1)
        return sr

    def optimize_parameters(self, step):
        # B, C, H, W = self.real_H.size()
        # lr, theta = self.netG(self.real_H)
        # out_dict = self.netA(lr)
        # lr_codec = out_dict['x_hat']
        # hr_rec, _ = self.netG(lr_codec, theta, reverse=True)

        # out_dict['x_hat'] = hr_rec
        # loss_rd = self.criterion(out_dict, self.real_H)
        # loss_rd['loss'].backward()
        # torch.nn.utils.clip_grad_norm_(self.netA.parameters(), max_norm=5)
        # torch.nn.utils.clip_grad_norm_(self.netG.parameters(), max_norm=5)
        # self.optimizer_A.step()
        # self.optimizer_G.step()

        # aux_loss = self.netA.module.aux_loss()
        # aux_loss.backward()
        # torch.nn.utils.clip_grad_norm_(self.netA.parameters(), max_norm=5)
        # self.optimizer_aux.step()

        # # set log
        # self.log_dict['distortion'] = loss_rd['bpp_loss'].item()
        # self.log_dict['rate'] = loss_rd['mse_loss'].item()
        # self.log_dict['rdloss'] = loss_rd['loss'].item()
        # qp = "37"

        # YUV train
        qp = self.train_opt['qp']
        self.real_H, self.ref_L = rgb2yuv420_down(self.real_H, 1 / self.scale)
        _, _, h, w = self.real_H.size()
        out_mv = None

        lr = self.ref_L
        lr_codec, out_mv = self.DecodeEncode(lr, qp, False)
        self.optimizer_A.zero_grad()
        lr_est, bpp = self.netA(x=lr)
        l_codec_fit = self.loss_forward(lr_est, lr_codec.clone().detach())

        rd_loss = l_codec_fit  # 0.05 * 255**2 * l_codec_fit + bpp  #  - 1e-5 * correlation_param
        rd_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.netA.parameters(), 5.0)
        self.optimizer_A.step()

        # aux_loss = self.netA.module.aux_loss()
        # aux_loss.backward()
        # torch.nn.utils.clip_grad_norm_(self.netA.parameters(), 1.0)
        # self.optimizer_C.step()

        self.optimizer_G.zero_grad()
        self.optimizer_C.zero_grad()
        lr = self.netG(x=self.real_H, up=False)  # forward downscaling
        lr_codec_est, bpp = self.netA(x=lr)
        # sr_est = self.yuvUpsample(lr_codec_est, h, w)
        # sr_est_codec = self.yuvUpsample(lr_codec, h, w)
        sr_est = self.netG(lr_codec_est, up=True)
        sr_est_codec = self.netG(lr_codec, up=True)

        l_size_fit = torch.mean(bpp)
        l_back_fit = self.loss_forward(sr_est, self.real_H) + self.loss_forward(sr_est_codec, self.real_H)
        l_forw_fit = self.loss_forward(lr, self.ref_L)

        loss = l_back_fit + 0.1 * l_forw_fit + 3e-4 * l_size_fit
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.netG.parameters(), 5.0)
        self.optimizer_G.step()
        self.optimizer_C.step()

        # set log
        self.log_dict['l_codec'] = l_codec_fit.item()
        self.log_dict['l_forw'] = l_forw_fit.item()
        self.log_dict['l_back'] = l_back_fit.item()
        self.log_dict['distortion'] = (l_back_fit + l_forw_fit).item()
        self.log_dict['l_size'] = l_size_fit.item()

    @torch.no_grad()
    def test(self):
        self.netG.eval()
        with torch.no_grad():
            qp = self.test_opt['qp']
            real_H, ref_L = rgb2yuv420_down(self.real_H, 1 / self.scale)
            self.ref_L = yuv4202rgb(ref_L)
            _, _, h, w = real_H.size()
            ref_L, _ = self.DecodeEncode(ref_L, qp, True)
            fake_H_bic = self.yuvUpsample(ref_L, h, w)
            self.fake_H_bic = yuv4202rgb(fake_H_bic)
            self.psnr_fix = 10 * torch.log10(1**2 / torch.mean((real_H - fake_H_bic)**2))
            self.ssim_fix = ms_ssim(real_H, fake_H_bic, data_range=1.0)

            lr = self.netG(x=real_H, up=False)  # forward downscaling
            self.forw_L = yuv4202rgb(lr)
            lr, _ = self.DecodeEncode(lr, qp, True)
            # fake_H = self.yuvUpsample(lr, h, w)
            fake_H = self.netG(x=lr, up=True)
            self.fake_H = yuv4202rgb(fake_H)
            self.psnr = 10 * torch.log10(1**2 / torch.mean((real_H - fake_H)**2))
            self.ssim = ms_ssim(real_H, fake_H, data_range=1.0)

    def downscale(self, HR_img):
        self.netG.eval()
        with torch.no_grad():
            LR_img = self.netG(x=HR_img, up=False)
        return LR_img

    def upscale(self, LR_img):
        self.netG.eval()
        with torch.no_grad():
            HR_img = self.netG(x=LR_img, up=True)
        return HR_img

    def get_current_log(self):
        return self.log_dict

    def get_current_visuals(self):
        out_dict = OrderedDict()
        out_dict['LR_ref'] = self.ref_L.detach()[0].float().cpu()
        out_dict['SR'] = self.fake_H.detach()[0].float().cpu()
        out_dict['LR'] = self.forw_L.detach()[0].float().cpu()
        out_dict['GT'] = self.real_H.detach()[0].float().cpu()
        out_dict['SR_bic'] = self.fake_H_bic.detach()[0].float().cpu()
        out_dict['PSNR_fix'] = self.psnr_fix.detach().item()
        out_dict['PSNR'] = self.psnr.detach().item()
        out_dict['SSIM_fix'] = self.ssim_fix.detach().item()
        out_dict['SSIM'] = self.ssim.detach().item()

        return out_dict

    def print_network(self):
        s, n = self.get_network_description(self.netG)
        if isinstance(self.netG, nn.DataParallel) or isinstance(self.netG, DistributedDataParallel):
            net_struc_str = '{} - {}'.format(self.netG.__class__.__name__, self.netG.module.__class__.__name__)
        else:
            net_struc_str = '{}'.format(self.netG.__class__.__name__)
        if self.rank <= 0:
            logger.info('Network G structure: {}, with parameters: {:,d}'.format(net_struc_str, n))
            logger.info(s)

    def load(self):
        load_path_G = self.opt['path']['pretrain_model_G']
        if load_path_G is not None:
            logger.info('Loading model for G [{:s}] ...'.format(load_path_G))
            self.load_network(load_path_G, self.netG, self.opt['path']['strict_load'])

        load_path_A = self.opt['path']['pretrain_model_A']
        if load_path_A is not None:
            logger.info('Loading model for A [{:s}] ...'.format(load_path_A))
            self.load_network(load_path_A, self.netA, self.opt['path']['strict_load'])

    def save(self, iter_label):
        self.save_network(self.netG, 'G', iter_label)
        self.save_network(self.netA, 'A', iter_label)
