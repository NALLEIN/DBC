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
import models.lr_scheduler as lr_scheduler
from .base_model import BaseModel
from models.modules.loss import ReconstructionLoss
# import data.util as util
import sys

from models.modules.resize import ResizeNet
from models.modules.google import ScaleHyperprior, RateDistortionLoss
from compressai.zoo.image import _load_model

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

        self.criterion = RateDistortionLoss(lmbda=1e-2, metrics='mse')
        self.netG = ResizeNet().to(self.device)
        # self.netA = ScaleHyperprior(192, 320).to(self.device)  # 128, 192
        self.netA = _load_model("bmshj2018-hyperprior", "mse", 1)

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
            optim_aux = []
            optim_rates = []
            for k, v in self.netA.named_parameters():
                if v.requires_grad:
                    if k.endswith(".quantiles"):
                        optim_aux.append(v)
                    else:
                        optim_rates.append(v)
                else:
                    if self.rank <= 0:
                        logger.warning('Params [{:s}] will not optimize.'.format(k))
            self.optimizer_A = torch.optim.Adam(optim_params, lr=train_opt['lr_G'], weight_decay=wd_G, betas=(train_opt['beta1'], train_opt['beta2']))
            self.optimizers.append(self.optimizer_A)
            self.optimizer_aux = torch.optim.Adam(optim_aux, lr=train_opt['lr_G'], weight_decay=wd_G, betas=(train_opt['beta1'], train_opt['beta2']))
            self.optimizers.append(self.optimizer_aux)

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

    # def gaussian_batch(self, dims):
    #     return torch.randn(tuple(dims)).to(self.device)

    # def loss_forward(self, out, y):
    #     l_forw_fit = self.train_opt['lambda_fit_forw'] * self.Reconstruction_forw(out, y)
    #     return l_forw_fit

    # def loss_backward(self, out, y):
    #     l_forw_fit = self.train_opt['lambda_rec_back'] * self.Reconstruction_back(out, y)
    #     return l_forw_fit

    def optimize_parameters(self, step):
        B, C, H, W = self.real_H.size()
        lr, theta = self.netG(self.real_H)
        out_dict = self.netA(lr)
        lr_codec = out_dict['x_hat']
        hr_rec, _ = self.netG(lr_codec, theta, reverse=True)

        out_dict['x_hat'] = hr_rec
        loss_rd = self.criterion(out_dict, self.real_H)
        loss_rd['loss'].backward()
        torch.nn.utils.clip_grad_norm_(self.netA.parameters(), max_norm=5)
        torch.nn.utils.clip_grad_norm_(self.netG.parameters(), max_norm=5)
        self.optimizer_A.step()
        self.optimizer_G.step()

        aux_loss = self.netA.module.aux_loss()
        aux_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.netA.parameters(), max_norm=5)
        self.optimizer_aux.step()

        # set log
        self.log_dict['distortion'] = loss_rd['bpp_loss'].item()
        self.log_dict['rate'] = loss_rd['mse_loss'].item()
        self.log_dict['rdloss'] = loss_rd['loss'].item()

    @torch.no_grad()
    def test(self):
        self.netG.eval()
        self.netA.eval()
        with torch.no_grad():
            lr, theta = self.netG(self.real_H)
            out_net = self.netA(lr)
            lr_codec = out_net['x_hat']
            hr_rec, _ = self.netG(lr_codec, theta, reverse=True)
            out_net['x_hat'] = hr_rec
            loss_rd = self.criterion(out_net, self.real_H)

            # used for visualization
            self.lr = lr
            self.lr_codec = lr_codec
            self.hr_rec = hr_rec
            self.gt = self.real_H
            self.bpp_fix = None
            self.bpp_net = loss_rd['loss']
            self.psnr_fix = None
            self.psnr_net = 10 * torch.log10(1**2 / torch.mean((self.real_H - hr_rec)**2))

    @torch.no_grad()
    def compress(self, x):
        self.netG.eval()
        self.netA.eval()
        with torch.no_grad():
            lr, theta = self.netA(x)
            out_net = self.netG.module.compress(lr)
        return out_net['strings'], out_net['shape'], theta

    @torch.no_grad()
    def decompress(self, strings, shape, theta):
        self.netG.eval()
        self.netA.eval()
        with torch.no_grad():
            lr_rec = self.netA.module.decompress(strings, shape)
            hr_rec = self.netG(lr_rec, theta, reverse=True)
        return hr_rec

    def get_current_log(self):
        return self.log_dict

    def get_current_visuals(self):
        out_dict = OrderedDict()
        out_dict['lr'] = self.lr.detach().float().cpu()
        out_dict['lr_codec'] = self.lr_codec.detach().float().cpu()
        out_dict['hr_rec'] = self.hr_rec.detach().float().cpu()
        out_dict['gt'] = self.real_H.detach().float().cpu()
        out_dict['bpp_net'] = self.bpp_net.detach().item()
        out_dict['bpp_fix'] = 100.0 # self.bpp_fix.detach().item()
        out_dict['PSNR_net'] = self.psnr_net.detach().item()
        out_dict['PSNR_fix'] = 100.0 # self.psnr_fix.detach().item()

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
