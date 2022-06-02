import torch
import logging
import math
logger = logging.getLogger('base')


def define_EDSR(opt):
    model = opt['model']
    if model == 'EDSR':
        from codes.models.modules.scalenet import scalenet
    netG = scalenet()
    return netG


def define_Analysis(opt):
    model = opt['model']
    if model == 'EDSR':
        from codes.models.modules.scalenet import analysis_arch
    netG = analysis_arch()
    return netG

