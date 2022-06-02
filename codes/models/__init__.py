import logging
logger = logging.getLogger('base')


def create_model(opt, model=None):
    model = opt['model']

    if model == 'IRN':
        from .IRN_model import IRNModel as M
    elif model == 'IRN2':
        from .IRN_model_2 import IRNModel as M
    elif model == 'IRN3':
        from .IRN_model_3 import IRNModel as M
    elif model == 'IRN3_4kvideo':
        from .IRN_model_3_4kvideo import IRNModel as M
    elif model == 'EDSR':
        from .EDSR_model_vimeo90k import EDSRModel as M
    elif model == 'ShiftNet':
        from .ShiftNet_model import ShiftNetModel as M
    elif model == 'rrdnn':
        from .rrdnn_model import RRDNNModel as M
    elif model == 'RAVC':
        from .RAVC_model import RAVCModel as M
    else:
        raise NotImplementedError('Model [{:s}] not recognized.'.format(model))
    m = M(opt)
    logger.info('Model [{:s}] is created.'.format(m.__class__.__name__))
    return m
