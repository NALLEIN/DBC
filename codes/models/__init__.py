import logging
logger = logging.getLogger('base')


def create_model(opt, model=None):
    model = opt['model']

    if model == 'resize':
        from .resize_net import ResizeModel as M
    else:
        raise NotImplementedError('Model [{:s}] not recognized.'.format(model))
    m = M(opt)
    logger.info('Model [{:s}] is created.'.format(m.__class__.__name__))
    return m
