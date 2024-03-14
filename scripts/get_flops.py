# Copyright (c) OpenMMLab. All rights reserved.
import argparse
import sys
sys.path.extend(['.', '..'])
import numpy as np
import torch
from mmengine.config import Config
from mmengine.logging import MMLogger
from mmengine.model import revert_sync_batchnorm
from mmengine.registry import init_default_scope

from model.model_wrapper import EncoderDecoder
try:
    from mmengine.analysis import get_model_complexity_info
    from mmengine.analysis.print_helper import _format_size
except ImportError:
    raise ImportError('Please upgrade mmengine >= 0.6.0')


def parse_args():
    parser = argparse.ArgumentParser(description='Get a detector flops')
    parser.add_argument('config', help='train config file path')
    args = parser.parse_args()
    return args



def inference(args):

    cfg = Config.fromfile(args.config)
    cfg = cfg.to_dict()

    init_default_scope(cfg.get('default_scope', 'mmdet'))

    result = {}
    avg_flops = []
    model = EncoderDecoder(**cfg.get('model').get('g_model'))


    if torch.cuda.is_available():
        model = model.cuda()
    model = revert_sync_batchnorm(model)
    model.eval()

    input_shape = (1, 3, 256, 256)
    device=next(model.parameters()).device
    inputs = torch.randn(input_shape, device=device)

    outputs = get_model_complexity_info(
        model,
        input_shape=None,
        inputs=inputs,
        show_table=True,
        show_arch=True)
    avg_flops.append(outputs['flops'])
    params = outputs['params']
    result['compute_type'] = 'dataloader: load a picture from the dataset'

    mean_flops = _format_size(int(np.average(avg_flops)))
    params = _format_size(params)
    result['flops'] = mean_flops
    result['params'] = params
    print(outputs['out_table'])
    


if __name__ == '__main__':
    args = parse_args()
    inference(args)
