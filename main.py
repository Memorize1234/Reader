# Copyright (c) 2022 IDEA. All Rights Reserved.
# ------------------------------------------------------------------------
import argparse
import json
import random
import os
import numpy as np
import torch

from mmengine.config import Config
from utils.logger import setup_logger
import utils.dist as dist
from utils.runner import Runner
from utils import global_vars as gl


def get_args_parser():
    parser = argparse.ArgumentParser('Reader', add_help=False)
    parser.add_argument('config', type=str)
    parser.add_argument('--ckpt', default=None)

    # training parameters
    parser.add_argument('--output_dir', default='work_dirs',
                        help='path where to save, empty for no saving')
    parser.add_argument('--seed', default=42, type=int)

    parser.add_argument('--test_mode', action="store_true")
    
    # distributed training parameters
    parser.add_argument("--local_rank", type=int, default=0, help='local rank for DistributedDataParallel')
    
    # comment for current experiment
    parser.add_argument("--comment", type=str, default='')
    
    return parser


def main(args):
    dist.init_distributed_mode()
    torch.cuda.set_device(dist.get_local_rank())
    torch.cuda.empty_cache()
    rank = int(os.getenv('RANK', '0'))

    # load cfg file and update the args
    print("Loading config file from {}".format(args.config))
    cfg = Config.fromfile(args.config)
    if args.ckpt is not None:
        ckpt_cfg = {'test_cfg.load_from': {'ckpt': args.ckpt}}
        cfg.merge_from_dict(ckpt_cfg)
    cfg_dict = cfg._cfg_dict.to_dict()
    cfg_dict['comment'] = args.comment
    use_gan = cfg.get('use_gan', False)
    face_finetune = cfg.get('face_finetune', False)
         
    # setup logger
    logger = setup_logger(output=os.path.join(args.output_dir, 'info.txt'), distributed_rank=rank, color=False, name="detr")
    if rank == 0:
        os.makedirs(args.output_dir, exist_ok=True)
        save_json_path = os.path.join(args.output_dir, "config.json")
        with open(save_json_path, 'w') as f:
            json.dump(cfg_dict, f, indent=2)
        logger.info("Full config saved to {}".format(save_json_path))
    logger.info('world size: {}'.format(dist.get_world_size()))
    logger.info('rank: {}'.format(dist.get_rank()))
    logger.info('local rank: {}'.format(dist.get_local_rank()))

    # fix the seed for reproducibility
    seed = args.seed + rank
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    
    # initialize global variable dict
    gl.init()

    # runner
    if not args.test_mode:
        runner = Runner(True, use_gan, face_finetune, cfg.get('model'), cfg.get('data'), cfg.get('train_cfg'), logger, out_dir=args.output_dir)
        runner.train_loop()
    else:
        runner = Runner(False, use_gan, face_finetune, cfg.get('model'), cfg.get('data'), cfg.get('test_cfg'), logger, out_dir=args.output_dir)
        runner.test_loop()


if __name__ == '__main__':
    parser = argparse.ArgumentParser('Training script', parents=[get_args_parser()])
    args = parser.parse_args()
    main(args)
    