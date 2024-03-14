# Copyright (c) 2022 IDEA. All Rights Reserved.
# ------------------------------------------------------------------------
import argparse
import os
import numpy as np
import torch
import cv2
import copy
import random
import torch.nn.functional as F
from PIL import Image
from tqdm import tqdm
from collections import OrderedDict
from mmengine.config import Config
from model.model_wrapper import EncoderDecoder
from dataset.imagenet_test import build_visual_dataloader
from utils.bbox import cxcywh_to_xyxy_torch
from utils import dist


def draw_mask(img, mask, fg_color=(255, 0, 0)):
    fg_color = np.array(fg_color).reshape(1, 1, -1)
    mask_color = np.expand_dims(mask, axis=2)
    bg_img = img * mask_color
    mask_color = (mask_color * fg_color).astype(np.uint8)
    img_mask = cv2.addWeighted(img, 0.7, mask_color, 0.3, 0)
    img_mask = cv2.addWeighted(img_mask, 1, bg_img, 0.3, 0)
    return img_mask

def masks2pano(masks):
    masks = masks.astype(np.uint8)
    pano_mask = np.zeros((*masks.shape[1:], 3), dtype=np.uint8)
    num = 0
    for mask in masks:
        if mask.any():
            num += 1
            mask = np.expand_dims(mask, axis=2)
            fg_color = COLOR[random.randint(0, len(COLOR)-1)]
            fg_color = np.array(fg_color).reshape(1, 1, -1)
            mask = (mask * fg_color).astype(np.uint8)
            pano_mask += mask
    return pano_mask, num


class Visualizer:
    def __init__(self, model, dataloader, out_dir, resize=False):
        self.model = model
        self.dataloader = dataloader
        self.out_dir = out_dir
        self.norm_mean = np.array([0.485, 0.456, 0.406], dtype=np.float32).reshape((1, 1, -1))
        self.norm_std = np.array([0.229, 0.224, 0.225], dtype=np.float32).reshape((1, 1, -1))
        self.resize = resize
    
    def _transform_keys(self, checkpoint, transform_keys):        
        new_checkpoint = OrderedDict()
        for n in checkpoint:
            for trans_k in transform_keys:
                if trans_k in n:
                    new_n = n.replace(trans_k, transform_keys[trans_k])
                    new_checkpoint[new_n] = checkpoint[n]
        return new_checkpoint
    
    def load(self, ckpt, transform_keys=None):
        self.ckpt_id = int(ckpt.split('/')[-1].split('.')[0])
        checkpoint = torch.load(ckpt, map_location='cpu')['model_state_dict']
        new_checkpoint = checkpoint
        if transform_keys is not None:
            new_checkpoint = self._transform_keys(new_checkpoint, transform_keys)
        self.model.load_state_dict(new_checkpoint, strict=True)
    
    def run(self):
        os.makedirs(os.path.join(self.out_dir, f'epoch_{self.ckpt_id}'), exist_ok=True)
        all_num = []
        with torch.no_grad():
            with tqdm(total=len(self.dataloader), desc=f'epoch_{self.ckpt_id}', 
                      disable=dist.get_rank() != 0) as pbar:
                for inputs, data_metas in self.dataloader:
                    pbar.update()
                    if inputs is None:
                        continue
                    samples = inputs.to(dist.dev())
                    results = self.model(samples)
                    all_num.append(results['enc_losses']['num_groups'])
                    self.postpocess(data_metas, results)
        all_num = torch.tensor(all_num).to(dist.dev())
        avg_num = all_num.mean()
        world_size = dist.get_world_size()
        if world_size > 1:
            torch.distributed.all_reduce(avg_num, torch.distributed.ReduceOp.SUM)
            avg_num = avg_num / world_size
        return avg_num.item()
    
    def postpocess(self, datas, results, img_net=False):
        recon_samples = results['recon_samples']
        recon_samples = recon_samples.permute(0, 2, 3, 1).cpu().numpy()
        for idx, data in enumerate(datas):
            out_file = data['filename'].split('/')[-1].split('.')[0]
            out_file = out_file + '.png'
            sample = recon_samples[idx]
            if self.resize:
                H, W = data['ori_shape']
                sample = cv2.resize(sample, (W, H))
            sample = (sample + 1 ) * 127.5
            sample = sample.clip(0, 255)
            sample = sample.astype(np.uint8)
            Image.fromarray(sample).save(os.path.join(self.out_dir, f'epoch_{self.ckpt_id}', out_file))
            
    def get_pano_results(self, ori_sample, masks_soft, masks_hard, H, W):
        masks_hard = F.interpolate(masks_hard.unsqueeze(0), size=(H, W), mode='nearest')[0].cpu().numpy()
        masks_hard_pano, _ = masks2pano(masks_hard)
        img_masks_hard = cv2.addWeighted(ori_sample, 0.5, masks_hard_pano, 0.5, 0)
        # hard_output = np.concatenate([img_masks_hard, masks_hard_pano], axis=0)

        masks_soft = F.interpolate(masks_soft.unsqueeze(0), size=(H, W), mode='bilinear')[0]
        index = masks_soft.max(0, keepdim=True)[1]
        masks_soft_hard = torch.zeros_like(masks_soft, memory_format=torch.legacy_contiguous_format).scatter_(0, index, 1.0)
        masks_soft_hard = masks_soft_hard.cpu().numpy()
        masks_soft_pano, _ = masks2pano(masks_soft_hard)
        img_masks_soft = cv2.addWeighted(ori_sample, 0.7, masks_soft_pano, 0.3, 0)
        # soft_output = np.concatenate([img_masks_soft, masks_soft_pano], axis=0)
        return masks_hard_pano, img_masks_hard, masks_soft_pano, img_masks_soft

def get_args_parser():
    parser = argparse.ArgumentParser('Set transformer detector', add_help=False)
    parser.add_argument('config', type=str)
    parser.add_argument('--ckpt_dir', type=str)
    parser.add_argument('--epoch_start', type=int)
    parser.add_argument('--epoch_end', type=int)
    parser.add_argument('--out_dir', default='visualize_results')
    return parser


def main(args):
    dist.init_distributed_mode()
    world_size = int(os.getenv('WORLD_SIZE', 1))
    rank = int(os.getenv('RANK', 0))
    
    print("Loading config file from {}".format(args.config))
    cfg = Config.fromfile(args.config)

    # init
    model_cfg = cfg.get('model').get('g_model')
    model = EncoderDecoder(**model_cfg).to(dist.dev())
    model.eval()

    set_cfg = cfg.get('data').get('val')
    set_cfg['world_size'] = world_size
    set_cfg['rank'] = rank
    dataloader = build_visual_dataloader(set_cfg, batch_size=10)
    visualizer = Visualizer(model, dataloader, args.out_dir)

    # load ckpt
    num_groups_dict = {}
    transform_keys = {'g_model.module.': ''}
    for i in range(args.epoch_start, args.epoch_end + 1):
        ckpt_path = os.path.join(args.ckpt_dir, f'{i}.pth')
        visualizer.load(ckpt_path, transform_keys)
        num_groups = visualizer.run()
        if dist.get_rank() == 0:
            print(f'num groups of {i}.pth: {num_groups}')
        num_groups_dict[i] = num_groups
        torch.distributed.barrier()
    if dist.get_rank() == 0:
        print('------------')
        print(num_groups_dict)


if __name__ == '__main__':
    parser = argparse.ArgumentParser('Test Imagenet Validation Reconstruct', parents=[get_args_parser()])
    args = parser.parse_args()
    main(args)