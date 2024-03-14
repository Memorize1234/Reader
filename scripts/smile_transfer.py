# Copyright (c) 2022 IDEA. All Rights Reserved.
# ------------------------------------------------------------------------
import argparse
import os
import numpy as np
import torch
import cv2
import random
from torch import nn
import torch.nn.functional as F
from PIL import Image
from tqdm import tqdm
from collections import OrderedDict
from mmengine.config import Config
from model.model_wrapper import EncoderDecoder
from dataset.celeba_smile import build_dataloader
import utils.dist as dist


def batch_masks_iou_torch(masks1, masks2):
    masks1 = masks1[:, None, ]
    masks2 = masks2[None, ]
    i = torch.logical_and(masks1, masks2).sum(-1).sum(-1)
    u = torch.logical_or(masks1, masks2).sum(-1).sum(-1)
    u[u==0] = 4096
    return i / u

class Runner:
    def __init__(self, model, dataloader, out_dir):
        self.model = model
        self.dataloader = dataloader
        self.out_dir = out_dir
        self.mask_folder = os.path.join(out_dir, 'mask')
        self.img_folder = os.path.join(out_dir, 'img')
        os.makedirs(self.mask_folder, exist_ok=True)
        os.makedirs(self.img_folder, exist_ok=True)
        self.lap_op = LaplacianOperator(1).cuda()
        self.norm_mean = np.array([0.485, 0.456, 0.406], dtype=np.float32).reshape((1, 1, 1, -1))
        self.norm_std = np.array([0.229, 0.224, 0.225], dtype=np.float32).reshape((1, 1, 1, -1))
    
    def _transform_keys(self, checkpoint, transform_keys):        
        new_checkpoint = OrderedDict()
        for n in checkpoint:
            for trans_k in transform_keys:
                if trans_k in n:
                    new_n = n.replace(trans_k, transform_keys[trans_k])
                    new_checkpoint[new_n] = checkpoint[n]
        return new_checkpoint
    
    def load(self, ckpt, transform_keys=None):
        print(f'loading checkpoint from {ckpt}')
        checkpoint = torch.load(ckpt, map_location='cpu')['model_state_dict']
        new_checkpoint = checkpoint
        if transform_keys is not None:
            new_checkpoint = self._transform_keys(new_checkpoint, transform_keys)
        self.model.load_state_dict(new_checkpoint, strict=True)
    
    def run(self, rank):
        with torch.no_grad():
            if rank == 0:
                pbar = tqdm(total=len(self.dataloader))
            for idx, data in enumerate(self.dataloader):
                if data is not None:
                    samples, anns, meta_infos = data
                    tgt_imgs = meta_infos['tgt_imgs'].to(dist.dev())
                    samples = samples.to(dist.dev())
                    inputs = torch.cat([samples, tgt_imgs], dim=0)
                    # inputs = samples
                    enc_outpus = self.model.encoder(inputs)
                    dec_inputs = self.transfer_smile_embed(enc_outpus, anns, meta_infos)
                    # dec_inputs = self.transfer_smile(enc_outpus, anns, meta_infos)
                    dec_outputs = self.model.decoder(**dec_inputs)
                    recon_samples = dec_outputs['recon_samples']
                    self.postpocess(samples, recon_samples, tgt_imgs, meta_infos)
                    if rank == 0:
                        pbar.update()
    
    def mask_align(self, mats, tgt_mask):
        scale_MAT, move_MAT = mats
        mask = tgt_mask.cpu().numpy().astype(np.uint8)
        mask = cv2.warpAffine(mask, scale_MAT, (128, 128))
        mask = cv2.warpAffine(mask, move_MAT, (128, 128))
        return torch.tensor(mask, dtype=torch.float32, device='cuda')
    
    def transfer_smile_embed(self, enc_outpus, anns, meta_infos):
        masks = enc_outpus['pred_masks']
        device = masks.device
        bs = len(masks) // 2
        mani_masks, mani_tokens = [], []
        pred_ori_masks, pred_tgt_masks = masks[:bs], masks[bs:]
        ori_embeds, tgt_embeds = enc_outpus['raw_tokens'][:bs], enc_outpus['raw_tokens'][bs:]
        for idx, (p_ori_masks, p_tgt_masks, ori_masks, tgt_masks) in enumerate(zip(pred_ori_masks, pred_tgt_masks, anns['ori_masks'], anns['tgt_masks'])):
            
            ori_masks, tgt_masks = ori_masks.to(device), tgt_masks.to(device)

            # remove ori        
            areas = p_ori_masks.any(-1).any(-1)
            used_inds = areas.nonzero().squeeze()
            ori_used_masks = p_ori_masks[used_inds]
            ori_used_tokens = ori_embeds[idx][used_inds]
            old_ori_pano = ori_used_masks.max(0)[1]
            matched_inds = find_corres(ori_masks, ori_used_masks)
            matched_inds = set(matched_inds.nonzero()[:, 1].tolist())
            old_ori_pano += 1
            new_id_start = old_ori_pano.max().item() + 1

            for ind in matched_inds:
                old_ori_pano[old_ori_pano == ind + 1] = 0

            # add tgt
            new_tokens = []
            areas = p_tgt_masks.any(-1).any(-1)
            used_inds = areas.nonzero().squeeze()
            tgt_used_masks = p_tgt_masks[used_inds]
            tgt_used_tokens = tgt_embeds[idx][used_inds]
            matched_inds = find_corres(tgt_masks, tgt_used_masks)
            matched_inds = set(matched_inds.nonzero()[:, 1].tolist())
            for num, ind in enumerate(matched_inds):
                tgt_mask = tgt_used_masks[ind]
                old_ori_pano[tgt_mask > 0] = new_id_start + num
                new_tokens.append(tgt_used_tokens[ind])
            new_tokens = torch.stack(new_tokens, dim=0)
            ori_used_tokens = torch.cat((ori_used_tokens, new_tokens), dim=0)

            # output
            out_masks, out_tokens = [], []
            id_max = old_ori_pano.max().item()
            for i in range(1, id_max+1):
                mask = old_ori_pano == i
                if mask.any():
                    out_masks.append(mask.float())
                    out_tokens.append(ori_used_tokens[i-1])
                    
            out_masks = torch.stack(out_masks, dim=0)
            mani_masks.append(out_masks)
            mani_tokens.append(torch.stack(out_tokens, dim=0))

        mani_masks = torch.stack(mani_masks, dim=0)
        mani_tokens = torch.stack(mani_tokens, dim=0)
        mani_attn_mask = torch.zeros((1, mani_masks.shape[1]), dtype=bool, device=mani_masks.device)
        mani_embed, _ = self.model.encoder.mask_embed(mani_masks, tokens=mani_tokens, extract=False)
        return dict(embeds=mani_embed, attn_mask=mani_attn_mask)
    
    def postpocess(self, ori_samples, recon_samples, tgt_imgs, meta_infos):
        samples = recon_samples.permute(0, 2, 3, 1).cpu().numpy()
        samples = samples * 0.5 + 0.5
        samples = samples.clip(0, 1)
        samples = (samples * 255).astype(np.uint8)
        
        tgt_imgs = tgt_imgs.permute(0, 2, 3, 1).cpu().numpy()
        tgt_imgs = tgt_imgs * 0.5 + 0.5
        tgt_imgs = (tgt_imgs * 255).astype(np.uint8)

        ori_samples = ori_samples.permute(0, 2, 3, 1).cpu().numpy()
        ori_samples = ori_samples * 0.5 + 0.5
        ori_samples = (ori_samples * 255).astype(np.uint8)

        for sample, tgt, ori, file in zip(samples, tgt_imgs, ori_samples, meta_infos['files']):
            file = file.replace('jpg', 'png')
            Image.fromarray(sample).save(os.path.join(self.img_folder, file))
            Image.fromarray(tgt).save(os.path.join(self.mask_folder, file))


class LaplacianOperator(nn.Module):
    def __init__(self, boundary_width=3):
        super().__init__()
        kernel_size = 2 * boundary_width + 1
        self.boundary_width = boundary_width
        self.kernel_size = kernel_size
        laplacian_kernel = - np.ones(shape=(kernel_size, kernel_size), dtype=np.float32)
        laplacian_kernel[boundary_width, boundary_width] = kernel_size ** 2 - 1
        self.laplacian_kernel = nn.Conv2d(1, 1, kernel_size=kernel_size, stride=1, padding=0, bias=False)
        self.laplacian_kernel.weight.data = torch.tensor(laplacian_kernel).unsqueeze(0).unsqueeze(0)
        self.laplacian_kernel.weight.requires_grad = False

    @torch.no_grad()
    def forward(self, mask_target):
        mask_target = mask_target.unsqueeze(0).unsqueeze(0)
        neg_mask_target = 1 - mask_target.clone()
        pad = (self.boundary_width, self.boundary_width, self.boundary_width, self.boundary_width)
        neg_mask_target = F.pad(neg_mask_target, pad, mode='constant', value=1)
        # neg_boundary
        neg_boundary_targets = self.laplacian_kernel(neg_mask_target)
        neg_boundary_targets = neg_boundary_targets.clamp(min=0) / float(self.kernel_size ** 2)
        neg_boundary_targets[neg_boundary_targets > 0.1] = 1
        neg_boundary_targets[neg_boundary_targets <= 0.1] = 0
        neg_boundary_targets[mask_target > 0] = 1
        return neg_boundary_targets.squeeze(0).squeeze(0)


def find_corres(ann_mask, masks, thr=0):
    ann_mask = ann_mask[:, None, ]
    masks = masks[None, ]
    inters = torch.logical_and(ann_mask, masks).sum(-1).sum(-1)
    return inters > thr

def get_args_parser():
    parser = argparse.ArgumentParser('Set transformer detector', add_help=False)
    parser.add_argument('config', type=str)
    parser.add_argument('ckpt', type=str)
    parser.add_argument('--out_dir', default='CelebA_output')
    return parser


def main(args):
    dist.init_distributed_mode()
    torch.cuda.set_device(dist.get_local_rank())
    torch.cuda.empty_cache()
    rank = int(os.getenv('RANK', '0'))
    if rank == 0:
        os.makedirs(args.out_dir, exist_ok=True)
        print("Loading config file from {}".format(args.config))
    cfg = Config.fromfile(args.config)

    # build model
    model_cfg = cfg.get('model').get('g_model')
    model = EncoderDecoder(**model_cfg).to(dist.dev())
    model.eval()

    # build dataset
    dataloader = build_dataloader(1, 1)

    # build runner
    runner = Runner(model, dataloader, args.out_dir)
    transform_keys = {'g_model.module.': ''}
    runner.load(args.ckpt, transform_keys)
    
    # run
    runner.run(rank)



if __name__ == '__main__':
    parser = argparse.ArgumentParser('Smile Transfer', parents=[get_args_parser()])
    args = parser.parse_args()
    main(args)