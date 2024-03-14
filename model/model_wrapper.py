import cv2
import os
import random
import torch
from torch import nn
import torch.nn.functional as F
from torch.optim import AdamW, SGD
import numpy as np
from mmdet.registry import MODELS
from torch.nn.parallel.distributed import DistributedDataParallel as DDP
from .losses.discriminator import GANLoss
from utils import global_vars as gl
from utils.model_param import get_train_param, get_named_params
import utils.dist as dist
import time
import math
import warnings
from PIL import Image
from torch.optim.lr_scheduler import CosineAnnealingLR
from .transformer import TransformerEncoderLayer

def isnan_dist(x: torch.Tensor):
    x = x.clone().detach()
    if dist.is_dist_avail_and_initialized():
        torch.distributed.all_reduce(x)
    return torch.isnan(x).item()

def cvt2scalar(value):
    if isinstance(value, (list, tuple)):
        if isinstance(value[0], np.ndarray):
            value = np.concatenate(value, axis=0)
            value = value.mean()
        elif isinstance(value[0], torch.Tensor):
            value = torch.cat(value, dim=0)
            value = value.mean().item()
        else:
            value = sum(value) / len(value)
    elif isinstance(value, np.ndarray):
        value = value.mean()
    elif isinstance(value, torch.Tensor):
        value = value.mean().item()
    return value

class EncoderDecoder(nn.Module):
    def __init__(self,
                 encoder,
                 decoder=None,
                 encoder_only=False,
                 enc_weight_cfg=None,
                 use_aug=False):
        
        super().__init__()
        self.encoder = MODELS.build(encoder)
        self.dynamic_enc_weight = False
        if enc_weight_cfg is not None:
            self.dynamic_enc_weight = True
            self.dynamic_start = enc_weight_cfg['start']
            self.dynamic_end = enc_weight_cfg['end']
        self.use_aug = use_aug
        self.encoder_only = encoder_only
        if not encoder_only:
            self.decoder = MODELS.build(decoder)
    
    def get_dec_aug_input(self, anns, outputs):
        new_attn_masks, new_pred_masks, local_imgs, loss_masks, gan_loss_weight, used_mask = masks_aug(
            outputs['pred_masks'], outputs['logits'], anns['aug_infos'], outputs['decoder_inputs']['attn_mask'])
        raw_decoder_inputs = outputs['decoder_inputs']
        aug_embed, _ = self.encoder.mask_embed(new_pred_masks, tokens=outputs['raw_tokens'][used_mask], extract=False)
        new_decoder_inputs = dict()
        new_decoder_inputs['attn_mask'] = torch.cat([raw_decoder_inputs['attn_mask'], new_attn_masks], dim=0)
        new_decoder_inputs['embeds'] = torch.cat([raw_decoder_inputs['embeds'], aug_embed], dim=0)
        new_decoder_inputs['loss_masks'] = loss_masks
        new_decoder_inputs['ori_bs'] = len(raw_decoder_inputs['attn_mask'])
        gan_loss_weight = gan_loss_weight[:len(new_decoder_inputs['attn_mask'])]
        return new_decoder_inputs, local_imgs, new_pred_masks, gan_loss_weight
        
    def forward(self, inputs, anns=None, enc_loss=False, dec_loss=False, get_local=False):
        outputs = self.encoder(inputs, anns=anns, get_loss=enc_loss)
        if not self.encoder_only:
            targets = inputs.clone().detach()
            if self.use_aug and self.training:
                decoder_inputs, local_imgs, new_pred_masks, gan_loss_weight = self.get_dec_aug_input(anns, outputs)
                targets = torch.cat([targets, local_imgs], dim=0)
                outputs['new_pred_masks'] = new_pred_masks
                outputs['gan_loss_weight'] = gan_loss_weight
                outputs['decoder_inputs'] = decoder_inputs
            else:
                decoder_inputs = outputs['decoder_inputs']
            outputs.update(self.decoder(**decoder_inputs, targets=targets, get_loss=dec_loss, anns=anns))
        # return outputs['recon_samples'] # for get_flops.py
        return outputs

@torch.no_grad()
def masks_aug(pred_masks, logits, aug_infos, attn_masks):
    bs = attn_masks.shape[0]
    pred_masks = pred_masks.clone().detach()
    logits = logits.clone().detach()
    attn_masks = attn_masks.clone().detach()
    soft_masks = logits.softmax(1)
    soft_masks = F.interpolate(soft_masks, (256, 256))
    new_pred_masks, new_attn_masks, local_imgs, loss_masks = [], [], [], []
    gan_loss_weight = torch.ones((bs*2,), device=attn_masks.device, dtype=torch.float32)
    used_mask = torch.zeros((bs,), device=attn_masks.device, dtype=bool)
    idx = 0
    for bs_id, (masks, soft_mask, attn_mask, aug_info) in enumerate(zip(pred_masks, soft_masks, attn_masks, aug_infos)):
        if aug_info is not None:
            valid_mask = find_corres(aug_info['aug_mask'], masks)
            inds = torch.nonzero(valid_mask).squeeze()
            if inds.numel() == 0 or inds.numel() == len(valid_mask):
                continue
            if aug_info['inpaint']:
                attn_mask[valid_mask] = True
                new_masks = masks
            else:
                attn_mask[~valid_mask] = True
                new_masks = warp_affine(soft_mask, inds, aug_info)
                gan_loss_weight[bs+idx] = 0
            new_attn_masks.append(attn_mask)
            new_pred_masks.append(new_masks)
            local_imgs.append(aug_info['local_img'])
            loss_masks.append(aug_info['loss_mask'])
            used_mask[bs_id] = True
            idx += 1
    new_attn_masks = torch.stack(new_attn_masks, dim=0)
    new_pred_masks = torch.stack(new_pred_masks, dim=0)
    local_imgs = torch.stack(local_imgs, dim=0).to(pred_masks.device)
    loss_masks = torch.stack(loss_masks, dim=0).to(pred_masks.device)
    return new_attn_masks, new_pred_masks, local_imgs, loss_masks, gan_loss_weight, used_mask

def find_corres(aug_mask, masks, thr=0.1):
    aug_mask = aug_mask.to(masks.device)
    areas = masks.sum(-1).sum(-1)
    areas[areas==0] = 1
    aug_mask = aug_mask[None,]
    inters = torch.logical_and(aug_mask, masks).sum(-1).sum(-1)
    frac = inters / areas
    return frac >= thr

def warp_affine(masks, inds, aug_info):
    dev = masks.device
    scaling_matrix = aug_info['scaling_matrix']
    translation_matrix = aug_info['translation_matrix']
    n, h, w = masks.shape[0], masks.shape[1], masks.shape[2]
    glb_masks = masks.max(0)[1]
    glb_masks = glb_masks.cpu().numpy()
    inds = inds.tolist()
    if not isinstance(inds, list):
        inds = [inds]
    new_masks = np.zeros((n, h, w), dtype=np.uint8)
    for ind in inds:
        mask = (glb_masks == ind).astype(np.uint8)
        mask = cv2.warpAffine(mask, scaling_matrix, (w, h)) # type: ignore
        mask = cv2.warpAffine(mask, translation_matrix, (w, h)) # type: ignore
        new_masks[ind] = mask
    new_masks = torch.tensor(new_masks, dtype=torch.float32, device=dev)
    new_masks = F.interpolate(new_masks.unsqueeze(1), (64, 64)).squeeze(1)
    return new_masks


class BaseWrapper(nn.Module):
    def __init__(self,
                 g_model: dict,
                 **kwargs):
        super().__init__()
        self.g_model = EncoderDecoder(**g_model)
        self.encoder_only = self.g_model.encoder_only
        self.use_ddp = False
    
    def freeze_g_params(self, params):
        for n, p in self.g_model.named_parameters():
            if n in params:
                p.requires_grad = False
    
    def train_init(self, lr_cfg, log_keys, pretrained_params, checkpoint=None, enc_weight=1.0, dec_weight=1.0):
        if self.use_ddp:
            model_without_ddp = self.g_model.module
        else:
            model_without_ddp = self.g_model

        g_params, lr = get_train_param(model_without_ddp, lr_cfg, pretrained_params)
        self.opt_g = AdamW(g_params, lr=lr, weight_decay=lr_cfg.get('weight_decay'))
        if checkpoint is not None and 'opt_g_state_dict' in checkpoint:
            self.opt_g.load_state_dict(checkpoint['opt_g_state_dict'])

        self.use_schedule = False
        schedule_cfg = lr_cfg.get('schedule_cfg', None)
        if schedule_cfg is not None:
            self.use_schedule = True
            self.max_epoch = schedule_cfg.get('max_epoch')
            self.lr_schedule_g = CosineAnnealingLR(self.opt_g, T_max=self.max_epoch)
            
            if checkpoint is not None and 'schedule_g_state_dict' in checkpoint:
                self.lr_schedule_g.load_state_dict(checkpoint['schedule_g_state_dict'])

        self.clip_grad = False
        if 'clip_grad' in lr_cfg:
            self.clip_grad = True
            self.clip_grad_cfg = lr_cfg.get('clip_grad').get('clip_grad_cfg')
            self.clip_grad_params = get_named_params(self.g_model.named_parameters(), lr_cfg.get('clip_grad').get('clip_grad_params'))
        
        self.use_warm_up = False
        if 'warm_up_cfg' in lr_cfg:
            self.use_warm_up = True
            self.warm_up_epoch = lr_cfg.get('warm_up_cfg').get('epoch')
            
        self.dec_weight = dec_weight
        self.enc_weight = enc_weight

        self.log_keys = set()
        if log_keys is not None:
            self.log_keys = set(log_keys)
    
    def to_ddp(self, device_ids, broadcast_buffers=True,
               find_unused_parameters=False, **kwargs):
        """Wrap models with separate MMDistributedDataParallel.

        It only wraps the modules with parameters.
        """
        self.use_ddp = True
        self.g_model = DDP(
            self.g_model,
            device_ids=device_ids,
            broadcast_buffers=broadcast_buffers,
            find_unused_parameters=find_unused_parameters,
            **kwargs)
    
    def enc_loss_process(self, outputs):
        l1_map = outputs['l1_map'].sum(1).clone().detach()
        l1_map = F.interpolate(l1_map.unsqueeze(1), size=(64, 64))
        avg_op = torch.ones((1, 1, 3, 3), device=l1_map.device) / 9
        l1_map = F.conv2d(l1_map, avg_op, padding=1)
        l1_max = l1_map.flatten(2).max(2, keepdim=True)[0]
        l1_map = l1_map.squeeze(1) / (l1_max + 1e-5)
        enc_weight = (1 - l1_map / l1_max) * 0.9 + 0.1
        outputs['enc_losses']['ce_loss'] = outputs['enc_losses']['ce_loss'] * enc_weight
        return outputs
        
    def forward(self, inputs, anns=None, step_info=None, **kwargs):
        inputs = inputs.to(dist.dev())
        # save_img(inputs, anns['local_imgs'])
        outputs = self.g_model(inputs, anns, **kwargs)

        if self.use_ddp:
            dynamic_enc_weight = self.g_model.module.dynamic_enc_weight
        else:
            dynamic_enc_weight = self.g_model.dynamic_enc_weight
        if dynamic_enc_weight:
            outputs = self.enc_loss_process(outputs, step_info)

        loss, loss_dict = self.loss_post_process(outputs, step_info)
        self.opt_g.zero_grad()
        loss.backward()
        if self.clip_grad:
            torch.nn.utils.clip_grad_norm_(parameters=self.clip_grad_params, **self.clip_grad_cfg)
        self.opt_g.step()
        # self.compute_g_grad_norm()
        return loss_dict
    
    def time_post_process(self, outputs):
        iou_time = sum(outputs['enc_losses']['iou_time'])
        hug_time = sum(outputs['enc_losses']['hug_time'])
        match_time = outputs['enc_losses']['match_time']
        enc_time = outputs['enc_time']
        dec_time = outputs['dec_time']
        return iou_time, hug_time, match_time, enc_time, dec_time
    
    def loss_post_process(self, outputs, step_info):
        log_vars = dict()
        cur_loss_frac = 1 
        if self.use_warm_up:
            if step_info['epoch'] < self.warm_up_epoch:
                cur_loss_frac = min(1, gl.get('global_step') / (self.warm_up_epoch * step_info['steps_per_epoch']))

        enc_losses, dec_losses = [], []
        if 'enc_losses' in outputs:
            for k, v in outputs['enc_losses'].items():
                if 'loss' in k:
                    enc_losses.append(self.enc_weight * v.mean())
                    log_vars[k] = v.mean().item()
                else:
                    log_vars[k] = cvt2scalar(v)
        if 'dec_losses' in outputs:
            for k, v in outputs['dec_losses'].items():
                if 'loss' in k:
                    dec_losses.append(self.dec_weight * v)
                    log_vars[k] = v.item()
                else:
                    log_vars[k] = cvt2scalar(v)
        assert (len(enc_losses) > 0) or (len(dec_losses) > 0)
        losses = sum([*enc_losses, *dec_losses]) * cur_loss_frac
        log_vars['losses'] = losses.item()
        return losses, log_vars
    
    def compute_g_grad_norm(self):
        g_sqsum = 0.0
        for p in self.g_model.parameters():
            g = p.grad
            if g is not None:
                g_sqsum += (g ** 2).sum().item()
        return dict(g_grad_norm=g_sqsum)

    def get_lr(self):
        return dict(lr_g=self.opt_g.state_dict()['param_groups'][0]['lr'])
    
    
class GANWraper(BaseWrapper):
    def __init__(self,
                 g_model: dict,
                 d_model: dict):
        super(GANWraper, self).__init__(g_model)
        self.d_model = GANLoss(**d_model)
    
    def train_init(self, lr_cfg, log_keys, pretrained_params, checkpoint=None, enc_weight=1.0, dec_weight=1.0):
        super(GANWraper, self).train_init(lr_cfg, log_keys, pretrained_params, checkpoint=checkpoint, enc_weight=enc_weight, dec_weight=enc_weight)
        d_params, lr = get_train_param(self.d_model, lr_cfg, None, is_d=True)
        self.opt_d = AdamW(d_params, lr=lr, weight_decay=lr_cfg.get('weight_decay'))
        if checkpoint is not None and 'opt_d_state_dict' in checkpoint:
            self.opt_d.load_state_dict(checkpoint['opt_d_state_dict'])
        
        if self.use_schedule:
            self.lr_schedule_d = CosineAnnealingLR(self.opt_d, T_max=self.max_epoch)
            if checkpoint is not None and 'schedule_d_state_dict' in checkpoint:
                self.lr_schedule_d.load_state_dict(checkpoint['schedule_d_state_dict'])

    def _get_last_layer(self):
        if self.use_ddp:
            g_model = self.g_model.module
        else:
            g_model = self.g_model
        return g_model.decoder.get_last_layer()
    
    def to_ddp(self, device_ids, broadcast_buffers=True,
               find_unused_parameters=False, **kwargs):
        """Wrap models with separate MMDistributedDataParallel.

        It only wraps the modules with parameters.
        """
        super(GANWraper, self).to_ddp(device_ids, 
                                      broadcast_buffers=broadcast_buffers, 
                                      find_unused_parameters=find_unused_parameters, 
                                      **kwargs)  
        self.d_model = DDP(
            self.d_model,
            device_ids=device_ids,
            broadcast_buffers=broadcast_buffers,
            find_unused_parameters=find_unused_parameters,
            **kwargs)
    
    def update_g(self, inputs, anns=None, step_info=None, **kwargs):
        outputs = self.g_model(inputs, anns, **kwargs)
        loss, loss_dict = self.loss_post_process(outputs, step_info)
        recon_samples = outputs['recon_samples']
        # vis_recon(inputs, recon_samples, outputs['pred_masks'])
        rec_loss = []
        for k in outputs['dec_losses']:
            if 'loss' in k:
                rec_loss.append(outputs['dec_losses'][k])
        rec_loss = sum(rec_loss)
        g_loss, g_loss_weight = self.d_model(
            inputs=None,
            reconstructions=recon_samples,
            update_g=True,
            global_step=gl.get('global_step'),
            sample_weight = outputs.get('gan_loss_weight'),
            last_layer=self._get_last_layer(),
            rec_loss=rec_loss)
        loss += g_loss
        loss_dict.update({'gan_loss_g': g_loss.item()})
        loss_dict.update({'gan_loss_g_weight': g_loss_weight.item()})
        self.opt_g.zero_grad()
        if isnan_dist(loss):
            gl.set('num_skips', gl.get('num_skips') + 1)
            if dist.get_rank() == 0:
                print(f'update_g got NaN loss, skipping')
            return loss_dict
        loss.backward()
        if self.clip_grad:
            torch.nn.utils.clip_grad_norm_(parameters=self.clip_grad_params, **self.clip_grad_cfg)
        self.opt_g.step()
        return loss_dict

    def update_d(self, inputs, anns=None):
        with torch.no_grad():
            outputs = self.g_model(inputs, anns)
        recon_samples = outputs['recon_samples']
        d_loss, logits_real, logits_fake = self.d_model(
            inputs=inputs,
            reconstructions=recon_samples,
            update_g=False,
            global_step=gl.get('global_step'),
            sample_weight = outputs.get('gan_loss_weight'),
            last_layer=None,
            rec_loss=None)
        loss_dict = {'gan_loss_d': d_loss.item(),
                     'logits_real': logits_real,
                     'logits_fake': logits_fake}
        self.opt_d.zero_grad()
        if isnan_dist(d_loss):
            gl.set('num_skips', gl.get('num_skips') + 1)
            if dist.get_rank() == 0:
                print(f'update_d got NaN loss, skipping')
            return loss_dict
        d_loss.backward()
        self.opt_d.step()
        return loss_dict

    def forward(self, inputs, anns=None, step_info=None, **kwargs):
        inputs = inputs.to(dist.dev())
        loss_dict = self.update_g(inputs, anns, step_info, **kwargs)
        loss_dict.update(self.update_d(inputs, anns))
        return loss_dict

    def compute_d_grad_norm(self):
        d_sqsum = 0.0
        for p in self.d_model.parameters():
            g = p.grad
            if g is not None:
                d_sqsum += (g ** 2).sum().item()
        return dict(d_grad_norm=d_sqsum)
