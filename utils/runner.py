import os
import re
import numpy as np
import torch
import torch.nn.functional as F
from collections import OrderedDict
from collections.abc import Iterable
from PIL import Image
from tqdm import tqdm
import cv2
import utils.dist as dist
from dataset.imagenet import EpochDataloader, get_val_datas
from dataset.celeba_train import EpochDataloaderFace
# from dataset.imagenet_vq import EpochDataloader, get_val_datas, build_test_dataloader
# from dataset.imagenet_hf import EpochDataloader
from utils import global_vars as gl
from model.model_wrapper import GANWraper, BaseWrapper, EncoderDecoder
from tensorboardX import SummaryWriter
import random
import copy
import time
import math

class Runner:
    def __init__(self,
                 train_mode,
                 use_gan,
                 face_finetune,
                 model_cfg,
                 data_cfg,
                 runner_cfg,
                 logger=None,
                 out_dir='work_dir'):
        self.train_mode = train_mode
        self.use_gan = use_gan
        self.face_finetune = face_finetune
        self.logger = logger
        self.out_dir = out_dir
        if train_mode:
            self._train_init(model_cfg, data_cfg, **runner_cfg)
        else:
            self._test_init(model_cfg, data_cfg, **runner_cfg)

    def _train_init(self,
                    model_cfg,
                    data_cfg,
                    lr_cfg,
                    log_interval,
                    save_interval,
                    gan_start_step=0,
                    num_epochs=300,
                    do_validate=False,
                    tensorboard_path=None,
                    resume_from=None,
                    load_from=None,
                    log_keys=None,
                    enc_loss=True,
                    dec_loss=True,
                    enc_weight=1.0,
                    dec_weight=1.0):
        if self.face_finetune:
            self.train_data = EpochDataloaderFace(self.logger, data_cfg.get('train'), data_cfg.get('train_loader'))
        else:
            self.train_data = EpochDataloader(self.logger, data_cfg.get('train'), data_cfg.get('train_loader'))
        self.steps_per_epoch = len(self.train_data.dataloader)
        gl.set('steps_per_epoch', self.steps_per_epoch)
        gl.set('num_skips', 0)
        if self.use_gan:
            self.model = GANWraper(**model_cfg).to(dist.dev())
        else:
            self.model = BaseWrapper(**model_cfg).to(dist.dev())
        self.log_interval = log_interval
        self.save_interval = save_interval
        self.gan_start_step = gan_start_step
        self.epoch = 0
        self.num_epochs = num_epochs
        
        pretrained_params = None
        checkpoint = None
        if resume_from is not None:
            pretrained_params, checkpoint = self.load_resume(**resume_from)
        elif load_from is not None:
            pretrained_params, checkpoint = self.load(**load_from)

        lr_mode = lr_cfg.get('type')
        if lr_mode == 'frozen_part':
            frozen_params = set(lr_cfg.get('frozen_params'))
            self.model.freeze_params(frozen_params)

        self.distributed = False
        world_size = dist.get_world_size()
        if world_size > 1:
            self.model.to_ddp([dist.get_local_rank()])
            self.distributed = True

        self.model.train_init(lr_cfg, log_keys, pretrained_params, checkpoint=checkpoint,
                              enc_weight=enc_weight, dec_weight=dec_weight)
        self._init_tensorboard(tensorboard_path)

        self.do_validate = do_validate
        if do_validate:
            val_inputs, val_metas = get_val_datas(data_cfg.get('val'))
            self.val_inputs = val_inputs.to(dist.dev())
            self.val_metas = val_metas
            self.norm_mean = np.array([0.485, 0.456, 0.406], dtype=np.float32).reshape((1, 1, -1))
            self.norm_std = np.array([0.229, 0.224, 0.225], dtype=np.float32).reshape((1, 1, -1))

        self.enc_loss = enc_loss
        self.dec_loss = dec_loss
    
    def _init_tensorboard(self, output_path=None):
        if output_path is None:
            output_path = self.out_dir
        if dist.get_rank() == 0:
            self.tensorboard_writer = SummaryWriter(logdir=output_path)
    
    def _write_tensorboard(self, info_dict, global_step):
        info_dict = dist.reduce_dict(info_dict)
        if dist.get_rank() == 0:
            for k, v in info_dict.items():
                self.tensorboard_writer.add_scalar(k, v, global_step=global_step)

    def _test_init(self, model_cfg, data_cfg, load_from, norm_cfg):
        self.model = EncoderDecoder(**model_cfg.get('g_model')).to(dist.dev())
        self.test_loader = build_test_dataloader(data_cfg.get('test'), data_cfg.get('test_loader'))
        self.load(**load_from)
        self.norm_mean = np.array(norm_cfg.get('mean'), dtype=np.float32).reshape((1, 1, -1))
        self.norm_std = np.array(norm_cfg.get('std'), dtype=np.float32).reshape((1, 1, -1))
    
    def load_resume(self, ckpt, ignored_keys=None, transform_keys=None, ckpt_inerpolation=None, 
                    rm_opt_g=False, rm_opt_d=False, rm_schedule_g=False, rm_schedule_d=False):
        if dist.get_rank() == 0:
            print(f'resume training from {ckpt}')
        resume_epoch = int(ckpt.split('/')[-1].split('.')[0])
        self.epoch = resume_epoch + 1
        checkpoint = torch.load(ckpt, map_location='cpu')
        model_state_dict = checkpoint['model_state_dict']

        new_model_state_dict = model_state_dict
        if ignored_keys is not None:
            new_model_state_dict, deleted_keys = self._delete_keys(new_model_state_dict, ignored_keys)
            if dist.get_rank() == 0:
                print(f'The following parameters have NOT been loaded: {deleted_keys}')
        if transform_keys is not None:
            new_model_state_dict = self._transform_keys(new_model_state_dict, transform_keys)
        if ckpt_inerpolation is not None:
            new_model_state_dict = self._ckpt_interpolation(new_model_state_dict, **ckpt_inerpolation)
        self.model.load_state_dict(new_model_state_dict, strict=False)
        
        if rm_opt_g and 'opt_g_state_dict' in checkpoint:
            del checkpoint['opt_g_state_dict']
        if rm_opt_d and 'opt_d_state_dict' in checkpoint:
            del checkpoint['opt_d_state_dict']
        if rm_schedule_g and 'schedule_g_state_dict' in checkpoint:
            del checkpoint['schedule_g_state_dict']
        if rm_schedule_d and 'schedule_d_state_dict' in checkpoint:
            del checkpoint['schedule_d_state_dict']
        
        return new_model_state_dict, checkpoint
    
    def load(self, ckpt, ignored_keys=None, transform_keys=None, ckpt_inerpolation=None,
             rm_opt_g=False, rm_opt_d=False, rm_schedule_g=False, rm_schedule_d=False):
        if dist.get_rank() == 0:
            print(f'loading checkpoint from {ckpt}')
        checkpoint = torch.load(ckpt, map_location='cpu')
        model_state_dict = checkpoint['model_state_dict']
        new_model_state_dict = model_state_dict
        if ignored_keys is not None:
            new_model_state_dict, deleted_keys = self._delete_keys(new_model_state_dict, ignored_keys)
            if dist.get_rank() == 0:
                print(f'The following parameters have NOT been loaded: {deleted_keys}')
        if transform_keys is not None:
            new_model_state_dict = self._transform_keys(new_model_state_dict, transform_keys)
        if ckpt_inerpolation is not None:
            new_model_state_dict = self._ckpt_interpolation(new_model_state_dict, **ckpt_inerpolation)
        self.model.load_state_dict(new_model_state_dict, strict=False)
        
        if rm_opt_g and 'opt_g_state_dict' in checkpoint:
            del checkpoint['opt_g_state_dict']
        if rm_opt_d and 'opt_d_state_dict' in checkpoint:
            del checkpoint['opt_d_state_dict']
        if rm_schedule_g and 'schedule_g_state_dict' in checkpoint:
            del checkpoint['schedule_g_state_dict']
        if rm_schedule_d and 'schedule_d_state_dict' in checkpoint:
            del checkpoint['schedule_d_state_dict']
        
        return new_model_state_dict, checkpoint
    
    def _delete_keys(self, checkpoint, ignored_keys):
        assert isinstance(ignored_keys, Iterable)
        deleted_keys = []
        for key in list(checkpoint.keys()):
            for ig_pattern in ignored_keys:
                if re.search(ig_pattern, key) is not None:
                    del checkpoint[key]
                    deleted_keys.append(key)
                    break
        return checkpoint, deleted_keys
    
    def _transform_keys(self, checkpoint, transform_keys):        
        new_checkpoint = OrderedDict()
        for n in checkpoint:
            for trans_k in transform_keys:
                if trans_k in n:
                    new_n = n.replace(trans_k, transform_keys[trans_k])
                    new_checkpoint[new_n] = checkpoint[n]
        return new_checkpoint
    
    def _ckpt_interpolation(self, checkpoint, key, new_size=None, scale_factor=None):
        if 'pos_embed' in key:
            old_pe = checkpoint[key]
            old_size, dim = old_pe.shape[1], old_pe.shape[2]
            old_size = int(math.sqrt(old_size))
            old_pe = old_pe.view(-1, old_size, old_size, dim).permute(0, 3, 1, 2).contiguous()
            new_pe= F.interpolate(old_pe, size=new_size, scale_factor=scale_factor, mode='bicubic', align_corners=True)
            new_pe = new_pe.permute(0, 2, 3, 1).view(1, -1, dim).contiguous()
            checkpoint[key] = new_pe
        elif 'patch_embed' in key:
            old_pe = checkpoint[key]
            new_pe= F.interpolate(old_pe, size=new_size, scale_factor=scale_factor, mode='bicubic', align_corners=True)
            checkpoint[key] = new_pe
        return checkpoint
    
    def train_loop(self):
        if dist.is_dist_avail_and_initialized():
            torch.distributed.barrier()
        for epoch in range(self.epoch, self.num_epochs):
            if self.distributed:
                self.train_data.sampler.set_epoch(epoch)
            for step, data in enumerate(self.train_data.dataloader):
                global_step = step + 1 + self.steps_per_epoch * epoch
                gl.set('global_step', global_step)
                step_info = dict(global_step = global_step, epoch=epoch, steps_per_epoch=self.steps_per_epoch)
                loss_dict = self.model(**data, step_info=step_info, enc_loss=self.enc_loss, dec_loss=self.dec_loss)
                if (step+1) % self.log_interval == 0:
                    info_dict = dict()
                    info_dict.update(self.model.compute_g_grad_norm())
                    info_dict.update(self.model.get_lr())
                    if self.use_gan:
                        info_dict.update(self.model.compute_d_grad_norm())
                    info_dict.update(loss_dict)
                    info_dict['num_skips'] = gl.get('num_skips')
                    self._log_info(info_dict, step=step+1, epoch=epoch)
                    self._write_tensorboard(info_dict, self.steps_per_epoch * epoch + step + 1)
            if (epoch+1) % self.save_interval == 0:
                self.save(epoch=epoch)
                if self.do_validate:
                    if self.use_linear or self.use_trans_enc:
                        self.val_cls(epoch)
                    else:
                        self.val_recon(epoch)

    def wt_tuning(self, epoch):
        if self.wt_tuning_epochs is not None:
            if epoch in self.wt_tuning_epochs:
                self.enc_weight = 0.5 * self.enc_weight

    def test_loop(self):
        rank = dist.get_rank()
        self.model.eval()
        with torch.no_grad():
            if rank== 0:
                pbar = tqdm(total=len(self.test_loader))
            num_groups_all = []
            for idx, data in enumerate(self.test_loader):
                samples, img_metas = data
                samples = samples.to(dist.dev())
                results = self.model(samples)
                num_groups = torch.tensor(results['enc_losses']['num_groups'], device=dist.dev())
                if dist.get_world_size() > 1:
                    torch.distributed.all_reduce(num_groups)
                    num_groups = num_groups / dist.get_world_size()
                num_groups_all.append(num_groups)
                self.save_recon_results(results, img_metas)
                if rank == 0:
                    pbar.update()
            if rank == 0:
                num_groups_all = torch.tensor(num_groups_all)
                print(f'avg num_groups: {num_groups_all.mean()}')
    
    def save_recon_results(self, results, img_metas):
        for recon_sample, img_meta in zip(results['recon_samples'], img_metas):  
            ori_shape = img_meta['ori_shape']
            filename = img_meta['filename']
            H, W = ori_shape
            sample = recon_sample.permute(1, 2, 0).cpu().numpy()
            sample = cv2.resize(sample, (W, H))
            sample = (sample * self.norm_std + self.norm_mean)
            sample = sample.clip(0, 1)
            sample = (sample * 255).astype(np.uint8)
            Image.fromarray(sample).save(os.path.join(self.out_dir, filename))
        
    def _log_info(self, info_dict, step=0, epoch=0):
        self.logger.info('-----------------------')
        self.logger.info(f'epoch: {epoch}, step: {step} / {self.steps_per_epoch}')
        for k, v in info_dict.items():
            self.logger.info(f'{k}: {v}')

    def save(self, epoch=0):
        if dist.get_rank() == 0:
            self.logger.info(f"saving epoch: {epoch} done!")
            ckpt_dict = {
                'model_state_dict': self.model.state_dict(),
                'opt_g_state_dict': self.model.opt_g.state_dict(),
            }
            if hasattr(self.model, 'opt_d'):
                ckpt_dict['opt_d_state_dict'] = self.model.opt_d.state_dict()
            if hasattr(self.model, 'lr_schedule_g'):
                ckpt_dict['schedule_g_state_dict'] = self.model.lr_schedule_g.state_dict()
            if hasattr(self.model, 'lr_schedule_d'):
                ckpt_dict['schedule_d_state_dict'] = self.model.lr_schedule_d.state_dict()
            ckpt_file = os.path.join(self.out_dir, f'{epoch}.pth')
            torch.save(ckpt_dict, ckpt_file)
        if self.model.use_ddp:
            torch.distributed.barrier()
    
    def val_cls(self, epoch):
        self.model.eval()
        if dist.get_rank() == 0:
            self.logger.info(f"doing validation after epoch: {epoch} .....")
        correct = 0
        total = 0
        with torch.no_grad():
            for data in self.val_data.dataloader:
                log_dict = self.model(**data)
                correct += log_dict['correct']
                total += log_dict['total']
        info_dict = dict(correct=correct, total=total)
        info_dict = dist.reduce_dict(info_dict, average=False)
        correct = info_dict['correct'].item()
        total = info_dict['total'].item()
        acc = correct / total
        self.logger.info(f'val_acc: {acc} ({correct}/{total})')
        if dist.get_rank() == 0:
            self.tensorboard_writer.add_scalar('val_acc', acc, global_step=epoch)
        self.model.train()
        if self.model.use_ddp:
            torch.distributed.barrier()

    def val_recon(self, epoch):
        self.model.eval()
        if dist.get_rank() == 0:
            self.logger.info(f"doing validation after epoch: {epoch} .....")
            with torch.no_grad():
                inputs = copy.deepcopy(self.val_inputs)
                results = self.model.model_without_ddp(inputs)
                self.write_val_img_to_tsbd(self.val_metas, results, epoch)
        self.model.train()
        if self.model.use_ddp:
            torch.distributed.barrier()
    
    def write_val_img_to_tsbd(self, datas, results, epoch):
        output_dir = os.path.join(self.out_dir, 'validation', f'epoch_{epoch}')
        os.makedirs(output_dir, exist_ok=True)
        for idx, data in enumerate(datas):
            H, W = data['ori_shape']
            recon_sample = results['recon_samples'][idx]
            recon_sample = recon_sample.permute(1, 2, 0).cpu().numpy()
            recon_sample = cv2.resize(recon_sample, (W, H))
            recon_sample = (recon_sample * self.norm_std + self.norm_mean)
            recon_sample = recon_sample.clip(0, 1)
            recon_sample = (recon_sample * 255).astype(np.uint8)
            ori_sample = data['ori_img']
            img_masks_hard, img_masks_soft, num_groups = self.get_pano_results(ori_sample, results['pred_masks_soft'][idx], results['pred_masks_hard'][idx], H, W)
            img = np.concatenate([ori_sample, recon_sample], axis=0)
            all_output = np.concatenate([img, img_masks_hard, img_masks_soft], axis=1)
            Image.fromarray(all_output).save(os.path.join(output_dir, f'{idx}_{num_groups}.png'))
            
    def get_pano_results(self, ori_sample, masks_soft, masks_hard, H, W):
        masks_hard = F.interpolate(masks_hard.unsqueeze(0), size=(H, W), mode='nearest')[0].cpu().numpy()
        masks_hard_pano, num_groups = masks2pano(masks_hard)
        img_masks_hard = cv2.addWeighted(ori_sample, 0.5, masks_hard_pano, 0.5, 0)
        hard_output = np.concatenate([img_masks_hard, masks_hard_pano], axis=0)

        masks_soft = F.interpolate(masks_soft.unsqueeze(0), size=(H, W), mode='bilinear')
        index = masks_soft.max(-3, keepdim=True)[1]
        masks_soft_hard = torch.zeros_like(masks_soft, memory_format=torch.legacy_contiguous_format).scatter_(-3, index, 1.0)
        masks_soft_hard = masks_soft_hard[0].cpu().numpy()
        masks_soft_pano, _ = masks2pano(masks_soft_hard)
        img_masks_soft = cv2.addWeighted(ori_sample, 0.5, masks_soft_pano, 0.5, 0)
        soft_output = np.concatenate([img_masks_soft, masks_soft_pano], axis=0)
        return hard_output, soft_output, num_groups


