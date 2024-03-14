import os
import cv2
import random
import numpy as np
import math
import torch
from torch import nn
import torch.nn.functional as F
from timm.models import VisionTransformer
from einops import rearrange
from mmdet.registry import MODELS
from PIL import Image
from scipy.optimize import linear_sum_assignment
from ..transformer import TransformerDecoderLayer, PositionEmbeddingSine, TwoWayTransformer, MLPBlock, LayerNorm2d, PatchEmbed


class DownsampleConv(nn.Module):
    def __init__(self, channels, mode='nearest') -> None:
        super().__init__()
        self.mode = mode
        out_conv = nn.Conv2d(channels*4, channels, kernel_size=1, stride=1, padding=0)
        nn.init.constant_(out_conv.weight.data, 0)
        nn.init.constant_(out_conv.bias.data, 0)
        self.conv_layers = nn.Sequential(
            nn.Conv2d(channels, channels*4, kernel_size=2, stride=2, padding=0),
            LayerNorm2d(channels*4),
            nn.GELU(),
            out_conv)

    def forward(self, x):
        out_x = F.interpolate(x, scale_factor=0.5, mode=self.mode)
        return out_x + self.conv_layers(x)


class MaskEmbedder(nn.Module):
    def __init__(self, img_size, patch_size, mask_size, rgb_dim, in_dim=256, out_dim=256, hidden_dim=512, masks_detach=False, add_rgb=False, add_downsample=False, use_attn=True) -> None:
        super().__init__()
        # rgb info

        self.use_attn = use_attn
        if use_attn:
            self.add_rgb = add_rgb
            if add_rgb:
                self.rgb_embed = PatchEmbed(img_size=img_size, patch_size=patch_size, embed_dim=rgb_dim)

            # img info grouping
            if add_rgb:
                self.src_embed = MLPBlock(rgb_dim+in_dim, mlp_dim=hidden_dim, out_dim=in_dim)
            else:
                self.src_embed = MLPBlock(in_dim, mlp_dim=hidden_dim, out_dim=in_dim)
            self.src_norm = nn.LayerNorm(in_dim)
        
            self.assign_attn = nn.MultiheadAttention(embed_dim=in_dim, num_heads=1, batch_first=True)
            self.attn_norm = nn.LayerNorm(in_dim)
            
        self.token_embed = nn.Linear(in_dim, hidden_dim)

        # region info
        self.add_downsample = add_downsample
        if add_downsample:
            self.down_layer = DownsampleConv(1)
        kernel_size = mask_size // 32
        self.kernel_size = kernel_size
        self.mask_conv = nn.Conv2d(1, 1, kernel_size=kernel_size, stride=kernel_size, padding=0)
        self.mask_embed = nn.Linear(1024, hidden_dim)
        self.hidden_norm = nn.LayerNorm(hidden_dim)
        self.out_embed = MLPBlock(hidden_dim, hidden_dim, out_dim=out_dim)
        self.masks_detach = masks_detach
    
    def get_attn_mask(self, masks):
        attn_mask = torch.logical_not(masks.bool()).flatten(-2)
        unused_flags = attn_mask.all(-1)
        attn_mask[unused_flags] = False
        return attn_mask

    def forward(self, masks, img=None, hr_feat=None, mask_token=None, tokens=None, extract=True):
        if extract:
            if self.use_attn:
                if self.add_rgb:
                    rgb = self.rgb_embed(img)
                    src = torch.cat([rgb, hr_feat], dim=1)
                else:
                    src = hr_feat
                src = src.flatten(-2).transpose(1, 2).contiguous()
                src = self.src_norm(self.src_embed(src))
                attn_mask = self.get_attn_mask(masks)
                tokens = self.assign_attn(mask_token, src, src, attn_mask=attn_mask)[0]
                tokens = self.attn_norm(tokens + mask_token)
            else:
                tokens = mask_token
            tokens = self.token_embed(tokens)
        else:
            assert tokens is not None

        if self.masks_detach:
            masks = masks.detach()
        masks = masks * 2 - 1
        bs, h, w = masks.shape[0], masks.shape[-2], masks.shape[-1]
        masks = masks.view(-1, 1, h, w)
        if self.add_downsample:
            masks = self.down_layer(masks)
        masks = self.mask_conv(masks)
        masks = masks.view(bs, -1, 32, 32)
        masks = masks.flatten(-2)
        mask_embeds = self.mask_embed(masks)
        new_tokens = tokens + mask_embeds
        return self.out_embed(self.hidden_norm(new_tokens)), tokens


class SimpleFPN(nn.Module):
    def __init__(self, in_ch, ch) -> None:
        super().__init__()
        self.ch = ch
        self.input_layer = nn.Sequential(
            nn.Conv2d(
                in_ch,
                ch,
                kernel_size=1,
                bias=False),
            LayerNorm2d(ch),
            nn.Conv2d(
                ch,
                ch,
                kernel_size=3,
                padding=1,
                bias=False),
            LayerNorm2d(ch))
        self.fpn1 = nn.Sequential(
            nn.ConvTranspose2d(ch, ch, kernel_size=2, stride=2),
            LayerNorm2d(ch),
            nn.GELU(),
            nn.ConvTranspose2d(ch, ch, kernel_size=2, stride=2))

        self.fpn2 = nn.ConvTranspose2d(ch, ch, kernel_size=2, stride=2)
        self.fpn3 = nn.Identity()
    
    def to1d(self, x, bs):
        return x.view(bs, self.ch, -1).transpose(1, 2).contiguous()

    def forward(self, x, min_size):
        bs = x.shape[0]
        x = self.input_layer(x)
        out_srcs = (
            self.to1d(self.fpn3(x), bs), 
            self.to1d(self.fpn2(x), bs), 
            self.to1d(self.fpn1(x), bs))
        out_sizes = (min_size, min_size*2, min_size*4)
        return out_sizes, out_srcs


class SimpleFPNTwoLayer(nn.Module):
    def __init__(self, in_ch, ch) -> None:
        super().__init__()
        self.ch = ch
        self.input_layer = nn.Sequential(
            nn.Conv2d(
                in_ch,
                ch,
                kernel_size=1,
                bias=False),
            LayerNorm2d(ch),
            nn.Conv2d(
                ch,
                ch,
                kernel_size=3,
                padding=1,
                bias=False),
            LayerNorm2d(ch))

        self.fpn1 = nn.Sequential(nn.ConvTranspose2d(ch, ch, kernel_size=2, stride=2), LayerNorm2d(ch))
        self.fpn2 = LayerNorm2d(ch)
    
    def to1d(self, x, bs):
        return x.view(bs, self.ch, -1).transpose(1, 2).contiguous()

    def forward(self, x, min_size):
        bs = x.shape[0]
        x = self.input_layer(x)
        out_srcs = (
            self.to1d(self.fpn2(x), bs), 
            self.to1d(self.fpn1(x), bs))
        out_sizes = (min_size, min_size*2)
        return out_sizes, out_srcs


class MaskDecoder(nn.Module):
    def __init__(self,
                 input_dim,
                 embed_dim, 
                 depth, 
                 num_heads,
                 num_queries,
                 rgb_cfg = None,
                 mlp_ratio = 4,
                 with_self_attn=True,
                 dropout=0.0,
                 get_internm=False,
                 assign_merge=False,
                 use_mlp_mix=False,
                 use_bidirect=True,
                 gumbel=False):
        super().__init__()
        self.depth = depth
        self.num_queries = num_queries
        self.num_heads = num_heads
        self.get_internm = get_internm

        if rgb_cfg is not None:
            self.use_rgb = True
            self.up_layers = SimpleFPNTwoLayer(input_dim, embed_dim)
            self.rgb_embed = PatchEmbed(**rgb_cfg)
            self.rgb_merge = nn.Sequential(
                nn.Conv2d(
                    rgb_cfg['embed_dim'] + embed_dim,
                    embed_dim,
                    kernel_size=1,
                    bias=False),
                LayerNorm2d(embed_dim),
                nn.Conv2d(
                    embed_dim,
                    embed_dim,
                    kernel_size=3,
                    padding=1,
                    bias=False),
                LayerNorm2d(embed_dim))
        else:
            self.use_rgb = False
            self.up_layers = SimpleFPN(input_dim, embed_dim)

        self.query_feat = nn.Embedding(num_queries, embed_dim)
        self.query_embed = nn.Embedding(num_queries, embed_dim)
        self.pe_layer = PositionEmbeddingSine(num_pos_feats = embed_dim // 2, normalize=True)

        self.layers = nn.ModuleList()
        for i in range(depth):
            self_attn = (i != 0) and with_self_attn
            self.layers.append(TransformerDecoderLayer(embed_dim, num_heads, with_self_attn=self_attn, mlp_ratio=mlp_ratio, dropout=dropout))
        self.embed_dim = embed_dim
        
        if get_internm:
            self.internm_embed = nn.Sequential(
                nn.Linear(embed_dim, embed_dim), nn.GELU(),
                nn.Linear(embed_dim, embed_dim))
        
        self.use_mlp_mix = use_mlp_mix
        if use_mlp_mix:
            self.mlp_inter = MLPBlock(num_queries, int(embed_dim*0.5), out_dim=num_queries)
            self.norm_post_tokens = nn.LayerNorm(embed_dim)

        self.use_bidirect = use_bidirect
        if use_bidirect:
            self.pre_assign_attn = TwoWayTransformer(
                depth = 2,
                embedding_dim = embed_dim,
                num_heads = num_heads)
        self.out_embed = nn.Sequential(
            nn.Linear(embed_dim, embed_dim), nn.GELU(),
            nn.Linear(embed_dim, embed_dim))
        
        self.assign_merge = assign_merge
        self.gumbel = gumbel
        if assign_merge:
            self.out_proj = nn.Linear(embed_dim, embed_dim)
            nn.init.constant_(self.out_proj.weight.data, 0)
            nn.init.constant_(self.out_proj.bias.data, 0)
            self.out_norm = nn.LayerNorm(embed_dim)


    def _reset_parameters(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)
    
    def get_mask(self, logits):
        tau = 1.0
        if self.training and self.gumbel:
            gumbel_dist = torch.distributions.gumbel.Gumbel(
                torch.tensor(0., device=logits.device, dtype=logits.dtype),
                torch.tensor(1., device=logits.device, dtype=logits.dtype))
            gumbels = gumbel_dist.sample(logits.shape)
            gumbels = (logits + gumbels) / tau  # ~Gumbel(logits,tau)
            y_soft = gumbels.softmax(1)
        else:
            y_soft = logits.softmax(1)
        index = y_soft.max(1, keepdim=True)[1]
        y_hard = torch.zeros_like(logits, memory_format=torch.legacy_contiguous_format).scatter_(1, index, 1.0)
        # Straight through.
        if self.training:
            ret = y_hard - y_soft.detach() + y_soft
        else:
            ret = y_hard
        return ret
    
    def project_group_token(self, group_tokens):
        # [B, S_2, C] <- [B, S_1, C]
        projected_group_tokens = self.mlp_inter(group_tokens.transpose(1, 2)).transpose(1, 2)
        projected_group_tokens = self.norm_post_tokens(projected_group_tokens)
        return projected_group_tokens
    
    def merge_after_assign(self, pred_masks, mask_token, mask_feat):
        weights = pred_masks.flatten(2)
        mask_feat = mask_feat.flatten(2).transpose(1, 2).contiguous()
        weights = weights / (weights.sum(dim=-1, keepdim=True) + 1.0)
        new_mask_token = weights @ mask_feat
        new_mask_token = mask_token + self.out_proj(new_mask_token)
        return self.out_norm(new_mask_token)
    
    def merge_rgb(self, srcs, img, bs, hw_size):
        rgb_info = self.rgb_embed(img)
        new_src_l = srcs[-1].transpose(1, 2).view(bs, -1, hw_size, hw_size).contiguous()
        new_src_l = torch.cat([rgb_info, new_src_l], dim=1)
        new_src_l = self.rgb_merge(new_src_l)
        return (*srcs[:-1], new_src_l.flatten(2).transpose(1, 2).contiguous())
    
    def get_last_layer(self):
        return self.out_embed[-1].weight

    def forward(self, img, src, bs):
        hw_size = int(math.sqrt(src.shape[1]))

        # up sample
        hw_sizes, srcs = self.up_layers(
            src.transpose(1, 2).view(bs, -1, hw_size, hw_size).contiguous(), hw_size)
        image_pes = []
        num_srcs = len(srcs)
        for ind in range(num_srcs):
            image_pes.append(self.pe_layer(bs, hw_sizes[ind], src.device))
        
        if self.use_rgb:
            srcs = self.merge_rgb(srcs, img, bs, hw_sizes[-1])
        
        # token to img attn
        query_embed = self.query_embed.weight.unsqueeze(0).repeat(bs, 1, 1)
        tgts = self.query_feat.weight.unsqueeze(0).repeat(bs, 1, 1)
        internm_masks = []
        if self.get_internm:
            interm_mask_feat = srcs[-1].transpose(1, 2).view(bs, -1, hw_sizes[-1],hw_sizes[-1]).contiguous()
        for layer_id, layer in enumerate(self.layers):
            src_id = layer_id % num_srcs
            tgts = layer(tgts, srcs[src_id], query_pos=query_embed, pos=image_pes[src_id])
            if self.get_internm:
                if src_id == num_srcs - 1:
                    interm_mask_token = self.internm_embed(tgts)
                    internm_masks.append(torch.einsum('bnc,bchw->bnhw', interm_mask_token, interm_mask_feat))
        
        # two way attn
        if self.use_mlp_mix:
            tgts = self.project_group_token(tgts)
        if self.use_bidirect:
            mask_token, mask_feat = self.pre_assign_attn(srcs[-1], image_pes[-1], tgts)
        else:
            mask_token, mask_feat = tgts, srcs[-1]

        # final prediction
        mask_token = self.out_embed(mask_token)
        mask_feat = mask_feat.transpose(1, 2).view(bs, -1, hw_sizes[-1], hw_sizes[-1]).contiguous()
        logits = torch.einsum('bnc,bchw->bnhw', mask_token, mask_feat)
        pred_masks = self.get_mask(logits)
        if self.assign_merge:
            mask_token = self.merge_after_assign(pred_masks, mask_token, mask_feat)
        return logits, pred_masks, mask_feat, mask_token, internm_masks


def build_vit(img_size, patch_size, embed_dim, depth, num_heads):
    vit = VisionTransformer(
        img_size = img_size,
        patch_size = patch_size,
        embed_dim = embed_dim,
        depth = depth,
        num_heads = num_heads,
        num_classes = 0,
        global_pool = '',
        class_token = False)
    return vit


@MODELS.register_module()
class ReaderEncoder(nn.Module):
    def __init__(self, backbone, mask_decoder, mask_embed, encoder_only=False, face_finetune=False) -> None:
        super().__init__()
        self.backbone = build_vit(**backbone)
        self.mask_decoder = MaskDecoder(**mask_decoder)
        self.encoder_only = encoder_only
        if not encoder_only:
            self.mask_embed = MaskEmbedder(**mask_embed)
        self.get_internm = self.mask_decoder.get_internm
        if self.get_internm:
            pass
        self.face_finetune = face_finetune
        if face_finetune:
            self.bnd_op = LaplacianOperator(boundary_width=2)
    
    def get_loss_interm(self, internm_masks, binary_gts, preds_inds):
        pass
    
    def get_loss(self, x, logits, pred_masks, internm_masks, anns):
        if 'gt_masks' in anns:
            gt_masks = anns['gt_masks']
        else:
            gt_masks = anns['low_lvl_masks']
        
        formulated_gts, ious, binary_gts, preds_inds = get_formulated_gt(pred_masks, gt_masks)
        # for idx, (img, mask) in enumerate(zip(x, formulated_gts)):
        #     visualize(img, mask, idx)
        losses = dict()
        if self.get_internm:
            self.get_loss_interm(internm_masks, binary_gts, preds_inds)
        if self.face_finetune:
            losses['ce_loss'] = F.cross_entropy(logits, formulated_gts)
        else:
            losses['ce_loss'] = F.cross_entropy(logits, formulated_gts, reduction='none')
        losses['ious'] = ious
        return losses
    
    def get_loss_face(self, x, logits, pred_masks, internm_masks, anns):
        if 'gt_masks' in anns:
            gt_masks = anns['gt_masks']
        else:
            gt_masks = anns['low_lvl_masks']
        
        pos_gts, neg_gts, ious = get_formulated_gt_face(pred_masks, gt_masks, self.bnd_op)
        # for idx, (img, mask) in enumerate(zip(x, formulated_gts)):
        #     visualize(img, mask, idx)
        losses = dict()
        losses['pos_ce_loss'] = F.cross_entropy(logits, pos_gts)
        neg_ce_loss = F.relu(F.softmax(logits, 1) * neg_gts - 1/256)
        weight = (pred_masks.shape[2]*pred_masks.shape[3]) / neg_gts.flatten(1).sum(-1)
        weight = weight.unsqueeze(-1)
        losses['neg_ce_loss'] = (neg_ce_loss.flatten(1) * weight).mean()
        losses['ious'] = ious
        return losses

    def get_attn_mask(self, masks):
        bs, n, _, _ = masks.shape
        attn_mask = torch.ones((bs, n), dtype=torch.bool, device=masks.device)
        used_flags = masks.any(-1).any(-1)
        attn_mask[used_flags] = False
        return attn_mask
    
    def get_decoder_inputs(self, img, pred_masks, hr_feat, mask_token):
        attn_mask = self.get_attn_mask(pred_masks)
        all_embeds, raw_tokens = self.mask_embed(pred_masks, img=img, hr_feat=hr_feat, mask_token=mask_token)
        
        num_groups = attn_mask.sum() / len(attn_mask)
        return dict(attn_mask = attn_mask, embeds = all_embeds), self.mask_decoder.num_queries - num_groups.item(), raw_tokens
    
    def forward(self, x, anns=None, get_loss=False):
        outputs = dict()
        bs = x.shape[0]
        img_embed = self.backbone(x)
        logits, pred_masks, hr_feat, mask_token, internm_masks = self.mask_decoder(x, img_embed, bs)
        # return logits
        outputs['pred_masks'] = pred_masks
        outputs['logits'] = logits
        outputs['enc_losses'] = dict()
        num_groups = None
        if not self.encoder_only:
            outputs['decoder_inputs'], num_groups, raw_tokens = self.get_decoder_inputs(x, pred_masks, hr_feat, mask_token)
            outputs['enc_losses'].update({'num_groups': num_groups})
            outputs['raw_tokens'] = raw_tokens
        if get_loss:
            if not self.face_finetune:
                enc_losses = self.get_loss(x, logits, pred_masks, internm_masks, anns)
            else:
                enc_losses = self.get_loss_face(x, logits, pred_masks, internm_masks, anns)
            outputs['enc_losses'].update(enc_losses)
        # return outputs['decoder_inputs']['embeds']
        if get_loss and self.face_finetune:
            neg_weight = calculate_adaptive_weight(outputs['enc_losses']['pos_ce_loss'], outputs['enc_losses']['neg_ce_loss'], self.mask_decoder.get_last_layer())
            outputs['enc_losses']['neg_ce_loss'] = outputs['enc_losses']['neg_ce_loss'] * neg_weight * 0.5
            outputs['enc_losses']['neg_weight'] = neg_weight
        return outputs


def calculate_adaptive_weight(ce_loss, neg_loss, last_layer):
    ce_grads = torch.autograd.grad(ce_loss, last_layer, retain_graph=True)[0]
    neg_grads = torch.autograd.grad(neg_loss, last_layer, retain_graph=True)[0]
    d_weight = torch.norm(ce_grads) / (torch.norm(neg_grads) + 1e-4)
    d_weight = torch.clamp(d_weight, 0.0, 1e4).detach()
    d_weight = d_weight
    return d_weight  


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

    def forward(self, mask_target):
        neg_mask_target = 1 - mask_target.clone()
        pad = (self.boundary_width, self.boundary_width, self.boundary_width, self.boundary_width)
        neg_mask_target = F.pad(neg_mask_target, pad, mode='constant', value=1)
        # neg_boundary
        neg_boundary_targets = self.laplacian_kernel(neg_mask_target)
        neg_boundary_targets = neg_boundary_targets.clamp(min=0) / float(self.kernel_size ** 2)
        neg_boundary_targets[neg_boundary_targets > 0.1] = 1
        neg_boundary_targets[neg_boundary_targets <= 0.1] = 0
        return neg_boundary_targets
    

@torch.no_grad()
def get_formulated_gt(preds, ori_gts, ignore_index=-100):
    formulated_gts = []
    all_matched_ious = []
    binary_gts = []
    preds_inds = []
    for bs_id, gt in enumerate(ori_gts):
        if gt is None:
            fake_gt = torch.ones((64, 64), dtype=preds.dtype, device=preds.device) * ignore_index
            formulated_gts.append(fake_gt)
            continue
        gt_masks = gt.masks
        pred_masks = preds[bs_id]
        gt_masks = torch.tensor(gt_masks, dtype=preds.dtype, device=preds.device)
        ious = batch_masks_iou_torch(pred_masks, gt_masks)
        iou_costs = (1 - ious).cpu().numpy()
        p, t = linear_sum_assignment(iou_costs)
        all_matched_ious.append(ious[p, t])
        p = torch.tensor(p, dtype=preds.dtype, device=preds.device).view(-1, 1, 1)
        formulated_gt = gt_masks[t] * p
        formulated_gt = formulated_gt.sum(0)
        invalid_mask = (gt_masks.sum(0) != 1)
        formulated_gt[invalid_mask] = ignore_index
        formulated_gts.append(formulated_gt)
        binary_gts.append(gt_masks[t])
        preds_inds.append(p)
    formulated_gts = torch.stack(formulated_gts, dim=0).long()
    return formulated_gts, all_matched_ious, binary_gts, preds_inds


@torch.no_grad()
def get_formulated_gt_face(preds, ori_gts, bnd_op, ignore_index=-100):
    pos_gts, pred_inds = [], []
    neg_gts = torch.zeros_like(preds)
    all_matched_ious = []
    for bs_id, gt in enumerate(ori_gts):
        gt_masks = gt.masks
        pred_masks = preds[bs_id]
        gt_masks = torch.tensor(gt_masks, dtype=preds.dtype, device=preds.device)
        boundary = bnd_op(gt_masks.unsqueeze(1)).squeeze(1)
        ious = batch_masks_iou_torch(pred_masks, gt_masks)
        iou_costs = (1 - ious).cpu().numpy()
        p, t = linear_sum_assignment(iou_costs)
        all_matched_ious.append(ious[p, t])
        p_torch = torch.tensor(p, dtype=preds.dtype, device=preds.device).view(-1, 1, 1)
        # pos gt
        pos_gt = gt_masks[t] * p_torch
        pos_gt = pos_gt.sum(0)
        pos_invalid_mask = (gt_masks.sum(0) != 1)
        pos_gt[pos_invalid_mask] = ignore_index
        pos_gts.append(pos_gt)
        # neg gt
        neg_gts[bs_id][p] = boundary[t]
        pred_inds.append(p)
    pos_gts = torch.stack(pos_gts, dim=0).long()
    return pos_gts, neg_gts, all_matched_ious

@torch.no_grad()
def batch_masks_iou_torch(masks1, masks2):
    masks1 = masks1[:, None, ]
    masks2 = masks2[None, ]
    i = torch.logical_and(masks1, masks2).sum(-1).sum(-1)
    u = torch.logical_or(masks1, masks2).sum(-1).sum(-1)
    u[u==0] = 4096
    return i / u

