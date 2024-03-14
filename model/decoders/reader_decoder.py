import os
import math
import torch
import numpy as np
from torch import nn
import torch.nn.functional as F
from mmdet.registry import MODELS
from PIL import Image
from ..losses.perceptual import LPIPS
from ..transformer import TransformerDecoderLayer, PositionEmbeddingSine, MLPBlock


class Upsample(nn.Module):
    def __init__(self, mode='nearest', use_conv=False, channels=None, zero_conv=True):
        super().__init__()
        self.use_conv = use_conv
        self.mode = mode
        if use_conv:
            assert channels is not None
            self.conv = nn.Conv2d(channels, channels, 3, padding=1)
            if zero_conv:
                nn.init.constant_(self.conv.weight.data, 0)
                nn.init.constant_(self.conv.bias.data, 0)

    def forward(self, x):
        x = F.interpolate(x, scale_factor=2, mode=self.mode)
        if self.use_conv:
            x = x + self.conv(x)
        return x


class LayerNorm2d(nn.Module):
    def __init__(self, num_channels: int, eps: float = 1e-6) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.ones(num_channels))
        self.bias = nn.Parameter(torch.zeros(num_channels))
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        u = x.mean(1, keepdim=True)
        s = (x - u).pow(2).mean(1, keepdim=True)
        x = (x - u) / torch.sqrt(s + self.eps)
        x = self.weight[:, None, None] * x + self.bias[:, None, None]
        return x
    

class ResBlock(nn.Module):

    def __init__(self, in_channels, out_channels, k=3, dropout=0, zero_out=False):
        super().__init__()
        if k == 3:
            pad = 1
        elif k == 1:
            pad = 0
        else:
            raise NotImplementedError()
        self.in_layers = nn.Sequential(
            LayerNorm2d(in_channels),
            nn.GELU(),
            nn.Conv2d(in_channels, out_channels, k, padding=pad),
        )
        out_conv = nn.Conv2d(out_channels, out_channels, k, padding=pad)
        if zero_out:
            nn.init.constant_(out_conv.weight.data, 0)
            nn.init.constant_(out_conv.bias.data, 0)
        self.out_layers = nn.Sequential(
            LayerNorm2d(out_channels),
            nn.GELU(),
            nn.Dropout(p=dropout),
            out_conv)

        if out_channels == in_channels:
            self.skip_connection = nn.Identity()
        else:
            self.skip_connection = nn.Conv2d(in_channels, out_channels, 1)

    def forward(self, x):
        h = self.out_layers(self.in_layers(x))
        return self.skip_connection(x) + h

class TransformerDecoder(nn.Module):
    def __init__(self,
                 latent_dim,
                 embed_dim, 
                 depth, 
                 num_heads,
                 num_queries,
                 mlp_ratio=4,
                 dropout=0.0,
                 add_up_sample=False):
                 # evo of #anchors
        super().__init__()
        self.depth = depth
        self.embed_dim = embed_dim
        self.num_queries = num_queries
        self.hw_size = int(math.sqrt(num_queries))
        self.num_heads = num_heads

        self.pe_layer = PositionEmbeddingSine(num_pos_feats = embed_dim // 2, normalize=True)
        self.tgt_embed = nn.Embedding(self.num_queries, embed_dim)
        nn.init.normal_(self.tgt_embed.weight.data)
        self.in_projs = nn.ModuleList()
        self.layers = nn.ModuleList()
        for i in range(depth):
            self_attn = (i != 0)
            self.in_projs.append(
                nn.Sequential(MLPBlock(latent_dim, latent_dim, out_dim=embed_dim), nn.LayerNorm(embed_dim)))
            self.layers.append(TransformerDecoderLayer(embed_dim, num_heads, with_self_attn=self_attn, mlp_ratio=mlp_ratio, dropout=dropout))
        self.add_up_sample = add_up_sample
        if add_up_sample:
            self.up_layer = Upsample(use_conv=True, channels=embed_dim)
            

    def formulate_attn_mask(self, attn_mask):
        bs, len_src = attn_mask.shape
        attn_mask = attn_mask[:, None, None,]
        attn_mask = torch.repeat_interleave(attn_mask, self.num_queries, dim=2)
        attn_mask = torch.repeat_interleave(attn_mask, self.num_heads, dim=1)
        attn_mask = attn_mask.view(bs*self.num_heads, self.num_queries, len_src)
        return attn_mask

    def _reset_parameters(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def forward(self, embeds, attn_mask=None):
        bs = embeds.shape[0]
        patch_pos = self.pe_layer(bs, self.hw_size, embeds.device)
        tgts = torch.repeat_interleave(self.tgt_embed.weight[None, :], bs, dim=0)
        # tgts = torch.repeat_interleave(self.tgt_embed.weight[None, :], bs.item(), dim=0) # for get_flops.py
        if attn_mask is not None:
            attn_mask = self.formulate_attn_mask(attn_mask)
        for l_id, layer in enumerate(self.layers):
            c_embeds = self.in_projs[l_id](embeds)
            tgts = layer(tgts, c_embeds, query_pos=patch_pos, memory_mask=attn_mask)
        tgts = tgts.transpose(1, 2).view(bs, -1, self.hw_size, self.hw_size).contiguous()
        if self.add_up_sample:
            tgts = self.up_layer(tgts)
        return tgts
    

@MODELS.register_module()
class ReaderDecoder(nn.Module):
    def __init__(self,
                 patch_size,
                 transformer,
                 lpips_loss,
                 l1_loss,
                 add_conv=False,
                 use_aug=False,
                 re_norm=False) -> None:
        super().__init__()
        embed_dim = transformer.get('embed_dim')
        self.transformer = TransformerDecoder(**transformer)
        self.add_conv = add_conv
        if add_conv:
            self.additional_cov = nn.Sequential(
                ResBlock(embed_dim, embed_dim, zero_out=True),
                LayerNorm2d(embed_dim),
                ResBlock(embed_dim, embed_dim, zero_out=True),
                LayerNorm2d(embed_dim),
                ResBlock(embed_dim, embed_dim, k=1, zero_out=True),
                LayerNorm2d(embed_dim))
        self.to_pixel = nn.ConvTranspose2d(embed_dim, 3, kernel_size=patch_size, stride=patch_size)
        self.lpips = LPIPS(**lpips_loss)
        self.l1_weight = l1_loss.get('weight', 1)
        self.use_aug = use_aug
        self.re_norm = re_norm
        if self.re_norm:
            norm_mean = np.array([0.485, 0.456, 0.406], dtype=np.float32).reshape((1, 3, 1, 1))
            norm_std = np.array([0.229, 0.224, 0.225], dtype=np.float32).reshape((1, 3, 1, 1))
            self.register_buffer("norm_mean", torch.tensor(norm_mean))
            self.register_buffer("norm_std", torch.tensor(norm_std))
    
    def re_normlize(self, x):
        x = x * self.norm_std + self.norm_mean
        x = x * 2 - 1
        return x
    
    def get_losses(self, samples, targets, loss_masks, ori_bs):
        losses = dict()
        if self.re_norm:
            # re-normlize for lpips and l1
            samples = self.re_normlize(samples)
            targets = self.re_normlize(targets)
        if self.use_aug:
            ori_samples = samples[:ori_bs]
            aug_samples = samples[ori_bs:]
            ori_targets = targets[:ori_bs]
            aug_targets = targets[ori_bs:]
            loss_weight = loss_masks.flatten(2).sum(-1).view(loss_masks.shape[0], 1, 1, 1)
            loss_weight = loss_masks.shape[-2] * loss_masks.shape[-1] / loss_weight
            loss_masks = loss_masks * loss_weight
            losses['loss_l1_local'] = self.l1_weight * ((aug_samples - aug_targets).abs() * loss_masks).mean()
        else:
            ori_samples = samples
            ori_targets = targets
        losses['loss_lpips'] = self.lpips(ori_samples, ori_targets, re_norm=False)
        l1_map = (ori_samples - ori_targets).abs()
        losses['loss_l1'] = self.l1_weight * l1_map.mean()
        return losses, l1_map

    def get_last_layer(self):
        return self.to_pixel.weight
    
    def forward(self, embeds, attn_mask, targets=None, loss_masks=None, anns=None, ori_bs=None, get_loss=False):
        embeds = self.transformer(embeds, attn_mask)
        if self.add_conv:
            embeds = self.additional_cov(embeds)
        recon = self.to_pixel(embeds)
        outputs = dict(recon_samples = recon)
        if get_loss:
            dec_losses, l1_map = self.get_losses(recon, targets, loss_masks, ori_bs)
            outputs.update({'dec_losses': dec_losses, 'l1_map': l1_map})
        return outputs
