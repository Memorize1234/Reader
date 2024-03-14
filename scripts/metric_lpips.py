"""Stripped version of https://github.com/richzhang/PerceptualSimilarity/tree/master/models"""

import torch
import torch.nn as nn
import torch.distributed as dist
from torch.utils.data import Dataset, DataLoader
from torch.utils.data.distributed import DistributedSampler
from torchvision import models
from torchvision import transforms as TF
from collections import namedtuple

import os
import cv2
import numpy as np
import argparse
from tqdm import tqdm

class LPIPS(nn.Module):
    # Learned perceptual metric
    def __init__(self, use_dropout=True, ckpt='', pretrained='', weight=1.0):
        super().__init__()
        self.scaling_layer = ScalingLayer()
        self.chns = [64, 128, 256, 512, 512]  # vg16 features
        self.net = vgg16(pretrained=pretrained, requires_grad=False)
        self.lin0 = NetLinLayer(self.chns[0], use_dropout=use_dropout)
        self.lin1 = NetLinLayer(self.chns[1], use_dropout=use_dropout)
        self.lin2 = NetLinLayer(self.chns[2], use_dropout=use_dropout)
        self.lin3 = NetLinLayer(self.chns[3], use_dropout=use_dropout)
        self.lin4 = NetLinLayer(self.chns[4], use_dropout=use_dropout)
        self.load_state_dict(torch.load(ckpt, map_location=torch.device("cpu")), strict=False)
        print("loaded pretrained LPIPS loss from {}".format(ckpt))
        for param in self.parameters():
            param.requires_grad = False
        self.weight = weight

    def forward(self, input, target, loss_mask=None, loss_weight=None):
        if loss_mask is not None:
            input = loss_mask * input
            target = loss_mask * target
        in0_input, in1_input = (self.scaling_layer(input), self.scaling_layer(target))
        outs0, outs1 = self.net(in0_input), self.net(in1_input)
        feats0, feats1, diffs = {}, {}, {}
        lins = [self.lin0, self.lin1, self.lin2, self.lin3, self.lin4]
        for kk in range(len(self.chns)):
            feats0[kk], feats1[kk] = normalize_tensor(outs0[kk]), normalize_tensor(outs1[kk])
            diffs[kk] = (feats0[kk] - feats1[kk]) ** 2

        if loss_weight is not None:
            res = [spatial_sum(lins[kk].model(diffs[kk]), keepdim=False) for kk in range(len(self.chns))]
            for l in range(len(self.chns)):
                res[l] = res[l] / loss_weight
        else:
            res = [spatial_average(lins[kk].model(diffs[kk]), keepdim=False) for kk in range(len(self.chns))]
        val = res[0]
        for l in range(1, len(self.chns)):
            val += res[l]
        return val # shape: (bs,)


class ScalingLayer(nn.Module):
    def __init__(self):
        super(ScalingLayer, self).__init__()
        self.register_buffer('shift', torch.Tensor([-.030, -.088, -.188])[None, :, None, None])
        self.register_buffer('scale', torch.Tensor([.458, .448, .450])[None, :, None, None])

    def forward(self, inp):
        return (inp - self.shift) / self.scale


class NetLinLayer(nn.Module):
    """ A single linear layer which does a 1x1 conv """
    def __init__(self, chn_in, chn_out=1, use_dropout=False):
        super(NetLinLayer, self).__init__()
        layers = [nn.Dropout(), ] if (use_dropout) else []
        layers += [nn.Conv2d(chn_in, chn_out, 1, stride=1, padding=0, bias=False), ]
        self.model = nn.Sequential(*layers)


class vgg16(torch.nn.Module):
    def __init__(self, requires_grad=False, pretrained=''):
        super(vgg16, self).__init__()
        vgg = models.vgg16(pretrained=False)
        vgg.load_state_dict(torch.load(pretrained, map_location='cpu'), strict=True)
        vgg_pretrained_features = vgg.features
        self.slice1 = torch.nn.Sequential()
        self.slice2 = torch.nn.Sequential()
        self.slice3 = torch.nn.Sequential()
        self.slice4 = torch.nn.Sequential()
        self.slice5 = torch.nn.Sequential()
        self.N_slices = 5
        for x in range(4):
            self.slice1.add_module(str(x), vgg_pretrained_features[x])
        for x in range(4, 9):
            self.slice2.add_module(str(x), vgg_pretrained_features[x])
        for x in range(9, 16):
            self.slice3.add_module(str(x), vgg_pretrained_features[x])
        for x in range(16, 23):
            self.slice4.add_module(str(x), vgg_pretrained_features[x])
        for x in range(23, 30):
            self.slice5.add_module(str(x), vgg_pretrained_features[x])
        if not requires_grad:
            for param in self.parameters():
                param.requires_grad = False

    def forward(self, X):
        h = self.slice1(X)
        h_relu1_2 = h
        h = self.slice2(h)
        h_relu2_2 = h
        h = self.slice3(h)
        h_relu3_3 = h
        h = self.slice4(h)
        h_relu4_3 = h
        h = self.slice5(h)
        h_relu5_3 = h
        vgg_outputs = namedtuple("VggOutputs", ['relu1_2', 'relu2_2', 'relu3_3', 'relu4_3', 'relu5_3'])
        out = vgg_outputs(h_relu1_2, h_relu2_2, h_relu3_3, h_relu4_3, h_relu5_3)
        return out


def normalize_tensor(x,eps=1e-10):
    norm_factor = torch.sqrt(torch.sum(x**2,dim=1,keepdim=True))
    return x/(norm_factor+eps)


def spatial_average(x, keepdim=True):
    return x.mean([-3, -2, -1],keepdim=keepdim)

def spatial_sum(x, keepdim=True):
    return x.sum([-3, -2, -1],keepdim=keepdim)

class ImageDataset(Dataset):
    def __init__(self, img_dir, target_dir, resize=None) -> None:
        super().__init__()
        self.img_dir = img_dir
        self.target_dir = target_dir
        img_list = os.listdir(img_dir)
        img_list = list(filter(lambda x: x.split('.')[1].lower() in ['png', 'jpg', 'jpeg'], img_list))
        img_list.sort()
        self.img_list = [os.path.join(img_dir, x) for x in img_list]
        self.target_list = [os.path.join(target_dir, x) for x in img_list]
        self.resize = resize
        mean = [0.485, 0.456, 0.406]
        std = [0.229, 0.224, 0.225]
        self.norm_mean = np.array(mean, dtype=np.float32).reshape((1, 1, -1))
        self.norm_std = np.array(std, dtype=np.float32).reshape((1, 1, -1))
    
    def __len__(self):
        return len(self.img_list)
        # return 1000
    
    def _normalize(self, img):
        img = img / 255.
        img = img * 2. - 1.
        if not img.flags.c_contiguous:
            img = np.ascontiguousarray(img.transpose(2, 0, 1))
            img = torch.tensor(img, device='cpu', dtype=torch.float32)
        else:
            img = torch.tensor(img, device='cpu', dtype=torch.float32).permute(2, 0, 1).contiguous()
        return img
    
    def __getitem__(self, index):
        img = cv2.imread(self.img_list[index])
        assert img is not None, f'Cannot read image: {self.img_list[index]}'
        img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        if self.resize is not None:
            img = cv2.resize(img, (self.resize, self.resize)).astype(np.float32)
        img = self._normalize(img)
        
        target = cv2.imread(self.target_list[index])
        assert target is not None, f'Cannot read image: {self.target_list[index]}'
        target = cv2.cvtColor(target, cv2.COLOR_RGB2BGR)
        if self.resize is not None:
            target = cv2.resize(target, (self.resize, self.resize)).astype(np.float32)
        target = self._normalize(target)
        
        index = torch.tensor(index, dtype=torch.long)
        
        return img, target, index

def get_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--img', type=str, required=True)
    parser.add_argument('--target', type=str, default='/root/vqgan_preprocess_val')
    parser.add_argument('--bs', type=int, default=10)
    return parser

if __name__ == '__main__':
    parser = get_parser()
    args = parser.parse_args()
    
    rank = int(os.environ.get('RANK', '0'))
    local_rank = int(os.environ.get('LOCAL_RANK', '0'))
    world_size = int(os.environ.get('WORLD_SIZE', '1'))
    print(f'rank: {rank}, local_rank: {local_rank}, world_size: {world_size}')
    if world_size > 1:
        dist.init_process_group('nccl')
    torch.cuda.set_device(local_rank)
    
    lpips = LPIPS(ckpt='./checkpoints/vgg/vgg_lpips.pth',
                  pretrained='./checkpoints/vgg/vgg16-397923af.pth')
    lpips = lpips.cuda()
    lpips = lpips.eval()
    
    dataset = ImageDataset(args.img, args.target)
    sampler = DistributedSampler(dataset, shuffle=False, drop_last=False)
    # assert len(dataset) % args.bs == 0, f'The length of the dataset must be divisible by the batch size. {len(dataset)} % {args.bs} != 0'
    dataloader = DataLoader(dataset, batch_size=args.bs, num_workers=4, sampler=sampler)
    if rank == 0:
        # dataloader = tqdm(dataloader)
        pbar = tqdm(dataloader)
    
    losses = torch.zeros(len(dataset), dtype=torch.float32).cuda()
    for i, (img, target, index) in enumerate(dataloader):
    # for img, target in dataloader:
        img = img.cuda()
        target = target.cuda()
        index = index.cuda()
        with torch.no_grad():
            loss = lpips(img, target) # shape: (bs,)
            losses[index] = loss
        if rank == 0:
            pbar.set_postfix(loss=loss.mean().item())
            pbar.update(1)
    if world_size > 1:
        dist.all_reduce(losses, op=dist.ReduceOp.SUM)
    if rank == 0:
        pbar.close()
    # losses = torch.tensor(losses)
    # torch.save(losses.cpu(), 'std.pth')
    assert (losses == 0).sum() == 0, f'found {(losses == 0).sum().item()} zeros in losses'
    if rank == 0:
        print(f'LPIPS loss:')
        print(f'  avg: {losses.mean():.6f}, std: {losses.std():.6f}, min: {losses.min():.6f}, max: {losses.max():.6f}')
        print(f'  {losses.mean():.6f}/{losses.std():.6f}/{losses.min():.6f}/{losses.max():.6f}')
