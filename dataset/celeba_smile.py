from typing import Any
import cv2
import torch
import numpy as np
import random
import os
import json
from torch.utils.data import DataLoader, Dataset
from torch.utils.data.distributed import DistributedSampler
import utils.dist as dist
import copy


def build_dataloader(bs, num_workers):
    dataset = CelebADataset()
    return DataLoader(dataset, batch_size=bs, shuffle=False, num_workers=num_workers, drop_last=False, collate_fn=my_collate)


def my_collate(batch_list):
    inputs = []
    ori_masks, tgt_masks, mats = [], [], []
    files, tgt_imgs, tgt_ids, dists = [], [], [], []
    for data in batch_list:
        if data is None:
            continue
        inputs.append(data['img'])
        ori_masks.append(data['ori_masks'])
        tgt_masks.append(data['tgt_masks'])
        files.append(data['img_file'])
        dists.append(data['dist'])
        tgt_imgs.append(data['tgt_img'])
        tgt_ids.append(data['tgt_id'])
        mats.append(data['mats'])
    inputs = torch.stack(inputs, dim=0)
    tgt_imgs = torch.stack(tgt_imgs, dim=0)
    anns = dict(ori_masks = ori_masks, tgt_masks = tgt_masks, mats=mats)
    meta_infos = dict(files = files, dists = dists, tgt_imgs = tgt_imgs, tgt_ids = tgt_ids)
    return inputs, anns, meta_infos

def mask2bbox(mask):
    x_any = mask.any(axis=0)
    y_any = mask.any(axis=1)
    x = np.where(x_any)[0]
    y = np.where(y_any)[0]
    x_1, x_2 = x[0], x[-1] + 1
    y_1, y_2 = y[0], y[-1] + 1
    return [x_1, y_1, x_2, y_2]

def mask2w(mask):
    x_any = mask.any(axis=0)
    x = np.where(x_any)[0]
    w = x[-1] + 1 - x[0]
    return w


class CelebADataset(Dataset):
    def __init__(self):
        super().__init__()
        img_root = 'path/to/CelebAMask-HQ/CelebA-HQ-img'
        json_file = 'path/to/CelebAMask-HQ/smile_pair_data.json'
        ann_root = 'path/to/CelebAMask-HQ/CelebAMask-HQ-mask-anno'
        with open(json_file, 'r') as f:
            all_pairs = json.load(f)
        self.ann_root = ann_root

        all_anns = dict()
        for i in range(30000):
            all_anns[i] = dict()
        ann_dirs = os.listdir(ann_root)
        for folder in ann_dirs:
            ann_dir = os.path.join(ann_root, folder)
            if not os.path.isdir(ann_dir):
                continue
            files = os.listdir(ann_dir)
            for file in files:
                if '.png' in file:
                    idx = int(file.split('_')[0])
                    if 'mouth' in file:
                        all_anns[idx]['mouth'] = os.path.join(ann_dir, file)
                    elif 'l_lip' in file:
                        all_anns[idx]['l_lip'] = os.path.join(ann_dir, file)
                    elif 'u_lip' in file:
                        all_anns[idx]['u_lip'] = os.path.join(ann_dir, file)
                        
        all_pos_dists = torch.zeros((len(all_pairs),), dtype=torch.float32, device='cpu')
        all_data = []
        for ind, img_id in enumerate(all_pairs):
            tgt_id = all_pairs[img_id]['inds']
            # load img
            non_smile = os.path.join(img_root, f"{img_id}.jpg")
            smile = os.path.join(img_root, f"{tgt_id}.jpg")
            all_pos_dists[ind] = all_pairs[img_id]['dists']
            data = dict(
                img_file = f"{img_id}.jpg",
                tgt_id = tgt_id,
                non_smile_img = non_smile,
                smile_img = smile,
                dist = all_pairs[img_id]['dists'],
                non_smile_anns = all_anns[int(img_id)],
                smile_anns = all_anns[tgt_id])
            all_data.append(data)
        _, indexes = torch.sort(all_pos_dists)
        self.all_data = []
        for i in range(400):
            self.all_data.append(all_data[indexes[i].item()])
        # self.all_data = all_data
        self.pipeline = DataPipeline()

    def __len__(self):
        return len(self.all_data)
        
    def __getitem__(self, idx):
        try:
            data_info = copy.deepcopy(self.all_data[idx])
            data = self.pipeline(data_info)
        except:
            data = None
        return data


class DataPipeline:
    def __init__(self, img_size=512):
        self.img_size = (img_size, img_size)
        mean = [0.485, 0.456, 0.406]
        std = [0.229, 0.224, 0.225]
        self.norm_mean = np.array(mean, dtype=np.float32).reshape((1, 1, -1))
        self.norm_std = np.array(std, dtype=np.float32).reshape((1, 1, -1))
    
    def mask_align(self, ori_mask, tgt_mask):
        h, w = ori_mask.shape
        w1 = mask2w(ori_mask)
        w2 = mask2w(tgt_mask)
        scale = w1 / w2
        scale_MAT = cv2.getRotationMatrix2D((w/2,h/2), angle=0, scale=scale)
        tgt_mask = cv2.warpAffine(tgt_mask.astype(np.uint8), scale_MAT, (w, h)).astype(bool)
        ori_box = mask2bbox(ori_mask)
        tgt_box = mask2bbox(tgt_mask)
        delta_x = int((ori_box[0] + ori_box[2] - tgt_box[0] - tgt_box[2]) / 2)
        delta_y_1 = ori_box[1] - tgt_box[1]
        delta_y_2 = int((ori_box[1] + ori_box[3] - tgt_box[1] - tgt_box[3]) / 2)
        delta_y = max(delta_y_1, delta_y_2)
        mov_MAT = np.float32([[1, 0, delta_x], [0, 1, delta_y]])
        return (scale_MAT, mov_MAT)
    
    def load_anns(self, data_info):
        ori_anns = data_info['non_smile_anns']
        tgt_anns = data_info['smile_anns']
        ori_masks, tgt_masks = [], []
        for k in ori_anns:
            mask = cv2.resize(cv2.imread(ori_anns[k], flags=0), dsize=(128, 128), interpolation=0)
            ori_masks.append(mask)
        for k in tgt_anns:
            mask = cv2.resize(cv2.imread(tgt_anns[k], flags=0), dsize=(128, 128), interpolation=0)
            tgt_masks.append(mask)
        ori_masks = np.stack(ori_masks, axis=0)
        pano_ori_masks = ori_masks.any(0)            
        tgt_masks = np.stack(tgt_masks, axis=0)
        pano_tgt_masks = tgt_masks.any(0)  
        data_info['ori_masks'] = torch.tensor(ori_masks, dtype=torch.float32, device='cpu') / 255
        data_info['tgt_masks'] = torch.tensor(tgt_masks, dtype=torch.float32, device='cpu') / 255
        mats = self.mask_align(pano_ori_masks, pano_tgt_masks)
        data_info['mats'] = mats
        return data_info
    
    def _resize(self, data_info, key='img'):
        img = data_info[key]
        data_info[key] = cv2.resize(img, self.img_size).astype(np.float32)
        return data_info
    
    def _normlize(self, data_info, key='img'):
        img = data_info[key]
        img = img / 255
        img = (img - 0.5) / 0.5
        if not img.flags.c_contiguous:
            img = np.ascontiguousarray(img.transpose(2, 0, 1))
            img = torch.tensor(img, device='cpu')
        else:
            img = torch.tensor(img, device='cpu').permute(2, 0, 1).contiguous()
        data_info[key] = img
        return data_info

    def inverse(self, data_info):
        smile_img, non_smile_img  = data_info['smile_img'], data_info['non_smile_img']
        ori_anns = data_info['non_smile_anns']
        tgt_anns = data_info['smile_anns']
        data_info['smile_img'] = non_smile_img
        data_info['non_smile_img'] = smile_img
        data_info['non_smile_anns'] = tgt_anns
        data_info['smile_anns'] = ori_anns
        return data_info
        
    def __call__(self, data_info):
        # loading
        # data_info = self.inverse(data_info)
        tgt_file = data_info['smile_img']
        tgt_img = cv2.imread(tgt_file)
        assert tgt_img is not None, f'img {tgt_file} is none'
        tgt_img = cv2.cvtColor(tgt_img, cv2.COLOR_RGB2BGR)
        data_info['tgt_img'] = tgt_img

        ori_file = data_info['non_smile_img']
        ori_img = cv2.imread(ori_file)
        assert ori_img is not None, f'img {ori_file} is none'
        ori_img = cv2.cvtColor(ori_img, cv2.COLOR_RGB2BGR)

        h, w = ori_img.shape[:2]
        data_info['ori_shape'] = (h, w)
        data_info['img'] = ori_img

        # loading anns
        data_info = self.load_anns(data_info)
        if data_info is None:
            return None

        # transforms
        data_info  = self._resize(data_info)
        data_info = self._normlize(data_info)
        data_info  = self._resize(data_info, key='tgt_img')
        data_info = self._normlize(data_info, key='tgt_img')
        return data_info
