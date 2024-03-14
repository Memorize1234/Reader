from typing import Any
import cv2
import torch
import numpy as np
import random
import os
import json
from torch.utils.data import DataLoader, Dataset
from torch.utils.data.distributed import DistributedSampler
from mmdet.structures.mask import BitmapMasks
import utils.dist as dist
import copy

try:
    from datasets import load_dataset
    has_datasets = True
except ImportError:
    has_datasets = False


def my_collate_train(batch_list):
    inputs = []
    gt_masks = []
    for data in batch_list:
        inputs.append(data['img'])
        gt_masks.append(data['gt_masks'])
    inputs = torch.stack(inputs, dim=0)
    return dict(inputs=inputs, anns=dict(gt_masks=gt_masks))


class EpochDataloaderFace:
    def __init__(self, logger, set_cfg, loader_cfg, test_mode=False):
        dataset = CelebADataset(**set_cfg, logger=logger)
        
        self.sampler = None
        if dist.get_world_size() > 1:
            self.sampler = DistributedSampler(dataset)
        batch_size = loader_cfg.get('batch_size', 1)
        num_workers = loader_cfg.get('num_workers', 1)
        self.dataloader = DataLoader(
            dataset, 
            batch_size=batch_size, 
            shuffle=(self.sampler is None), 
            num_workers=num_workers, 
            drop_last=True, 
            collate_fn=my_collate_train, 
            sampler=self.sampler)
        

def my_collate_val(batch_list):
    inputs, data_metas = [], []
    for data in batch_list:
        if data is None:
            continue
        data_metas.append(data)
        inputs.append(data['img'])
    if len(inputs) == 0:
        return None, None
    inputs = np.stack(inputs, axis=0)
    inputs = torch.tensor(inputs)
    return inputs, data_metas

def build_visual_dataloader(set_cfg, loader_cfg):
    dataset = CelebADataset(**set_cfg)
    return DataLoader(dataset, batch_size=loader_cfg.get('batch_size'), shuffle=False, num_workers=loader_cfg.get('num_workers'), drop_last=False, collate_fn=my_collate_val)


class CelebADataset(Dataset):
    def __init__(self, img_root, ann_root, test_filelist=None, img_ids=None, logger=None, pipeline_cfg=None, use_hf=False, hf_cache_dir=None, visual_mode=False):
        super().__init__()
        used_cats = ['nose', 'l_eye', 'r_eye', 'l_brow', 'r_brow', 'l_ear', 'r_ear', 'u_lip', 'l_lip', 'mouth', 'skin', 'neck']
        
        self.img_root = img_root
        self.ann_root = ann_root
        self.logger = logger
        
        self.use_hf = use_hf
        if self.use_hf:
            assert has_datasets, f'Cannot find hugging face\'s datasets, try `pip install datasets`'
            self.logger.info('Using hugging face\'s `datasets` to load data. Note that `img_root` and `ann_root` will be ignored.')
            all_data = load_dataset('./celeba_reader_new', split='train', cache_dir=hf_cache_dir)
        else:
            if test_filelist is not None:             
                with open(test_filelist, "r") as f:
                    split_lines = f.read().splitlines()
                test_files = set(split_lines)
            
            all_data = []
            if visual_mode:
                for img_file in split_lines[:100]:
                    img_id = int(img_file.split('.')[0])
                    img_file = os.path.join(img_root, img_file)
                    data_info = dict(img_file = img_file, img_id=img_id)
                    all_data.append(data_info)
            else:
                for img_id in range(30000):
                    img_file = f'{img_id}.jpg'
                    if img_file not in test_files:
                        img_file = os.path.join(img_root, img_file)
                        data_info = dict(img_file = img_file, img_id=img_id)
                        ann_files = dict()
                        ann_dir = str(img_id // 2000)
                        img_id = str(img_id)
                        ann_prefix = (5-len(img_id))*'0' + img_id
                        for key in used_cats:
                            ann_files[key] = os.path.join(ann_root, ann_dir, f'{ann_prefix}_{key}.png')
                        data_info['ann_files'] = ann_files
                        all_data.append(data_info)
        
        self.visual_mode = visual_mode
        self.all_data = all_data
        self.pipeline = DataPipeline(**pipeline_cfg, visual_mode=visual_mode)

    def __len__(self):
        return len(self.all_data)
        
    def __getitem__(self, idx):
        need_load = not self.use_hf
        # data_info = copy.deepcopy(self.all_data[idx])
        # data = self.pipeline(data_info, need_load=need_load)
        while True:
            try:
                data_info = copy.deepcopy(self.all_data[idx])
                data = self.pipeline(data_info, need_load=need_load)
            except Exception as e:
                data_info = copy.deepcopy(self.all_data[idx])
                if need_load:
                    file = data_info['image']['path']
                else:
                    file = data_info['img_file']
                self.logger.info(f'load data {file} error: {e}')
                idx = random.randint(0, len(self) - 1)
            else:
                if data is not None:
                    break
                else:
                    idx = random.randint(0, len(self) - 1)
        return data

class DataPipeline:
    def __init__(self, img_size=512, visual_mode=False):
        self.img_size = (img_size, img_size)
        mask_size = img_size // 4
        self.mask_size = (mask_size, mask_size)
        self.visual_mode = visual_mode
    
    def load_anns(self, data_info, need_load=True):
        masks = []
        if not need_load:
            ann_files = data_info['anno']
            for k in ann_files:
                item = ann_files[k]
                if item is None:
                    continue
                mask_bytes = item['bytes']
                mask = cv2.imdecode(np.frombuffer(mask_bytes, dtype=np.uint8), flags=0)
                mask = (mask >= 127.5).astype(np.uint8)
                masks.append(mask)
        else:
            ann_files = data_info['ann_files']
            for k in ann_files:
                mask_file = ann_files[k]
                if os.path.exists(mask_file):
                    mask = cv2.imread(mask_file, flags=0)
                    mask = (mask >= 127.5).astype(np.uint8)
                    masks.append(mask)
        masks = BitmapMasks(masks, 512, 512)
        masks = masks.resize(self.mask_size)
        areas = masks.areas
        valid_inds = areas > 3
        if not valid_inds.any():
            return None
        data_info['gt_masks'] = masks[valid_inds]
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
    
    def _remove_overlap(self, data_info):
        old_masks = data_info['gt_masks'].masks
        areas = data_info['gt_masks'].areas
        inds = np.argsort(areas)[::-1]
        old_masks = old_masks[inds]
        all_mask = np.zeros_like(old_masks[0])
        for idx, mask in enumerate(old_masks):
            all_mask[mask > 0] = idx + 1
        new_masks = []
        for idx in range(1, all_mask.max()+1):
            new_masks.append((all_mask == idx).astype(np.uint8))
        new_bit_masks = BitmapMasks(new_masks, *self.mask_size)
        data_info['gt_masks'] = new_bit_masks
        return data_info
        
    def __call__(self, data_info, need_load=True):
        if need_load:
            # loading
            img_file = data_info['img_file']
            img = cv2.imread(img_file)
            assert img is not None, f'img {img} is none'
            img_id = data_info['img_id']
        else:
            img_bytes = data_info['image']['bytes']
            img = cv2.imdecode(np.frombuffer(img_bytes, dtype=np.uint8), cv2.IMREAD_COLOR)
            img_id = int(data_info['image']['path'].split('/')[-1].split('.')[0])
        
        img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        data_info['img'] = img
        h, w = img.shape[:2]
        data_info['ori_shape'] = (h, w)
        data_info['filename'] = f'{img_id}.jpg'
        
        # loading anns
        if not self.visual_mode:
            data_info = self.load_anns(data_info, need_load=need_load)
            data_info = self._remove_overlap(data_info)
            if data_info is None:
                return None

        # transforms
        data_info  = self._resize(data_info)
        data_info = self._normlize(data_info)
        return data_info
    

