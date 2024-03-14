from typing import Any
import cv2
import torch
import numpy as np
import random
import os
import json
from torch.utils.data import DataLoader, Dataset
import dataset.transforms as T
from torchvision.transforms import transforms
import utils.dist as dist
from PIL import Image, ImageFilter
from mmdet.structures.bbox import HorizontalBoxes
from mmdet.structures.mask import BitmapMasks
import pycocotools.mask as maskUtils
from mmengine.structures import InstanceData
from torch.utils.data.distributed import DistributedSampler
from PIL import Image
import copy


try:
    from datasets import load_dataset
    has_datasets = True
except ImportError:
    has_datasets = False

    

def build_test_dataloader(set_cfg, loader_cfg):
    dataset = TestDatasetWrapper(**set_cfg)
    return DataLoader(dataset, batch_size=loader_cfg.get('batch_size'), shuffle=False, num_workers=loader_cfg.get('num_workers'), drop_last=False, collate_fn=my_collate_test)

def my_collate_test(batch_list):
    inputs, data_metas = [], []
    for data in batch_list:
        data_metas.append(data)
        inputs.append(data['img'])
    inputs = np.stack(inputs, axis=0)
    inputs = torch.tensor(inputs)
    return inputs, data_metas


class TestDatasetWrapper(Dataset):
    def __init__(self, img_root, img_sets):
        super().__init__()
        self.img_root = img_root
        all_imgs = []
        all_imgs_dict = dict()
        with open(img_sets, "r") as f:
            ori_imgs = f.read().splitlines()
            for idx, file in enumerate(ori_imgs):
                all_imgs.append(file)
                img_id = int(file.split('/')[-1].split('.')[0].split('_')[-1])
                all_imgs_dict.update({img_id: file})
        imgs = []
        for img_id in range(len(all_imgs)):
            if img_id in all_imgs_dict:
                imgs.append(all_imgs_dict[img_id])
        shard = dist.get_rank()
        num_shards = dist.get_world_size()
        self.imgs = imgs[shard:][::num_shards]
        self.norm_mean = np.array([0.485, 0.456, 0.406], dtype=np.float32).reshape((1, 1, -1))
        self.norm_std = np.array([0.229, 0.224, 0.225], dtype=np.float32).reshape((1, 1, -1))

    def __len__(self):
        return len(self.imgs)

    def __getitem__(self, idx):
        data_info = dict()
        img_name = self.imgs[idx]
        img = cv2.imread(os.path.join(self.img_root, img_name))
        img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        data_info['ori_img'] = img
        h, w = img.shape[:2]
        data_info['filename'] = img_name.split('/')[-1]
        data_info['ori_shape'] = (h, w)
        img = cv2.resize(img, (256, 256)).astype(np.float32)
        img = img / 255
        img = (img - self.norm_mean) / self.norm_std
        data_info['img'] = img.transpose(2, 0, 1)
        return data_info