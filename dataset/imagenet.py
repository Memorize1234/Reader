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


def my_collate_train_with_anns(batch_list):
    inputs, aug_infos, high_lvl_masks, low_lvl_masks = [], [], [], []
    has_aug_info = 'aug_info' in batch_list[0]
    for data in batch_list:
        inputs.append(data['img'])
        high_lvl_masks.append(data['high_lvl_masks'])
        low_lvl_masks.append(data['low_lvl_masks'])
        if has_aug_info:
            aug_infos.append(data['aug_info'])

    inputs = torch.stack(inputs, dim=0)
    return dict(inputs=inputs, anns=dict(high_lvl_masks=high_lvl_masks, low_lvl_masks=low_lvl_masks, aug_infos=aug_infos))

def my_collate_train(batch_list):
    inputs = []
    for data in batch_list:
        inputs.append(data['img'])
    inputs = torch.stack(inputs, dim=0)
    return dict(inputs=inputs, anns=None)


class EpochDataloader:
    def __init__(self, logger, set_cfg, loader_cfg, test_mode=False):
        dataset = ImageNetDataset(
            logger,
            **set_cfg,
            test_mode = test_mode)
        
        self.sampler = None
        if dist.get_world_size() > 1:
            self.sampler = DistributedSampler(dataset)
        batch_size = loader_cfg.get('batch_size', 1)
        num_workers = loader_cfg.get('num_workers', 1)
        need_anns = set_cfg.get('pipeline_cfg').get('need_anns')
        if need_anns:
            collate = my_collate_train_with_anns
        else:
            collate = my_collate_train
        if not test_mode:
            self.dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=(self.sampler is None), num_workers=num_workers, drop_last=True, collate_fn=collate, sampler=self.sampler)
        else:
            self.dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers, drop_last=False, collate_fn=collate)


class ImageNetDataset(Dataset):
    def __init__(self, logger, img_root, ann_root=None, pipeline_cfg=dict(), test_mode=False, use_hf=False, hf_cache_dir=None):
        super().__init__()
        self.img_root = img_root
        self.ann_root = ann_root
        self.logger = logger
        self.pipeline = DataPipeline(**pipeline_cfg)
        
        self.use_hf = use_hf
        if self.use_hf:
            assert has_datasets, f'Cannot find hugging face\'s datasets, try `pip install datasets`'
            assert not test_mode, f'Hugging face\'s `datasets` only supports training mode now'
            self.logger.info('Using hugging face\'s `datasets` to load data. Note that `img_root` and `ann_root` will be ignored.')
            self.local_data = load_dataset('./imagenet_reader_old', split='train', cache_dir=hf_cache_dir)
        else:
            all_data = []
            all_dirs = os.listdir(ann_root)
            for dir in all_dirs:
                ann_dir = os.path.join(ann_root, dir)
                img_dir = os.path.join(img_root, dir)
                files = os.listdir(ann_dir)
                for file in files:
                    all_data.append(
                        {'img_file': os.path.join(img_dir, file.replace('json', 'JPEG')),
                        'json_file': os.path.join(ann_dir, file)})
            self.local_data = all_data
            self.test_mode = test_mode

    def __len__(self):
        return len(self.local_data)

    def __getitem__(self, idx):
        need_load = not self.use_hf
        while True:
            try:
                data_info = copy.deepcopy(self.local_data[idx])
                data = self.pipeline(data_info, need_load=need_load)
            except Exception as e:
                data_info = copy.deepcopy(self.local_data[idx])
                if need_load:
                    file = data_info['img_file']
                else:
                    file = data_info['image']['path']
                self.logger.info(f'load data {file} error: {e}')
                idx = random.randint(0, len(self) - 1)
            else:
                if data is not None:
                    break
                else:
                    idx = random.randint(0, len(self) - 1)
        return data


class DataPipeline:
    def __init__(self, img_size, need_anns=False, local_input=False, mask_size_factor=1/8, max_high_num=16, max_low_num=256, flip=0.5, norm_type='imagenet', dilate=0):
        self.img_size = (img_size, img_size)
        self.flip = flip
        self.need_anns = need_anns
        self.local_input = local_input
        self.max_high_num = max_high_num
        self.max_low_num = max_low_num
        self.mask_size = (int(img_size*mask_size_factor), int(img_size*mask_size_factor))
        self.high_area_min = 256 * 256 / 16
        self.high_area_max = 256 * 256 / 2
        self.low_area_min = 2
        self.low_area_max = 64 * 64 / 16
        self.norm_type = norm_type
        assert norm_type in ['imagenet', 'standard']
        mean_in = [0.485, 0.456, 0.406]
        std_in = [0.229, 0.224, 0.225]
        self.norm_mean_in = np.array(mean_in, dtype=np.float32).reshape((1, 1, -1))
        self.norm_std_in = np.array(std_in, dtype=np.float32).reshape((1, 1, -1))
        if dilate > 0:
            self.use_dilate = True
            self.dilate_op = np.ones((dilate, dilate), np.uint8)
            self.dilate_op_l = np.ones((dilate*4, dilate*4), np.uint8)
        else:
            self.use_dilate = False

    
    def get_masks(self, belong_rela, rles):
        h, w = rles[0]['size']
        gt_masks = maskUtils.decode(rles[:256]).transpose(2, 0, 1)
        gt_masks = BitmapMasks(gt_masks, h, w)
        high_lvl_masks_inds = set(belong_rela)
        low_lvl_masks = []
        masks = gt_masks.masks
        areas = gt_masks.areas
        for ind, mask in enumerate(masks):
            if str(ind) not in high_lvl_masks_inds:
                low_lvl_masks.append(mask)
        if len(low_lvl_masks) == 0:
            return None
        low_lvl_masks = BitmapMasks(low_lvl_masks, h, w)
        high_lvl_masks = None
        if self.local_input:
            valid_masks = (areas < self.high_area_max) & (areas > self.high_area_min)
            if valid_masks.any():
                valid_mask_inds = np.nonzero(valid_masks)[0]
                high_lvl_masks = masks[valid_mask_inds]
                high_lvl_masks = BitmapMasks(high_lvl_masks, h, w)                    
        return high_lvl_masks, low_lvl_masks

    def load_anns(self, data_info):
        if not self.need_anns:
            return data_info
        with open(data_info['json_file']) as f:
            anns = json.load(f)
        gt_masks = self.get_gt_masks_from_anns(anns)
        if gt_masks is None:
            return None
        high_lvl_masks, low_lvl_masks = gt_masks
        data_info['high_lvl_masks'] = high_lvl_masks
        data_info['low_lvl_masks'] = low_lvl_masks
        return data_info

    def get_gt_masks_from_anns(self, anns):
        if len(anns['segmentation']) == 0:
            return None
        res = self.get_masks(anns['belong_rela'], anns['segmentation'])
        return res
    
    def _resize(self, data_info):
        if data_info is None:
            return None
        img = data_info['img']
        img = cv2.resize(img, self.img_size).astype(np.float32)
        data_info['img'] = img

        if self.need_anns:
            low_lvl_masks = data_info['low_lvl_masks'].resize(self.mask_size)
            areas = low_lvl_masks.areas
            valid_inds = (areas < self.low_area_max) & (areas >= self.low_area_min)
            if not valid_inds.any():
                data_info['low_lvl_masks'] = None
            else:
                data_info['low_lvl_masks'] = low_lvl_masks[valid_inds]
            if self.local_input:
                data_info['high_lvl_masks_ori'] = data_info['high_lvl_masks']
                if data_info['high_lvl_masks'] is not None:
                    data_info['high_lvl_masks'] = data_info['high_lvl_masks'].resize(self.mask_size)
        return data_info
    
    def _normlize(self, data_info, key='img'):
        if data_info is None:
            return None
        img = data_info[key]
        img = img / 255
        if self.norm_type == 'imagenet':
            img = (img - self.norm_mean_in) / self.norm_std_in
        else:
            img = img * 2.0 - 1.0
        if not img.flags.c_contiguous:
            img = np.ascontiguousarray(img.transpose(2, 0, 1))
            img = torch.tensor(img, device='cpu')
        else:
            img = torch.tensor(img, device='cpu').permute(2, 0, 1).contiguous()
        data_info[key] = img
        return data_info
    
    def _flip(self, data_info):
        if data_info is None:
            return None
        img = data_info['img']
        fliped = False
        if self.flip > 0:
            if np.random.rand() < self.flip:
                img = cv2.flip(img, 1)
                fliped = True
                if self.need_anns:
                    data_info['low_lvl_masks'] = data_info['low_lvl_masks'].flip()
                    if self.local_input:
                        if data_info['high_lvl_masks'] is not None:
                            # data_info['high_lvl_masks'] will be None if self.local_input is False
                            data_info['high_lvl_masks'] = data_info['high_lvl_masks'].flip()
        data_info['img'] = img
        data_info['fliped'] = fliped
        return data_info
    
    def _get_local_img(self, data_info):
        if data_info['high_lvl_masks'] is None:
            data_info['aug_info'] = None
            return data_info
        model_size_masks = data_info['high_lvl_masks'].masks
        img_size_masks = data_info['high_lvl_masks_ori'].masks
        
        num = len(model_size_masks)
        inpaint = True
        chosen_idx = random.randrange(num)
        if np.random.rand() < 0.5:
            inpaint = False

        aug_info = dict()
        aug_info['inpaint'] = inpaint
        model_size_mask = model_size_masks[chosen_idx]
        img_size_mask_np = img_size_masks[chosen_idx]
        
        if inpaint:
            if self.use_dilate:
                img_size_mask_np = cv2.dilate(img_size_mask_np, self.dilate_op_l, iterations=1)
                model_size_mask = cv2.dilate(model_size_mask, self.dilate_op, iterations=1)
            img_size_mask_torch = torch.tensor(img_size_mask_np, device='cpu').unsqueeze(0)
            loss_mask = 1 - img_size_mask_torch
            local_img = loss_mask * data_info['img']
        else:
            img_size_mask_torch = torch.tensor(img_size_mask_np, device='cpu').unsqueeze(0)
            local_img = img_size_mask_torch * data_info['img']
            local_img, loss_mask, scaling_matrix, translation_matrix = translation_and_scaling(local_img, img_size_mask_np)
            aug_info['scaling_matrix'] = scaling_matrix
            aug_info['translation_matrix'] = translation_matrix

        model_size_mask = torch.tensor(model_size_mask, device='cpu')
        aug_info['aug_mask'] = model_size_mask
        aug_info['local_img'] = local_img
        aug_info['loss_mask'] = loss_mask
        data_info['aug_info'] = aug_info
        return data_info
        
    def __call__(self, data_info, need_load=True):
        if need_load:
            # loading
            img_file = data_info['img_file']
            img = cv2.imread(img_file)
            assert img is not None, f'img {img_file} is none'

            # loading anns
            data_info = self.load_anns(data_info)
            if data_info is None:
                return None
        else:
            img_bytes = data_info['image']['bytes']
            img = cv2.imdecode(np.frombuffer(img_bytes, dtype=np.uint8), cv2.IMREAD_COLOR)
            anns = data_info['anno']
            anns['belong_rela'] = json.loads(anns['belong_rela'])
            gt_masks = self.get_gt_masks_from_anns(anns)
            if gt_masks is None:
                return None
            else:
                high_lvl_masks, low_lvl_masks = gt_masks
                data_info['high_lvl_masks'] = high_lvl_masks
                data_info['low_lvl_masks'] = low_lvl_masks
        
        img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        h, w = img.shape[:2]
        data_info['ori_shape'] = (h, w)
        data_info['img'] = img

        # transforms
        data_info = self._flip(data_info)
        data_info = self._resize(data_info)
        data_info = self._normlize(data_info)
        if self.local_input:
            data_info = self._get_local_img(data_info)
        return data_info

def random_one_connect(masks, num):
    inds = list(range(num))
    random.shuffle(inds)
    for ind in inds:
        num_labels, _ = cv2.connectedComponents(masks[ind])
        if num_labels == 0:
            return ind
    return None

def foreground_distances(binary_mask):
    foreground_indices = np.transpose(np.nonzero(binary_mask))
    top_distance = foreground_indices[:, 0].min()
    bottom_distance = binary_mask.shape[0] - 1 - foreground_indices[:, 0].max()
    left_distance = foreground_indices[:, 1].min()
    right_distance = binary_mask.shape[1] - 1 - foreground_indices[:, 1].max()
    return top_distance, bottom_distance, left_distance, right_distance

def find_center(binary_mask):
    H, W = binary_mask.shape
    foreground_indices = np.transpose(np.nonzero(binary_mask))
    y_1, y_2 = foreground_indices[:, 0].min(), foreground_indices[:, 0].max()
    x_1, x_2 = foreground_indices[:, 1].min(), foreground_indices[:, 1].max()
    y_c = (y_1 + y_2) // 2
    x_c = (x_1 + x_2) // 2
    max_y = max((H - y_c) / (y_2 - y_c), y_c / (y_c - y_1))
    max_x = max((W - x_c) / (x_2 - x_c), x_c / (x_c - x_1))
    return x_c.item(), y_c.item(), max_x, max_y

def translation_and_scaling(img, mask, min_scale=0.5, max_scale=1.5):
    h, w = mask.shape[0], mask.shape[1]
    angle = 0
    x_c, y_c, max_x, max_y = find_center(mask)
    max_scale = min([max_scale, max_x, max_y])
    scale = random.uniform(min_scale, max_scale)
    scaling_matrix = cv2.getRotationMatrix2D((x_c, y_c), angle=angle, scale=scale)
    mask = cv2.warpAffine(mask, scaling_matrix, (w, h)) # type: ignore
    y_1, y_2, x_1, x_2 = foreground_distances(mask)
    delta_x, delta_y = int(random.uniform(-x_1, x_2)), int(random.uniform(-y_1, y_2))
    translation_matrix = np.float32(
        [[1, 0, delta_x],
        [0, 1, delta_y]]) # type: ignore
    mask = cv2.warpAffine(mask, translation_matrix, (w, h))
    img_np = img.numpy().transpose(1, 2, 0)
    img_np = cv2.warpAffine(img_np, scaling_matrix, (w, h))
    img_np = cv2.warpAffine(img_np, translation_matrix, (w, h))
    img = np.ascontiguousarray(img_np.transpose(2, 0, 1))
    img = torch.tensor(img, device='cpu')
    mask = torch.tensor(mask, device='cpu').unsqueeze(0)
    return img, mask, scaling_matrix, translation_matrix
    

def get_val_datas(set_cfg):
    inputs, data_metas = [], []
    dataset = VisualDatasetWrapper(**set_cfg)
    for idx in range(len(dataset)):
        data = dataset[idx]
        data_metas.append(data)
        inputs.append(data['img'])
    inputs = np.stack(inputs, axis=0)
    inputs = torch.tensor(inputs)
    return inputs, data_metas


def build_visual_dataloader(set_cfg, img_ids=None):
    dataset = VisualDatasetWrapper(**set_cfg, img_ids=img_ids)
    return DataLoader(dataset, batch_size=1, shuffle=False, num_workers=1, drop_last=False, collate_fn=my_collate_vis)

def my_collate_vis(batch_list):
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


class VisualDatasetWrapper(Dataset):
    def __init__(self, img_root, img_sets=None, img_ids=None, num=None, norm_type='standard'):
        super().__init__()
        self.img_root = img_root
        self.norm_type = norm_type
        all_imgs = []
        all_imgs_dict = dict()
        with open(img_sets, "r") as f:
            ori_imgs = f.read().splitlines()
            for idx, file in enumerate(ori_imgs):
                all_imgs.append(file)
                img_id = int(file.split('/')[-1].split('.')[0].split('_')[-1])
                all_imgs_dict.update({img_id: file})
        input_imgs = []
        if img_ids is not None:
            for img_id in img_ids:
                input_imgs.append(all_imgs_dict[int(img_id)])
        else:
            if num < len(all_imgs):
                input_imgs = all_imgs[:int(num)]
            else:
                input_imgs = all_imgs

        self.imgs = input_imgs
        self.norm_mean = np.array([0.485, 0.456, 0.406], dtype=np.float32).reshape((1, 1, -1))
        self.norm_std = np.array([0.229, 0.224, 0.225], dtype=np.float32).reshape((1, 1, -1))

    def __len__(self):
        return len(self.imgs)

    def __getitem__(self, idx):
        data_info = dict()
        img_name = self.imgs[idx]
        img = cv2.imread(os.path.join(self.img_root, img_name))
        assert img is not None, f'cannot load image: {os.path.join(self.img_root, img_name)}'
        h, w = img.shape[:2]
        # if h < w:
        #     return None
        img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        data_info['ori_img'] = img
        
        data_info['filename'] = img_name
        data_info['ori_shape'] = (h, w)
        img = cv2.resize(img, (256, 256)).astype(np.float32)
        img = img / 255
        if self.norm_type == 'imagenet':
            img = (img - self.norm_mean) / self.norm_std
        else:
            img = img * 2.0 - 1.0
        data_info['img'] = img.transpose(2, 0, 1)
        return data_info
    