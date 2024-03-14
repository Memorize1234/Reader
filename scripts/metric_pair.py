import os
import cv2
import argparse
import numpy as np
from concurrent.futures import ThreadPoolExecutor, as_completed
from skimage.metrics import structural_similarity, peak_signal_noise_ratio
from tqdm import tqdm

def get_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--gt_dir', type=str, required=True)
    parser.add_argument('--recon_dir', type=str, required=True)
    parser.add_argument('--resize', type=int, default=None)
    parser.add_argument('--n_workers', type=int, default=1)
    return parser

def check_ext(filename, ext):
    for e in ext:
        if filename.lower().endswith(e.lower()):
            return True
    return False

def compute_metric(filenames, args):
    ssim_all = []
    psnr_all = []
    l1_all = []
    for filename in filenames:
        gt = cv2.imread(os.path.join(args.gt_dir, filename))
        recon = cv2.imread(os.path.join(args.recon_dir, filename))
        assert gt.shape == recon.shape, f'{filename}: {gt.shape} != {recon.shape}'
        if args.resize is not None:
            size = (args.resize, args.resize)
            gt = cv2.resize(gt, dsize=size)
            recon = cv2.resize(recon, dsize=size)
        ssim = structural_similarity(gt, recon, channel_axis=2)
        psnr = peak_signal_noise_ratio(gt, recon)
        l1_loss = np.abs(gt - recon).mean()
        ssim_all.append(ssim)
        psnr_all.append(psnr)
        l1_all.append(l1_loss)
    return ssim_all, psnr_all, l1_all

if __name__ == '__main__':
    parser = get_parser()
    args = parser.parse_args()
    filenames = os.listdir(args.gt_dir)
    filenames = [filename for filename in filenames if check_ext(filename, ['.jpg', '.jpeg', '.png'])]
    print(f'found {len(filenames)} images in ground truth directory')
    recon_not_found = []
    for filename in filenames:
        if not os.path.exists(os.path.join(args.recon_dir, filename)):
            recon_not_found.append(filename)
    print(f'images present in ground truth directory but not in reconstruction directory: {recon_not_found}')
    filenames = [filename for filename in filenames if filename not in recon_not_found]
    print(f'start processing {len(filenames)} image pairs')
    
    ssim, psnr, l1_loss = [], [], []
    pool = ThreadPoolExecutor(max_workers=args.n_workers)
    futures = []
    for i in range(args.n_workers):
        filenames_sub = filenames[i:][::args.n_workers]
        if i == args.n_workers - 1:
            filenames_sub = tqdm(filenames_sub)
        future = pool.submit(compute_metric, filenames_sub, args)
        futures.append(future)
    for future in as_completed(futures):
        res = future.result()
        ssim += res[0]
        psnr += res[1]
        l1_loss += res[2]
        
    ssim = np.array(ssim)
    print(f'SSIM results:')
    print(f'  avg: {ssim.mean():.6f}, std: {ssim.std():.6f}, min: {ssim.min():.6f}, max: {ssim.max():.6f}')
    print(f'  {ssim.mean():.6f}/{ssim.std():.6f}/{ssim.min():.6f}/{ssim.max():.6f}')
    psnr = np.array(psnr)
    print(f'PSNR results:')
    print(f'  avg: {psnr.mean():.6f}, std: {psnr.std():.6f}, min: {psnr.min():.6f}, max: {psnr.max():.6f}')
    print(f'  {psnr.mean():.6f}/{psnr.std():.6f}/{psnr.min():.6f}/{psnr.max():.6f}')
    l1_loss = np.array(l1_loss)
    print(f'L1 loss results:')
    print(f'  avg: {l1_loss.mean():.6f}, std: {l1_loss.std():.6f}, min: {l1_loss.min():.6f}, max: {l1_loss.max():.6f}')
    print(f'  {l1_loss.mean():.6f}/{l1_loss.std():.6f}/{l1_loss.min():.6f}/{l1_loss.max():.6f}')
