# Reader

This is the official pytorch implementation of ***Reader***.

## Todo List

- [x] ~~Release source code~~
- [ ] Upload sam pseudo annotations on ImageNet training set
- [ ] Retrain an end-to-end version and upload model weights
- [ ] Release editing ui


## Installation

tested on Python 3.8.16 and PyTorch 1.12.1
```bash
pip install torch==1.12.1+cu113 torchvision==0.13.1+cu113 torchaudio==0.12.1 --extra-index-url https://download.pytorch.org/whl/cu113
pip install -U openmim
mim install mmengine
mim install "mmcv>=2.0.0"
mim install mmdet
pip install -r requirements.txt
```

## Directory of Important Files

```bash
Reader
├── configs
│   ├── reader_tiny_tiny_celeba_512.py  # finetuning on the CelebaMask-HQ dataset
│   ├── reader_tiny_tiny_in_256_aug.py  # finetuning on the ImageNet dataset
│   └── reader_tiny_tiny_in_256.py      # pretraining on the ImageNet dataset
├── scripts
│   ├── metric_lpips.py       # evaluate the lpips metric of reconstructed images
│   ├── metric_pair.py        # evaluate other metrics of reconstructed images
│   ├── smile_transfer.py     # simle transferring script
│   └── test_recon.py         # ImageNet reconstruction script
└── main.py    # training script
```
