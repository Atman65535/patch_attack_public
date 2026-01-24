"""
File: lpips_test.py
Author: Atman
Date: 1/24/26
Description:
    
"""
import torch
import lpips
import torchvision.io

from usr.datasets.rellis_pytorch import Rellis3DDatasetTorch
loss_fn_alex = lpips.LPIPS(net='alex') # best forward scores

import torch
img0 = torch.rand(1,3, 64, 64) * 2 - 1 # image should be RGB, IMPORTANT: normalized to [-1,1]
img1 = torch.rand(1,3,64,64) * 2 - 1
d = loss_fn_alex(img0, img1)
print(d)

from omegaconf import OmegaConf

cfg = OmegaConf.load()
