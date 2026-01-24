"""
File: rellis_pytorch.py
Author: Atman
Date: 1/22/26
Description:
    
"""
import warnings

import torch
import torchvision
import cv2
import numpy as np
import os
import sys
from torch.utils.data import Dataset

class Rellis3DDatasetTorch(Dataset):
    def __init__(self,
                 ignore_label=255,
                 crop_sizeHW=(1024, 1024),
                 base_addr="/home/atman/a_workspace/mmlab/mmsegmentation/data/rellis3d",
                 mode = "train" # "train/test/val"
                 ):
        self.img_shape                  = (1200, 1920) # H, W
        self.ignore_label               = ignore_label
        self.crop_size                  = crop_sizeHW
        self.base_addr                  = base_addr
        self.mode                       = mode
        self.device                     = torch.device("cpu")

        self.img_addrs                  = []
        self.cal_addrs                  = []
        self.label_mapping = {0: 0,
                               1: 0,
                               3: 1,
                               4: 2,
                               5: 3,
                               6: 4,
                               7: 5,
                               8: 6,
                               9: 7,
                               10: 8,
                               12: 9,
                               15: 10,
                               17: 11,
                               18: 12,
                               19: 13,
                               23: 14,
                               27: 15,
                               31: 16,
                               33: 17,
                               34: 18}
        self.lut = torch.ones(256, dtype=torch.long) * self.ignore_label

        lst_path = os.path.join(self.base_addr, f"{self.mode}.lst")
        with open(lst_path, "r") as file_in:
            for line in file_in:
                img_pth, cal_pth = line.strip().split(' ') # this is the .lst property
                self.img_addrs.append(img_pth)
                self.cal_addrs.append(cal_pth)
        for k, v in self.label_mapping.items():
            self.lut[k] = v

    def __len__(self):
        return len(self.img_addrs)

    def __getitem__(self, item):
        img_addr = os.path.join(self.base_addr, self.img_addrs[item])
        cal_addr = os.path.join(self.base_addr, self.cal_addrs[item])
        img = cv2.imread(img_addr, cv2.IMREAD_COLOR)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        cal = cv2.imread(cal_addr, cv2.IMREAD_GRAYSCALE)
        if max(self.crop_size) > min(self.img_shape):
            if max(self.crop_size) <= max(self.img_shape):
                warnings.warn("your img slice is not a square")
            else:
                raise ValueError("Img slice should smaller than original img")

        h_start = np.random.randint(0, self.img_shape[0] - self.crop_size[0])
        w_start = np.random.randint(0, self.img_shape[1] - self.crop_size[1])
        img = img[h_start:h_start + self.crop_size[0], w_start:w_start + self.crop_size[1], :]
        cal = cal[h_start:h_start + self.crop_size[0], w_start:w_start + self.crop_size[1]]
        img = torch.tensor(img.transpose(2, 0, 1), device=self.device, dtype=torch.float32)
        cal = torch.tensor(cal, device=self.device, dtype=torch.float32)
        cal = self.lut[cal.long()]
        img = img / 255
        cal = cal.long()
        return img, cal

# TODO FINISH this loader, just return tensor and gt.
# Elegant but troublesome
class RellisDataloader:
    def __init__(self):
        pass

if __name__ == "__main__":
    sys.path.append(os.path.dirname(os.path.dirname("/home/atman/a_workspace/mmlab/mmsegmentation")))
    from src.utils import Visualizer

    dataset = Rellis3DDatasetTorch(255, crop_sizeHW=(512, 512))
    loader = torch.utils.data.DataLoader(dataset, batch_size=2)
    vis = Visualizer()
    for _, j in enumerate(loader, 0):
        vis.RGB_01_show(j[0][0])
        vis.gt_show(j[1][0])
    print("pass validation")
