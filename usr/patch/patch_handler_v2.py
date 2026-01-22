"""
File: patch.py
Author: Atman
Date: 1/21/26
Description:
    
"""
import cv2
import numpy as np
import os
import torch
import random
import matplotlib.pyplot as plt

import torchvision.transforms as transforms

from mmengine import Config
from usr.utils import Visualizer

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname("/home/atman/a_workspace/mmlab/mmsegmentation")))

class PatchHandler:
    def __init__(self, cfg):
        """summary: get float32 trainable tensor Patch.
        Args:
            cfg: mmengine.ConfigDict
        """
        self.vis                = Visualizer()

        self.optim_lr           = cfg.lr;
        self.optim_optim_name   = cfg.optim_name

        self.patch_dump_path    = cfg.dump_path
        self.patch_patch_size   = cfg.patch_size
        self.patch_alpha        = cfg.alpha
        self.patch_ignore_label = cfg.ignore_label
        self.patch_location     = cfg.location # "random" or "center"

        self.eot_enable_eot     = cfg.enable_eot
        self.eot_rot_deg        = cfg.rot_deg
        self.eot_scaling        = cfg.scale
        self.eot_translate      = cfg.max_translate

        self.anchor             : tuple[int, int] # h_start, w_start

        # EOT Transformation Build
        max_pad = int((np.cos(np.pi / 4) - 1 / 2 )* self.patch_patch_size + self.eot_translate * self.patch_patch_size)
        self.eot_transform      = transforms.Compose([
            transforms.Pad(padding=max_pad, padding_mode="reflect"),
            transforms.RandomRotation(degrees=(-self.eot_rot_deg, self.eot_rot_deg)),
            transforms.RandomAffine(degrees=0, translate=(self.eot_translate, self.eot_translate)),
            transforms.CenterCrop(self.patch_patch_size),
            transforms.RandomResizedCrop(size=self.patch_patch_size, scale=self.eot_scaling),
        ])

        # read patch or generate it, get float32 type
        if self.patch_dump_path and os.path.exists(self.patch_dump_path):
            self.patch = cv2.imread(self.patch_dump_path) # uint8
            self.patch = cv2.cvtColor(self.patch, cv2.COLOR_BGR2RGB)
            if self.patch.shape[1] != self.patch_patch_size:
                self.patch = np.random.rand(self.patch_patch_size, self.patch_patch_size, 3)* 255
        else:
            self.patch = np.random.rand(256, 256, 3) * 255
        self.patch = self.patch.astype(np.float32)

        # np to tensor
        self.patch = self.patch.transpose(2, 0, 1)  # HWC -> CHW
        self.patch = torch.tensor(self.patch, dtype=torch.float32, requires_grad=True)
        if self.patch.max() > 255.001:
            raise ValueError(f"PatchHandler: expect from 0 to 255 patch, but get {self.patch.max()}")

        # build optim
        optcls = getattr(torch.optim, self.optim_optim_name)
        self.optimizer = optcls([self.patch,], self.optim_lr)
        self.optimizer.zero_grad()

    def apply_patch(self, tensor, gt):
        """
        Args:
            tensor: float32, [BCHW] original data.
            gt    : ground truth for classifier (Segmentor).
        Returns   : Patched tensor and a copy of gt with ignore
                    label on the patch area.
        """
        if tensor.ndim != 4:
            raise RuntimeError(f"Expected input BCHW tensor, but get ndim {tensor.ndim}")
        b, c, h, w = tensor.shape
        h_max = h - self.patch_patch_size
        w_max = w - self.patch_patch_size
        if self.patch_location == 'random':
            h_start = random.randint(0, h_max)
            w_start = random.randint(0, w_max)
        elif self.patch_location == "center":
            h_start = int((h - self.patch_patch_size) / 2)
            w_start = int((w - self.patch_patch_size)/ 2)
        h_end = h_start + self.patch_patch_size
        w_end = w_start + self.patch_patch_size
        self.anchor = (h_start, w_start) # This is IMPORTANT!!!!!

        # add patch to image
        patch_transformed = self.eot_transform(self.patch)
        tensor[:, :, h_start:h_end, w_start:w_end] =\
            (1-self.patch_alpha) * tensor[:, :, h_start:h_end, w_start:w_end] + self.patch_alpha * patch_transformed
        gt_patched = gt.clone()
        gt_patched[:, h_start:h_end, w_start:w_end] = self.patch_ignore_label

        return tensor, gt_patched

    def dump(self):
        # Save patch
        img = self.patch.permute(1, 2, 0).detach().cpu().numpy()
        img.astype(np.uint8)
        img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        cv2.imwrite(self.patch_dump_path, img)
        print(f"cv2: write patch at {self.patch_dump_path}")


    # TODO Update method
    def update(self):
        pass

if __name__ == "__main__":
    cfg = Config.fromfile("/home/atman/a_workspace/mmlab/mmsegmentation/usr/configs/exp/patch_config.py")
    t = torch.rand(2, 3, 512, 512)* 256
    gt = torch.full((2, 512, 512), 7)
    gt = gt.to(torch.uint8)
    patch = PatchHandler(cfg.patch_handler)
    t, gt = patch.apply_patch(t, gt)
    patch.vis.RGB_tensor_show(t[0].to(torch.uint8))
    patch.vis.gt_show(gt[0])
    patch.dump()
