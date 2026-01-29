"""
File: patch.py
Author: Atman
Date: 1/21/26
Description:
    
"""
import warnings

import cv2
import numpy as np
import os
import torch
import random

import torchvision.transforms as transforms
from src.utils import Visualizer

class PatchHandler:
    def __init__(self, cfg):
        """summary: get float32 trainable tensor Patch.
                    Patch is at [0, 1] region
        Args:
            cfg: OmegaConf configdict
        """
        self.device             = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.vis                = Visualizer()

        self.optim_lr           = cfg.lr
        self.optim_optim_name   = cfg.optim_name

        self.patch_load_from    = cfg.load_from
        self.patch_patch_size   = cfg.patch_size
        self.rfes_edge          = cfg.rfes_edge
        self.patch_ignore_label = cfg.ignore_label
        self.patch_location     = cfg.location          # "random" or "center"

        self.eot_enable_eot     = cfg.enable_eot
        self.patch_alpha        = cfg.alpha
        self.eot_rot_deg        = cfg.rot_deg
        self.eot_scaling        = cfg.scale
        self.eot_translate      = cfg.max_translate

        self.anchor             = (-1, -1) # h_start, w_start

        # EOT Transformation Build
        if not self.eot_enable_eot:
            warnings.warn("EOT Transformation Disabled, Using Fixed Patch")

        max_pad = int((np.cos(np.pi / 4) - 1 / 2 )* self.patch_patch_size + self.eot_translate * self.patch_patch_size)
        self.eot_transform      = transforms.Compose([
            transforms.Pad(padding=max_pad, padding_mode="reflect"),
            transforms.RandomRotation(degrees=(-self.eot_rot_deg, self.eot_rot_deg)),
            transforms.RandomAffine(degrees=0, translate=(self.eot_translate, self.eot_translate)),
            transforms.CenterCrop(self.patch_patch_size),
            transforms.RandomResizedCrop(size=self.patch_patch_size, scale=self.eot_scaling),
            transforms.RandomVerticalFlip(p=0.5),
            transforms.RandomHorizontalFlip(p=0.5),
        ])

        # read patch or generate it, get float32 type
        if self.patch_load_from and os.path.exists(self.patch_load_from):
            self.patch = cv2.imread(self.patch_load_from) # uint8
            self.patch = cv2.cvtColor(self.patch, cv2.COLOR_BGR2RGB)
            self.patch = self.patch.astype(np.float32) / 255
            if self.patch.shape[1] != self.patch_patch_size:
                self.patch = np.random.rand(self.patch_patch_size, self.patch_patch_size, 3)
        else:
            self.patch = np.random.rand(self.patch_patch_size, self.patch_patch_size, 3)
        self.patch = self.patch.astype(np.float32)

        # np to tensor
        self.patch = self.patch.transpose(2, 0, 1)  # HWC -> CHW
        self.patch = torch.tensor(self.patch, dtype=torch.float32, requires_grad=True, device=self.device)
        if self.patch.max() > 1.001:
            raise ValueError(f"PatchHandler: expect from 0 to 1 patch, but get {self.patch.max()}")

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
        self._check_range(tensor)
        b, c, h, w = tensor.shape
        h_max = h - self.patch_patch_size - self.rfes_edge - 1 # extreme case : rfes at edge of image
        w_max = w - self.patch_patch_size - self.rfes_edge - 1
        if self.patch_location == 'random':
            h_start = random.randint(self.rfes_edge + 1, h_max)
            w_start = random.randint(self.rfes_edge + 1, w_max)
        elif self.patch_location == "center":
            h_start = int((h - self.patch_patch_size) / 2)
            w_start = int((w - self.patch_patch_size)/ 2)
        else:
            raise NotImplementedError(f"Only support center and random patch mode, {self.patch_location} is invalid")
        h_end = h_start + self.patch_patch_size
        w_end = w_start + self.patch_patch_size
        self.anchor = (h_start, w_start) # This is IMPORTANT!!!!!

        # add patch to image
        if self.eot_enable_eot:
            patch_transformed = self.eot_transform(self.patch)
        else:
            patch_transformed = self.patch
        tensor_adv = tensor.clone()
        tensor_adv[:, :, h_start:h_end, w_start:w_end] =\
            (1-self.patch_alpha) * tensor_adv[:, :, h_start:h_end, w_start:w_end] + self.patch_alpha * patch_transformed
        gt_patched = gt.clone()
        gt_patched[:, h_start:h_end, w_start:w_end] = self.patch_ignore_label

        return tensor_adv, gt_patched

    def dump(self, path= None):
        # Save patch
        img = self.patch.permute(1, 2, 0).detach().cpu().numpy()
        img = img * 255
        img.astype(np.uint8)
        img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        if path is None:
            path = self.patch_load_from
        cv2.imwrite(path, img)
        print(f"cv2: write patch at {path}")

    def _check_range(self, tensor):
        """
        Check the [0, 1] range of tensor(img)
        """
        if torch.max(self.patch) > 1.001 or torch.min(self.patch) < -0.0001:
            raise ValueError(f"the patch min is {torch.min(self.patch)}, max is {torch.max(self.patch)}, expected [0, 1]")
        if torch.max(tensor) > 1.0001 or torch.min(tensor) < -0.0001:
            raise ValueError(f"the input tensor should in [0, 1] but get min {torch.min(tensor)}, max {torch.max(tensor)}")

    def step_zerograd(self):
        self.optimizer.step()
        self.optimizer.zero_grad()
        with torch.no_grad():
            self.patch = torch.clamp_(self.patch, 0.00, 1.00)

#if __name__ == "__main__":
    # import sys
    # import os
    # sys.path.append(os.path.dirname(os.path.dirname("/home/atman/a_workspace/mmlab/mmsegmentation")))
    # cfg = Config.fromfile("/home/atman/a_workspace/mmlab/mmsegmentation/src/configs/exp/patch_config_local.py")
    # t = torch.rand(2, 3, 512, 512)* 256
    # gt = torch.full((2, 512, 512), 7)
    # gt = gt.to(torch.uint8)
    # patch = PatchHandler(cfg.patch_handler)
    # t, gt = patch.apply_patch(t, gt)
    # patch.vis.RGB_01_show(t[0].to(torch.uint8))
    # patch.vis.gt_show(gt[0])
    # patch.dump()
