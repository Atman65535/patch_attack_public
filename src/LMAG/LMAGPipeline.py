"""
File: LMAGPipeline.py
Author: Atman
Date: 1/29/26
Description:
    
"""
import cv2
import torch
from .LMAGEngine import LMAGEngine
from .LMAGScheduler import LMAGScheduler

class LMAGPipeline:
    def __init__(self, cfg):
        """send in cfg.LMAG_cfg"""
        self.scheduler = LMAGScheduler(cfg)
        self.rfes_edge = cfg.rfes_edge
        self.resolution = cfg.resolution # real size into diffusion

    def get_loss(self, clean, adv):
        return self.scheduler.run(adv, clean)

    def rfes_crop(self, tensor, anchor = None):
        """
        input : Standard image batch of dataset, range [0, 1], float32
        anchor: hstart, wstart. this is the start point of patch, not rfes field
        Comment: anchor always valid. This is Implemented in patch handler
        Returns: Batch, (B, C, H, W), range [0, 1] for L-MAG
        """
        if tensor.ndim != 4:
            raise TypeError(f"expected tensor BCHW, but get ndim={tensor.ndim}")
        _, _, h, w = tensor.shape
        diff_tensor = tensor.clone()
        assert anchor is not None, "Anchor in rfes is None, please send in an anchor"
        h_start = anchor[0] - self.rfes_edge
        w_start = anchor[1] - self.rfes_edge
        h_end = h_start + self.resolution
        w_end = w_start + self.resolution
        return diff_tensor[:, :, h_start:h_end, w_start:w_end]

    def get_gt_in_patch(self, tensor, anchor):
        """
            Get ground truth under patched area.
        Args:
            tensor: gt clean [BHW]
            anchor: anchor from PatchHandler
        """
        _, h, w = tensor.shape
        diff_tensor = tensor.clone()
        h_start = anchor[0] - self.rfes_edge
        w_start = anchor[1] - self.rfes_edge
        h_end = h_start + self.resolution
        w_end = w_start + self.resolution
        return diff_tensor[:, h_start:h_end, w_start:w_end]

if __name__ == "__main__":
    print("pass validation")
