"""
File: test.py
Author: Atman
Date: 1/23/26
Description:
    
"""

import torch
import cv2
import numpy as np
from utils import Visualizer
from mmengine import RUNNERS
from mmengine.runner import Runner
from mmengine import Config


if __name__ == "__main__":

    from mmseg.utils import register_all_modules
    register_all_modules()
    vis = Visualizer()
    cfg = Config.fromfile("/home/atman/a_workspace/mmlab/mmsegmentation/usr/configs/exp/patch_config.py")
    data_loader = Runner.build_dataloader(cfg.train_dataloader)
    date_iter  = iter(data_loader)
    img = cv2.imread('/home/atman/a_workspace/mmlab/mmsegmentation/bisenet/frame000000-1581623790_349.png', cv2.IMREAD_GRAYSCALE)
    ten = torch.tensor(img)
    vis.gt_show(ten)
    vis.visualize_palette()
    print("pass validation")
