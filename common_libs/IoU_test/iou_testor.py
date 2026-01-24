"""
File: iou_testor.py
Author: Atman
Date: 1/24/26
Description:
    
"""
import segmentation_models_pytorch

# Target: send IoU out
import segmentation_models_pytorch as smp
import torch
from torch.utils.data import DataLoader
from usr.datasets.rellis_pytorch import Rellis3DDatasetTorch

dataset = Rellis3DDatasetTorch(crop_sizeHW=(1024, 1024), mode="test")

model_list = []

unet_plus_plus = smp.UnetPlusPlus(encoder_name="resnet34",
                                  encoder_depth=5,
                                  encoder_weights="imagenet",
                                  in_channels=3,
                                  classes=19)
ckpt = ""