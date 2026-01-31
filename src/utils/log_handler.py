"""
File: log_handler.py
Author: Atman
Date: 1/25/26
Description:
    
"""
import sys

import torch
from loguru import logger
import time
import wandb
from omegaconf import OmegaConf
from torch.onnx.symbolic_opset8 import full
from src.utils import Visualizer
import cv2
import numpy as np


class LogAssistant:
    def __init__(self, full_cfg, init=True):
        if init:
            wandb.init(
                project=full_cfg.project,
                name=full_cfg.exp_name,
                config={
                    "PEX": 0,
                    "model_arch": full_cfg.model_arch,
                    "patch_size": full_cfg.patch_size,
                }
            )
            config_dict = OmegaConf.to_container(full_cfg, resolve=True)
            wandb.config.update(config_dict)
            logger.remove()
            logger.add(sink=sys.stdout,
                       format='<green>{time:YY-MM-DD HH:mm}</green> | <level>{level}</level> | - <level>{message}</level>')

            logger.add(sink=f"logs/{time.strftime('%m%d_%H%M')}_exp_{full_cfg.exp_name}.log",
                       format='<green>{time:YY-MM-DD HH:mm}</green> | <level>{level}</level> | - <level>{message}</level>',
                       rotation="50 MB")

        self.log_iter = full_cfg.log_iter
        self.global_steps = 0
        self.total_loss = 0
        self.total_classify_loss = 0
        self.total_self_loss = 0
        self.vis = Visualizer()

    @torch.no_grad()
    def clear(self):
        self.total_loss = 0
        self.total_classify_loss = 0
        self.total_self_loss = 0

    @torch.no_grad()
    def wandb_loss_push(self):
        wandb.log({
            "Loss/Total": self.total_loss / self.log_iter,
            "Loss/Classifier": self.total_classify_loss /self.log_iter,
            "Loss/Self-Attention": self.total_self_loss / self.log_iter,
        }, step=self.global_steps)
        self.clear()

    @torch.no_grad()
    def wandb_image_push_aug(self, img_adv, pred, gt_clean, patch, clean_self, adv_self):
        log_dict = {
            "Visuals/CleanImage": wandb.Image(img_adv.permute(1, 2, 0).detach().cpu().numpy()),
            "Visuals/Predictions": wandb.Image(self.vis.gt_show(pred,return_array=True)),
            "Visuals/GroundTruth": wandb.Image(self.vis.gt_show(gt_clean, return_array=True)),
            "Visuals/Raw_Patch": wandb.Image(patch.permute(1, 2, 0).detach().cpu().numpy(), caption="Generated Patch Texture"),
            "Visuals/Self_Attention_clean": wandb.Image(add_color(clean_self, "clean_self"), caption="self attn clean"),
            "Visuals/Self_Attention_adv":   wandb.Image(add_color(adv_self, "adv_self"), caption="self attn adv"),
        }
        wandb.log(log_dict, step=self.global_steps)

def add_color(attn, txt=None):
    if type(attn) is torch.Tensor:
        attn = attn.detach().cpu().numpy()
    attn = attn - attn.min()
    attn = attn * 255 / attn.max()
    image = cv2.applyColorMap(attn.astype(np.uint8), cv2.COLORMAP_JET)
    image = cv2.resize(image, (256, 256), interpolation=cv2.INTER_NEAREST)
    if txt:
        font = cv2.FONT_HERSHEY_SIMPLEX
        cv2.putText(image, txt, (5, 20), font, 0.6, (0, 255, 0), 2)
    return image

if __name__ == "__main__":
    print("pass validation")
