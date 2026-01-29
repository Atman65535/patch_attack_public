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
from src.utils import Visualizer
from src.LMAG.LMAGScheduler import view_one


class LogAssistant:
    def __init__(self, full_cfg):
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
        self.status_table = wandb.Table(columns=["Step", "Prompts"], log_mode='MUTABLE')

        self.log_iter = full_cfg.log_iter
        self.global_steps = 0
        self.total_loss = 0
        self.total_classify_loss = 0
        self.total_self_loss = 0
        self.total_cross_ip_loss = 0
        self.total_cross_txt_loss = 0
        self.vis = Visualizer()

    @torch.no_grad()
    def clear(self):
        self.total_loss = 0
        self.total_classify_loss = 0
        self.total_self_loss = 0
        self.total_cross_txt_loss = 0
        self.total_cross_ip_loss = 0

    @torch.no_grad()
    def wandb_loss_push(self):
        wandb.log({
            "Loss/Total": self.total_loss / self.log_iter,
            "Loss/Classifier": self.total_classify_loss /self.log_iter,
            "Loss/Self-Attention": self.total_self_loss / self.log_iter,
            "Loss/txt-Cross-Attention": self.total_cross_txt_loss / self.log_iter,
            "Loss/ip-Cross-Attention": self.total_cross_ip_loss / self.log_iter,
        }, step=self.global_steps)
        self.clear()

    @torch.no_grad()
    def wandb_image_push(self, img_adv, pred, gt_clean, patch, clean_self, adv_self, cross_attn):
        log_dict = {
            "Visuals/CleanImage": wandb.Image(img_adv.permute(1, 2, 0).detach().cpu().numpy()),
            "Visuals/Predictions": wandb.Image(self.vis.gt_show(pred,return_array=True)),
            "Visuals/GroundTruth": wandb.Image(self.vis.gt_show(gt_clean, return_array=True)),
            "Visuals/Raw_Patch": wandb.Image(patch.permute(1, 2, 0).detach().cpu().numpy(), caption="Generated Patch Texture"),
            "Visuals/Self_Attention_clean": wandb.Image(self.vis.visualize_self_attn_map(clean_self), caption="self attn clean"),
            "Visuals/Self_Attention_adv": wandb.Image(self.vis.visualize_self_attn_map(adv_self), caption="self attn adv"),
            "Visuals/Cross_Attention": wandb.Image(self.vis.visualize_cross_attn_map(cross_attn), caption="cross attn"),
            "Monitor/Status" : self.status_table
        }
        wandb.log(log_dict, step=self.global_steps)

    @torch.no_grad()
    def wandb_image_push_aug(self, img_adv, pred, gt_clean, patch, clean_self, adv_self, cross_attn_ip, cross_attn_txt):
        log_dict = {
            "Visuals/CleanImage": wandb.Image(img_adv.permute(1, 2, 0).detach().cpu().numpy()),
            "Visuals/Predictions": wandb.Image(self.vis.gt_show(pred,return_array=True)),
            "Visuals/GroundTruth": wandb.Image(self.vis.gt_show(gt_clean, return_array=True)),
            "Visuals/Raw_Patch": wandb.Image(patch.permute(1, 2, 0).detach().cpu().numpy(), caption="Generated Patch Texture"),
            "Visuals/Self_Attention_clean": wandb.Image(view_one("", clean_self, 0, None, True), caption="self attn clean"),
            "Visuals/Self_Attention_adv":   wandb.Image(view_one("", adv_self, 0, None, True), caption="self attn adv"),
            "Visuals/Cross_Attention_IP":   wandb.Image(view_one("", cross_attn_ip, 0, None, True), caption="cross attn"),
            "Visuals/Cross_Attention_txt":  wandb.Image(view_one("", cross_attn_txt, 0, None, True ), caption="cross attn"),
        }
        wandb.log(log_dict, step=self.global_steps)
if __name__ == "__main__":
    print("pass validation")
