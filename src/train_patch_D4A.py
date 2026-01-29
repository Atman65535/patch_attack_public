"""
File: train_patch_D4A.py
Author: Atman
Date: 1/29/26
Description:
    
"""
import sys
import os

import torch
from omegaconf import OmegaConf
from torch.utils.data import DataLoader

from src.utils                      import Visualizer
from src.patch                      import PatchHandler
from src.classifier_pipeline        import Classifier
from src.datasets.rellis_pytorch    import Rellis3DDatasetTorch
from src.diffusion_loss_neat.diff_loss_pipeline \
    import DiffLossTools
from src.utils                      import MetricsKit
from src.utils                      import LogAssistant
from src.LMAG.LMAGPipeline import LMAGPipeline
torch.autograd.set_detect_anomaly(True)

import wandb
from loguru import logger
import time
os.chdir('/home/atman/a_workspace/D4A')

OmegaConf.register_new_resolver("eval", eval)
config_file = "./src/configs/D4A_config_local.yaml"
cfg = OmegaConf.load(config_file)



@logger.catch
def main():
    dataset                 = Rellis3DDatasetTorch(ignore_label=cfg.dataset_cfg.ignore_label,
                                                   crop_sizeHW=cfg.dataset_cfg.crop_sizeHW,
                                                   base_addr=cfg.dataset_cfg.base_addr,
                                                   mode=cfg.dataset_cfg.mode)
    dataloader              = DataLoader(dataset, shuffle=cfg.dataloader_cfg.shuffle,
                                         batch_size=cfg.dataloader_cfg.batch_size,
                                         num_workers=cfg.dataloader_cfg.num_workers)
    classifier              = Classifier(cfg.classifier_cfg)
    patch_handler           = PatchHandler(cfg.patch_handler_cfg)
    # diff_tools = DiffLossTools(cfg.diffusion_cfg)
    diffusion_pipeline = LMAGPipeline(cfg.LMAG_cfg)
    metrics                 = MetricsKit(cfg.metrics_cfg)
    log_ass                 = LogAssistant(cfg)
    gradient_cnt            = 0
    # train iter
    for e in range(cfg.epochs):
        for img_clean, gt_clean in dataloader: # 01, RGB
            img_clean = img_clean.to("cuda")
            gt_clean = gt_clean.to("cuda")
            loss_iter = 0
            img_adv, patched_gt = patch_handler.apply_patch(img_clean, gt_clean)
            anchor = patch_handler.anchor # patch left up corner
            pred, logits = classifier.inference(img_adv)

            classifier_loss = classifier.class_loss(logits, gt_clean, anchor)
            loss_iter = loss_iter + classifier_loss
            log_ass.total_loss += classifier_loss.item()
            log_ass.total_classify_loss += classifier_loss.item()

            diff_clean  = diffusion_pipeline.rfes_crop(img_clean, anchor)
            diff_adv    = diffusion_pipeline.rfes_crop(img_adv, anchor)
            diff_gt     = diffusion_pipeline.get_gt_in_patch(gt_clean, anchor)
            pack = (None, None, None, None)
            for clean, adv, label in zip(diff_clean, diff_adv, diff_gt):
                clean = clean.unsqueeze(0) # diffusion batch_size = 1, so align to BCHW
                adv = adv.unsqueeze(0)
                self_loss, cross_loss, clean_self, pack = diffusion_pipeline.get_loss(clean, adv, label)
                loss_iter = loss_iter + self_loss + cross_loss
                log_ass.total_loss += self_loss.item() + cross_loss.item()
                log_ass.total_self_loss += self_loss.item()
                log_ass.total_cross_loss += cross_loss.item()

            loss_iter.backward()
            gradient_cnt += 1

            if gradient_cnt == cfg.gradient_storage:
                gradient_cnt = 0
                patch_handler.step_zerograd()
            # Visualization and logging
            log_ass.global_steps += 1
            with torch.no_grad():
                if log_ass.global_steps % cfg.log_iter == 0:
                    logger.info(
                        f"Step: {log_ass.global_steps:05d}" +
                        f"| ASR: {metrics.asr_score(pred, gt_clean):.4f}" +
                        f"| LPIPS: {metrics.lpips_score(img_clean, img_adv):.4f}" +
                        f"| mIoU: {metrics.miou_score(pred, gt_clean):.4f} " +
                        f"| ClLoss: {log_ass.total_classify_loss / cfg.log_iter:.4f} " +
                        f"| SelfLoss: {log_ass.total_self_loss / cfg.log_iter:.4f} "+
                        f"| CrossLoss: {log_ass.total_cross_loss / cfg.log_iter:.4f}")
                    log_ass.wandb_loss_push()
                if log_ass.global_steps % (cfg.log_iter * 2) == 0:
                    log_ass.status_table.add_data(log_ass.global_steps, diffusion_pipeline.cur_prompt)
                    log_ass.wandb_image_push_aug(img_adv[0], pred[0], gt_clean[0],
                                             patch_handler.patch,
                                             **pack)

                    log_ass.clear()
        patch_handler.dump(path=f"./patch/patch_{e}.png")

if __name__ == "__main__":
    main()
