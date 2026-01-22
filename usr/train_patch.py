# system and tools
import os
import os.path as osp
from typing import List, Dict, Tuple, Optional, Union
import argparse
import logging
import pickle

import torch
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import numpy as np

import mmseg.apis
from mmengine import Config

from usr.diffusion_loss_neat.diff_loss_pipeline import DiffLossTools
from usr.classifier_pipeline import Classifier
from usr.patch import PatchHandler
from usr.metrics import PatchMetrics
from usr.utils import Utils, LossHandler
from usr.utils import Visualizer
from usr.datasets import Rellis3DDataset

def main():
    torch.autograd.set_detect_anomaly(True)
    vis = Visualizer()
    # cfg init
    config_file = "/home/atman/a_workspace/mmlab/mmsegmentation/usr/configs/exp/patch_config.py"
    cfg = Config.fromfile(config_file)
    classifier = Classifier(cfg)
    # patch associated
    # patch_handler = PatchHandler(cfg)
    patch_handler = PatchHandler(cfg.patch_handler) # new version
    patch_metrics = PatchMetrics()
    # data associated
    # todo add to config file
    dataset = Rellis3DDataset(crop_sizeHW=(1024, 1024))
    data_loader = torch.utils.data.DataLoader(dataset,batch_size=4)


    diffusion_loss_pipeline = DiffLossTools(cfg.diffusion_config)
    # train iter
    loss_iter = cfg.loss_back_iter # 和下面的cnt一起用于梯度累积，暂时没有用上
    loss_iter_cnt = 0
    #loss = LossHandler(cfg.weight_config)
    for e in range(cfg.epochs):
        # contains ['pred_sem_seg', 'seg_logits']
        for _, batch in enumerate(data_loader, 0):
            total_loss = 0
            loss_iter_cnt += 1
            # preprocess: normalize and apply patch
            classify_loss, gt_batch = classifier_pipeline(patch_handler, model, patch_metrics, preprocessor, batch)
            diff_batch = diffusion_loss_pipeline.image_preprocessor01(batch['inputs'], 1024, 1024)
            clean_batch, adv_batch, gt = patch_handler.apply_patch(diff_batch, gt_batch, classifier=False)
            # -1 to 1 img
            clean_batch = clean_batch * 2.0 - 1.0
            adv_batch = adv_batch * 2.0 - 1.0
            loss.update(classifier=classify_loss)
            total_loss = classify_loss * loss.classifier_weight
            for clean, adv, gt in zip(clean_batch, adv_batch, gt):
                clean = clean.unsqueeze(0)
                adv = adv.unsqueeze(0)
                self_loss, cross_loss = diffusion_loss_pipeline.get_loss(clean, adv, gt)
                loss.update(self_attn=self_loss, cross=cross_loss)
                total_loss = total_loss +self_loss * loss.self_attn_weight + cross_loss * loss.cross_attn_weight
           
            # 这块已经糊了，随便写的梯度累积（实则并没有累积）   
            total_loss.backward()
            patch_handler.patch_optim_step()
            loss.log(e)
            loss.reset()

        # save
        os.makedirs("./saved_patches", exist_ok=True)
        pickle.dump(patch_handler, open(osp.join("./saved_patches", f"patch_epoch_{e}.pat"), "wb"))

if __name__ == "__main__":

    from mmseg.utils import register_all_modules
    register_all_modules()

    main()