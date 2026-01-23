import argparse
import logging
import os
import os.path as osp

import torch
from mmengine import Config
from torch.utils.data import DataLoader

from usr.utils                      import Visualizer
from usr.patch                      import PatchHandler
from usr.classifier_pipeline        import Classifier
from usr.datasets.rellis_pytorch    import Rellis3DDatasetTorch
from usr.diffusion_loss_neat.diff_loss_pipeline \
                                    import DiffLossTools
from usr.utils.ElegantLogger        import ElegantLogger
torch.autograd.set_detect_anomaly(True)

def main():
    config_file = "/home/atman/a_workspace/mmlab/mmsegmentation/usr/configs/patch_config.py"
    vis                     = Visualizer()
    cfg                     = Config.fromfile(config_file)
    dataset                 = Rellis3DDatasetTorch(**cfg.dataset_cfg)
    dataloader              = DataLoader(dataset, **cfg.dataloader_cfg)
    classifier              = Classifier(cfg.classifier_cfg)
    patch_handler           = PatchHandler(cfg.patch_handler_cfg)
    diffusion_loss_pipeline = DiffLossTools(cfg.diffusion_cfg)
    logger                  = ElegantLogger()
    # train iter
    for e in range(cfg.epochs):
        step = 0
        for img_clean, gt_clean in dataloader: # 01, RGB
            loss_iter = 0
            img_adv, patched_gt = patch_handler.apply_patch(img_clean, gt_clean)
            anchor = patch_handler.anchor # patch left up corner
            pred, logits = classifier.inference(img_adv)

            classifier_loss = classifier.class_loss(logits, gt_clean, anchor)
            loss_iter = loss_iter + classifier_loss
            logger.update(classifier_loss=classifier_loss)

            diff_clean  = diffusion_loss_pipeline.std_image_to_diffusion_format(img_clean, anchor)
            diff_adv    = diffusion_loss_pipeline.std_image_to_diffusion_format(img_adv, anchor)
            diff_gt     = diffusion_loss_pipeline.get_gt_in_patch(gt_clean, anchor)
            for clean, adv, label in zip(diff_clean, diff_adv, diff_gt):
                clean = clean.unsqueeze(0) # diffusion batch_size = 1, so align to BCHW
                adv = adv.unsqueeze(0)
                self_loss, cross_loss = diffusion_loss_pipeline.get_loss(clean, adv, label)
                loss_iter = loss_iter + self_loss + cross_loss
                logger.update(self_loss=self_loss, cross_loss=cross_loss)

            loss_iter.backward()
            patch_handler.step_zerograd()
            step += 1
            if step % 10 == 0:
                logger.flush(e, step)
        patch_handler.dump(path=f"./patch/patch_{e}.png")

if __name__ == "__main__":
    main()