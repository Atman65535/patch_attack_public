import argparse
import logging
import os
import os.path as osp

import torch
from omegaconf import OmegaConf
from torch.utils.data import DataLoader

from usr.utils                      import Visualizer
from usr.patch                      import PatchHandler
from usr.classifier_pipeline        import Classifier
from usr.datasets.rellis_pytorch    import Rellis3DDatasetTorch
from usr.diffusion_loss_neat.diff_loss_pipeline \
                                    import DiffLossTools
from usr.utils                      import ElegantLogger
torch.autograd.set_detect_anomaly(True)

import wandb

config_file = "/home/atman/a_workspace/mmlab/mmsegmentation/usr/configs/patch_config_local.yaml"
cfg = OmegaConf.load(config_file)
wandb.init(
    project=cfg.project,
    name=cfg.exp_name,
    config={
        "PEX": 0,
        "model_arch": cfg.model_arch,
        "patch_size": (cfg.patch_size, cfg.patch_size),
    }
)
config_dict = OmegaConf.to_container(cfg, resolve=True)
wandb.config.update(config_dict)

def main():

    vis                     = Visualizer()
    dataset                 = Rellis3DDatasetTorch(ignore_label=cfg.dataset_cfg.ignore_label,
                                                   crop_sizeHW=cfg.dataset_cfg.crop_sizeHW,
                                                   base_addr=cfg.dataset_cfg.base_addr,
                                                   mode=cfg.dataset_cfg.mode)
    dataloader              = DataLoader(dataset, shuffle=cfg.dataloader_cfg.shuffle,
                                         batch_size=cfg.dataloader_cfg.batch_size,
                                         num_workers=cfg.dataloader_cfg.num_workers)
    classifier              = Classifier(cfg.classifier_cfg)
    patch_handler           = PatchHandler(cfg.patch_handler_cfg)
    diffusion_loss_pipeline = DiffLossTools(cfg.diffusion_cfg)
    logger                  = ElegantLogger()
    # train iter
    global_steps = 0
    total_loss = 0
    total_classify_loss = 0
    total_self_loss = 0
    total_cross_loss = 0
    for e in range(cfg.epochs):
        for img_clean, gt_clean in dataloader: # 01, RGB
            loss_iter = 0
            img_adv, patched_gt = patch_handler.apply_patch(img_clean, gt_clean)
            anchor = patch_handler.anchor # patch left up corner
            pred, logits = classifier.inference(img_adv)

            classifier_loss = classifier.class_loss(logits, gt_clean, anchor)
            loss_iter = loss_iter + classifier_loss
            total_loss += classifier_loss.item()
            total_classify_loss += classifier_loss.item()
            logger.update(classifier_loss=classifier_loss)

            diff_clean  = diffusion_loss_pipeline.std_image_to_diffusion_format(img_clean, anchor)
            diff_adv    = diffusion_loss_pipeline.std_image_to_diffusion_format(img_adv, anchor)
            diff_gt     = diffusion_loss_pipeline.get_gt_in_patch(gt_clean, anchor)
            for clean, adv, label in zip(diff_clean, diff_adv, diff_gt):
                clean = clean.unsqueeze(0) # diffusion batch_size = 1, so align to BCHW
                adv = adv.unsqueeze(0)
                self_loss, cross_loss, clean_self, adv_self, cross_attn = diffusion_loss_pipeline.get_loss(clean, adv, label)
                loss_iter = loss_iter + self_loss + cross_loss
                total_loss += self_loss.item() + cross_loss.item()
                total_self_loss += self_loss.item()
                total_cross_loss += cross_loss.item()
                logger.update(self_loss=self_loss, cross_loss=cross_loss)

            loss_iter.backward()
            patch_handler.step_zerograd()
            global_steps += 1
            if global_steps % 5 == 0:
                logger.flush(e, global_steps)
                wandb.log({
                    "Loss/Total": total_loss,
                    "Loss/Classifier": total_classify_loss,
                    "Loss/Self-Attention": total_self_loss,
                    "Loss/Cross-Attention": total_cross_loss,
                    "Epoch": e,
                }, step=global_steps)
                log_dict = {
                    "Visuals/CleanImage": wandb.Image(img_adv[0].permute(1, 2, 0).detach().cpu().numpy()),
                    "Visuals/Predictions": wandb.Image(vis.gt_show(pred[0],return_array=True)),
                    "Visuals/GroundTruth": wandb.Image(vis.gt_show(gt_clean[0], return_array=True)),
                    "Visuals/Raw_Patch": wandb.Image(patch_handler.patch.permute(1, 2, 0).detach().cpu().numpy(), caption="Generated Patch Texture"),
                    "Visuals/Self_Attention_clean": wandb.Image(vis.visualize_self_attn_map(clean_self), caption="self attn clean"),
                    "Visuals/Self_Attention_adv": wandb.Image(vis.visualize_self_attn_map(adv_self), caption="self attn adv"),
                    "Visuals/Cross_Attention": wandb.Image(vis.visualize_cross_attn_map(cross_attn), caption="cross attn"),
                    "Visuals/Prompt": diffusion_loss_pipeline.get_prompt_from_gt(diff_gt[0]),
                }
                wandb.log(log_dict, step=global_steps)
        patch_handler.dump(path=f"./patch/patch_{e}.png")

if __name__ == "__main__":
    main()