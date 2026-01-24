"""
File: TrainMonitor.py
Author: Atman
Date: 1/23/26
Description:
    
"""
import wandb
import numpy as np

class WBVisualizer:
    def __init__(self, project_name, config=None):
        # 1. 初始化云端项目
        wandb.init(project=project_name, config=config)

        # 你的调色板定义（W&B 的分割图可以直接支持 class_labels 字典）
        self.class_labels = {
            1: "dirt", 3: "grass", 4: "tree", 5: "pole", # ... 依此类推
        }

    def log_scalar(self, tag, value, step=None):
        # W&B 会自动维护 step，也可以手动指定
        wandb.log({tag: value}, step=step)

    def log_seg_images(self, image_tensor, mask_tensor, pred_tensor, tag="Validation"):
        """
        image_tensor: [3, H, W]
        mask_tensor: [H, W] (GT)
        pred_tensor: [H, W] (模型输出)
        """
        img = image_tensor.detach().cpu().numpy().transpose(1, 2, 0)
        gt = mask_tensor.detach().cpu().numpy()
        pred = pred_tensor.detach().cpu().numpy()

        # W&B 的特色：在一个面板上开关 GT 和 Pred
        wandb.log({
            tag: wandb.Image(img, masks={
                "predictions": {
                    "mask_data": pred,
                    "class_labels": self.class_labels
                },
                "ground_truth": {
                    "mask_data": gt,
                    "class_labels": self.class_labels
                }
            })
        })

    def finish(self):
        wandb.finish()

if __name__ == "__main__":
    print("pass validation")
