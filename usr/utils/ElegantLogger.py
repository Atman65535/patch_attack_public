"""
File: ElegantLogger.py
Author: Atman & Gemini
Date: 1/23/26
Description:
    
"""

import time
import os
from collections import defaultdict
import torch

class ElegantLogger:
    """
    ['classifier_loss', 'self_loss', 'cross_loss']
    please send into loss terms like format above
    """
    def __init__(self, save_path="logs"):
        os.makedirs(save_path, exist_ok=True)
        self.log_file = os.path.join(save_path, f"exp_{time.strftime('%m%d_%H%M')}.log")
        self.metrics = defaultdict(float)

    def update(self, **kwargs):
        """支持一次传入多个loss，例如 logger.update(classifier=0.5, self_attn=0.2)"""
        for k, v in kwargs.items():
            if isinstance(v, torch.Tensor):
                v = v.detach().item() # 彻底切断计算图，防止显存泄漏
            self.metrics[k] += v

    def flush(self, epoch, step=None):
        """格式化输出并重置累加器"""
        timestamp = time.strftime("%Y-%m-%d %H:%M:%S")
        log_str = f"[{timestamp}] Epoch: {epoch:03d}"
        if step is not None:
            log_str += f" | Step: {step:04d}"

        # 拼接三种 Loss
        loss_details = []
        for name in ['classifier_loss', 'self_loss', 'cross_loss']:
            avg_val = self.metrics[name]
            loss_details.append(f"{name}: {avg_val:.4f}")

        log_str += " | " + " | ".join(loss_details)

        # 1. 打印到终端 (带一点颜色更漂亮)
        print(f"\033[1;34m{log_str}\033[0m")

        # 2. 写入文件
        with open(self.log_file, "a") as f:
            f.write(log_str + "\n")

        # 3. 重置累加器
        self.metrics.clear()

if __name__ == "__main__":
    logger = ElegantLogger()

# # 训练循环中
#     for images, masks in loader:
#         # 1. 真正的 BP Loss
#         total_loss = classifier_loss + ...
#         total_loss.backward()
#
#         # 2. 只用于记录的 Loss (直接丢进去，它会自动 detach)
#         logger.update(classifier=classifier_loss, self=self_loss, cross=cross_loss)
#
#         # 3. 按照你的步长打 Log
#         if step % 50 == 0:
#             logger.flush(epoch, step)
    print("pass validation")
