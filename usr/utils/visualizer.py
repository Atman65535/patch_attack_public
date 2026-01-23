"""
File: visualizer.py
Author: Atman
Date: 1/21/26
Description:
    
"""
from pickletools import uint8

import cv2
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import torch

class Visualizer:
    def __init__(self):
        self.palette = {0: {"color": [0, 0, 0],         "name": "void"},
                        1: {"color": [108, 64, 20],     "name": "dirt"},
                        3: {"color": [0, 102, 0],       "name": "grass"},
                        4: {"color": [0, 255, 0],       "name": "tree"},
                        5: {"color": [0, 153, 153],     "name": "pole"},
                        6: {"color": [0, 128, 255],     "name": "water"},
                        7: {"color": [0, 0, 255],       "name": "sky"},
                        8: {"color": [255, 255, 0],     "name": "vehicle"},
                        9: {"color": [255, 0, 127],     "name": "object"},
                        10: {"color": [64, 64, 64],     "name": "asphalt"},
                        12: {"color": [255, 0, 0],      "name": "building"},
                        15: {"color": [102, 0, 0],      "name": "log"},
                        17: {"color": [204, 153, 255],  "name": "person"},
                        18: {"color": [102, 0, 204],    "name": "fence"},
                        19: {"color": [255, 153, 204],  "name": "bush"},
                        23: {"color": [170, 170, 170],  "name": "concrete"},
                        27: {"color": [41, 121, 255],   "name": "barrier"},
                        31: {"color": [134, 255, 239],  "name": "puddle"},
                        33: {"color": [99, 66, 34],     "name": "mud"},
                        34: {"color": [110, 22, 138],   "name": "rubble"}}
        self.color_list = [
            [108, 64, 20],
            [0, 102, 0],
            [0, 255, 0],
            [0, 153, 153],
            [0, 128, 255],
            [0, 0, 255],
            [255, 255, 0],
            [255, 0, 127],
            [64, 64, 64],
            [255, 0, 0],
            [102, 0, 0],
            [204, 153, 255],
            [102, 0, 204],
            [255, 153, 204],
            [170, 170, 170],
            [41, 121, 255],
            [134, 255, 239],
            [99, 66, 34],
            [110, 22, 138],
        ]
        while len(self.color_list) < 256:
            self.color_list.append([0,0,0])
        self.color_list = np.array(self.color_list, dtype=np.uint8)
    def RGB_01_show(self, tensor):
        """
        Img show, Only accept standart 01 torch tensor or np array
        Args:
            tensor: What ever you want to shou
        """
        if type(tensor) is torch.Tensor:
            t = tensor.detach().cpu().permute(1, 2, 0).numpy()
        if t.dtype is np.float32:
            t = (t + 1.)/2
            t *= 255
            t =t.astype(np.uint8)
        plt.imshow(t)
        plt.show()

    def gt_show(self, gt):
        # [H, W]
        if gt.dtype is not torch.uint8:
            gt = gt.to(torch.uint8)
        im_out = self.color_list[gt.detach().cpu()]
        plt.imshow(im_out)
        plt.show()

    def visualize_palette(self, class_dict = None):
        """
        class_dict: 格式为 {id: {"color": [R, G, B], "name": "xxx"}, ...}
        """
        if class_dict == None:
            class_dict = self.palette
         # 提取信息
        ids = sorted(class_dict.keys())
        colors = [np.array(class_dict[i]["color"]) / 255.0 for i in ids] # 归一化到 0-1
        names = [class_dict[i]["name"] for i in ids]

        # 创建图例句柄 (Handles)
        legend_handles = []
        for color, name, idx in zip(colors, names, ids):
            # 创建一个色块
            patch = mpatches.Patch(color=color, label=f"{idx}: {name}")
            legend_handles.append(patch)

        # 创建画布
        # 根据类别数量动态调整高度
        fig = plt.figure(figsize=(6, len(ids) * 0.4))

        # 在画布上添加图例
        # loc='center' 让图例居中显示
        # frameon=False 去掉边框
        plt.legend(handles=legend_handles, loc='center', fontsize=12, frameon=False)

        plt.axis('off') # 隐藏主坐标轴
        plt.title("Rellis3D Category Legend", fontsize=14, pad=20)

        plt.tight_layout()
        plt.show()

    def show_cross_attention_map(self, attn_map, file_name = "map"):
        scale = attn_map.shape[0]
        tmp_img = torch.mean(attn_map, dim=-1)# expand to image
        tmp_img = tmp_img * 255 / tmp_img.max()
        tmp_img = tmp_img.to(torch.uint8)
        tmp_img = tmp_img.detach().cpu().unsqueeze(-1).expand(scale, scale, 3).numpy()
        tmp_img = cv2.cvtColor(tmp_img, cv2.COLOR_RGB2BGR)
        tmp_img = cv2.applyColorMap(tmp_img, cv2.COLORMAP_JET)
        tmp_img = cv2.resize(tmp_img, (256, 256), interpolation=cv2.INTER_NEAREST)
        cv2.imwrite(f"./heat_map/{file_name}.png", tmp_img)

    def show_self_attention_map(self, attn_map, file_name="map"):
        output_size = (256, 256)
        if attn_map.shape[-1] != attn_map.shape[0] * attn_map.shape[1]:
            raise ValueError("expected attention area is a square!")
        if torch.is_tensor(attn_map):
            attn_map = attn_map.detach().cpu().numpy()
        flat_map = attn_map.reshape(attn_map.shape[-1], attn_map.shape[-1]) # a big
        u, s, vh = np.linalg.svd(flat_map - np.mean(flat_map, axis=1, keepdims=True))
        res_map = u[:, 0].reshape(8, 8)
        res_map = (res_map - res_map.min()) / (res_map.max() - res_map.min() + 1e-8)
        res_map = (res_map * 255).astype(np.uint8)
        res_map = np.stack([res_map] * 3, axis=-1)
        vis_large = cv2.resize(res_map, output_size, interpolation=cv2.INTER_NEAREST)
        color_map = cv2.applyColorMap(vis_large, cv2.COLORMAP_JET)
        cv2.imwrite(f"./heat_map/{file_name}.png", color_map)

if __name__ == "__main__":
    print("pass validation")
