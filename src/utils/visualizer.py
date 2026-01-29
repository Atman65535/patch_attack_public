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
        """this palette and color list is remapped definition"""
        self.id_palette_dict = {
                        # 0 : {"color": [0, 0, 0],         "name": "void"},
                        0 : {"color": [108, 64, 20],     "name": "dirt"},
                        1 : {"color": [0, 102, 0],       "name": "grass"},
                        2 : {"color": [0, 255, 0],       "name": "tree"},
                        3 : {"color": [0, 153, 153],     "name": "pole"},
                        4 : {"color": [0, 128, 255],     "name": "water"},
                        5 : {"color": [0, 0, 255],       "name": "sky"},
                        6 : {"color": [255, 255, 0],     "name": "vehicle"},
                        7 : {"color": [255, 0, 127],     "name": "object"},
                        8 : {"color": [64, 64, 64],      "name": "asphalt"},
                        9 : {"color": [255, 0, 0],       "name": "building"},
                        10: {"color": [102, 0, 0],       "name": "log"},
                        11: {"color": [204, 153, 255],   "name": "person"},
                        12: {"color": [102, 0, 204],     "name": "fence"},
                        13: {"color": [255, 153, 204],   "name": "bush"},
                        14: {"color": [170, 170, 170],   "name": "concrete"},
                        15: {"color": [41, 121, 255],    "name": "barrier"},
                        16: {"color": [134, 255, 239],   "name": "puddle"},
                        17: {"color": [99, 66, 34],      "name": "mud"},
                        18: {"color": [110, 22, 138],    "name": "rubble"}}
        self.id_name_dict = { 0  : "dirt", 1  : "grass", 2  : "tree",
                              3  : "pole", 4  : "water", 5  : "sky",
                              6  : "vehicle", 7  : "object", 8  : "asphalt",
                              9  : "building", 10 : "log", 11 : "person",
                              12 : "fence", 13 : "bush", 14 : "concrete",
                              15 : "barrier", 16 : "puddle", 17 : "mud", 18 : "rubble"}
        self.color_list = []
        while len(self.color_list) < 256:
            self.color_list.append(self.id_palette_dict.get(len(self.color_list), {"color": [0, 0, 0]})['color'])
        self.color_list = np.array(self.color_list, dtype=np.uint8)
    def RGB_01_show(self, tensor, return_array=False):
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
        if return_array:
            return t
        plt.imshow(t)
        plt.show()

    def gt_show(self, gt, return_array=False):
        # [H, W]
        if gt.dtype is not torch.uint8:
            gt = gt.to(torch.uint8)
        im_out = self.color_list[gt.detach().cpu()]
        if return_array:
            return im_out
        plt.imshow(im_out)
        plt.show()

    def visualize_palette(self, class_dict = None):
        """
        class_dict: 格式为 {id: {"color": [R, G, B], "name": "xxx"}, ...}
        """
        if class_dict == None:
            class_dict = self.id_palette_dict
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

    def visualize_cross_attn_map(self, attn_map, file_name=None, prompt=None):
        """
        Args:
            attn_map: attention map from attention pipeline
            file_name: if this is not None, store the result in
                        f'heat_map/{filename}.png', else return ndarray
        Returns: ndarray [HWC], you can use cv2 or mpl to visualize it
        """
        output_size = (256, 256)
        if attn_map.ndim == 3:
            scale = attn_map.shape[0]
            res_map = torch.mean(attn_map, dim=-1)# expand to image
            res_map = res_map * 255 / res_map.max()
            res_map = res_map.to(torch.uint8)
            res_map = res_map.detach().cpu().unsqueeze(-1).expand(scale, scale, 3).numpy()
            res_map = cv2.cvtColor(res_map, cv2.COLOR_RGB2BGR)
            res_map = cv2.applyColorMap(res_map, cv2.COLORMAP_JET)
            res_map = cv2.resize(res_map, output_size, interpolation=cv2.INTER_NEAREST)
            if prompt:
                font = cv2.FONT_HERSHEY_SIMPLEX
                # 参数：图像, 文字, 位置(x,y), 字体, 缩放, 颜色(B,G,R), 厚度
                cv2.putText(res_map, prompt, (5, 20), font, 0.6, (0, 255, 0), 2)
            if file_name is not None:
                cv2.imwrite(f"./heat_map/{file_name}.png", res_map)
            else:
                return res_map
        elif attn_map.ndim == 2:
            if torch.is_tensor(attn_map):
                res_map = attn_map.detach().cpu().numpy()
            res_map = (res_map - res_map.min()) / (res_map.max() - res_map.min() + 1e-8)
            res_map = (res_map * 255).astype(np.uint8)
            res_map = cv2.applyColorMap(res_map, cv2.COLORMAP_JET)
            res_map = cv2.resize(res_map, output_size, interpolation=cv2.INTER_NEAREST)
            if prompt:
                font = cv2.FONT_HERSHEY_SIMPLEX
                # 参数：图像, 文字, 位置(x,y), 字体, 缩放, 颜色(B,G,R), 厚度
                cv2.putText(res_map, prompt, (5, 20), font, 0.6, (0, 255, 0), 2)
            if file_name is not None:
                cv2.imwrite(f"./heat_map/{file_name}.png", res_map)
            else:
                return res_map

    def visualize_self_attn_map(self, attn_map, file_name=None):
        """
        Transform original attention map to visiable picture or save it
        Args:
            attn_map: attention map from attention pipeline
            file_name: if set this none, return the np array, else store in
                       f'heat_map/{filename}.png'

        Returns: ndarray. [HWC], 256 x 256, attention map
        """
        output_size = (256, 256)
        if attn_map.ndim == 3:
            if attn_map.shape[-1] != attn_map.shape[0] * attn_map.shape[1]:
                raise ValueError(f"expected attention area is a square! get {attn_map.shape}")
            if torch.is_tensor(attn_map):
                attn_map = attn_map.detach().cpu().numpy()
            h, w, c = attn_map.shape
            flat_map = attn_map.reshape(attn_map.shape[-1], attn_map.shape[-1]) # a big
            u, s, vh = np.linalg.svd(flat_map - np.mean(flat_map, axis=1, keepdims=True))
            res_map = u[:, 0].reshape(h, w)
            res_map = (res_map - res_map.min()) / (res_map.max() - res_map.min() + 1e-8)
            res_map = (res_map * 255).astype(np.uint8)
            res_map = np.stack([res_map] * 3, axis=-1)
            res_map = cv2.resize(res_map, output_size, interpolation=cv2.INTER_NEAREST)
            res_map = cv2.applyColorMap(res_map, cv2.COLORMAP_JET)
            if file_name is not None:
                cv2.imwrite(f"./heat_map/{file_name}.png", res_map)
            else:
                return res_map
        elif attn_map.ndim == 2:
            if torch.is_tensor(attn_map):
                res_map = attn_map.detach().cpu().numpy()
            res_map = (res_map - res_map.min()) / (res_map.max() - res_map.min() + 1e-8)
            res_map = (res_map * 255).astype(np.uint8)
            res_map = cv2.applyColorMap(res_map, cv2.COLORMAP_JET)
            res_map = cv2.resize(res_map, output_size, interpolation=cv2.INTER_NEAREST)
            if file_name is not None:
                cv2.imwrite(f"./heat_map/{file_name}.png", color_map)
            else:
                return res_map

if __name__ == "__main__":
    print("pass validation")
