"""
File: UAT.py
Author: Atman
Date: 1/27/26
Description:
    Ultimate Attention Terminal, track all attentions you need, your
    best champ.
"""
import os.path

import torch
import numpy as np
from diffusers.models.attention_processor import AttnProcessor2_0, IPAdapterAttnProcessor2_0
from torch.fx.experimental.sym_node import magic_methods_on_operator_with_trailing_underscore

from .LMAGProcessor import LMAGProcessor

class LMAGAttentionTerminal:
    def __init__(self, cfg):
        self.cond_maps_clean = self.clean_dict()
        self.cond_maps_adv = self.clean_dict()
        if not cfg.enable_cfg:
            raise ValueError("Make sure cfg is enabled")
        self.batch_size = 4
        self.target_resolution = cfg.img_height // cfg.target_downscale

    def save_attention(self, attn_prob, block, category):
        """We agree that prompt is [negative(uncond), cond], latent is [adversarial, clean]
            [uncond_adv, uncond_clean, cond_adv, cond_clean]
        """
        assert (block in ["down", "mid", "up"]), f"invalid block name {block}"
        assert category in ["self", "txt", "ip"]
        key = f"{block}_{category}"
        b_h, size, fea = attn_prob.shape
        if size != self.target_resolution ** 2:
            return
        # heads = b_h // self.batch_size # 8
        prob_uncond = attn_prob[0:b_h//2, ...]
        prob_cond = attn_prob[b_h//2:, ...]
        map_adv_cond = prob_cond[0:prob_cond.shape[0]//2, ...]
        map_clean_cond = prob_cond[prob_cond.shape[0]//2:, ...]

        res = int(np.sqrt(attn_prob.shape[1]))
        map_adv_cond = torch.mean(map_adv_cond, dim=0).reshape(res, res, -1)
        map_clean_cond = torch.mean(map_clean_cond, dim=0).reshape(res, res, -1)

        # The tomb for cross cfg. No them anymore...

        self.cond_maps_adv[key].append(map_adv_cond)
        self.cond_maps_clean[key].append(map_clean_cond)

    def reset(self):
        self.cond_maps_clean = self.clean_dict()
        self.cond_maps_adv = self.clean_dict()

    def replace_unet(self, unet):
        def recr_replace(func, name):
            if func.__class__.__name__ == "Attention":
                ins = LMAGProcessor(func.processor, name, self)
                func.processor = ins
                return
            if func.children() is not None:
                for _name, ch in func.named_children():
                    recr_replace(ch, f"{name}.{_name}")
        recr_replace(unet, "")

    def get_attn_map(self,
                     category="self",
                     which="adv",
                     stage=None): # No IP Now
        """category='self', 'txt', or 'ip'
            stage:None, down, up, mid, None for average"""
        if which == "adv":
            store = self.cond_maps_adv.items()
        else: # clean
            store = self.cond_maps_clean.items()
        if not stage:
            maps = []
            for name, val in store:
                if category in name:
                    for submap in val:
                        maps.append(submap)
            assert len(maps) > 0, "check target resolution"
            return torch.stack(maps).mean(dim=0)
        else:
            maps = []
            for name, val in store:
                if category in name and stage in name:
                    for submap in val:
                        maps.append(submap)
            assert len(maps) > 0, "check target resolution"
            return torch.stack(maps).mean(dim=0)


    @staticmethod
    def clean_dict():
        return {
            "down_self":[], "down_txt":[], "down_ip":[],
            "mid_self":[], "mid_txt":[], "mid_ip":[],
            "up_self":[], "up_txt":[], "up_ip":[]
        }


import cv2
def store_attention_map(dir_path,
                        attn,
                        category="self",
                        res=256,
                        mode="mean" # or discrete
                        ):
    if not os.path.isdir(dir_path):
        os.mkdir(dir_path)
    if category == "self":
        attn = attn.detach().cpu().numpy()
        attn = attn.reshape(attn.shape[-1], attn.shape[-1])
        u, s, vh = np.linalg.svd(attn - np.mean(attn, axis=1, keepdims=True))
        for i in range(5):
            image = vh[i].reshape(res, res)
            image = image - image.min()
            image = 255 * image / image.max()
            #image = np.repeat(np.expand_dims(image, axis=2), 3, axis=2).astype(np.uint8)
            #image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
            image = cv2.applyColorMap(image.astype(np.uint8), cv2.COLORMAP_JET)
            image = cv2.resize(image, (256, 256), interpolation=cv2.INTER_NEAREST)
            cv2.imwrite(os.path.join(dir_path, f"self{i}.png"), image)
    if category == "txt" or category=='ip':
        if mode == "mean":
            image = attn[:, :, 1:attn.shape[-1]].mean(dim=-1)
            image = image.detach().cpu().numpy()
            image = 255 * image / image.max()
            image = cv2.applyColorMap(image.astype(np.uint8), cv2.COLORMAP_JET)
            image = cv2.resize(image, (256, 256), interpolation=cv2.INTER_NEAREST)
            cv2.imwrite(os.path.join(dir_path, f"{category}mean.png"), image)
            image, _ = attn[:, :, 1:attn.shape[-1]].max(dim=-1)
            image = image.detach().cpu().numpy()
            image = 255 * image / image.max()
            image = cv2.applyColorMap(image.astype(np.uint8), cv2.COLORMAP_JET)
            image = cv2.resize(image, (256, 256), interpolation=cv2.INTER_NEAREST)
            cv2.imwrite(os.path.join(dir_path, f"{category}max.png"), image)
        if mode == "discrete":
            for i in range(attn.shape[-1]):
                image = attn[:, :, i]
                image = image - image.min()
                image = 255 * image / image.max()
                image = image.detach().cpu().numpy()
                image = cv2.applyColorMap(image.astype(np.uint8), cv2.COLORMAP_JET)
                image = cv2.resize(image, (256, 256), interpolation=cv2.INTER_NEAREST)
                cv2.imwrite(os.path.join(dir_path, f"{category}{i}.png"), image)

if __name__ == "__main__":
    img1 = cv2.imread("")
    img2 = cv2.imread("")

    print("pass validation")
