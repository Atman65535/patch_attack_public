"""
File: LMAGScheduler.py
Author: Atman
Date: 1/28/26
Description:
    
"""
import os

import torch
from .LMAGProcessor import LMAGProcessor
from .LMAGAttentionTerminal import LMAGAttentionTerminal
from .LMAGEngine import LMAGEngine

class AtomScheduler:
    def __init__(self, threshold):
        self.cos = torch.nn.CosineSimilarity(dim=0)
        self.damping = torch.nn.functional.sigmoid
        self.threshold = threshold

    @torch.no_grad()
    def __call__(self, ref, manipulated):
        """Returns: weight for self attn, weight for IP-Adapter"""
        if ref.ndim != 1:
            ref = ref.view(-1)
        if manipulated.ndim != 1:
            manipulated = manipulated.view(-1)
        sim = self.cos(ref, manipulated)
        weight = 1 - self.damping(sim - self.threshold)
        if type(weight) is torch.Tensor:
            weight = weight.item()
        return 1, weight


class LMAGScheduler:
    def __init__(self, cfg):
        self.terminal = LMAGAttentionTerminal(cfg)
        self.terminal.reset()
        self.scheduler = AtomScheduler(cfg.confidence_threshold)
        self.engine = LMAGEngine(cfg)

        self.self_criterions_dict = {
            "CosineSimilarity": torch.nn.CosineSimilarity,}
        self.cross_criterions_dict = {
            "var": torch.var,
            "kl_div": torch.nn.functional.kl_div
        }

        if cfg.self_criterion in self.self_criterions_dict.keys():
            attr = self.self_criterions_dict[cfg.self_criterion]
            self.self_criterion = attr(**cfg.self_criterion_kwargs)
        if cfg.cross_criterion in self.cross_criterions_dict.keys():
            self.cross_criterion = self.cross_criterions_dict[cfg.cross_criterion]

        self.ip_stage = cfg.ip_stage
        self.txt_stage = cfg.txt_stage
        self.self_pca_layers = cfg.self_pca_layers
        self.txt_mode  = cfg.txt_mode
        self.ip_token_idx = cfg.ip_token_idx
        self.self_pca_start = cfg.self_pca_start
        self.debug = cfg.get("debug", False)
        self.enable_self_adapt_weight = cfg.enable_self_adapt_weight
        self.self_attn_weight = 1
        self.ip_cross_weight = cfg.ip_cross_weight
        self.txt_cross_weight = cfg.txt_cross_weight
        self.cross_uniformization = cfg.cross_uniformization

        self.terminal.replace_unet(self.engine.unet)

    def run(self, img_adv, img_clean, prompt, ip_adapter_image, **kwargs):
        self.engine(img_adv, img_clean, prompt, ip_adapter_image, **kwargs)
        with torch.no_grad():
            tokens = self.engine.tokenizer.encode(prompt)
            token_len = len(tokens) - 2
        self_loss, cross_txt_loss, cross_ip_loss, attn_bag = self.attn_loss(token_len)
        self.terminal.reset()
        return self_loss, cross_txt_loss, cross_ip_loss, attn_bag

    def self_pca(self, selfattn):
        """[resolution, resolution] mean"""
        shape = selfattn.shape[0]
        selfattn = selfattn.reshape(selfattn.shape[-1], selfattn.shape[-1])
        u, s, vh = torch.linalg.svd(selfattn - torch.mean(selfattn, dim=1, keepdim=True))
        end = self.self_pca_start + self.self_pca_layers
        attn = vh[self.self_pca_start:end, :].mean(dim=0, keepdim=False)
        attn = attn.reshape(shape, shape)
        attn = (attn - attn.min())
        attn = attn / attn.max()
        return attn

    def attn_loss(self, token_len):
        selfattn_cl = self.terminal.get_attn_map(category="self", which="clean")
        selfattn_adv = self.terminal.get_attn_map(category="self", which="adv")
        crossattn_ip = self.terminal.get_attn_map(category="ip", which="adv", stage=self.ip_stage)
        crossattn_txt = self.terminal.get_attn_map(category="txt", which="adv", stage=self.txt_stage)

        selfattn_adv = self.self_pca(selfattn_adv)
        selfattn_cl = self.self_pca(selfattn_cl)
        crossattn_txt = crossattn_txt[:, :, 1:1+token_len]
        crossattn_txt = (torch.max(crossattn_txt, dim=-1)[0]
                         if self.txt_mode == "max"
                         else torch.mean(crossattn_txt, dim=-1))
        crossattn_ip = crossattn_ip[:, :, self.ip_token_idx]

        if self.enable_self_adapt_weight:
            _, ip_weight = self.scheduler(selfattn_adv, crossattn_ip)
            _, txt_weight = self.scheduler(selfattn_adv, crossattn_txt)
        else:
            ip_weight = self.ip_cross_weight
            txt_weight = self.txt_cross_weight
        self_loss = self.self_attn_loss(selfattn_cl, selfattn_adv)
        cross_txt_loss = self.cross_attn_loss(crossattn_txt, weight=txt_weight)
        cross_ip_loss = self.cross_attn_loss(crossattn_ip, ip_weight)
        # bag for visualize only
        attn_bag = (selfattn_cl.detach(), selfattn_adv.detach(), crossattn_ip.detach(), crossattn_txt.detach())
        return self_loss, cross_txt_loss, cross_ip_loss, attn_bag

    def self_attn_loss(self, self_clean, self_adv, weight=1):
        """Restrict self structure
           Compatible for MSE and COS"""
        loss = self.self_criterion(self_clean.view(-1), self_adv.view(-1))
        if type(self.self_criterion) is torch.nn.CosineSimilarity:
            return -1 * loss
        else: return loss

    def cross_attn_loss(self, cross, weight=0):
        assert weight != 0, "send weight in"
        if self.cross_uniformization:
            cross = cross - cross.min().item()
            cross = cross / cross.max().item()
        # if self.debug:
        #     print(f"mean{cross.mean()}, std{cross.std()}, weight{weight}")
        return self.cross_criterion(cross) * weight / cross.numel()

def view_one(path, attn, i, txt=None, return_img=False):
    import cv2
    import numpy as np
    import os.path as osp
    if type(attn) is torch.Tensor:
        attn = attn.detach().cpu().numpy()
    attn = attn - attn.min()
    attn = attn * 255 / attn.max()
    image = cv2.applyColorMap(attn.astype(np.uint8), cv2.COLORMAP_JET)
    image = cv2.resize(image, (256, 256), interpolation=cv2.INTER_NEAREST)

    if txt:
        font = cv2.FONT_HERSHEY_SIMPLEX
        cv2.putText(image, txt, (5, 20), font, 0.6, (0, 255, 0), 2)

    if return_img: return image
    cv2.imwrite(os.path.join(path, f"self{i}.png"), image)
    return

def view_bag(bag, path):
    import os.path as osp
    import numpy as np
    if not osp.isdir(path):
        os.mkdir(path)
    for idx in range(len(bag)):
        view_one(path, bag[idx], idx)
    # font = cv2.FONT_HERSHEY_SIMPLEX
    # # 参数：图像, 文字, 位置(x,y), 字体, 缩放, 颜色(B,G,R), 厚度
    # cv2.putText(img_bgr, text, (5, 20), font, 0.6, (0, 255, 0), 2)

if __name__ == "__main__":
    print("pass validation")
