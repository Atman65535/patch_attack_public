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
    def __init__(self):
        self.cos = torch.nn.CosineSimilarity(dim=0)
        self.damping = torch.nn.functional.sigmoid

class LMAGScheduler:
    def __init__(self, cfg):
        self.terminal = LMAGAttentionTerminal(cfg)
        self.terminal.reset()
        self.scheduler = AtomScheduler()
        self.engine = LMAGEngine(cfg)

        self.self_criterions_dict = {
            "CosineSimilarity": torch.nn.CosineSimilarity,
            "MSELoss": torch.nn.MSELoss
        }

        if cfg.self_criterion in self.self_criterions_dict.keys():
            attr = self.self_criterions_dict[cfg.self_criterion]
            self.self_criterion = attr(**cfg.self_criterion_kwargs)

        self.self_pca_layers = cfg.self_pca_layers
        self.txt_mode  = cfg.txt_mode
        self.self_pca_start = cfg.self_pca_start
        self.debug = cfg.get("debug", False)

        self.terminal.replace_unet(self.engine.unet)

    def run(self, img_adv, img_clean, **kwargs):
        self.engine(img_adv, img_clean, **kwargs)
        self_loss, attn_bag = self.attn_loss()
        self.terminal.reset()
        return self_loss, attn_bag

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

    def attn_loss(self):
        selfattn_cl = self.terminal.get_attn_map(category="self", which="clean")
        selfattn_adv = self.terminal.get_attn_map(category="self", which="adv")

        selfattn_adv = self.self_pca(selfattn_adv)
        selfattn_cl = self.self_pca(selfattn_cl)
        self_loss = self.self_attn_loss(selfattn_cl, selfattn_adv)
        # bag for visualize only
        attn_bag = (selfattn_cl.detach(), selfattn_adv.detach())
        return self_loss, attn_bag

    def self_attn_loss(self, self_clean, self_adv, weight=1):
        """Restrict self structure
           Compatible for MSE and COS"""
        loss = self.self_criterion(self_clean.view(-1), self_adv.view(-1))
        if type(self.self_criterion) is torch.nn.CosineSimilarity:
            return -1 * loss
        else: return loss

if __name__ == "__main__":
    print("pass validation")
