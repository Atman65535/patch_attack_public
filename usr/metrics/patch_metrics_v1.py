from typing import List, Optional, Union

import torch
import torch.nn as nn
from torch import Tensor
import torch.nn.functional as F

from mmengine.config import Config

from mmseg.structures import SegDataSample
from mmseg.registry import METRICS
from mmseg.models.losses import CrossEntropyLoss

class PatchMetrics():
    def __init__(self, cfg):
        """__init__ int function 

        Arguments:
            cfg {ConfigDict} -- supreme config
        """        
        self.cfg = cfg
        config = cfg.patch_metrics

        self.ignore_lable = config.ignore_label
        self.patch_size = config.patch_size
        self.meta_info = config.meta_info

        self.weight_classify_loss = config.classify_loss.weight
        self.self_attention_loss = config.self_attention_loss
        self.cross_attention_loss = config.cross_attention_loss

        self.ce_loss = nn.CrossEntropyLoss(ignore_index=self.ignore_lable)
        self.ce_loss_map = nn.CrossEntropyLoss(ignore_index=self.ignore_lable,
                                               reduction='none')

    def _get_crossentropy_map(self, logits, gt_label):
        """_get_crossentropy_map get crossentropy loss map

        Arguments:
            logits {tensor} -- logits [B, 19, H, W]
            gt_label {tensor} -- [B, H, W]
        """        
        ce_loss = self.ce_loss_map
        loss_map = ce_loss(logits, gt_label)
        return loss_map
    
    def classify_loss(self, logits, gt_label, location:tuple):
        """classify_loss get class loss after inference

        Arguments:
            logits {tensor} -- logits after inference [B, 19, H, W]
            gt_label {tensor} -- gt, [B, H, W]
            location {tuple} -- patch_anchor in patch handler

        Returns:
            float -- classify loss
        """        
        h_start, w_start = location
        h_end = h_start + self.patch_size
        w_end = w_start + self.patch_size

        loss = self.ce_loss(logits[:, :, h_start:h_end, w_start:w_end], 
                            gt_label[:, h_start:h_end, w_start:w_end])
        
        return loss * self.weight_classify_loss

    #TODO the fourth loss term
    def l1_regularization_loss(self):
        pass

    @staticmethod
    def l2_regularization_loss(img: torch.Tensor, width, height):
        area = width * height
        loss_map = torch.pow(img, 2)
        loss = loss_map.sum()
        loss = loss / area
        return loss

    @staticmethod
    def smooth_loss(img: torch.Tensor, width, height):
        """

        Args:
            img: the tensor for calculating loss,
                shape [C, H, W]

        Returns: value of this loss

        """
        area = width * height
        smooth_horizontal = torch.pow(img[:, 1:, :] - img[:, :-1, :], 2)
        smooth_vertical = torch.pow(img[:, :, 1:] - img[:, :, :-1], 2)
        loss = (torch.sum(smooth_vertical) + torch.sum(smooth_horizontal)) / area
        return loss
