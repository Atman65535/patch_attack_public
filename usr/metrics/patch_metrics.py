from typing import List, Optional, Union

import torch
import torch.nn as nn
from torch import Tensor
import torch.nn.functional as F

from mmengine.config import Config

from mmseg.structures import SegDataSample
from mmseg.registry import METRICS
from mmseg.models.losses import CrossEntropyLoss

import kornia.color as kc

class PatchMetrics():
    config: Config
    config = None
    def __init__(self, config):
        self.weight:Tensor
        self.ignore_index = config['ignore_label'] # 255 by default
        self.weight = torch.tensor(config['meta_info']['weight'],
                                   requires_grad=False)
        self.classes = config['meta_info']['classes']

    def update_prediction(self, model_output:SegDataSample, source:SegDataSample):
        self.logits = torch.stack([logits.seg_logits.data for logits in model_output], dim=0)
        self.prediction = torch.stack([pred.pred_sem_seg.data for pred in model_output], dim=0)
        labels = torch.stack([label.gt_sem_seg.data for label in source['data_samples']], dim=0)
        self.labels = labels.squeeze()

    def loss(self, logits:Tensor, gt_label:Tensor):
        ce_loss = self._mean_crossentropy_loss(logits,gt_label)
        entropy_loss = self._prediction_entropy_loss(logits)


    def get_lossmap(self,
                predict:torch.Tensor, 
                label:torch.Tensor):
        ce_loss = nn.CrossEntropyLoss(reduction='none',
                                      ignore_index=self.ignore_index)
        
        pass

    # this function return the entropy of prediction
    # uniform the distribution, pull down the confidence of model
    @staticmethod
    def _prediction_entropy_loss(logits: Tensor) -> float:
        probabilities = F.softmax(logits, dim=1).clamp_min(1e-12)
        entropy_map = -(probabilities * probabilities.log()).sum(dim=1)
        entropy_mean = entropy_map[self.valid].mean() if self.valid.any() else entropy_map.mean()

        return self.weight_entropy * entropy_mean
    
    @staticmethod
    def _mean_crossentropy_loss(logits=None, label=None):
        ce_loss = nn.CrossEntropyLoss(weight=self.weight, 
                                      reduction='mean',
                                      ignore_index=self.ignore_index)
        loss = ce_loss(logits, label)

    @staticmethod
    def _topk_mean_crossentropy_loss(logits, ratio):
        ce_map = nn.CrossEntropyLoss(weight=None,
                                     ignore_index=self.ignore_index,
                                     reduction="none")
        valid_values = ce_map[mask]
        assert valid_values.numel() > 0, f"too strict map makes valid values 0!"
        k = max(1, int(ratio * valid_values.numel()))
        return torch.topk(valid_values, k=k, largest=True).values.mean()
    
    # TODO finetune this method for lab euclidian distance
    @staticmethod
    def _LAB_regularzation_loss(patch, img_behind):
        patch_lab = kc.rgb_to_lab(patch)
        original_lab = kc.rgb_to_lab(img_behind)
        diff = original_lab - patch_lab
        loss = torch.norm(diff, p=2, dim=1).mean()
        return loss * config.lab_weight
    
    @staticmethod
    def _RGB_regularzation_loss(patch, img_behind):
        color_diff = patch - img_behind
        l2_loss = torch.norm(color_diff, p=2, dim=1).mean()
        luminance_patch = PatchMetrics._rgb2luminance(patch)
        luminance_img = PatchMetrics._rgb2luminance(img_behind)
        l1_loss = torch.abs(luminance_img - luminance_patch).mean()

        return config.l1_weight * l1_loss + config.l2_weight * l2_loss
        
    @staticmethod
    def _rgb2luminance(img: Tensor):
        R = img[:, 0]
        G = img[:, 1]
        B = img[:, 2]
        return 0.299 * R + 0.587 * G + 0.114 * B

    def get_pix_acc(self, model_output, label):
        pass
    
    def get_miou(self, predicts=None, labels=None):
        ious = self._class_iou()
        return ious.mean()

    def get_fw_iou(self, model_output, label):
        class_iou = self._class_iou()
        return (class_iou * self.weight).sum / self.weight.sum()
        pass

    def _class_iou(self, classes=19, predicts=None, labels=None, ignore_label=255):
        ious = []
        if predicts == None or labels == None:
            predicts = self.prediction
            labels = self.labels
        
        for idx in range(classes):
            if idx == ignore_label:
                continue
            pred = (predicts == idx)
            gt = (labels == idx)
            intersection = (pred & gt).sum()
            union = (pred | gt).sum() + 1e-12
            ious.append(intersection / union)
        
        return ious
            
