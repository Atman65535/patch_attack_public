from typing import List, Optional, Union

import torch
import torch.nn as nn
from torch import Tensor

from mmseg.structures import SegDataSample
from mmseg.registry import METRICS
from mmseg.models.losses import CrossEntropyLoss

class PatchMetrics():
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

    def get_loss(self,
                predict:torch.Tensor, 
                label:torch.Tensor):
        ce_loss = nn.CrossEntropyLoss(reduction='none',
                                      ignore_index=self.ignore_index)
        
        pass
    
    def _mean_ce_loss(self, logits=None, label=None):
        if logits == None or label == None:
            logits = self.logits
            label = self.labels
        ce_loss = nn.CrossEntropyLoss(weight=self.weight, 
                                      reduction='mean',
                                      ignore_index=self.ignore_index)
        loss = ce_loss(logits, label)


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
            
