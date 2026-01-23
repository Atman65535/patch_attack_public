"""
File: classifier_pipeline.py
Author: Atman
Date: 1/22/26
Description:
    
"""
import warnings
from typing import Optional

import cv2
import torch
import torch.nn
import torchvision.transforms as transforms

from mmengine import Config
import segmentation_models_pytorch as smp

def invoke(*argc, **argv):
    print("call")

class Classifier:
    def __init__(self, cfg):
        """
        Args:
            cfg: this config file is the modelpart of config
        """
        # model
        model_name = cfg.model
        init_params = cfg.argv
        load_from = cfg.load_from
        # basic settings
        self.ignore_label       = cfg.ignore_label
        self.patch_size         = cfg.patch_size
        self.outer_enhance      = cfg.outer_enhance
        self.patch_supress_weight = cfg.patch_supress_weight
        self.loss_weight        = cfg.loss_weight
        self.mean               = cfg.mean
        self.std                = cfg.std

        self.device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
        self.normalize = transforms.Compose([transforms.Normalize(mean=self.mean, std=self.std)])

        if getattr(smp, model_name) is None:
            raise AttributeError(f"No attribution smp.{model_name}")
        attr = getattr(smp, model_name)
        self.model = attr(**init_params)
        state_dict = torch.load(load_from)
        self.model.load_state_dict(state_dict)
        self.model.eval().to(self.device)
        for param in self.model.parameters():
            param.requires_grad_(False)  # freeze
        # default
        self.ce_loss = torch.nn.CrossEntropyLoss(ignore_index=self.ignore_label, reduction="none")

    def _preprocess(self, tensor):
        tensor = tensor.to(self.device)
        # tensor = self.normalize(tensor)
        return tensor

    def _postprocess(self, tensor):
        pass

    def inference(self, tensor):
        """
        Quick inference with segmentor
        Args:
            tensor: data tensor, accept Float32, 0~1
        Returns: prediction and logits.
        """
        tensor = self._preprocess(tensor)
        logits = self.model(tensor)
        pred = torch.argmax(logits, dim=1)
        self._postprocess(logits)
        return pred, logits

    def class_loss(self, seg_logits, gt, patch_anchor=None):
        # outer_enhance=False, patch_anchor:Optional[tuple[int, int]]=None
        """
        This classifier loss enhance the loss out of Patch (Optional)
        Args:
            outer_enhance: bool, if this is true, we will increase the weight of loss out of patch
                           else we only calc the loss on whole image.
            patch_anchor: (h_start, w_start), then we can ignore the covered area or not
                          you can find this anchor in PatchHandler Class
        Returns: the classifier loss value. NEGATIVE !!!!!!!!!!!!!!!!!!!!!!!
        """
        loss_map = self.ce_loss(seg_logits.to(self.device), gt.to(self.device))
        # TODO add this loss enhance method
        if (self.outer_enhance): # here we except the patched area
            if patch_anchor is None:
                raise   KeyError("When you want add outer enhance, send in patch anchor from PatchHandler")
            h_start = patch_anchor[0]
            w_start = patch_anchor[1]
            h_end = h_start + self.patch_size
            w_end = w_start + self.patch_size
            weight_mask = torch.ones_like(loss_map)
            weight_mask[:, h_start:h_end, w_start:w_end] = self.patch_supress_weight
            loss = loss_map * weight_mask

        # Negative ! Negative !
        return -1  *  loss_map.mean() * self.loss_weight

if __name__ == "__main__":
    from usr.utils import Visualizer
    vis = Visualizer()
    from usr.datasets.rellis_pytorch import Rellis3DDatasetTorch
    from torch.utils.data import DataLoader
    from mmengine import Config
    cfg = Config.fromfile("/home/atman/a_workspace/mmlab/mmsegmentation/usr/configs/patch_config.py")
    ds = Rellis3DDatasetTorch(crop_sizeHW=(512, 512))
    dl = DataLoader(ds, batch_size=1)
    classifier = Classifier(cfg.classifier_cfg)
    for img, gt in dl:
        pred, logits = classifier.inference(img)
        vis.gt_show(gt[0])
        vis.gt_show(pred[0])
        vis.RGB_01_show(img[0])
        print(f"loss = {classifier.class_loss(logits, gt)}")
        break;
    print("pass validation")
