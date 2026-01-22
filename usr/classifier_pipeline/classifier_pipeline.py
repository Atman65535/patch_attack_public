"""
File: classifier_pipeline.py
Author: Atman
Date: 1/22/26
Description:
    
"""
import warnings
from typing import Optional

import torch
import torch.nn
from torch.utils.data import DataLoader
import torchvision.transforms as transforms

from mmengine.runner import Runner
from mmengine import Config
from mmseg.registry import  MODELS
from mmseg.structures import SegDataSample

from usr.utils import Visualizer
from usr.datasets import Rellis3DDataset

def invoke(*argc, **argv):
    print("call")

class Classifier:
    def __init__(self, cfg):
        """
        Args:
            cfg: this config file is the base config, not config.model
        """
        if not hasattr(cfg, "model"):
            raise ValueError("the config of Classifier should have model attribution")
        self.model_cfg          = cfg.model

        cfg = cfg.classifier_cfg
        self.ignore_label       = cfg.ignore_label
        self.patch_size         = cfg.patch_size
        self.outer_enhance      = cfg.outer_enhance
        self.patch_supress_weight = cfg.patch_supress_weight
        self.weight             = cfg.weight
        self.mean               = [0.485, 0.456, 0.406]
        self.std                = [0.229, 0.224, 0.225]

        self.device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
        self.normalize = transforms.Compose([transforms.Normalize(mean=self.mean, std=self.std)])

        self.model = MODELS.build(self.model_cfg).to(self.device)
        self.model.data_preprocessor = None
        self.model.eval()
        for param in self.model.parameters():
            param.requires_grad_(False)  # freeze
        # default
        self.ce_loss = torch.nn.CrossEntropyLoss(ignore_index=self.ignore_label, reduction="none")

    def inference(self, tensor, gt):
        """
        Quick inference with segmentor
        Args:
            tensor: data tensor, accept Float32, 0~1
            gt:     ground truth
        Returns: prediction and logits.
        """
        tensor = self.normalize(tensor)
        logits = self.model.predict(tensor.to(self.device))
        print(f"Logits range: {logits.min().item()} to {logits.max().item()}")
        pred = torch.argmax(logits, dim=1)
        print(torch.unique(pred))

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
        return -1  *  loss_map.mean() * self.weight

if __name__ == "__main__":
    import sys
    import os

    sys.path.append(os.path.dirname(os.path.dirname("/home/atman/a_workspace/mmlab/mmsegmentation")))
    from mmseg.utils import register_all_modules
    register_all_modules()
    cfg = Config.fromfile("/home/atman/a_workspace/mmlab/mmsegmentation/usr/configs/exp/patch_config.py")
    dataset = Rellis3DDataset()
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=4, num_workers=4)
    model = Classifier(cfg)
    vis = Visualizer()
    for _, data in enumerate(dataloader, 0):
        im = data[0]
        gt = data[1]
        pred, logit = model.inference(im, gt)
        vis.gt_show(pred[0])
        vis.gt_show(gt[0])
        vis.RGB_tensor_show((im[0]*255).to(torch.uint8))
        #loss = model.class_loss(logit, gt)

    # from mmseg.apis import init_model, inference_model,show_result_pyplot
    #
    # config_path = '/home/atman/a_workspace/mmlab/mmsegmentation/usr/configs/exp/bisenetv2_rellis1024x1024.py'
    # checkpoint_path = '/home/atman/a_workspace/mmlab/mmsegmentation/usr/configs/pretrained/bisenetv2.pth'
    # img_path = '/home/atman/a_workspace/mmlab/mmsegmentation/data/rellis3d/00000/pylon_camera_node/frame000000-1581624652_750.png'
    #
    #
    # model = init_model(config_path, checkpoint_path)
    # result = inference_model(model, img_path)
    # vis_image = show_result_pyplot(model, img_path, result)
    print("pass validation")
