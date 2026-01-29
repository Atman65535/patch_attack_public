"""
File: LMAGPipeline.py
Author: Atman
Date: 1/29/26
Description:
    
"""
import cv2
import torch
@torch.no_grad()
def _get_top2_labels(gt_map: torch.Tensor,
                     ignore_label=255,
                     thres=0.5):
    """
    从gt_map中获得标签prompt
    Args:
        gt_map: gt_map of current image, which should be valid
        through all pixels.
        NO IGNORE LABEL! FOR VALID DIFFUSING
        ignore_label: you shouldn't use this
        thres: threshold for return 1 or 2 label

    Returns:the label of top1 or 2 type.
    usage : label_dict[top1] + "and" label_dict[top2]

    """
    labels = gt_map.reshape(-1)
    unique_labels, counts = labels.unique(return_counts=True)
    if len(unique_labels) > 1:
        top2 = torch.topk(counts, k=2).indices
        ind1 = top2[0]
        ind2 = top2[1]
        if counts[ind2] > counts[ind1] * thres:
            return unique_labels[ind1].item(), unique_labels[ind2].item()
        else:
            return unique_labels[ind1].item(), None
    else:
        # this part should put outside.
        # if unique_labels[0] == ignore_label:
        #
        #     return None, None
        # else:
        return unique_labels[0].item(), None

from .LMAGEngine import LMAGEngine
from .LMAGScheduler import LMAGScheduler

class LMAGPipeline:
    def __init__(self, cfg):
        """send in cfg.LMAG_cfg"""
        self.scheduler = LMAGScheduler(cfg)
        self.label_dict = cfg.label_dict
        self.rfes_edge = cfg.rfes_edge
        self.resolution = cfg.resolution # real size into diffusion
        self.cur_prompt = None
        self.cur_img_prompt = None

    def get_loss(self, clean, adv, gt):
        labels = self.get_prompt_from_gt(gt)
        assert labels is not None, "return None label, check your image or labels"
        prompt = labels['prompt']
        img = cv2.imread(labels['img_prompt'])
        self.cur_prompt = labels['prompt']
        self.cur_img_prompt = labels['img_prompt']
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        return self.scheduler.run(adv, clean, prompt, img)

    def get_prompt_from_gt(self, gt):
        top1, top2 = _get_top2_labels(gt)
        if self.label_dict[top1]['valid']:
            return self.label_dict[top1]
        elif self.label_dict[top2]['valid']:
            return self.label_dict[top2]
        else:
            return None

    def rfes_crop(self, tensor, anchor = None):
        """
        input : Standard image batch of dataset, range [0, 1], float32
        Returns: Batch, (B, C, H, W), range [0, 1] for L-MAG
        """
        if tensor.ndim != 4:
            raise TypeError(f"expected tensor BCHW, but get ndim={tensor.ndim}")
        _, _, h, w = tensor.shape
        diff_tensor = tensor.clone()
        if anchor is None:
            return diff_tensor

        if anchor[0] - self.rfes_edge  <= 0 or anchor[1] - self.rfes_edge <= 0: # left out
            h_start = anchor[0]
            w_start = anchor[1]
            h_end = h_start + self.resolution
            w_end = w_start + self.resolution
        elif anchor[0] + self.resolution + self.rfes_edge >= h or anchor[1] + self.resolution + self.rfes_edge >= w: # right out
            h_start = anchor[0] - 2 * self.rfes_edge
            w_start = anchor[1] - 2 * self.rfes_edge
            h_end = h_start + self.resolution
            w_end = w_start + self.resolution
        else:
            h_start = anchor[0] - self.rfes_edge
            w_start = anchor[1] - self.rfes_edge
            h_end = h_start + self.resolution
            w_end = w_start + self.resolution
        return diff_tensor[:, :, h_start:h_end, w_start:w_end]

    def get_gt_in_patch(self, tensor, anchor):
        """
            Get ground truth under patched area.
        Args:
            tensor: gt clean [BHW]
            anchor: anchor from PatchHandler
        """
        _, h, w = tensor.shape
        if anchor[0] - self.rfes_edge  <= 0 or anchor[1] - self.rfes_edge <= 0: # left out
            h_start = anchor[0]
            w_start = anchor[1]
            h_end = h_start + self.resolution
            w_end = w_start + self.resolution
        elif anchor[0] + self.resolution + self.rfes_edge >= h or anchor[1] + self.resolution + self.rfes_edge >= w: # right out
            h_start = anchor[0] - 2 * self.rfes_edge
            w_start = anchor[1] - 2 * self.rfes_edge
            h_end = h_start + self.resolution
            w_end = w_start + self.resolution
        else:
            h_start = anchor[0] - self.rfes_edge
            w_start = anchor[1] - self.rfes_edge
            h_end = h_start + self.resolution
            w_end = w_start + self.resolution
        diff_tensor = tensor.clone()
        return diff_tensor[:, h_start:h_end, w_start:w_end]
    
if __name__ == "__main__":
    print("pass validation")
