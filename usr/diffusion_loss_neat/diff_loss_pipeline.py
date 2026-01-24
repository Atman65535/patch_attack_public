"""
File: diff_loss_pipeline.py
Author: Atman
Date: 12/22/25
Description:
    
"""
import torch
from typing import List
import torch.nn.functional as F
from ..utils import Visualizer

from mmengine import ConfigDict

from .ddim_reverse import ddim_reverse, ddim_reverse_no_grad
from .UNet_patch import register_attention_control, reset_attention_control
from .attention_catcher import AttentionCatcher
from .diffuison_utils import (diffusion_image_checker, ddim_denoise, build_diffusion_model,
                              build_conditional_embeddings, build_unconditional_embeddings)


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


class DiffLossTools:
    """ 
    主要对接主程序的类，包括整个加噪去噪得到loss的流程
    """
    def __init__(self,
                 cfg: ConfigDict):
        if cfg.label_dict is None:
            raise ValueError("DiffLossTools: label dict must not none!")

        self.label_dict = cfg.label_dict
        self.attn_catcher = AttentionCatcher(batch_size=cfg.batch_size_of_diffusion * 2,
                                             resolution=cfg.diffusion_resolution,
                                             target_map_resolution=None,
                                             checked=False)
        self.attn_catcher.reset_all()
        self.resolution = cfg.diffusion_resolution
        self.batch_size = cfg.batch_size_of_diffusion
        self.num_inference_steps = cfg.num_inference_steps
        self.guidance_scale = cfg.guidance_scale
        self.intermediate_steps = cfg.intermediate_steps
        self.self_weight = cfg.self_weight
        self.cross_weight = cfg.cross_weight
        self.device = torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")
        self.model = build_diffusion_model()
        self.vis = Visualizer()

    def get_loss(self, clean, adv, gt):
        self.attn_catcher.reset_all()
        clean = clean.to(self.device)
        adv = adv.to(self.device)
        diffusion_image_checker(clean, resolution=self.resolution, strict=True)
        diffusion_image_checker(adv, resolution=self.resolution, strict=True)

        cond_prompt = self.get_prompt_from_gt(gt) # conditional prompt
        latent_clean = ddim_reverse(clean,
                                    cond_prompt,
                                    self.model,
                                    batch_size=1,
                                    num_inference_steps=self.num_inference_steps,
                                    guidance_scale=self.guidance_scale,
                                    intermediate_steps=self.intermediate_steps,
                                    resolution=self.resolution)
        latent_adv = ddim_reverse(adv,
                                cond_prompt,
                                self.model,
                                batch_size=1,
                                num_inference_steps=self.num_inference_steps,
                                guidance_scale=self.guidance_scale,
                                intermediate_steps=self.intermediate_steps,
                                resolution=self.resolution)
        token_len = len(self.model.tokenizer.encode(cond_prompt))
        # unconditional embeddings and conditional one.
        uncond_emb = build_unconditional_embeddings(self.model, self.batch_size * 2 )
        cond_emb = build_conditional_embeddings(self.model, self.batch_size * 2, cond_prompt)
        assert latent_adv.ndim == 4, "invalid latent dimension!"
        latent = torch.cat([latent_clean, latent_adv])
        # accumulate attention maps
        register_attention_control(self.model, self.attn_catcher)
        for ind, time in enumerate(self.model.scheduler.timesteps[self.num_inference_steps - self.intermediate_steps:]):
            latent = ddim_denoise(self.model, latent, uncond_emb, cond_emb, time, self.guidance_scale)
        reset_attention_control(self.model)

        self_attn_loss = self.attn_catcher.self_attn_loss.loss
        adv_cross_map = self.attn_catcher.extract_attn_map(("up", "down"), is_cross=True)[1, :, :, 1:token_len - 1]
        clean_self_map, adv_self_map = self.attn_catcher.extract_attn_map(stages=("up", "down"), is_cross=False)
        # self.vis.show_cross_attention_map(adv_cross_map, "cross")
        # self.vis.show_self_attention_map(adv_self_map, "advself")
        # self.vis.show_self_attention_map(clean_self_map,"cleanself")
        self.attn_catcher.reset_all()

        cross_attn_loss = adv_cross_map.var()
        return self.self_weight*self_attn_loss, self.cross_weight* cross_attn_loss, clean_self_map, adv_self_map, adv_cross_map

    def get_prompt_from_gt(self, gt):
        """
        从gt_map中获取prompt
        """
        top1, top2 = _get_top2_labels(gt)
        if top2:
            return self.label_dict[top1] + " and " + self.label_dict[top2]
        return self.label_dict[top1]

    def get_gt_in_patch(self, tensor, anchor):
        """
            Get ground truth under patched area.
        Args:
            tensor: gt clean [BHW]
            anchor: anchor from PatchHandler
        """
        diff_tensor = tensor.clone()
        return diff_tensor[:, anchor[0]:anchor[0]+self.resolution, anchor[1]:anchor[1]+self.resolution]

    def std_image_to_diffusion_format(self, tensor, anchor = None):
        """
        手动实现的预处理类，主要是希望预处理行为可控
        input : Standard image batch of dataset, range [0, 1], float32
        Returns: Batch, (B, C, H, W), range [-1, 1] for diffusion
        """
        if tensor.ndim != 4:
            raise TypeError(f"expected tensor BCHW, but get ndim={tensor.ndim}")
        diff_tensor = tensor.clone()
        diff_tensor = diff_tensor * 2.0 - 1.0
        if anchor is None:
            return diff_tensor
        return diff_tensor[:, :, anchor[0]:anchor[0]+self.resolution, anchor[1]:anchor[1]+self.resolution]

if __name__ == "__main__":
    #  = DiffLossTools()
    # lis = []
    # t = torch.rand(3, 250, 250)
    # lis.append(t)
    # lis.append(t)
    # #lis = tools.image_processor(lis, 256, 256, 0.0)
    # print(lis.shape)
    # print("pass validation")
    pass
