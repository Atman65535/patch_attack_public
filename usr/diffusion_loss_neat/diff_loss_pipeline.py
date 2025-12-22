"""
File: diff_loss_pipeline.py
Author: Atman
Date: 12/22/25
Description:
    
"""
import torch
from typing import List
import torch.nn.functional as F

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
        self.model = build_diffusion_model()

    def get_loss(self, clean, adv, gt):
        self.attn_catcher.reset_all()

        diffusion_image_checker(clean, resolution=self.resolution, strict=True)
        diffusion_image_checker(adv, resolution=self.resolution, strict=True)

        cond_prompt = self._get_prompt_from_gt(gt)
        latent_clean = ddim_reverse(clean,
                                    cond_prompt,
                                    self.model,
                                    batch_size=1,
                                    num_inference_steps=self.num_inference_steps,
                                    guidance_scale=self.guidance_scale,
                                    intermediate_steps=self.intermediate_steps,
                                    resolution=self.resolution)
        latent_adv = ddim_reverse_no_grad(adv,
                                          cond_prompt,
                                          self.model,
                                          batch_size=1,
                                          num_inference_steps=self.num_inference_steps,
                                          guidance_scale=self.guidance_scale,
                                          intermediate_steps=self.intermediate_steps,
                                          resolution=self.resolution)
        token_len = len(self.model.tokenizer.encode(cond_prompt))
        uncond_emb = build_unconditional_embeddings(self.model, self.batch_size * 2 )
        cond_emb = build_conditional_embeddings(self.model, self.batch_size * 2, cond_prompt)
        assert latent_adv.ndim == 4, "invalid latent dimension!"
        latent = torch.cat([latent_clean, latent_adv])
        register_attention_control(self.model, self.attn_catcher)
        for ind, time in enumerate(self.model.scheduler.timesteps[self.num_inference_steps - self.intermediate_steps:]):
            latent = ddim_denoise(self.model, latent, uncond_emb, cond_emb, time, self.guidance_scale)
        reset_attention_control(self.model)
        self_attn_loss = self.attn_catcher.self_attn_loss.loss
        cross_map = self.attn_catcher.extract_cross_attn_map(("up", "down"))[1:token_len - 1]
        self.attn_catcher.reset_all()
        cross_attn_loss = cross_map.var()
        return self_attn_loss, cross_attn_loss

    def _get_prompt_from_gt(self, gt):
        top1, top2 = _get_top2_labels(gt)
        if top2:
            return self.label_dict[top1] + "and" + self.label_dict[top2]
        return self.label_dict[top1]

    @staticmethod
    def image_preprocessor01(list_of_raw_imgs: List[torch.Tensor], W, H, pad_val=0.0):
        """
        input List[ (3, H, W), ...]
        Returns: Batch, (B, C, H, W)
        """
        out = []
        device = torch.device("cuda", 0)
        for img in list_of_raw_imgs:
            if img.ndim != 3:
                raise ValueError("expected input image ndim=3")
            img_trans = img[[2, 1, 0], ...]
            if img.shape[2] > W or img.shape[1] > H:
                raise ValueError("img.shape is larger than padding target!")
            pad_w = W - img.shape[2]
            pad_h = H - img.shape[1]
            img_trans = F.pad(img_trans, (0, pad_w, 0, pad_h), value=pad_val)
            img_trans = img_trans / 255.0
            out.append(img_trans)
        return torch.stack(out).to(device=device)


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
