"""
File: LMAGEngine.py
Author: Atman
Date: 1/27/26
Description:
    
"""
import warnings
from typing import Union, Optional, List, Dict, Callable

import torch
import numpy as np
from diffusers.pipelines import StableDiffusionPipeline
from omegaconf import OmegaConf

class LMAGEngine:
    def __init__(self, cfg):
        if not torch.cuda.is_available():
            raise RuntimeError("Use cuda, rather than cpu")
        self.device = torch.device("cuda")

        self.pipe = StableDiffusionPipeline.from_pretrained(cfg.diffusion_model_path,
                                                            safety_checker=None)
        self.pipe.to(self.device)

        components = [self.pipe.unet, self.pipe.vae, self.pipe.text_encoder]
        for model in components:
            if model is not None:
                model.eval()
                model.requires_grad_(False)

        self.vae = self.pipe.vae
        self.encode_prompt = self.pipe.encode_prompt
        self.prepare_ip_adapter_image_embeds = self.pipe.prepare_ip_adapter_image_embeds
        self.unet = self.pipe.unet
        self.scheduler = self.pipe.scheduler
        self.tokenizer = self.pipe.tokenizer

        self.timestep = torch.tensor(cfg.timestep, device=self.device)
        self.img_height = cfg.img_height
        self.img_width = cfg.img_width

        self.vae_scale_factor =( 2 ** (len(self.vae.config.block_out_channels) - 1)
                                 if getattr(self, "vae", None)
                                 else 8)


        # check
        if self.img_width % 32 or self.img_height % 32:
            raise ValueError("img input size must be multiple of 32")
        assert self.img_height is not None and self.img_height == self.img_width,\
            "Square diffusion size, and don't accept None input"

    def __call__(self,
                 img_adv: torch.Tensor,
                 img_clean: torch.Tensor,
                 **kwargs
                 ):
        """Without DDIM Inversion.
        Args:
            img_adv: BCHW, Only accept 1, 3, H, W, H==W, [0, 1]
            img_clean: same as img_adv
            prompt: string, only string
            ip_adapter_image: torch.tensor, CHW, RGB
            **kwargs:
        """
        img_adv = img_adv.clone()
        img_clean = img_clean.clone()
        batch_size = 2
        prompt_embeds, negative_prompt_embeds = self.encode_prompt(
            [' '] * batch_size,
            self.device,
            1,
            do_classifier_free_guidance=True,
        )

        latent_clean = self._encode_image(img_clean)
        latent_adv = self._encode_image(img_adv)

        latents = torch.cat([latent_adv, latent_clean])
        latent_model_input = torch.cat([latents] * 2)
        model_prompts = torch.cat([negative_prompt_embeds, prompt_embeds])

        latent_model_input = self.scheduler.scale_model_input(latent_model_input)
        self.unet(
            latent_model_input,
            self.timestep,
            encoder_hidden_states=model_prompts
        )
        #noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)

    def _encode_image(self, img):
        img = img * 2.0 - 1.0
        img = img.to(device=self.vae.device, dtype=self.vae.dtype)
        latents = self.vae.encode(img).latent_dist.sample()
        return latents * self.vae.config.scaling_factor

    def _decode_latent(self, latent):
        latent = latent / self.vae.config.scaling_factor
        img = self.vae.decode(latent).sample
        return (img / 2.0 + 0.5).clamp(0.0, 1.0)
