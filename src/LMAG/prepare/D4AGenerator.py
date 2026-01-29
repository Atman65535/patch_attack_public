"""
File: D4AGenerator.py
Author: Atman
Date: 1/27/26
Description:
    
"""
from transformers.models.clip.modeling_clip import clip_loss

"""
File: D4AEngine.py
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

from UAT import UltimateAttnProcessor

class D4AGenerator:
    def __init__(self, cfg):
        if not torch.cuda.is_available():
            raise RuntimeError("Use cuda, rather than cpu")
        self.device = torch.device("cuda")

        self.pipe = StableDiffusionPipeline.from_pretrained(cfg.diffusion_model_path,
                                                            safety_checker=None)
        self.pipe.load_ip_adapter(
            cfg.IP_Adapter_path,
            subfolder="models",
            weight_name="ip-adapter_sd15.bin"
        )
        self.pipe.to(self.device)

        self.vae = self.pipe.vae
        self.encode_prompt = self.pipe.encode_prompt
        self.prepare_ip_adapter_image_embeds = self.pipe.prepare_ip_adapter_image_embeds
        self.unet = self.pipe.unet
        self.scheduler = self.pipe.scheduler

        self.timestep = torch.tensor(cfg.timestep, device=self.device)
        self.img_height = cfg.img_height
        self.img_width = cfg.img_width
        self.sigmas = cfg.sigmas
        self.num_images_per_prompt = 1
        self.eta = cfg.eta
        self.guidance_rescale = cfg.guidance_rescale
        self.lora_scale = cfg.lora_scale
        self.clip_skip = cfg.clip_skip

        self.vae_scale_factor =( 2 ** (len(self.vae.config.block_out_channels) - 1)
                                 if getattr(self, "vae", None)
                                 else 8)
        # check
        if self.img_width % 32 or self.img_height % 32:
            raise ValueError("img input size must be multiple of 32")
        assert self.img_height is not None and self.img_height == self.img_width, \
            "Square diffusion size, and don't accept None input"
        assert self.guidance_rescale == 0, "we don't use diffusion for generative, don't change this arg"

    def __call__(self,
                 img_adv: torch.Tensor,
                 img_clean: torch.Tensor,
                 prompt: str = None,
                 ip_adapter_image: Optional[torch.tensor] = None,
                 **kwargs
                 ):
        """

        Args:
            img_adv: BCHW, Only accept 1, 3, H, W, H==W, [0, 1]
            img_clean: same as img_adv
            prompt: string, only string
            ip_adapter_image: torch.tensor, CHW, RGB
            **kwargs:

        Returns:

        """

        img_adv = img_adv.clone()
        img_clean = img_clean.clone()

        assert type(prompt) is str, "please input string type prompt"

        # prompt_embeds, negative_prompt_embeds = self.encode_prompt(
        #     prompt,
        #     self.device,
        #     self.num_images_per_prompt,
        #     do_classifier_free_guidance=False,
        #     lora_scale=self.lora_scale,
        #     clip_skip=self.clip_skip
        # )
        prompt_embeds, negative_prompt_embeds = self.encode_prompt(
            [prompt] * 2,
            self.device,
            self.num_images_per_prompt,
            do_classifier_free_guidance=True,
            lora_scale=self.lora_scale,
            clip_skip=self.clip_skip
        )
        model_prompts = torch.cat([negative_prompt_embeds, prompt_embeds])

        image_embeds = self.prepare_ip_adapter_image_embeds(
            ip_adapter_image,
            None,
            self.device,
            2*self.num_images_per_prompt,
            True
        )

        # latent_clean = self._encode_image(img_clean)
        # latent_adv = self._encode_image(img_adv)
        # latent_model_input = torch.cat([latent_clean, latent_adv])
        # prompt_embeds_input = torch.cat([prompt_embeds] * 2)
        #image_embeds = torch.cat([image_embeds] * 2)
        added_cond_kwargs = {"image_embeds": image_embeds}

        self.scheduler.set_timesteps(1)

        generator = torch.manual_seed(42)
        latents = torch.randn((2, self.unet.config.in_channels, self.img_height // 8, self.img_width // 8), generator=generator).to("cuda")


        for t in self.scheduler.timesteps:
            # 扩展 Latent 用于 CFG
            latent_model_input = torch.cat([latents] * 2)
            latent_model_input = self.scheduler.scale_model_input(latent_model_input, t)

            # 预测噪声 - 这一步会经过你的 AttentionProcessor
            with torch.no_grad():
                noise_pred = self.unet(latent_model_input,
                                       t,
                                       encoder_hidden_states=model_prompts,
                                       timestep_cond=None,
                                       cross_attention_kwargs=None,
                                       added_cond_kwargs=added_cond_kwargs,
                                       ).sample

                # 执行 CFG
                noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
                noise_pred = noise_pred_uncond + 7.4 * (noise_pred_text - noise_pred_uncond)

                # 计算上一步的 Latent (x_t -> x_{t-1})
                latents = self.scheduler.step(noise_pred, t, latents).prev_sample

        # 4. 最后的解码
        with torch.no_grad():
            image = self.vae.decode(1 / 0.18215 * latents).sample[1]

        img = (image + 1) /2
        img = img.clamp(0, 1).permute(1, 2, 0)
        import matplotlib.pyplot as plt

        plt.imshow(img.cpu().numpy())
        plt.axis("off")
        plt.show()
                    # self.unet(
                    #     latent_model_input,
                    #     self.timestep,
                    #     encoder_hidden_states=prompt_embeds_input,
                    #     timestep_cond=None,
                    #     cross_attention_kwargs=None,
                    #     added_cond_kwargs=added_cond_kwargs,
                    #     return_dict=False



    def _encode_image(self, img):
        img = img * 2 - 1.0
        img = img.to(device=self.vae.device, dtype=self.vae.dtype)
        latents = self.vae.encode(img).latent_dist.sample()
        return latents * self.vae.config.scaling_factor

    def _decode_latent(self, latent):
        latent = latent / self.vae.config.scaling_factor
        img = self.vae.decode(latent).sample
        return (img / 2 + 0.5).clamp(0, 1)

if __name__ == "__main__":
    import cv2
    cfg = OmegaConf.load("D4AConfig.yaml")
    runner = D4AGenerator(cfg)
    img_prompt = cv2.imread("/home/atman/a_workspace/patch_attack_public/data/science_1.jpg")
    img_prompt = cv2.cvtColor(img_prompt, cv2.COLOR_BGR2RGB)
    from UAT import UltimateAttnTerminal
    terminal = UltimateAttnTerminal(cfg)
    terminal.replace_unet(runner.unet)
    img = cv2.imread("/home/atman/a_workspace/patch_attack_public/data/kobe256.jpg")
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = torch.tensor(img).permute(2, 0, 1).unsqueeze(0)
    runner(img, img.clone(), "Kobe, basketball player", img_prompt)
    selfattn = terminal.get_attn_map(category="self")
    ipattn = terminal.get_attn_map(category="ip")
    textattn=terminal.get_attn_map(category="txt")
    from UAT import store_attention_map
    store_attention_map("/home/atman/a_workspace/D4A/maps/KobeWithKobe\'sprompt/self", selfattn, "self", 16)
    store_attention_map("/home/atman/a_workspace/D4A/maps/KobeWithKobe\'sprompt/txt", textattn, "txt", 16)
    store_attention_map("/home/atman/a_workspace/D4A/maps/KobeWithKobe\'sprompt/ip", ipattn, "ip", 16)
    print("pass validation")
    print("pass validation")

