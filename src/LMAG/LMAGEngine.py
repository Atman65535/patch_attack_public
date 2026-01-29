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
        self.pipe.load_ip_adapter(
            cfg.IP_Adapter_path,
            subfolder="models",
            weight_name="ip-adapter_sd15.bin"
        )
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
        assert self.img_height is not None and self.img_height == self.img_width,\
            "Square diffusion size, and don't accept None input"
        assert self.guidance_rescale == 0, "we don't use diffusion for generative, don't change this arg"

    def __call__(self,
                 img_adv: torch.Tensor,
                 img_clean: torch.Tensor,
                 prompt: str = None,
                 ip_adapter_image: Optional[torch.tensor] = None,
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

        assert type(prompt) is str, "please input string type prompt"

        batch_size = 2
        prompt_embeds, negative_prompt_embeds = self.encode_prompt(
            [prompt] * batch_size,
            self.device,
            self.num_images_per_prompt,
            do_classifier_free_guidance=True,
            lora_scale=self.lora_scale,
            clip_skip=self.clip_skip
        )


        image_embeds = self.prepare_ip_adapter_image_embeds(
            ip_adapter_image,
            None,
            self.device,
            batch_size * self.num_images_per_prompt,
            True
        )
        added_cond_kwargs = {"image_embeds": image_embeds}


        latent_clean = self._encode_image(img_clean)
        latent_adv = self._encode_image(img_adv)

        latents = torch.cat([latent_adv, latent_clean])
        latent_model_input = torch.cat([latents] * 2)
        model_prompts = torch.cat([negative_prompt_embeds, prompt_embeds])

        latent_model_input = self.scheduler.scale_model_input(latent_model_input)
        self.unet(
            latent_model_input,
            self.timestep,
            encoder_hidden_states=model_prompts,
            timestep_cond=None,
            cross_attention_kwargs=None,
            added_cond_kwargs=added_cond_kwargs,
            return_dict=False
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

if __name__ == "__main__":
    import cv2
    cfg = OmegaConf.load("/src/configs/D4A_config_local.yaml")

    from LMAGScheduler import LMAGScheduler


    img_prompt = cv2.imread("/home/atman/a_workspace/patch_attack_public/data/science_1.jpg")
    img_prompt = cv2.cvtColor(img_prompt, cv2.COLOR_BGR2RGB)
    img = cv2.imread("/home/atman/a_workspace/patch_attack_public/data/human.jpg")
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = torch.tensor(img).permute(2, 0, 1).unsqueeze(0)
    # with torch.no_grad():
    #     runner = D4AEngine(cfg.D4A_cfg)
    #
    #     from UAT import UltimateAttnTerminal
    #     terminal = UltimateAttnTerminal(cfg.D4A_cfg)
    #     terminal.replace_unet(runner.unet)
    #
    #     runner(img, img.clone(), "green grass land and some mud", img_prompt)
    #     selfattn = terminal.get_attn_map(category="self")
    #     ipattn = terminal.get_attn_map(category="ip",stage="down")
    #     textattn=terminal.get_attn_map(category="txt")
    #     from UAT import store_attention_map
    #     store_attention_map("/home/atman/a_workspace/D4A/maps/rellistest/self", selfattn, "self", 16)
    #     store_attention_map("/home/atman/a_workspace/D4A/maps/rellistest/txt", textattn, "txt", 16)
    #     store_attention_map("/home/atman/a_workspace/D4A/maps/rellistest/ip", ipattn, "ip", 16)
    #     store_attention_map("/home/atman/a_workspace/D4A/maps/rellistest/ip", ipattn, "ip", 16, "discrete")
    #     print("pass validation")


    # seed = 42
    # torch.manual_seed(seed)
    # torch.cuda.manual_seed(seed)
    # torch.cuda.manual_seed_all(seed) # 如果你有多个 GPU
    #
    # # 3. 强制 CUDA 使用确定性算法（关键！）
    # # 注意：这可能会让你的运行速度稍微变慢一点
    # torch.backends.cudnn.deterministic = True
    # torch.backends.cudnn.benchmark = False

    # from UAS import UltimateAttnScheduler
    # with torch.no_grad():
    #
    #     scheduler = UltimateAttnScheduler(cfg.D4A_cfg)
    #     engine = D4AEngine(cfg.D4A_cfg)
    #     scheduler.terminal.replace_unet(engine.unet)
    #     engine(img, img.clone(), "grass", img_prompt)
    #     loss1, loss2, loss3, pack = scheduler.attn_loss()
    #     print(loss1, loss2, loss3)
    #     from UAS import view_bag
    #     view_bag(bag=pack, path="/home/atman/a_workspace/D4A/maps/test")
    #     print("pass validation")
    with torch.no_grad():
        scheduler = LMAGScheduler(cfg.D4A_cfg)
        loss1, loss2, loss3, pack = scheduler.run(img, img.clone(), "dirt", img_prompt)
        print(loss1, loss2, loss3)
        from .LMAGScheduler import view_bag
        view_bag(bag=pack, path="/home/atman/a_workspace/patch_attack_public/maps/test")
        print("pass validation")