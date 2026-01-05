"""
File: diffuison_utils.py
Author: Atman
Date: 12/14/25
Description:
    utilities for diffusion loss
Modules:
    1. build stable diffusion pipeline.
    2. transform image to latent via VAE encoder.

"""

import torch
from diffusers import StableDiffusionPipeline, DDIMScheduler
import warnings

# Part 1. Valid.

SD_LATENT_SCALE = 0.18215

def build_diffusion_model(model_path="Manojb/stable-diffusion-2-base"):
    """
    Build diffusion model, no grad and no optimizable parameters.
    Returns: model: StableDiffusionPipeline
    """
    if not torch.cuda.is_available():
        raise ValueError("ldm stable must run on cuda !")
    device = torch.device("cuda")
    ldm_stable = StableDiffusionPipeline.from_pretrained(model_path).to(device)
    ldm_stable.scheduler = DDIMScheduler.from_config(ldm_stable.scheduler.config)
    ldm_stable.vae.requires_grad_(False)
    ldm_stable.text_encoder.requires_grad_(False)
    ldm_stable.unet.requires_grad_(False)
    return ldm_stable

def vae_encoder(image: torch.Tensor, model):
    # vae encode image to latent[BCHW]
    generator = torch.Generator().manual_seed(1479)
    gpu_generator = torch.Generator(device=image.device)
    gpu_generator.manual_seed(generator.initial_seed())
    return SD_LATENT_SCALE * model.vae.encode(image).latent_dist.sample(generator=gpu_generator)

@torch.no_grad()
def diffusion_image_checker(image:torch.Tensor, resolution, strict=True):
    # check diffusion image type
    if image.device.type != "cuda":
        raise RuntimeError("vae_encoder: move image to cuda!")
    if image.ndim != 4:
        raise TypeError(f"diffusion_image_checker: image.ndim should = 4! but get {image.ndim}")
    B, C, H, W = image.shape
    if C != 3 or H != resolution or W != resolution:
        raise TypeError(f"diffusion_image_checker: expected BCHW, H and W = {resolution} but get {image.shape}")
    if strict:
        max = torch.max(image)
        min = torch.min(image)
        if max > 1.2 or min < -1.02:
            raise ValueError(f"diffusion_image_checker: expected img range [-1, 1], but get [{min}, {max}]")

def build_unconditional_embeddings(model, batch_size):
    token_max_length = model.tokenizer.model_max_length
    if token_max_length != 77:
        warnings.warn(f"expected token length is 77 but get {token_max_length}", RuntimeWarning)

    uncond_input = model.tokenizer(
        [""] * batch_size, padding="max_length", max_length=token_max_length, return_tensors="pt"
    )
    uncond_embeddings = model.text_encoder(uncond_input.input_ids.to(model.device))[0]
    return uncond_embeddings

def build_conditional_embeddings(model, batch_size, prompt:str):
    # here prompt is just a string. Don't multiply it
    prompt = [prompt] * batch_size
    text_input = model.tokenizer(
        prompt,
        padding="max_length",
        max_length=model.tokenizer.model_max_length,
        truncation=True,
        return_tensors="pt",
    )
    text_embeddings = model.text_encoder(text_input.input_ids.to(model.device))[0]
    return text_embeddings


# Part 2
def cfg_predict_noise(model, latent_input, uncond_embeddings, cond_embeddings, timestep, guidance_scale):
    """
    Use Classifier Free Guidance.
    Return: noise predicted.
    """
    context = torch.cat([uncond_embeddings, cond_embeddings])
    latent_cfg = torch.cat([latent_input] * 2)
    noise_uncond, noise_cond = model.unet(latent_cfg, timestep, encoder_hidden_states=context)["sample"].chunk(2)
    noise_pred = noise_uncond + guidance_scale * (noise_cond - noise_uncond)
    return noise_pred

def ddim_denoise(model, latent_input, uncond_embed, cond_embed, timestep, guidance_scale):
    noise = cfg_predict_noise(model, latent_input, uncond_embed, cond_embed, timestep, guidance_scale)
    latent_out = model.scheduler.step(noise, timestep, latent_input)["prev_sample"]
    return latent_out

#########################################
        # This part only for dbg
#########################################
def _dummy_img(shape):
    device = torch.device("cuda", 0)
    return torch.rand(shape, device=device).clamp(-1, 1)

def part1_dbg():
    batch_size = 2
    res = 256

    dummy_image = _dummy_img([2, 3, res, res])
    dummy_image = dummy_image.clamp(-1, 1)
    model = build_diffusion_model()
    diffusion_image_checker(dummy_image, res,True)
    print("model builder pass Check")

    latent = vae_encoder(dummy_image, model)
    print(f"vae encoder latent shape {latent.shape}")
    print("vae passed!")

    uncond = build_unconditional_embeddings(model, batch_size)
    print(uncond.shape)
    print("\nuncond building pass!\n")

    cond = build_conditional_embeddings(model, batch_size, "grass land with mud")
    print(f"conditional shape {cond.shape}")
    print("\ncond builder pass !\n")

    context = torch.cat([uncond, cond])
    print(f"cat shape is {context.shape} (uncond, cond)")

    print("pass validation")

def part2_dbg():
    res = 256
    batch_size = 2
    dummy_image = _dummy_img([2, 3, res, res])
    dummy_image = dummy_image.clamp(-1, 1)
    model = build_diffusion_model()
    model.scheduler.set_timesteps(1000)
    latent = vae_encoder(dummy_image, model)
    print(f"vae encoder latent shape {latent.shape}")
    print("vae passed!")

    uncond = build_unconditional_embeddings(model, batch_size)
    print(uncond.shape)
    print("\nuncond building pass!\n")

    cond = build_conditional_embeddings(model, batch_size, "grass land with mud")
    print(f"conditional shape {cond.shape}")
    print("\ncond builder pass !\n")

    ddim_denoise(model, latent, uncond, cond, 30, 4)
    print("ddim denoise pass")


if __name__ == "__main__":
    #part1_dbg()
    part2_dbg()