"""
File: ddim_reverse.py
Author: Atman
Date: 12/14/25
Description:
    Implement of DDIM reverse process
"""
import torch
from diffusers import StableDiffusionPipeline
from .diffuison_utils import (vae_encoder, diffusion_image_checker,
                             build_unconditional_embeddings, build_conditional_embeddings,
                             cfg_predict_noise)

@torch.no_grad()
def ddim_reverse_no_grad(original_image: torch.Tensor,
                         prompt: str,
                         model: StableDiffusionPipeline,
                         batch_size=1,
                         num_inference_steps=50,
                         guidance_scale=4.5,
                         intermediate_steps=5,
                         resolution=256):
    """
    This function transfer an RGB BCHW square image to latent space, with ddim reverse.
    Args:
        batch_size: Batch size of image, maybe 1 for ordinary use, but 2 [clean, adv] for attack
        num_inference_steps: this name align to diffusion pipeline.
        guidance_scale: for CFG process, always 3-10
        intermediate_steps: steps for denoise and attention map calculation
        resolution:
    Returns: latent variable of several steps before. Noisy one.
    """
    if intermediate_steps > 0.15 * num_inference_steps:
        raise ValueError(f"ddim_reverse: too much intermediate steps! less than {0.15 * num_inference_steps}")
    diffusion_image_checker(original_image, resolution, strict=True)
    # embeddings
    uncond_embeddings = build_unconditional_embeddings(model, batch_size)
    cond_embeddings = build_conditional_embeddings(model, batch_size, prompt)
    # timestep
    model.scheduler.set_timesteps(num_inference_steps)
    start_latent = vae_encoder(original_image, model)
    # 50: tensor([1, 21, 41, 61, 81, 101,..., 981])
    timesteps = model.scheduler.timesteps.flip(0)
    # CFG process
    first_timestep = timesteps[0]
    noise_pred = cfg_predict_noise(model, start_latent, uncond_embeddings, cond_embeddings, first_timestep, guidance_scale)
    # reverse over intermediate steps
    jump_to_timestep = timesteps[intermediate_steps]  # eg index = 5, then 5 denoise steps
    alpha_bar_jump_to = model.scheduler.alphas_cumprod[jump_to_timestep]
    # reverse x_0
    reverse_x0 = (1 / torch.sqrt(model.scheduler.alphas_cumprod[first_timestep]) * (
            start_latent - noise_pred * torch.sqrt(1 - model.scheduler.alphas_cumprod[first_timestep]))
                  )
    jump_to = reverse_x0 * torch.sqrt(alpha_bar_jump_to) + torch.sqrt(1 - alpha_bar_jump_to) * noise_pred
    return jump_to

@torch.enable_grad()
def ddim_reverse(original_image: torch.Tensor,
                 prompt: str,
                 model: StableDiffusionPipeline,
                 batch_size=1,
                 num_inference_steps=50,
                 guidance_scale=4.5,
                 intermediate_steps=5,
                 resolution=256):
    """
    This function transfer an RGB BCHW square image to latent space, with ddim reverse.
    Args:
        batch_size: Batch size of image, maybe 1 for ordinary use, but 2 [clean, adv] for attack
        num_inference_steps: this name align to diffusion pipeline.
        guidance_scale: for CFG process, always 3-10
        intermediate_steps: steps for denoise and attention map calculation
        resolution:
    Returns: latent variable of several steps before. Noisy one.
    """
    if intermediate_steps > 0.15 * num_inference_steps:
        raise ValueError(f"ddim_reverse: too much intermediate steps! less than {0.15 * num_inference_steps}")
    diffusion_image_checker(original_image, resolution, strict=True)
    # embeddings
    uncond_embeddings = build_unconditional_embeddings(model, batch_size)
    cond_embeddings = build_conditional_embeddings(model, batch_size, prompt)
    # timestep
    model.scheduler.set_timesteps(num_inference_steps)
    start_latent = vae_encoder(original_image, model)
    # 50: tensor([1, 21, 41, 61, 81, 101,..., 981])
    timesteps = model.scheduler.timesteps.flip(0)
    # CFG process
    first_timestep = timesteps[0]
    noise_pred = cfg_predict_noise(model, start_latent, uncond_embeddings, cond_embeddings, first_timestep, guidance_scale)
    # reverse over intermediate steps
    jump_to_timestep = timesteps[intermediate_steps]  # eg index = 5, then 5 denoise steps
    alpha_bar_jump_to = model.scheduler.alphas_cumprod[jump_to_timestep]
    # reverse x_0
    reverse_x0 = (1 / torch.sqrt(model.scheduler.alphas_cumprod[first_timestep]) * (
            start_latent - noise_pred * torch.sqrt(1 - model.scheduler.alphas_cumprod[first_timestep]))
                  )
    jump_to = reverse_x0 * torch.sqrt(alpha_bar_jump_to) + torch.sqrt(1 - alpha_bar_jump_to) * noise_pred
    return jump_to


if __name__ == "__main__":
    from diffuison_utils import build_diffusion_model, _dummy_img
    res = 256

    model = build_diffusion_model()
    dummy_image = _dummy_img([1, 3, res, res])
    latent = ddim_reverse(dummy_image, "grass land with rocks", model, resolution=res)
    print(f"latent reversed shape = {latent.shape}")
    print("\033[32m DDIM reverse pass!\033[0m")
    print("pass validation")
