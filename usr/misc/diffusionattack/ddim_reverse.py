import torch

from diffusers import StableDiffusionPipeline, DDIMScheduler
from mmengine.config import Config, ConfigDict

from diffusion_utils import aggregate_attention, preprocess, encoder

#TODO finish model loading
@torch.enable_grad()
def ddim_reverse_to_attack_start_latent(image,
                                        cond_prompt,
                                        model,
                                        num_inference_steps=100,
                                        guidance_scale=2.5,
                                        res=256,
                                        transition_steps=5):
    """
    Args:
        image: preprocessed image, RGB, BCHW.
        cond_prompt: USER label prompt, not the uncond one.
        model: ldm_model
        num_inference_steps: the sample point for model, from 1 to 1000 uniformly
        guidance_scale: for Classifier Free Guidance.
        res: resolution of image
        transition_steps: between z_0 and z_attack, target of self attn

    Returns: the attack start latent, such as z_6, for 5 transition steps.
    """
    if type(image) != torch.Tensor:
        raise TypeError(f"Expected torch.Tensor, but get type(image)={type(image)}")
    #batch_size = image.shape[0]
    batch_size = 1 # need rethink
    max_length = model.tokenizer.model_max_length # shape of uncond_embedding [batch size, max_length, 1024]
    uncond_input_token = model.tokenizer(
        [""] * batch_size, padding='max_length', max_length=max_length, return_tensors='pt'
    ) # 'pt' means pytorch tensors.
    #uncond_embeddings = model.text_encoder(uncond_input_token.input_ids.to(model.device))[0]
    uncond_embeddings = model.text_encoder(uncond_input_token.input_ids.to(model.device)).last_hidden_state

    cond_input = model.tokenizer(
        cond_prompt, # prompt: ['soup_bowl', 'soup_bowl'], cond_prompt: 'soup_bowl'
        padding="max_length",
        max_length=max_length,
        truncation=True,
        return_tensors="pt",
    )
    cond_embeddings = model.text_encoder(cond_input.input_ids.to(model.device)).last_hidden_state

    context = torch.cat([uncond_embeddings, cond_embeddings])

    model.scheduler.set_timesteps(num_inference_steps)
    timesteps = model.scheduler.timesteps.flip(0) # dim = 0 # [1, 51, 101....951] , 20 steps.

    latent_z0 = encoder(image, model, res=res)

    #classifier free guidance
    latents_cfg = torch.cat([latent_z0, latent_z0])
    timestep = timesteps[0] # the very first one, tensor(1)
    attack_start_timestep = timesteps[0 + transition_steps + 1]# start = transition + 1
    noise_pred = model.unet(latents_cfg, timestep, encoder_hidden_states=context)["sample"]

    noise_pred_uncond, noise_pred_cond = noise_pred.chunk(2)
    noise_pred_cfg = noise_pred_uncond + guidance_scale * (noise_pred_cond - noise_pred_uncond)

    if attack_start_timestep > model.scheduler.config.num_train_timesteps:
        raise ValueError(f"Expected an val < T_max, got {attack_start_timestep}")
    alpha_bar_attack_start_timestep = model.scheduler.alphas_cumprod[attack_start_timestep]
    reverse_x0 = (1 / torch.sqrt(model.scheduler.alphas_cumprod[t]) * (
            latents - noise_pred * torch.sqrt(1 - model.scheduler.alphas_cumprod[t])))
    attack_start_latent = (reverse_x0 * torch.sqrt(alpha_bar_attack_start_timestep) +
                           torch.sqrt(1 - alpha_bar_attack_start_timestep) * noise_pred_cfg)
    return attack_start_latent


@torch.no_grad()
def ddim_reverse_get_transition_steps(image,
                                      cond_prompt,
                                      model,
                                      num_inference_steps=100,
                                      guidance_scale=2.5,
                                      res=256,
                                      transition_steps=5):
    """

    Args:
        image: preprocessed image, BCHW
        cond_prompt: USER prompt, no uncond one.
        model:
        num_inference_steps: T // num_inf_stp = gap
        guidance_scale: for classifier free guidance
        res: resolution
        transition_steps: between z_0 and z_attack, target of self attn

    Returns: a list of latents.
            such as [z_5, z_4, z_3, z_2, z_1] under the condition that transition stp = 5

    """
    if type(image) != torch.Tensor:
        raise TypeError(f"Expected torch.Tensor, but get type(image)={type(image)}")
    #batch_size = image.shape[0]
    batch_size = 1 # this need to rethink in later works.
    max_length = model.tokenizer.model_max_length
    uncond_input = model.tokenizer(
        [""] * batch_size, padding="max_length", max_length=max_length, return_tensors="pt"
    )
    uncond_embeddings = model.text_encoder(uncond_input.input_ids.to(model.device))[0]

    text_input = model.tokenizer(
        cond_prompt,
        padding="max_length",
        max_length=max_length,
        truncation=True,
        return_tensors="pt",
    )
    text_embeddings = model.text_encoder(text_input.input_ids.to(model.device))[0]

    context = [uncond_embeddings, text_embeddings]
    context = torch.cat(context)

    model.scheduler.set_timesteps(num_inference_steps)

    latents = encoder(image, model, res=res)
    timesteps = model.scheduler.timesteps.flip(0)

    sequential_latents = [latents] # z_0, z_1 ...

    #  Not inverse the last step, as the alpha_bar_next will be set to 0 which is not aligned to its real value (~0.003)
    #  and this will lead to a bad result.
    for t in tqdm(timesteps[:transition_steps], desc="DDIM_inverse"):
        latents_input = torch.cat([latents] * 2)
        noise_pred = model.unet(latents_input, t, encoder_hidden_states=context)["sample"]

        noise_pred_uncond, noise_prediction_text = noise_pred.chunk(2)
        noise_pred = noise_pred_uncond + guidance_scale * (noise_prediction_text - noise_pred_uncond)

        next_timestep = t + model.scheduler.config.num_train_timesteps // model.scheduler.num_inference_steps
        alpha_bar_next = model.scheduler.alphas_cumprod[next_timestep] \
            if next_timestep <= model.scheduler.config.num_train_timesteps else torch.tensor(0.0)

        "leverage reversed_x0"
        reverse_x0 = (1 / torch.sqrt(model.scheduler.alphas_cumprod[t]) * (
                latents - noise_pred * torch.sqrt(1 - model.scheduler.alphas_cumprod[t])))

        latents = reverse_x0 * torch.sqrt(alpha_bar_next) + torch.sqrt(1 - alpha_bar_next) * noise_pred

        sequential_latents.append(latents)

    inverted_sequence_latents = sequential_latents[:0:-1]
    return inverted_sequence_latents # [z_5, z_4, ...]

def init_latent(latent, model, height, width, batch_size):
    latents = latent.expand(batch_size, model.unet.in_channels, height // 8, width // 8).to(model.device)
    return latent, latents

def diffusion_step(model, latents, context, t, guidance_scale):
    latents_input = torch.cat([latents] * 2)
    noise_pred = model.unet(latents_input, t, encoder_hidden_states=context)["sample"]
    noise_pred_uncond, noise_prediction_text = noise_pred.chunk(2)
    noise_pred = noise_pred_uncond + guidance_scale * (noise_prediction_text - noise_pred_uncond)
    latents = model.scheduler.step(noise_pred, t, latents)["prev_sample"]
    return latents

if __name__ == "__main__":
    test_tensor = torch.rand((1, 3, 256, 256))

    ddim_reverse_get_transition_steps()
