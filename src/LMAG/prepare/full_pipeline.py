import torch
from diffusers import UNet2DConditionModel, AutoencoderKL, LMSDiscreteScheduler
from transformers import CLIPTextModel, CLIPTokenizer

# 1. 加载最基础的组件 (以 SD 1.5 为例)
model_id = "./runwayml/stable-diffusioin-v1-5"
tokenizer = CLIPTokenizer.from_pretrained(model_id, subfolder="tokenizer")
text_encoder = CLIPTextModel.from_pretrained(model_id, subfolder="text_encoder").cuda()
vae = AutoencoderKL.from_pretrained(model_id, subfolder="vae").cuda()
unet = UNet2DConditionModel.from_pretrained(model_id, subfolder="unet").cuda()
scheduler = LMSDiscreteScheduler.from_pretrained(model_id, subfolder="scheduler")

# 2. 准备最纯净的输入
prompt = ["a clean start"] * 2
height, width = 512, 512
guidance_scale = 7.5
generator = torch.manual_seed(42)

# 编码文本
text_input = tokenizer(prompt, padding="max_length", max_length=tokenizer.model_max_length, truncation=True, return_tensors="pt")
text_embeddings = text_encoder(text_input.input_ids.to("cuda"))[0]

# 准备初始噪声 (潜空间 64x64)
latents = torch.randn((1, unet.config.in_channels, height // 8, width // 8), generator=generator).to("cuda")
latents = latents * scheduler.init_noise_sigma

# 3. 纯净的手动去噪循环 (这里就是你的战场)
scheduler.set_timesteps(30)

for t in scheduler.timesteps:
    # 扩展 Latent 用于 CFG
    latent_model_input = torch.cat([latents] * 2)
    latent_model_input = scheduler.scale_model_input(latent_model_input, t)

    # 预测噪声 - 这一步会经过你的 AttentionProcessor
    with torch.no_grad():
        noise_pred = unet(latent_model_input, t, encoder_hidden_states=text_embeddings).sample

    # 执行 CFG
    noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
    noise_pred = noise_pred_uncond + guidance_scale * (noise_pred_text - noise_pred_uncond)

    # 计算上一步的 Latent (x_t -> x_{t-1})
    latents = scheduler.step(noise_pred, t, latents).prev_sample

# 4. 最后的解码
with torch.no_grad():
    image = vae.decode(1 / 0.18215 * latents).sample
    print(image)