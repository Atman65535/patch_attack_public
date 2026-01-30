import torch
from diffusers import StableDiffusionPipeline
import PIL

pipe = StableDiffusionPipeline.from_pretrained(
    "stable-diffusion-v1-5/stable-diffusion-v1-5", dtype=torch.float16
)
pipe = pipe.to("cuda")

prompt ="Kobe Bryant"
image = pipe(prompt).images[0]
image.show()