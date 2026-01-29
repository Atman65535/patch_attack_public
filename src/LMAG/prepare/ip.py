"""
File: ip.py
Author: Atman
Date: 1/26/26
Description:
    
"""
import torch
import cv2
from diffusers import StableDiffusionPipeline, UNet2DConditionModel
from diffusers.models.attention_processor import IPAdapterAttnProcessor, AttnProcessor

# 1. 军需清单：确定模型 ID
model_id = "./models/runwayml/stable-diffusion-v1-5"
ip_adapter_ckpt = "./models/h94/IP-Adapter" # 权重所在仓库

# 2. 初始化 Pipeline (作为组件容器)
# 我们只用它来快速加载 VAE, Tokenizer 和 Text Encoder
pipe = StableDiffusionPipeline.from_pretrained(
    model_id,
    torch_dtype=torch.float16,
    safety_checker=None
).to("cuda")

# 3. 核心抓手：加载 IP-Adapter 权重
# 这一步会干两件事：
# a) 在 pipe 级别加载 Image Projection 层 (将 CLIP 特征映射到 UNet 空间)
# b) 给 UNet 的每个 Cross-Attention 层换上 IPAdapterAttnProcessor
pipe.load_ip_adapter(
    ip_adapter_ckpt,
    subfolder="models",
    weight_name="ip-adapter_sd15.bin"
)

# 4. 建立索引：现在我们来看看你的“领土”
unet = pipe.unet
image_proj = pipe.image_encoder # 如果加载了的话

print(f"--- 逻辑帝国资产清单 ---")
print(f"UNet 结构类型: {type(unet)}")
print(f"IP-Adapter 投影层类型: {type(pipe.image_encoder)}")

# 5. 关键调试：看看 Attention 处理器长什么样
# 这就是你明天要“动手术”的地方
print("\n--- 正在检查第一个 Cross-Attention 层 ---")
for name, proc in unet.attn_processors.items():
    if "attn2" in name: # attn2 通常是处理文本/图像引导的交叉注意力
        print(f"层名: {name}")
        print(f"处理器类型: {type(proc)}")
        # IPAdapterAttnProcessor 里面藏着 scale 和权重

    # 6. 验证运行 (简单的推理测试)
# 这一步能确保你的基础环境是鲁棒的
image = cv2.imread("./data/digger.jpg")
image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
prompt = "Kobe Bryant on excavator"
output = pipe(prompt=prompt, ip_adapter_image=image, num_inference_steps=20).images[0]
output.show()