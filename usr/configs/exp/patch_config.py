_base_ = [
    './bisenetv2_rellis1024x1024.py',
    #'./bisenetv1_rellis.py',
    #'./bisenetv2_rellis.py',
    #'./mask2former_swin-l_rellis.py',
    #'./pidnet-s_rellis.py'
    #'./pidnet-l_rellis.py',
    #'./segformer_mit-b5_rellis.py'
]
val_evaluator = dict(type='IoUMetric', iou_metrics=['mIoU']) # 这个暂时没用到

# 加载预训练全量模型，这个已经全训好了
load_from = "/home/atman/a_workspace/mmlab/mmsegmentation/usr/configs/pretrained/bisenetv2.pth"

import os
cwd = os.getcwd()
patch_path = os.path.join(cwd, "usr/patch/patch.pat") # 暂为了存储patch的pickle，但是还没写存储的功能

# 宏观patch train参数
epochs = 30
lr = 0.005 
ignore_label = 255
batch_size = 2
loss_back_iter = 10 # 这个是为梯度累积设计的，暂时没用上

patch_size = 256 # 保证32的倍数，VAE和UNet降采样需求

# loss权重，后面还有一个写到loss的地方，因为代码写冲突了，所有请这两个权重保持一致
weight_config = dict(
    classifier = 1.0,
    self = 2.0,
    cross = 150.0,
)

train_dataloader = dict(batch_size=batch_size,
                         num_workers=4)

patch_config = dict(
    # Basic settings
    lr = lr,
    batch_size = batch_size,
    # Patch geometry 
    patch_path = patch_path,
    patch_size = patch_size,
    # 暂时只支持灰度patch
    patch_mode = "gray_scale", # "rgb" or "gray_scale"
    # EOT part 暂时没用上EOT
    enable_eot = False,
    rot_deg = 20,
    scale = (0.8, 1.2),
    max_translate = 0.05,
    location = "default", # TODO here we need more implement
    ignore_label = ignore_label
)
patch_metrics = dict(
    ignore_label = ignore_label,
    patch_size = patch_size,
    meta_info = dict(
        weight=[],
        classes = 19,
    ),
    #这块的loss请不要改，除非对整个patch metric类做修改
    classify_loss = dict(
        type = "class_loss",
        weight = -weight_config["classifier"], 
    ),
    self_attention_loss = dict(
        weight = weight_config["self"],
    ),
    cross_attention_loss = dict(
        weight = weight_config["cross"],
    ),
)


diffusion_config = dict(

    batch_size_of_diffusion=1, # 每次输入单张图进入Pipeline
    diffusion_resolution=patch_size,
    num_inference_steps=50,
    guidance_scale=4.5,
    intermediate_steps=5, # 中介步骤，从潜空间回来一共去噪几次
    # 为提示词服务的字典。可以增加一些描述，不过也可以不增加
    label_dict = {
        0: "dirt",
        1: "grass",
        2: "tree",
        3: "pole",
        4: "water",
        5: "sky",
        6: "vehicle",
        7: "object",
        8: "asphalt",
        9: "building",
        10: "log",
        11: "person",
        12: "fence",
        13: "bush",
        14: "concrete",
        15: "barrier",
        16: "puddle",
        17: "mud",
        18: "rubble",
    }
)

