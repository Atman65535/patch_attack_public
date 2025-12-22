_base_ = [
    './bisenetv2_rellis1024x1024.py',
    #'./bisenetv1_rellis.py',
    #'./bisenetv2_rellis.py',
    #'./mask2former_swin-l_rellis.py',
    #'./pidnet-s_rellis.py'
    #'./pidnet-l_rellis.py',
    #'./segformer_mit-b5_rellis.py'
]
val_evaluator = dict(type='IoUMetric', iou_metrics=['mIoU'])

load_from = "/home/atman/a_workspace/mmlab/mmsegmentation/usr/configs/pretrained/bisenetv2.pth"

import os
cwd = os.getcwd()

patch_path = os.path.join(cwd, "usr/patch/patch.pat")

epochs = 30
lr = 0.001
ignore_label = 255
batch_size = 2
loss_back_iter = 10 # batchs

patch_size = 256
# crop_size is assigned

weight_config = dict(
    classifier = 1.0,
    self = 1.0,
    cross = 1.0,
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
    #out_size = (200, 200), # ? what is this
    patch_mode = "gray_scale", # "rgb" or "gray_scale"
    # EOT part
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
    classify_loss = dict(
        type = "class_loss",
        weight = -1.0,
    ),
    self_attention_loss = dict(
        weight = 1.0,
    ),
    cross_attention_loss = dict(
        weight = 1.0,
    ),
)
diffusion_config = dict(

    batch_size_of_diffusion=1, # only for image. But
    diffusion_resolution=patch_size,
    num_inference_steps=50,
    guidance_scale=4.5,
    intermediate_steps=5,

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

