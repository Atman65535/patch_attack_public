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

import os
cwd = os.getcwd()

patch_path = os.path.join(cwd, "usr/patch/patch.pat")

epochs = 30
lr = 0.001
ignore_label = 255
batch_size = 4

patch_size = 200
# crop_size is assigned 

train_dataloader = dict(batch_size=batch_size,
                         num_workers=4)

patch_config = dict(
    # Basic settings
    lr = lr,
    batch_size = batch_size,
    # Patch geometry 
    patch_path = patch_path,
    patch_size = patch_size,
    out_size = (200, 200),
    patch_mode = "gray_scale", # "rgb" or "gray_scale"
    # EOT part
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


