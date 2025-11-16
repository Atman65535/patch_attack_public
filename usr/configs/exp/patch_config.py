_base_ = [
    '../bisenetv2/bisenetv2_fcn_4xb4-amp-160k_rellis-1024x1024.py'
]


val_evaluator = dict(type='IoUMetric', iou_metrics=['mIoU'])

epochs = 1
lr = 0.001
ignore_label = 255

patch_config = dict(
    # Basic settings
    lr = lr,
    patch_size = 200,
    patch_path = "/media/atman/atman_2T/patch.jpg",
    # EOT part
    rot_deg = 20,
    translate = (0.2, 0.2), 
    scaling = (0.8, 1.2),
    location = "default", # TODO here we need more implement
    ignore_label = ignore_label
)
patch_metrics = dict(
    ignore_label = ignore_label,
    meta_info = dict(
        weight=[],
        classes = 19,
    ),
)


