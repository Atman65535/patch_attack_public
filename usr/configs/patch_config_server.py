"""
File: patch_config_local.py
Author: Atman
Date: 1/23/26
Description:

"""
#********************* USAGE *********************#
# When load a module, use cfg.module as its initializor
# eg: classifier_pipeline = Classifier(cfg.classifier_cfg)



# ************************************************#

import os
cwd = os.getcwd()
patch_path = os.path.join(cwd, "usr/patch/patch.pat") # 暂为了存储patch的pickle，但是还没写存储的功能

#************* TRAIN SETTINGS ****************#
epochs              = 30
lr                  = 0.007
ignore_label        = 255
batch_size          = 4
num_workers         = 4
patch_size          = 256               # 保证32的倍数，VAE和UNet降采样需求
crop_sizeHW         = (1024, 1024)        # Crop Size for Dataset, (Height, Width)

hyper_params = dict(
    # Loss Control
    classifier_loss_weight = 1,
    self_loss_weight = 30,
    cross_loss_weight = 1e5,
    # Classifier: outer enhancement
    outer_enhancement = True,
    outer_enhan_patch_supress_weight = 0.8,
    # ImageNet Normalize
    mean = [0.485, 0.456, 0.406],
    std  = [0.229, 0.224, 0.225],
    # patch
    patch_alpha = 0.5,
    # diffusion
    intermediate_steps = 1,
    RFES_edge = 16, # 0, 16, 32, or more.
)

paths = dict(
    patch_load_from = "",
    dataset_base_addr = "/home/atman/a_workspace/mmlab/mmsegmentation/data/rellis3d",
    model_pretrained = "/home/atman/a_workspace/mmlab/mmsegmentation/usr/configs/pretrained/UNet++_rellis_50e_512x512.pth"
)
dataset_cfg = dict(
    ignore_label=ignore_label,
    crop_sizeHW=crop_sizeHW,
    base_addr= paths['dataset_base_addr'],
    mode="train", # train, test, val
)
dataloader_cfg = dict(
    shuffle=False,
    batch_size=batch_size,
    num_workers=num_workers
)

classifier_cfg = dict(
    # model
    model="UnetPlusPlus",
    argv= dict(encoder_name     = "resnet34",
               encoder_weights  = "imagenet",     # 自动下载并加载ImageNet权重
               in_channels      = 3,
               classes          = 19
               ),
    load_from = paths['model_pretrained'],
    # basic settings
    ignore_label = ignore_label,
    patch_size = patch_size,
    outer_enhance = hyper_params['outer_enhancement'],
    patch_supress_weight = hyper_params['outer_enhan_patch_supress_weight'],
    loss_weight = hyper_params['classifier_loss_weight'],
    mean    = hyper_params['mean'], # ImageNet Pretrained, RGB
    std     = hyper_params['std'],
    # preprocessor
)

# This version only support RGB 3 ch Patch
# with FIXED transparency as hyperparameter
patch_handler_cfg = dict(
    # optim settings: self.optim_
    lr = lr,
    batch_size = batch_size,
    optim_name = "Adam",
    # Patch geometry
    load_from = paths['patch_load_from'],
    patch_size = patch_size,
    # 0.05 patch + (1-0,05) Img
    alpha = hyper_params['patch_alpha'],
    enable_eot = False,
    rot_deg = 20,
    scale = (0.8, 1.2),
    max_translate = 0.05,
    location = "center",
    ignore_label = ignore_label
)

diffusion_cfg = dict(

    batch_size_of_diffusion=1, # 每次输入单张图进入Pipeline
    diffusion_resolution=patch_size,
    num_inference_steps=50,
    guidance_scale=4.5,
    intermediate_steps=hyper_params['intermediate_steps'], # 中介步骤，从潜空间回来一共去噪几次
    self_weight = hyper_params['self_loss_weight'],
    cross_weight = hyper_params['cross_loss_weight'],
    RFES_edge = hyper_params['RFES_edge'],
    # 为提示词服务的字典。可以增加一些描述，不过也可以不增加
    label_dict = {
        0: "dirt",
        1: "green grass land",
        2: "tree",
        3: "pole",
        4: "water",
        5: "blue sky and shallow clouds on the top",
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


if __name__ == "__main__":
    print("pass validation")
