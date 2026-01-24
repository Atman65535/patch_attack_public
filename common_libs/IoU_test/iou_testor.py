"""
File: iou_testor.py
Author: Atman
Date: 1/24/26
Description:
    
"""
import segmentation_models_pytorch

# Target: send IoU out
import segmentation_models_pytorch as smp
import torch
from torch.utils.data import DataLoader
from src.datasets.rellis_pytorch import Rellis3DDatasetTorch
import torchvision.transforms as transforms
from torchvision.transforms import Normalize
from torchmetrics.classification import MulticlassJaccardIndex
from tqdm import tqdm


model_list = []
ckpt_list = []

unet_plus_plus = smp.UnetPlusPlus(encoder_name="resnet34",
                                  encoder_depth=5,
                                  encoder_weights="imagenet",
                                  in_channels=3,
                                  classes=19)
ckpt_upp = "/home/atman/a_workspace/mmlab/mmsegmentation/src/configs/pretrained/UNet++_rellis_50e_512x512.pth"
model_list.append(unet_plus_plus)
ckpt_list.append(ckpt_upp)

dataset = Rellis3DDatasetTorch(crop_sizeHW=(1024, 1024), mode="test")
norm = transforms.Compose([Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])
metric = MulticlassJaccardIndex(num_classes=19, average = "weighted", ignore_index=255).to("cuda:0")

for model, ckpt in zip(model_list, ckpt_list):
    state_dict = torch.load(ckpt, map_location="cpu")
    model.load_state_dict(state_dict)
    model.eval().to(torch.device("cuda:0"))
    ds = Rellis3DDatasetTorch(crop_sizeHW=(1024, 1024))
    dl = DataLoader(ds, num_workers=1, batch_size=1)
    mIoU = 0
    cnt = 0
    for sample, gt in tqdm(dl, desc=f"testing model {model.__class__.__name__}"):
        res = model(norm(sample).to(torch.device("cuda:0")))
        mIoU += metric(res, gt.to("cuda:0"))
        cnt += 1

    print(f"{model.__class__.__name__}: mIou : \033[32m{mIoU/cnt:.4f}\033[0m")