"""
File: test_smp.py
Author: Atman
Date: 1/23/26
Description:
    
"""
import segmentation_models_pytorch as smp
import torch
from torch.utils.data import DataLoader
from usr.utils import Visualizer
from usr.datasets.rellis_pytorch import Rellis3DDatasetTorch as Rellis3DDataset

def main():
    model = smp.UnetPlusPlus(
        encoder_name="resnet34",
        encoder_weights="imagenet",     # 自动下载并加载ImageNet权重
        in_channels=3,
        classes=19,                     # Rellis3D 训练类别
    )
    ckpt = "./smp_rellis_epoch_50.pth"
    state_dict = torch.load(ckpt, map_location="cpu")
    model.load_state_dict(state_dict)
    model.eval().to(torch.device("cuda:0"))

    ds = Rellis3DDataset(crop_sizeHW=(512,512))
    dl = DataLoader(ds, num_workers=4, batch_size=1)
    for idx, sample in enumerate(dl):
        res = model(sample)
    print("pass")

if __name__ == "__main__":
    main()
    print("pass validation")
