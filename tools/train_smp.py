"""
File: train_smp.py
Author: Atman
Date: 1/23/26
Description:
    
"""
import os
import torch
import segmentation_models_pytorch as smp
from torch.utils.data import DataLoader
from src.rellis_pytorch import Rellis3DDatasetTorch as Rellis3DDataset
import torchvision.transforms as transforms

os.chdir("/home/atman/a_workspace/D4A")

upp = smp.UnetPlusPlus(
    encoder_name="resnet34",
    encoder_weights="imagenet",
    in_channels=3,
    classes=19,
)

dpt = smp.DPT()
segformer = smp.Segformer()
smp.


dataset = Rellis3DDataset(ignore_label=255,
                          crop_sizeHW=(1024, 1024),
                          base_addr="./data/rellis3d",
                          mode="train")
dataloader = DataLoader(dataset, batch_size=4, shuffle=True, num_workers=2)



for model in model_list:

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    criterion = smp.losses.DiceLoss(mode='multiclass')
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

    train_set = Rellis3DDataset(ignore_label=255, crop_sizeHW=(512, 512))
    train_loader = DataLoader(train_set, batch_size=8, shuffle=True, num_workers=4)
    normalize = transforms.Compose([transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])





def train_one_epoch(model, loader, optimizer, criterion):
    model.train()
    total_loss = 0
    for images, masks in loader: # images: [B, 3, H, W], masks: [B, H, W]
        images = images.to(device)
        masks = masks.to(device).long()

        optimizer.zero_grad()
        images = normalize(images)
        output = model(images)  # 输出是 Raw Logits
        loss = criterion(output, masks)

        loss.backward()
        optimizer.step()

        total_loss += loss.item()
    return total_loss / len(loader)


print("Starting training...")
for epoch in range(50):
    loss = train_one_epoch(model, train_loader, optimizer, criterion)
    print(f"Epoch {epoch+1}, Loss: {loss:.4f}")

    torch.save(model.state_dict(), f"smp_rellis_epoch_{epoch+1}.pth")

