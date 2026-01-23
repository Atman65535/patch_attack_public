"""
File: train_smp.py
Author: Atman
Date: 1/23/26
Description:
    
"""
import torch
import segmentation_models_pytorch as smp
from torch.utils.data import DataLoader
from usr.datasets.rellis_pytorch import Rellis3DDatasetTorch as Rellis3DDataset

# 1. 初始化模型
# 使用 Unet++ 结构，配合 ResNet34 (权衡了速度和性能)
model = smp.UnetPlusPlus(
    encoder_name="resnet34",
    encoder_weights="imagenet",     # 自动下载并加载ImageNet权重
    in_channels=3,
    classes=19,                     # Rellis3D 训练类别
)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

# 2. 定义损失函数和优化器
# DiceLoss + CrossEntropy 是分割任务的黄金搭档
criterion = smp.losses.DiceLoss(mode='multiclass')
optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

train_set = Rellis3DDataset(ignore_label=255, crop_sizeHW=(512, 512))
# 3. 准备数据 (假设你已经有了 train_ds)
train_loader = DataLoader(train_set, batch_size=8, shuffle=True, num_workers=4)

# 4. 极简训练循环
def train_one_epoch(model, loader, optimizer, criterion):
    model.train()
    total_loss = 0
    for images, masks in loader: # images: [B, 3, H, W], masks: [B, H, W]
        images = images.to(device)
        masks = masks.to(device).long()

        optimizer.zero_grad()
        output = model(images)  # 输出是 Raw Logits
        loss = criterion(output, masks)

        loss.backward()
        optimizer.step()

        total_loss += loss.item()
    return total_loss / len(loader)

# 5. 立即开始训练
print("Starting training...")
for epoch in range(50):
    loss = train_one_epoch(model, train_loader, optimizer, criterion)
    print(f"Epoch {epoch+1}, Loss: {loss:.4f}")

    # 每个Epoch存一次，防止断电
    torch.save(model.state_dict(), f"smp_rellis_epoch_{epoch+1}.pth")

