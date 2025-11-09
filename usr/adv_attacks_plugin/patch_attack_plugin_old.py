"""
mmseg plugin: patch_attack_plugin.py

Usage:
  1) copy this file into your mmseg repo, e.g. mmseg/plugins/patch_attack_plugin.py
  2) in your training entrypoint (before launching train), import the plugin to register the hook:
       import mmseg.plugins.patch_attack_plugin  # path according to where you placed the file
  3) add a custom_hooks item into your mmseg config (example below).

This plugin implements an end-to-end learnable adversarial patch with EOT and patch loss
for semantic segmentation models in mmseg.
"""
import os
from typing import Optional, Tuple, Dict, Any

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms.functional as TF
# from mmcv.runner import HOOKS, Hook    老版本
from mmengine.hooks import Hook
from mmengine.registry import HOOKS

from PIL import Image
import numpy as np

print('[PatchAttackHook] loaded from:', __file__)

def total_variation(x: torch.Tensor) -> torch.Tensor:
    """Total variation for patch smoothness. x: [C,H,W] or [B,C,H,W]."""
    if x.dim() == 3:
        x = x.unsqueeze(0)
    b = x.size(0)
    tv_h = torch.abs(x[:, :, 1:, :] - x[:, :, :-1, :]).sum()
    tv_w = torch.abs(x[:, :, :, 1:] - x[:, :, :, :-1]).sum()
    return (tv_h + tv_w) / b

import math

def make_affine_matrix(angle_deg: float, scale: float, tx: float, ty: float, device='cpu'):
    theta = torch.zeros(2, 3, device=device, dtype=torch.float32)
    rad = torch.tensor(angle_deg, device=device, dtype=theta.dtype) * (math.pi / 180.0)
    c = torch.cos(rad) * scale
    s = torch.sin(rad) * scale
    theta[0, 0] = c
    theta[0, 1] = -s
    theta[0, 2] = torch.tensor(tx, device=device, dtype=theta.dtype)
    theta[1, 0] = s
    theta[1, 1] = c
    theta[1, 2] = torch.tensor(ty, device=device, dtype=theta.dtype)
    return theta

def random_eot_transform_batch(patch: torch.Tensor, batch_size: int, out_size: Tuple[int, int],
                               rot_degree: float = 20.0, scale_min: float = 0.8, scale_max: float = 1.2,
                               max_translate: float = 0.2, device: Optional[str] = None) -> torch.Tensor:
    if device is None:
        device = patch.device
    # --- add: 统一成 [C,p,p] 并放到 device ---
    if patch.dim() == 2:
        patch = patch.unsqueeze(0)  # [1,p,p]
    patch = patch.to(device)
    C, p, _ = patch.shape
    # create a patch batch [B, C, p, p]
    patch_batch = patch.unsqueeze(0).repeat(batch_size, 1, 1, 1).to(device)

    H_out, W_out = out_size
    # We'll transform patch into its own p x p grid first, then we will place onto canvas by sampling into H_out x W_out
    transformed = []
    for i in range(batch_size):
        angle = (torch.rand(1).item() * 2 - 1) * rot_degree
        scale = float(scale_min + (scale_max - scale_min) * torch.rand(1).item())
        # translation in normalized coordinates relative to patch size: range [-max_translate, max_translate]
        tx = (torch.rand(1).item() * 2 - 1) * max_translate
        ty = (torch.rand(1).item() * 2 - 1) * max_translate
        # Build theta for grid_sample: but we sample patch into a patch-sized grid (so translation applies as fraction of patch)
        theta = make_affine_matrix(angle, scale, tx, ty, device=device).unsqueeze(0)  # [1,2,3]
        grid = F.affine_grid(theta, size=(1, C, p, p), align_corners=False)  # [-1,1] coords
        warped = F.grid_sample(patch_batch[i:i+1], grid, align_corners=False)  # [1,C,p,p]
        # Now place warped onto a canvas of H_out x W_out: scale up to desired size with center placement (no additional translate)
        # We'll create an empty canvas and paste resized warped patch at center
        # Resize warped to desired patch size in output canvas by interpolation
        warped_resized = F.interpolate(warped, size=(min(H_out, p), min(W_out, p)), mode='bilinear', align_corners=False)
        canvas = torch.zeros(1, C, H_out, W_out, device=device)
        ph, pw = warped_resized.shape[2], warped_resized.shape[3]
        y0 = (H_out - ph) // 2
        x0 = (W_out - pw) // 2
        canvas[:, :, y0:y0+ph, x0:x0+pw] = warped_resized
        transformed.append(canvas)
    out = torch.cat(transformed, dim=0)  # [B,C,H_out,W_out]
    return out


class LearnablePatch(nn.Module):
    """
    A learnable patch with optional alpha mask.
    - rgb_param: raw RGB values, optimized and clamped via sigmoid to [0,1]
    - mask_param: raw mask logits, converted via sigmoid to (0,1) alpha mask
    """
    def __init__(self, patch_size: int = 64, init: str = 'rand', device: str = 'cuda'):
        super().__init__()
        self.patch_size = patch_size
        self.device = device
        if init == 'zeros':
            init_rgb = torch.zeros(3, patch_size, patch_size)
            init_mask = torch.zeros(1, patch_size, patch_size)
        elif init == 'ones':
            init_rgb = torch.ones(3, patch_size, patch_size) * 0.5
            init_mask = torch.ones(1, patch_size, patch_size) * 0.9
        else:
            init_rgb = torch.rand(3, patch_size, patch_size) * 0.5 + 0.25
            init_mask = torch.rand(1, patch_size, patch_size) * 0.2 + 0.05

        self.rgb_param = nn.Parameter(init_rgb)  # optimized
        self.mask_param = nn.Parameter(init_mask)  # optimized

    def forward(self):
        rgb = torch.sigmoid(self.rgb_param)  # [3,H,W] in (0,1)
        mask = torch.sigmoid(self.mask_param)  # [1,H,W] in (0,1)
        return rgb, mask


class PatchApplier:
    """
    Apply a transformed patch (with alpha mask) onto images.
    images: [B,C,H,W], rgb_patch: [B,3,Hp,Wp] or [3,Hp,Wp]
    mask: [B,1,Hp,Wp] or [1,Hp,Wp]
    location_mode: 'random'|'center'|(x,y) absolute pixel coordinates for top-left
    """
    def __init__(self, ignore_label: int = 255):
        self.ignore_label = ignore_label

    def apply_to_image_batch(self,
                             images: torch.Tensor,
                             labels: torch.Tensor,
                             rgb_patch: torch.Tensor,
                             mask: torch.Tensor,
                             location_mode: str = 'random') -> Tuple[torch.Tensor, torch.Tensor]:
        B, C, H, W = images.shape
        # accept rgb_patch as [3,ph,pw] or [B,3,ph,pw]
        if rgb_patch.dim() == 3:
            ph, pw = rgb_patch.shape[1], rgb_patch.shape[2]
            rgb_batch = rgb_patch.unsqueeze(0).repeat(B, 1, 1, 1)
        else:
            rgb_batch = rgb_patch
            ph, pw = rgb_patch.shape[2], rgb_patch.shape[3]

        if mask.dim() == 3:
            mask_batch = mask.unsqueeze(0).repeat(B, 1, 1, 1)
        else:
            mask_batch = mask

        patched_images = images.clone()
        # patched_labels = labels.clone()
        patched_labels = labels.clone().long()
        if patched_labels.dim() == 4 and patched_labels.size(1) == 1:
            patched_labels = patched_labels.squeeze(1)  # [B,H,W]

        for i in range(B):
            if location_mode == 'center':
                x = (W - pw) // 2
                y = (H - ph) // 2
            elif isinstance(location_mode, (tuple, list)):
                x, y = int(location_mode[0]), int(location_mode[1])
                x = max(0, min(W - pw, x))
                y = max(0, min(H - ph, y))
            else:  # random
                x = int(torch.randint(0, max(1, W - pw + 1), (1,)).item())
                y = int(torch.randint(0, max(1, H - ph + 1), (1,)).item())

            alpha = mask_batch[i]  # [1,ph,pw]
            rgb = rgb_batch[i]  # [3,ph,pw]
            # blend: out = alpha * patch + (1-alpha) * image_region
            region = patched_images[i, :, y:y+ph, x:x+pw]
            # ensure same device/dtype
            region = region.to(rgb.device)
            alpha_broadcast = alpha.repeat(3, 1, 1)
            blended = alpha_broadcast * rgb + (1.0 - alpha_broadcast) * region
            patched_images[i, :, y:y+ph, x:x+pw] = blended
            # set label region to ignore
            patched_labels[i, y:y+ph, x:x+pw] = self.ignore_label

        # restore labels shape [B,1,H,W] if original had that shape
        return patched_images, patched_labels


class PatchLoss:
    """
    Loss wrapper:
      - primary: segmentation CE loss (we maximize it -> in hook we minimize -CE)
      - reg: lambda_tv * TV(mask) + lambda_l2 * ||patch||_2^2 + lambda_mask_l1 * ||mask||_1
    """
    def __init__(self,
                 ignore_index: int = 255,
                 lambda_tv: float = 1e-4,
                 lambda_l2: float = 1e-6,
                 lambda_mask_l1: float = 1e-3):
        self.ignore_index = ignore_index
        self.ce = nn.CrossEntropyLoss(ignore_index=ignore_index)
        self.lambda_tv = lambda_tv
        self.lambda_l2 = lambda_l2
        self.lambda_mask_l1 = lambda_mask_l1

    def __call__(self, logits: torch.Tensor, labels: torch.Tensor, patch_rgb: torch.Tensor, patch_mask: torch.Tensor):
        """
        logits: [B, num_classes, H, W]
        labels: [B, H, W] (long)
        patch_rgb: [C, ph, pw] or [B,C,ph,pw]
        patch_mask: [1,ph,pw] or [B,1,ph,pw]
        """
        # --- 新增：保证 logits 与 labels 空间对齐、标签清洗为合法索引 ---
        # labels: [B,1,H,W] -> [B,H,W]
        if labels.dim() == 4 and labels.size(1) == 1:
            labels = labels.squeeze(1)

        # 若模型输出分辨率与标签不一致，先插到 labels 的 H,W
        if logits.shape[-2:] != labels.shape[-2:]:
            logits = F.interpolate(logits, size=labels.shape[-2:], mode='bilinear', align_corners=False)

        # 保险：把所有越界标签(非 ignore_index，且不在[0, C-1])置为 ignore_index，避免 CE 触发 device assert
        num_classes = logits.size(1)
        labels = labels.long().contiguous()
        invalid = (labels != self.ignore_index) & ((labels < 0) | (labels >= num_classes))
        if invalid.any():
            labels = labels.clone()
            labels[invalid] = self.ignore_index

        # segmentation loss
        seg_loss = self.ce(logits, labels)


        # regularizers computed on single-patch (take mean if batch)
        if patch_rgb.dim() == 4:
            pr = patch_rgb.mean(dim=0)  # [C,ph,pw]
        else:
            pr = patch_rgb
        if patch_mask.dim() == 4:
            pm = patch_mask.mean(dim=0)
        else:
            pm = patch_mask

        tv = total_variation(pr.unsqueeze(0))  # returns scalar
        l2 = torch.sum(pr ** 2)
        mask_l1 = torch.sum(torch.abs(pm))

        reg = self.lambda_tv * tv + self.lambda_l2 * l2 + self.lambda_mask_l1 * mask_l1

        return seg_loss, reg


@HOOKS.register_module()
class PatchAttackHook(Hook):
    """
    mmcv Hook to learn a patch during training.

    cfg keys (example):
      enabled: True
      patch_size: 64
      lr: 0.3
      steps_per_iter: 1
      eot: {rot:20, scale_min:0.9, scale_max:1.1, max_translate:0.15}
      location: 'random'|'center'|(x,y)
      ignore_label: 255
      reg: {lambda_tv:1e-4, lambda_l2:1e-6, lambda_mask_l1:1e-3}
      clamp: {min:0.0, max:1.0}
      save_dir: './work_dirs/patches'
      normalize: None or dict(mean=[..], std=[..])  # if images are normalized in pipeline, set here
      update_model: False  # if True, allow model gradients during inner loop (advanced)
    """
    def __init__(self, cfg: Dict[str, Any], device: Optional[str] = None):
        super().__init__()
        cfg = cfg or {}
        self.enabled = cfg.get('enabled', True)
        if not self.enabled:
            return
        self.device = device if device is not None else ('cuda' if torch.cuda.is_available() else 'cpu')
        self.patch_size = int(cfg.get('patch_size', 64))
        self.lr = float(cfg.get('lr', 0.3))
        self.steps_per_iter = int(cfg.get('steps_per_iter', 1))
        eot_cfg = cfg.get('eot', {})
        self.eot_rot = float(eot_cfg.get('rot', 20.0))
        self.eot_scale_min = float(eot_cfg.get('scale_min', 0.9))
        self.eot_scale_max = float(eot_cfg.get('scale_max', 1.1))
        self.eot_max_translate = float(eot_cfg.get('max_translate', 0.15))
        self.location = cfg.get('location', 'random')
        self.ignore_label = int(cfg.get('ignore_label', 255))
        self.reg_cfg = cfg.get('reg', {})
        self.lambda_tv = float(self.reg_cfg.get('lambda_tv', 1e-4))
        self.lambda_l2 = float(self.reg_cfg.get('lambda_l2', 1e-6))
        self.lambda_mask_l1 = float(self.reg_cfg.get('lambda_mask_l1', 1e-3))
        clamp_cfg = cfg.get('clamp', {})
        self.clip_min = float(clamp_cfg.get('min', 0.0))
        self.clip_max = float(clamp_cfg.get('max', 1.0))
        self.save_dir = cfg.get('save_dir', './work_dirs/patches')
        self.normalize_cfg = cfg.get('normalize', None)
        self.update_model = bool(cfg.get('update_model', False))  # default: do not update model inside inner loop

        # create modules
        self.patch_net = LearnablePatch(self.patch_size, init=cfg.get('init', 'rand'), device=self.device).to(self.device)
        self.applier = PatchApplier(self.ignore_label)
        self.criterion = PatchLoss(ignore_index=self.ignore_label,
                                   lambda_tv=self.lambda_tv,
                                   lambda_l2=self.lambda_l2,
                                   lambda_mask_l1=self.lambda_mask_l1)
        # optimizer for patch params
        self.patch_optimizer = torch.optim.Adam(self.patch_net.parameters(), lr=self.lr)
        os.makedirs(self.save_dir, exist_ok=True)

    def before_run(self, runner):
        if not self.enabled:
            return
        # ensure model on right device
        try:
            runner.model.to(self.device)
        except Exception:
            pass

    # def _prepare_images_labels(self, data_batch: Dict[str, torch.Tensor]) -> Tuple[torch.Tensor, torch.Tensor]:
    #     """Convert mmseg data_batch to image and label tensors on device.
    #        Expects data_batch['img'] and data_batch['gt_semantic_seg'] present.
    #     """
    #     images = data_batch['img'].to(self.device)  # [B,C,H,W]
    #     labels = data_batch['gt_semantic_seg'].to(self.device)  # [B,1,H,W] or [B,H,W]
    #     if labels.dim() == 4 and labels.size(1) == 1:
    #         labels = labels.squeeze(1)
    #     return images, labels

    def _denorm_patch_for_save(self, rgb: torch.Tensor) -> np.ndarray:
        # rgb: [3,ph,pw] in [0,1] -> uint8
        arr = (rgb.detach().cpu().clamp(0, 1).permute(1, 2, 0).numpy() * 255).astype('uint8')
        return arr

    def _prepare_images_labels(self, data_batch):
        """兼容 mmseg 1.x( PackSegInputs ) 和老格式两种 batch 结构，返回 images[B,C,H,W], labels[B,H,W]."""
        if isinstance(data_batch, dict) and 'inputs' in data_batch and 'data_samples' in data_batch:
            inputs = data_batch['inputs']

            # --- 统一成 list[Tensor(C,H,W)] 并统计本 batch 最大高宽 ---
            if isinstance(inputs, (list, tuple)):
                tensor_list = []
                Hmax, Wmax = 0, 0
                for x in inputs:
                    x = torch.as_tensor(x)
                    # 如果是 [H,W,C] 转成 [C,H,W]
                    if x.dim() == 3 and x.shape[-1] == 3 and x.shape[0] != 3:
                        x = x.permute(2, 0, 1).contiguous()
                    x = x.float()
                    tensor_list.append(x)
                    Hmax = max(Hmax, x.shape[1])
                    Wmax = max(Wmax, x.shape[2])

                # --- 右下角补零 pad 到相同尺寸，再 stack ---
                padded_imgs = []
                for x in tensor_list:
                    ph = Hmax - x.shape[1]
                    pw = Wmax - x.shape[2]
                    # pad 顺序: (left, right, top, bottom)
                    x_pad = F.pad(x, (0, pw, 0, ph), mode='constant', value=0.0)
                    padded_imgs.append(x_pad)
                images = torch.stack(padded_imgs, dim=0).to(self.device)  # [B,C,Hmax,Wmax]
            else:
                images = inputs.to(self.device)  # 已是 Tensor[B,C,H,W]

            # --- 从 data_samples 取 label，并按同样的 Hmax/Wmax 做 pad，再 stack ---
            labels_list = []
            for ds in data_batch['data_samples']:
                if hasattr(ds, 'gt_sem_seg') and hasattr(ds.gt_sem_seg, 'data'):
                    lab = ds.gt_sem_seg.data
                elif hasattr(ds, 'gt_semantic_seg') and hasattr(ds.gt_semantic_seg, 'data'):
                    lab = ds.gt_semantic_seg.data
                else:
                    raise KeyError('gt_sem_seg / gt_semantic_seg not found in data_samples')

                lab_t = torch.as_tensor(lab)
                # [1,H,W] -> [H,W]
                if lab_t.dim() == 3 and lab_t.size(0) == 1:
                    lab_t = lab_t.squeeze(0)

                ph = Hmax - lab_t.shape[-2]
                pw = Wmax - lab_t.shape[-1]
                # 先扩一维到 [1,H,W]，pad 后再去掉
                lab_t = lab_t.unsqueeze(0)  # [1,H,W]
                lab_t = F.pad(lab_t, (0, pw, 0, ph), mode='constant', value=self.ignore_label)
                lab_t = lab_t.squeeze(0).long()  # [Hmax,Wmax]
                labels_list.append(lab_t)

            labels = torch.stack(labels_list, dim=0).to(self.device)  # [B,Hmax,Wmax]
            return images, labels

        # ===== 老格式：直接给 'img' 和 'gt_semantic_seg' =====
        if isinstance(data_batch, dict) and 'img' in data_batch and 'gt_semantic_seg' in data_batch:
            images = data_batch['img'].to(self.device)
            labels = data_batch['gt_semantic_seg']
            labels = torch.as_tensor(labels).to(self.device)
            if labels.dim() == 4 and labels.size(1) == 1:
                labels = labels.squeeze(1)
            return images, labels

        raise KeyError(f'Unsupported data_batch format. Keys: {list(data_batch.keys())}')




    # def before_train_iter(self, runner):
    #     if not self.enabled:
    #         return
    #     data_batch = runner.data_batch
    #     if 'img' not in data_batch or 'gt_semantic_seg' not in data_batch:
    #         return

    def before_train_iter(self, runner, batch_idx: int, data_batch=None, **kwargs):
        if not self.enabled:
            return
        # 优先用 mmengine 传进来的 data_batch；兜底从 runner 拿
        if data_batch is None:
            data_batch = getattr(runner, 'data_batch', None)
            if data_batch is None:
                return

        images, labels = self._prepare_images_labels(data_batch)
        B, C, H, W = images.shape

        # === 新增：清洗 data_batch 里的原始标签，避免主训练 step 的越界 ===
        try:
            # 取解码头的 num_classes
            dec_head = getattr(getattr(runner.model, 'module', runner.model), 'decode_head', None)
            num_classes = int(getattr(dec_head, 'num_classes')) if dec_head is not None else None
            ign = int(self.ignore_label)

            if num_classes is not None and isinstance(data_batch, dict) and 'data_samples' in data_batch:
                for ds in data_batch['data_samples']:
                    # mmseg 1.x 的 PixelData: gt_sem_seg.data
                    lab = None
                    if hasattr(ds, 'gt_sem_seg') and hasattr(ds.gt_sem_seg, 'data'):
                        lab = ds.gt_sem_seg.data
                    elif hasattr(ds, 'gt_semantic_seg') and hasattr(ds.gt_semantic_seg, 'data'):
                        lab = ds.gt_semantic_seg.data

                    if lab is None:
                        continue

                    lab_t = torch.as_tensor(lab)  # 保持原 device（通常是 CPU）
                    # [1,H,W] -> [H,W]
                    if lab_t.dim() == 3 and lab_t.size(0) == 1:
                        lab_t = lab_t.squeeze(0)

                    # 越界（非 ignore）一律置 ignore
                    invalid = (lab_t != ign) & ((lab_t < 0) | (lab_t >= num_classes))
                    if invalid.any():
                        lab_t = lab_t.clone()
                        lab_t[invalid] = ign

                    # dtype 统一为 long
                    lab_t = lab_t.long()

                    # 回写回 data_samples（保持原结构）
                    if hasattr(ds, 'gt_sem_seg') and hasattr(ds.gt_sem_seg, 'data'):
                        ds.gt_sem_seg.data = lab_t
                    elif hasattr(ds, 'gt_semantic_seg') and hasattr(ds.gt_semantic_seg, 'data'):
                        ds.gt_semantic_seg.data = lab_t
        except Exception:
            # 清洗是“尽力而为”，失败不影响后续，但建议在调试时打开异常查看
            pass
        # === 新增结束 ===


        # If training pipeline normalizes inputs, we must apply same normalization to patch when placing it.
        # Option: if normalize_cfg provided, patch will be normalized before blending.
        normalize_mean = None
        normalize_std = None
        scale_255 = 1.0
        if self.normalize_cfg:
            normalize_mean = torch.tensor(self.normalize_cfg.get('mean'), dtype=images.dtype).view(1, C, 1, 1).to(self.device)
            normalize_std = torch.tensor(self.normalize_cfg.get('std'), dtype=images.dtype).view(1, C, 1, 1).to(self.device)
            # 如果 mean/std 明显是 0-255 量级，则把输入从[0,1]放大到[0,255]
            if float(normalize_mean.max().item()) > 2.0 or float(normalize_std.max().item()) > 2.0:
                scale_255 = 255.0


        # inner optimization loop (opt patch to maximize segmentation loss)
        for _ in range(self.steps_per_iter):
            # get current patch rgb and mask
            rgb_patch, mask_patch = self.patch_net()  # [3,ph,pw], [1,ph,pw]
            ph, pw = rgb_patch.shape[1], rgb_patch.shape[2]

            # Build transformed patch batch via EOT (we will get B x C x H x W canvases with small centered patch)
            transformed = random_eot_transform_batch(patch=rgb_patch,
                                                    batch_size=B,
                                                    out_size=(ph, pw),
                                                    rot_degree=self.eot_rot,
                                                    scale_min=self.eot_scale_min,
                                                    scale_max=self.eot_scale_max,
                                                    max_translate=self.eot_max_translate,
                                                    device=self.device)
            # transformed: [B,C,ph,pw] (Note: function creates B,C,H_out,W_out equal to ph,pw)
            # Do the same for mask (transform mask logits similarly using grid_sample)
            mask_transformed = random_eot_transform_batch(
                patch=mask_patch,            # 保持 [1, ph, pw]
                batch_size=B,
                out_size=(ph, pw),
                rot_degree=self.eot_rot,
                scale_min=self.eot_scale_min,
                scale_max=self.eot_scale_max,
                max_translate=self.eot_max_translate,
                device=self.device)
            # mask_transformed now shape [B,1,ph,pw] but function returns [B, C, H, W]; for mask we used 1 channel so okay.

            # We now need to place transformed patch patches into the full image canvas at chosen locations.
            # For efficiency we'll create small canvases sized ph x pw and then let PatchApplier paste them at random places.
            # Applier expects rgb_patch either [3,ph,pw] or [B,3,ph,pw]
            rgb_batch_for_applier = transformed  # [B,3,ph,pw]
            mask_batch_for_applier = mask_transformed  # [B,1,ph,pw]

            # Apply to images
            patched_images, patched_labels = self.applier.apply_to_image_batch(images, labels, rgb_batch_for_applier, mask_batch_for_applier, location_mode=self.location)

            # If normalization is used in pipeline, normalize patched_images accordingly before forward
            # ……（贴补丁操作后）
            # 为防止数值飘出范围，先夹到[0,1]
            patched_images = patched_images.clamp(0.0, 1.0)

            forward_images = patched_images
            if normalize_mean is not None and normalize_std is not None:
                forward_images = (forward_images * scale_255 - normalize_mean) / normalize_std

            # Forward pass
            runner.model.train()  # ensure train mode for normalization layers etc.
            # Try common mmseg forward signature to get logits: model(return_loss=False, img=...)
            try:
                outputs = runner.model(return_loss=False, img=forward_images)
                logits = outputs  # expected to be [B,num_classes,H,W]
            except Exception:
                try:
                    outputs = runner.model.module(return_loss=False, img=forward_images)
                    logits = outputs
                except Exception:
                    # As fallback, try decode_head
                    try:
                        # logits = runner.model.module.decode_head(forward_images)
                        # logits = runner.model._run_forward(forward_images, data_samples=None, mode='tensor')
                        logits = runner.model.predict(forward_images, data_samples=None, mode='tensor')

                    except Exception:
                        # if cannot obtain logits, skip patch update
                        return

            # Compute loss (segmentation loss + reg)
            seg_loss, reg = self.criterion(logits, patched_labels, rgb_patch, mask_patch)
            # We want to maximize seg_loss -> minimize -seg_loss (+ reg)
            loss = -seg_loss + reg

            # zero grads
            self.patch_optimizer.zero_grad()
            # optionally freeze model parameters to avoid updating them when computing grads (typical)
            model_requires_grad = []
            if not self.update_model:
                for p in runner.model.parameters():
                    model_requires_grad.append(p.requires_grad)
                    p.requires_grad = False

            loss.backward()
            self.patch_optimizer.step()

            # restore model grad flags
            if not self.update_model:
                for p, flag in zip(runner.model.parameters(), model_requires_grad):
                    p.requires_grad = flag

            # clamp patch parameters to [clip_min, clip_max] in forward via sigmoid, but also we can clamp raw params to limit
            with torch.no_grad():
                # raw params are unconstrained; but we can clip sigmoid outputs indirectly by clamping raw param
                self.patch_net.rgb_param.data.clamp_(-6.0, 6.0)
                self.patch_net.mask_param.data.clamp_(-6.0, 6.0)

        # logging: store last seg_loss
        try:
            runner.log_buffer.update({'patch_seg_loss': float(seg_loss.detach().cpu().item()),
                                      'patch_reg': float(reg.detach().cpu().item())})
        except Exception:
            pass

    def after_train_epoch(self, runner):
        if not self.enabled:
            return
        # save current patch and mask for inspection
        rgb, mask = self.patch_net()
        rgb_img = self._denorm_patch_for_save(rgb)
        mask_arr = (mask.detach().cpu().squeeze(0).clamp(0, 1).numpy() * 255).astype('uint8')
        im = Image.fromarray(rgb_img)
        im_mask = Image.fromarray(mask_arr)
        epoch = getattr(runner, 'epoch', 'final')
        im.save(os.path.join(self.save_dir, f'patch_epoch_{epoch}.png'))
        im_mask.save(os.path.join(self.save_dir, f'patch_mask_epoch_{epoch}.png'))

    def after_run(self, runner):
        if not self.enabled:
            return
        rgb, mask = self.patch_net()
        rgb_img = self._denorm_patch_for_save(rgb)
        im = Image.fromarray(rgb_img)
        im.save(os.path.join(self.save_dir, 'final_patch.png'))
        mask_arr = (mask.detach().cpu().squeeze(0).clamp(0, 1).numpy() * 255).astype('uint8')
        Image.fromarray(mask_arr).save(os.path.join(self.save_dir, 'final_patch_mask.png'))
