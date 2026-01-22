# mmseg/plugins/patch_attack_plugin.py
"""
Patch Attack plugin (APT-style, transferability-oriented)

- 只改插件本文件；不改 mmseg 主框架
- 目标：实现类似 adversarial-patch-transferability 的思路：
    * EOT（多重随机仿射/抖动）
    * 自适应困难像素挖掘的交叉熵（Top-k CE）
    * 预测熵最大化（提高可迁移性）
    * 贴图区域可选仅在补丁内统计 or 全图统计
    * 正则：TV/L2/Mask-L1
- 兼容 mmseg 1.x 的 PackSegInputs / data_samples 结构
- 训练阶段只更新补丁参数（默认），不更新模型参数
"""

import os
import math
from typing import Optional, Tuple, Dict, Any, List

import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
import numpy as np

from mmengine.hooks import Hook
from mmengine.registry import HOOKS

print('[PatchAttackHook/APT] loaded from:', __file__)


# ----------------------------- Utils -----------------------------

def total_variation(x: torch.Tensor) -> torch.Tensor:
    """Total variation for patch smoothness. x: [C,H,W] or [B,C,H,W]."""
    if x.dim() == 3:
        x = x.unsqueeze(0)
    tv_h = torch.abs(x[:, :, 1:, :] - x[:, :, :-1, :]).sum()
    tv_w = torch.abs(x[:, :, :, 1:] - x[:, :, :, :-1]).sum()
    return tv_h + tv_w


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


def _random_color_jitter(x: torch.Tensor, jitter: float) -> torch.Tensor:
    """轻量级颜色抖动：x∈[0,1]，乘以 (1±jitter) 并加上小偏置，再裁剪回 [0,1]."""
    if jitter <= 0:
        return x
    b = x.new_empty(x.size(0), 1, 1, 1).uniform_(1 - jitter, 1 + jitter)
    a = x.new_empty(x.size(0), 1, 1, 1).uniform_(-jitter, jitter) * 0.05
    y = x * b + a
    return y.clamp(0.0, 1.0)


def random_eot_transform_batch(
    patch: torch.Tensor,                 # [C,p,p] 或 [1,p,p]
    batch_size: int,
    out_size: Tuple[int, int],
    rot_degree: float = 20.0,
    scale_min: float = 0.8,
    scale_max: float = 1.2,
    max_translate: float = 0.2,
    color_jitter: float = 0.1,
    device: Optional[torch.device] = None
) -> torch.Tensor:
    """对补丁做一次 EOT 采样并放入指定大小画布（通常仍为 p×p），返回 [B,C,H_out,W_out]."""
    if device is None:
        device = patch.device
    if patch.dim() == 2:
        patch = patch.unsqueeze(0)  # [1,p,p]
    patch = patch.to(device)
    C, p, _ = patch.shape

    H_out, W_out = out_size
    out = torch.zeros(batch_size, C, H_out, W_out, device=device, dtype=patch.dtype)

    for i in range(batch_size):
        angle = float((torch.rand(1, device=device) * 2 - 1) * rot_degree)
        scale = float(scale_min + (scale_max - scale_min) * torch.rand(1, device=device))
        tx = float((torch.rand(1, device=device) * 2 - 1) * max_translate)
        ty = float((torch.rand(1, device=device) * 2 - 1) * max_translate)

        theta = make_affine_matrix(angle, scale, tx, ty, device=device).unsqueeze(0)  # [1,2,3]
        grid = F.affine_grid(theta, size=(1, C, p, p), align_corners=False)
        warped = F.grid_sample(patch.unsqueeze(0), grid, align_corners=False)  # [1,C,p,p]

        # 颜色抖动（提升可迁移性）
        warped = _random_color_jitter(warped, color_jitter)

        # 居中贴到 out 画布
        ph, pw = min(H_out, p), min(W_out, p)
        wr = F.interpolate(warped, size=(ph, pw), mode='bilinear', align_corners=False)  # [1,C,ph,pw]
        y0 = (H_out - ph) // 2
        x0 = (W_out - pw) // 2
        out[i, :, y0:y0+ph, x0:x0+pw] = wr[0]
    return out


# ----------------------------- Modules -----------------------------

class LearnablePatch(nn.Module):
    """可学习补丁（rgb + alpha mask，经 sigmoid 映射到 [0,1]）"""
    def __init__(self, patch_size: int = 64, init: str = 'rand', device: str = 'cuda'):
        super().__init__()
        self.patch_size = patch_size
        if init == 'zeros':
            init_rgb = torch.zeros(3, patch_size, patch_size)
            init_mask = torch.zeros(1, patch_size, patch_size)
        elif init == 'ones':
            init_rgb = torch.ones(3, patch_size, patch_size) * 0.5
            init_mask = torch.ones(1, patch_size, patch_size) * 0.9
        else:
            init_rgb = torch.rand(3, patch_size, patch_size) * 0.5 + 0.25
            init_mask = torch.rand(1, patch_size, patch_size) * 0.2 + 0.05

        self.rgb_param = nn.Parameter(init_rgb)
        self.mask_param = nn.Parameter(init_mask)

    def forward(self):
        rgb = torch.sigmoid(self.rgb_param)   # [3,H,W] in (0,1)
        mask = torch.sigmoid(self.mask_param) # [1,H,W] in (0,1)
        return rgb, mask


class PatchApplier:
    """
    将补丁贴到图像上。
    返回：(patched_images, patched_labels, placed_alpha)
      - placed_alpha: [B,1,H,W]，为贴补丁后的 alpha 映射（便于只在补丁区域计算损失）
    """
    def __init__(self, ignore_label: int = 255):
        self.ignore_label = ignore_label

    def apply_to_image_batch(
        self,
        images: torch.Tensor,       # [B,C,H,W] in [0,1]
        labels: torch.Tensor,       # [B,H,W] (long)
        rgb_patch: torch.Tensor,    # [B,3,ph,pw] 或 [3,ph,pw]
        mask: torch.Tensor,         # [B,1,ph,pw] 或 [1,ph,pw]
        location_mode: Any = 'random'
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        B, C, H, W = images.shape

        # 统一 batch 化
        if rgb_patch.dim() == 3:
            ph, pw = rgb_patch.shape[-2:]
            rgb_b = rgb_patch.unsqueeze(0).repeat(B, 1, 1, 1)
        else:
            rgb_b = rgb_patch
            ph, pw = rgb_patch.shape[-2:]

        if mask.dim() == 3:
            mask_b = mask.unsqueeze(0).repeat(B, 1, 1, 1)
        else:
            mask_b = mask

        imgs = images.clone()
        labs = labels.clone().long()
        if labs.dim() == 4 and labs.size(1) == 1:
            labs = labs.squeeze(1)

        placed_alpha = torch.zeros(B, 1, H, W, dtype=imgs.dtype, device=imgs.device)

        for i in range(B):
            if location_mode == 'center':
                x = (W - pw) // 2
                y = (H - ph) // 2
            elif isinstance(location_mode, (tuple, list)):
                x, y = int(location_mode[0]), int(location_mode[1])
                x = max(0, min(W - pw, x))
                y = max(0, min(H - ph, y))
            else:
                x = int(torch.randint(0, max(1, W - pw + 1), (1,)).item())
                y = int(torch.randint(0, max(1, H - ph + 1), (1,)).item())

            alpha = mask_b[i]                     # [1,ph,pw] in [0,1]
            rgb   = rgb_b[i]                      # [3,ph,pw]
            region = imgs[i, :, y:y+ph, x:x+pw]   # [C,ph,pw]
            alpha3 = alpha.repeat(3, 1, 1)
            blended = alpha3 * rgb + (1.0 - alpha3) * region
            imgs[i, :, y:y+ph, x:x+pw] = blended

            # label 区域置 ignore
            labs[i, y:y+ph, x:x+pw] = self.ignore_label

            # 记录贴图 alpha
            placed_alpha[i, :, y:y+ph, x:x+pw] = alpha

        return imgs, labs, placed_alpha


# ----------------------------- Loss (APT style) -----------------------------

class PatchLossAPT:
    """
    APT 风格损失：
      attack_obj = w_ce * TopK_CE + w_ent * Entropy
      loss = -attack_obj + lambda_tv*TV + lambda_l2*||patch||^2 + lambda_mask_l1*||mask||_1
    说明：
      - TopK_CE：对非忽略像素的 per-pixel CE 取 Top-k(比例) 后平均
      - Entropy：对模型预测分布的熵（全类），在有效像素上平均
      - region: 'all' | 'in_patch'（熵与 CE 的统计范围）
    """
    def __init__(
        self,
        ignore_index: int = 255,
        topk_ratio: float = 0.2,
        w_ce: float = 0.5,
        w_ent: float = 0.5,
        region: str = 'all',
        lambda_tv: float = 1e-4,
        lambda_l2: float = 1e-6,
        lambda_mask_l1: float = 1e-3
    ):
        self.ignore_index = ignore_index
        self.topk_ratio = float(topk_ratio)
        self.w_ce = float(w_ce)
        self.w_ent = float(w_ent)
        self.region = region
        self.lambda_tv = lambda_tv
        self.lambda_l2 = lambda_l2
        self.lambda_mask_l1 = lambda_mask_l1

    @staticmethod
    def _valid_mask_from_labels(labels: torch.Tensor, ignore_index: int) -> torch.Tensor:
        # labels: [B,H,W]
        return (labels != ignore_index)

    def _maybe_restrict_region(
        self,
        valid: torch.Tensor,           # [B,H,W] bool
        placed_alpha: Optional[torch.Tensor]  # [B,1,H,W] or None
    ) -> torch.Tensor:
        if self.region == 'in_patch' and placed_alpha is not None:
            in_patch = placed_alpha.squeeze(1) > 1e-6  # [B,H,W]
            return valid & in_patch
        return valid

    def _topk_mean(self, x: torch.Tensor, mask: torch.Tensor, ratio: float) -> torch.Tensor:
        # x: [B,H,W], mask: [B,H,W] bool
        vals = x[mask]
        if vals.numel() == 0:
            return x.new_tensor(0.0)
        k = max(1, int(math.ceil(ratio * vals.numel())))
        topk = torch.topk(vals, k=k, largest=True).values
        return topk.mean()

    def __call__(
        self,
        logits: torch.Tensor,         # [B,C,H,W]
        labels: torch.Tensor,         # [B,H,W] (long)
        placed_alpha: Optional[torch.Tensor],  # [B,1,H,W]
        patch_rgb: torch.Tensor,      # [3,ph,pw] or [B,3,ph,pw]
        patch_mask: torch.Tensor      # [1,ph,pw] or [B,1,ph,pw]
    ):
        B, C, H, W = logits.shape
        # 对齐空间分辨率
        if logits.shape[-2:] != labels.shape[-2:]:
            logits = F.interpolate(logits, size=labels.shape[-2:], mode='bilinear', align_corners=False)

        # 清洗越界标签，避免 CE 触发 device assert
        num_classes = logits.size(1)
        labels = labels.long().contiguous()
        invalid = (labels != self.ignore_index) & ((labels < 0) | (labels >= num_classes))
        if invalid.any():
            labels = labels.clone()
            labels[invalid] = self.ignore_index

        # 基础有效像素 mask
        valid = self._valid_mask_from_labels(labels, self.ignore_index)
        valid = self._maybe_restrict_region(valid, placed_alpha)

        # per-pixel CE（reduction='none'）
        ce_map = F.cross_entropy(logits, labels, ignore_index=self.ignore_index, reduction='none')  # [B,H,W]
        ce_topk = self._topk_mean(ce_map, valid, self.topk_ratio)

        # 预测熵（全类）
        with torch.no_grad():
            # 为数值稳定，可不 detach；这里按论文思路希望对熵也反传梯度，故不 detach
            pass
        probs = F.softmax(logits, dim=1) + 1e-12
        ent_map = -(probs * probs.log()).sum(dim=1)  # [B,H,W]
        ent_mean = ent_map[valid].mean() if valid.any() else ent_map.mean()

        # APT 攻击目标：越大越好
        attack_obj = self.w_ce * ce_topk + self.w_ent * ent_mean

        # 正则项（对单个补丁取均值）
        if patch_rgb.dim() == 4:
            pr = patch_rgb.mean(dim=0)  # [C,ph,pw]
        else:
            pr = patch_rgb
        if patch_mask.dim() == 4:
            pm = patch_mask.mean(dim=0)  # [1,ph,pw]
        else:
            pm = patch_mask

        tv = total_variation(pr.unsqueeze(0))
        l2 = torch.sum(pr ** 2)
        mask_l1 = torch.sum(torch.abs(pm))
        reg = self.lambda_tv * tv + self.lambda_l2 * l2 + self.lambda_mask_l1 * mask_l1

        # 我们在外层最小化：loss = -attack_obj + reg
        return attack_obj, reg


# ----------------------------- Hook -----------------------------

@HOOKS.register_module()
class PatchAttackHook(Hook):
    """
    使用 APT 风格的补丁攻击
    cfg 示例（放到 custom_hooks 中）：
    custom_hooks = [
      dict(
        type='PatchAttackHook',
        cfg=dict(
            enabled=True,
            patch_size=96,
            lr=0.3,
            steps_per_iter=1,
            eot=dict(rot=20, scale_min=0.9, scale_max=1.1, max_translate=0.15, color_jitter=0.1, samples=4),
            location='random',          # 'random' | 'center' | (x,y)
            ignore_label=255,
            reg=dict(lambda_tv=1e-4, lambda_l2=1e-6, lambda_mask_l1=1e-3),
            loss=dict(
                topk_ratio=0.2, w_ce=0.5, w_ent=0.5, region='all'  # or 'in_patch'
            ),
            clamp=dict(min=0.0, max=1.0),
            save_dir='./work_dirs/patches',
            normalize=None,             # or dict(mean=[..], std=[..]) 与数据流水线一致
            update_model=False          # 只更新补丁
        )
      )
    ]
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
        self.eot_color_jitter = float(eot_cfg.get('color_jitter', 0.1))
        self.eot_samples = int(eot_cfg.get('samples', 1))  # 每个 iter 对同一 batch 进行多少次 EOT 累积

        self.location = cfg.get('location', 'random')
        self.ignore_label = int(cfg.get('ignore_label', 255))

        reg_cfg = cfg.get('reg', {})
        self.lambda_tv = float(reg_cfg.get('lambda_tv', 1e-4))
        self.lambda_l2 = float(reg_cfg.get('lambda_l2', 1e-6))
        self.lambda_mask_l1 = float(reg_cfg.get('lambda_mask_l1', 1e-3))

        loss_cfg = cfg.get('loss', {})
        self.topk_ratio = float(loss_cfg.get('topk_ratio', 0.2))
        self.w_ce = float(loss_cfg.get('w_ce', 0.5))
        self.w_ent = float(loss_cfg.get('w_ent', 0.5))
        self.loss_region = str(loss_cfg.get('region', 'all'))

        clamp_cfg = cfg.get('clamp', {})
        self.clip_min = float(clamp_cfg.get('min', 0.0))
        self.clip_max = float(clamp_cfg.get('max', 1.0))

        self.save_dir = cfg.get('save_dir', './work_dirs/patches')
        self.normalize_cfg = cfg.get('normalize', None)
        self.update_model = bool(cfg.get('update_model', False))

        # modules
        self.patch_net = LearnablePatch(self.patch_size, init=cfg.get('init', 'rand'), device=self.device).to(self.device)
        self.applier = PatchApplier(self.ignore_label)
        self.criterion = PatchLossAPT(
            ignore_index=self.ignore_label,
            topk_ratio=self.topk_ratio,
            w_ce=self.w_ce,
            w_ent=self.w_ent,
            region=self.loss_region,
            lambda_tv=self.lambda_tv,
            lambda_l2=self.lambda_l2,
            lambda_mask_l1=self.lambda_mask_l1
        )
        self.patch_optimizer = torch.optim.Adam(self.patch_net.parameters(), lr=self.lr)
        os.makedirs(self.save_dir, exist_ok=True)

    # ---------------- I/O helpers ----------------

    def _denorm_patch_for_save(self, rgb: torch.Tensor) -> np.ndarray:
        arr = (rgb.detach().cpu().clamp(0, 1).permute(1, 2, 0).numpy() * 255).astype('uint8')
        return arr

    def _prepare_images_labels(self, data_batch):
        """兼容新旧 batch 结构，返回 images[B,C,H,W]∈[0,1], labels[B,H,W](long, ignore=255)."""
        if isinstance(data_batch, dict) and 'inputs' in data_batch and 'data_samples' in data_batch:
            inputs = data_batch['inputs']
            # 统一成 [B,C,Hmax,Wmax]
            if isinstance(inputs, (list, tuple)):
                tensor_list = []
                Hmax, Wmax = 0, 0
                for x in inputs:
                    x = torch.as_tensor(x)
                    if x.dim() == 3 and x.shape[-1] == 3 and x.shape[0] != 3:
                        x = x.permute(2, 0, 1).contiguous()
                    x = x.float()
                    tensor_list.append(x)
                    Hmax = max(Hmax, x.shape[1]); Wmax = max(Wmax, x.shape[2])
                padded = []
                for x in tensor_list:
                    ph = Hmax - x.shape[1]; pw = Wmax - x.shape[2]
                    x_pad = F.pad(x, (0, pw, 0, ph), mode='constant', value=0.0)
                    padded.append(x_pad)
                images = torch.stack(padded, dim=0).to(self.device)
            else:
                images = inputs.to(self.device).float()

            labels_list: List[torch.Tensor] = []
            for ds in data_batch['data_samples']:
                lab = None
                if hasattr(ds, 'gt_sem_seg') and hasattr(ds.gt_sem_seg, 'data'):
                    lab = ds.gt_sem_seg.data
                elif hasattr(ds, 'gt_semantic_seg') and hasattr(ds.gt_semantic_seg, 'data'):
                    lab = ds.gt_semantic_seg.data
                if lab is None:
                    raise KeyError('gt_sem_seg / gt_semantic_seg not found in data_samples')

                lab_t = torch.as_tensor(lab)
                if lab_t.dim() == 3 and lab_t.size(0) == 1:
                    lab_t = lab_t.squeeze(0)

                ph = images.shape[-2] - lab_t.shape[-2]
                pw = images.shape[-1] - lab_t.shape[-1]
                lab_t = F.pad(lab_t.unsqueeze(0), (0, pw, 0, ph), mode='constant', value=self.ignore_label).squeeze(0).long()
                labels_list.append(lab_t)
            labels = torch.stack(labels_list, dim=0).to(self.device)
            return images, labels

        if isinstance(data_batch, dict) and 'img' in data_batch and 'gt_semantic_seg' in data_batch:
            images = data_batch['img'].to(self.device).float()
            labels = torch.as_tensor(data_batch['gt_semantic_seg']).to(self.device).long()
            if labels.dim() == 4 and labels.size(1) == 1:
                labels = labels.squeeze(1)
            return images, labels

        raise KeyError(f'Unsupported data_batch format. Keys: {list(data_batch.keys())}')

    def before_run(self, runner):
        if not self.enabled:
            return
        try:
            runner.model.to(self.device)
        except Exception:
            pass

    # ---------------- Main step ----------------

    def before_train_iter(self, runner, batch_idx: int, data_batch=None, **kwargs):
        if not self.enabled:
            return
        if data_batch is None:
            data_batch = getattr(runner, 'data_batch', None)
            if data_batch is None:
                return

        images, labels = self._prepare_images_labels(data_batch)
        B, C, H, W = images.shape

        # 清洗 data_samples 中的标签（防止主损失里越界）
        try:
            dec_head = getattr(getattr(runner.model, 'module', runner.model), 'decode_head', None)
            num_classes = int(getattr(dec_head, 'num_classes')) if dec_head is not None else None
            ign = int(self.ignore_label)
            if num_classes is not None and isinstance(data_batch, dict) and 'data_samples' in data_batch:
                for ds in data_batch['data_samples']:
                    lab = None
                    if hasattr(ds, 'gt_sem_seg') and hasattr(ds.gt_sem_seg, 'data'):
                        lab = ds.gt_sem_seg.data
                    elif hasattr(ds, 'gt_semantic_seg') and hasattr(ds.gt_semantic_seg, 'data'):
                        lab = ds.gt_semantic_seg.data
                    if lab is None:
                        continue
                    lab_t = torch.as_tensor(lab)
                    if lab_t.dim() == 3 and lab_t.size(0) == 1:
                        lab_t = lab_t.squeeze(0)
                    invalid = (lab_t != ign) & ((lab_t < 0) | (lab_t >= num_classes))
                    if invalid.any():
                        lab_t = lab_t.clone(); lab_t[invalid] = ign
                    if hasattr(ds, 'gt_sem_seg') and hasattr(ds.gt_sem_seg, 'data'):
                        ds.gt_sem_seg.data = lab_t.long()
                    elif hasattr(ds, 'gt_semantic_seg') and hasattr(ds.gt_semantic_seg, 'data'):
                        ds.gt_semantic_seg.data = lab_t.long()
        except Exception:
            pass

        # 归一化（与数据流水线保持一致）
        normalize_mean = None
        normalize_std = None
        scale_255 = 1.0
        if self.normalize_cfg:
            normalize_mean = torch.tensor(self.normalize_cfg.get('mean'), dtype=images.dtype).view(1, C, 1, 1).to(self.device)
            normalize_std = torch.tensor(self.normalize_cfg.get('std'), dtype=images.dtype).view(1, C, 1, 1).to(self.device)
            if float(normalize_mean.max().item()) > 2.0 or float(normalize_std.max().item()) > 2.0:
                scale_255 = 255.0

        # 内循环优化补丁
        for _ in range(self.steps_per_iter):
            rgb_patch, mask_patch = self.patch_net()     # [3,ph,pw], [1,ph,pw]
            ph, pw = rgb_patch.shape[-2:]

            attack_obj_acc = 0.0
            reg_acc = 0.0

            # 多次 EOT 累积，提升可迁移性
            for _e in range(max(1, self.eot_samples)):
                # 对补丁做 EOT 变换（此处输出仍为 ph×pw，真正放入整图在 applier 完成）
                rgb_eot = random_eot_transform_batch(
                    patch=rgb_patch, batch_size=B, out_size=(ph, pw),
                    rot_degree=self.eot_rot, scale_min=self.eot_scale_min, scale_max=self.eot_scale_max,
                    max_translate=self.eot_max_translate, color_jitter=self.eot_color_jitter, device=self.device
                )
                mask_eot = random_eot_transform_batch(
                    patch=mask_patch, batch_size=B, out_size=(ph, pw),
                    rot_degree=self.eot_rot, scale_min=self.eot_scale_min, scale_max=self.eot_scale_max,
                    max_translate=self.eot_max_translate, color_jitter=0.0, device=self.device
                )

                # 贴到整图
                patched_images, patched_labels, placed_alpha = self.applier.apply_to_image_batch(
                    images, labels, rgb_eot, mask_eot, location_mode=self.location
                )
                patched_images = patched_images.clamp(0.0, 1.0)

                # 与流水线一致的归一化
                forward_images = patched_images
                if normalize_mean is not None and normalize_std is not None:
                    forward_images = (forward_images * scale_255 - normalize_mean) / normalize_std

                # 前向，拿到 logits
                runner.model.train()
                logits = None
                try:
                    outputs = runner.model(return_loss=False, img=forward_images)
                    logits = outputs
                except Exception:
                    try:
                        outputs = runner.model.module(return_loss=False, img=forward_images)
                        logits = outputs
                    except Exception:
                        try:
                            logits = runner.model.predict(forward_images, data_samples=None, mode='tensor')
                        except Exception:
                            return

                # 计算 APT 攻击目标与正则
                attack_obj, reg = self.criterion(logits, patched_labels, placed_alpha, rgb_patch, mask_patch)

                attack_obj_acc = attack_obj_acc + attack_obj
                reg_acc = reg_acc + reg

            # 求平均
            attack_obj_acc = attack_obj_acc / max(1, self.eot_samples)
            reg_acc = reg_acc / max(1, self.eot_samples)

            # 最终优化目标：最小化 loss = -attack_obj + reg
            loss = -attack_obj_acc + reg_acc

            # 只更新补丁
            self.patch_optimizer.zero_grad()
            model_flags = []
            if not self.update_model:
                for p in runner.model.parameters():
                    model_flags.append(p.requires_grad)
                    p.requires_grad = False
            loss.backward()
            self.patch_optimizer.step()
            if not self.update_model:
                for p, f in zip(runner.model.parameters(), model_flags):
                    p.requires_grad = f

            # 稳定参数范围（原始参数裁剪到 [-6,6]，对应 sigmoid 后接近 [0,1]）
            with torch.no_grad():
                self.patch_net.rgb_param.data.clamp_(-6.0, 6.0)
                self.patch_net.mask_param.data.clamp_(-6.0, 6.0)

        # 记录日志
        try:
            runner.log_buffer.update({
                'apt_attack_obj': float(attack_obj_acc.detach().cpu().item()),
                'apt_reg': float(reg_acc.detach().cpu().item())
            })
        except Exception:
            pass

    # ---------------- Save ----------------

    def after_train_epoch(self, runner):
        if not self.enabled:
            return
        rgb, mask = self.patch_net()
        rgb_img = self._denorm_patch_for_save(rgb)
        mask_arr = (mask.detach().cpu().squeeze(0).clamp(0, 1).numpy() * 255).astype('uint8')
        epoch = getattr(runner, 'epoch', 'final')
        Image.fromarray(rgb_img).save(os.path.join(self.save_dir, f'patch_epoch_{epoch}.png'))
        Image.fromarray(mask_arr).save(os.path.join(self.save_dir, f'patch_mask_epoch_{epoch}.png'))

    def after_run(self, runner):
        if not self.enabled:
            return
        rgb, mask = self.patch_net()
        rgb_img = self._denorm_patch_for_save(rgb)
        Image.fromarray(rgb_img).save(os.path.join(self.save_dir, 'final_patch.png'))
        mask_arr = (mask.detach().cpu().squeeze(0).clamp(0, 1).numpy() * 255).astype('uint8')
        Image.fromarray(mask_arr).save(os.path.join(self.save_dir, 'final_patch_mask.png'))
