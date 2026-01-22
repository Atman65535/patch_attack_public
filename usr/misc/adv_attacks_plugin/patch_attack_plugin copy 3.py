# mmseg/plugins/patch_attack_plugin.py
"""
Patch Attack plugin (APT-style, transferability-oriented)

Compatibility: mmseg 1.x PackSegInputs / data_samples structure
Only modify this plugin file (no changes to mmseg core).

Highlights:
- Robust label coercion from SegDataSample / PixelData / Tensor to [B,H,W] int64.
- Memory-friendly: eval() + freeze params during attack, FP16 AMP (bf16 off),
  attack-time downsample, micro-batch streaming backward, aggressive tensor cleanup.
- Forward dispatch without using 'mode=' kw (works across mmseg versions).
"""

from __future__ import annotations

import os
import math
from typing import Optional, Tuple, Dict, Any, List, Union

import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
import numpy as np

from mmengine.hooks import Hook
from mmengine.registry import HOOKS
from torch.cuda.amp import autocast

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
    """Construct 2x3 affine matrix for F.affine_grid (tx,ty in [-1,1])."""
    if isinstance(device, (str,)):
        device = torch.device(device)
    theta = torch.zeros(2, 3, device=device, dtype=torch.float32)
    rad = torch.tensor(angle_deg, device=device, dtype=theta.dtype) * (math.pi / 180.0)
    c = torch.cos(rad) * scale
    s = torch.sin(rad) * scale
    theta[0, 0] = c; theta[0, 1] = -s; theta[0, 2] = float(tx)
    theta[1, 0] = s; theta[1, 1] =  c; theta[1, 2] = float(ty)
    return theta


def _random_color_jitter(x: torch.Tensor, jitter: float) -> torch.Tensor:
    """Light color jitter for patches. x in [0,1]."""
    if jitter <= 0:
        return x
    b = x.new_empty(x.size(0), 1, 1, 1).uniform_(1 - jitter, 1 + jitter)
    a = x.new_empty(x.size(0), 1, 1, 1).uniform_(-jitter, jitter) * 0.05
    y = x * b + a
    return y.clamp(0.0, 1.0)


def random_eot_transform_batch(
    patch: torch.Tensor,                 # [C,p,p] or [1,p,p]
    batch_size: int,
    out_size: Tuple[int, int],
    rot_degree: float = 20.0,
    scale_min: float = 0.8,
    scale_max: float = 1.2,
    max_translate: float = 0.2,
    color_jitter: float = 0.1,
    device: Optional[Union[str, torch.device]] = None
) -> torch.Tensor:
    """Apply EOT transforms on patch and return [B,C,H_out,W_out]."""
    if device is None:
        device = patch.device
    if isinstance(device, str):
        device = torch.device(device)
    patch = patch.to(device)
    if patch.dim() == 2:
        patch = patch.unsqueeze(0)  # [1,p,p]
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
        warped = F.grid_sample(patch.unsqueeze(0), grid, mode='bilinear',
                               padding_mode='zeros', align_corners=False)  # [1,C,p,p]
        warped = _random_color_jitter(warped, color_jitter)
        ph, pw = min(H_out, p), min(W_out, p)
        wr = F.interpolate(warped, size=(ph, pw), mode='bilinear', align_corners=False) if (ph != p or pw != p) else warped
        y0 = (H_out - ph) // 2; x0 = (W_out - pw) // 2
        out[i, :, y0:y0+ph, x0:x0+pw] = wr[0]
    return out


# ----------------------------- Modules -----------------------------

class LearnablePatch(nn.Module):
    """Learnable patch: rgb + alpha mask in logits (sigmoid applied in forward)."""
    def __init__(self, patch_size: int = 64, init: str = 'rand'):
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
    Apply patch to image batch.
    Returns (patched_images, patched_labels, placed_alpha)
    - placed_alpha: [B,1,H,W] alpha map of placed patch area
    """
    def __init__(self, ignore_label: int = 255):
        self.ignore_label = ignore_label

    def apply_to_image_batch(
        self,
        images: torch.Tensor,       # [B,C,H,W] in [0,1]
        labels: torch.Tensor,       # [B,H,W] (long)
        rgb_patch: torch.Tensor,    # [B,3,ph,pw] or [3,ph,pw]
        mask: torch.Tensor,         # [B,1,ph,pw] or [1,ph,pw]
        location_mode: Any = 'random'
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        B, _, H, W = images.shape

        # unify batch for rgb_patch
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
                x = (W - pw) // 2; y = (H - ph) // 2
            elif isinstance(location_mode, (tuple, list)):
                x, y = int(location_mode[0]), int(location_mode[1])
                x = max(0, min(W - pw, x)); y = max(0, min(H - ph, y))
            else:
                x = int(torch.randint(0, max(1, W - pw + 1), (1,)).item())
                y = int(torch.randint(0, max(1, H - ph + 1), (1,)).item())

            alpha = mask_b[i]                   # [1,ph,pw]
            rgb   = rgb_b[i]                    # [3,ph,pw]
            region = imgs[i, :, y:y+ph, x:x+pw] # [C,ph,pw]
            if region.shape[1] != ph or region.shape[2] != pw:
                ph2 = region.shape[1]; pw2 = region.shape[2]
                alpha = alpha[:, :ph2, :pw2]; rgb = rgb[:, :ph2, :pw2]
            blended = alpha.repeat(3, 1, 1) * rgb + (1.0 - alpha.repeat(3, 1, 1)) * region
            imgs[i, :, y:y+ph, x:x+pw] = blended

            labs[i, y:y+ph, x:x+pw] = self.ignore_label
            placed_alpha[i, :, y:y+ph, x:x+pw] = alpha

        return imgs, labs, placed_alpha


# ----------------------------- Loss (APT style) -----------------------------

class PatchLossAPT:
    """
    APT objective (maximize):
      attack_obj = w_ce * TopK_CE + w_ent * Entropy
    Final loss (minimize):
      loss = -attack_obj + lambda_tv*TV + lambda_l2*||patch||^2 + lambda_mask_l1*||mask||_1
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
        return (labels != ignore_index)

    def _maybe_restrict_region(
        self, valid: torch.Tensor, placed_alpha: Optional[torch.Tensor]
    ) -> torch.Tensor:
        if self.region == 'in_patch' and placed_alpha is not None:
            in_patch = placed_alpha.squeeze(1) > 1e-6  # [B,H,W]
            return valid & in_patch
        return valid

    # ---------- Robust label coercion ----------
    @staticmethod
    def _coerce_labels(labels_in: Any, device: torch.device, ignore_index: int) -> torch.Tensor:
        """
        Convert various carriers to [B,H,W] LongTensor on `device`.
        Accepts: torch.Tensor | list/tuple of above | SegDataSample | PixelData | dict with 'data'.
        """
        # 1) Tensor directly
        if isinstance(labels_in, torch.Tensor):
            lab = labels_in.to(device=device, dtype=torch.long, non_blocking=True)
            if lab.dim() == 4 and lab.size(1) == 1:
                lab = lab.squeeze(1)   # [B,1,H,W] -> [B,H,W]
            return lab

        # 2) List/Tuple (e.g., list of per-sample tensors or data samples)
        if isinstance(labels_in, (list, tuple)):
            stacked: List[torch.Tensor] = []
            for it in labels_in:
                stacked.append(PatchLossAPT._coerce_labels(it, device, ignore_index))
            return torch.stack(stacked, dim=0) if stacked[0].dim() == 2 else torch.cat(stacked, dim=0)

        # 3) SegDataSample / PixelData-like
        # Try common attributes in order:
        for attr in ('gt_sem_seg', 'gt_semantic_seg', 'sem_seg'):
            if hasattr(labels_in, attr):
                obj = getattr(labels_in, attr)
                data = getattr(obj, 'data', obj)  # PixelData.data or raw
                t = torch.as_tensor(data, device=device).long()
                if t.dim() == 3 and t.size(0) == 1:  # [1,H,W] -> [H,W]
                    t = t.squeeze(0)
                if t.dim() == 2:
                    t = t.unsqueeze(0)              # -> [1,H,W]
                return t

        # 4) dict-like with 'data'
        if isinstance(labels_in, dict) and 'data' in labels_in:
            t = torch.as_tensor(labels_in['data'], device=device).long()
            if t.dim() == 4 and t.size(1) == 1:
                t = t.squeeze(1)
            if t.dim() == 3 and t.size(0) == 1:
                t = t.squeeze(0)
            if t.dim() == 2:
                t = t.unsqueeze(0)
            return t

        raise TypeError(f'Unsupported labels type for coercion: {type(labels_in)}')

    # ---------- Objectives ----------
    def attack_obj_only(
        self,
        logits: torch.Tensor,         # [B,C,H,W]
        labels: Any,                  # robust
        placed_alpha: Optional[torch.Tensor],
    ) -> torch.Tensor:
        labels = self._coerce_labels(labels, device=logits.device, ignore_index=self.ignore_index)

        if logits.shape[-2:] != labels.shape[-2:]:
            logits = F.interpolate(logits, size=labels.shape[-2:], mode='bilinear', align_corners=False)

        num_classes = logits.size(1)
        labels = labels.long().contiguous()
        invalid = (labels != self.ignore_index) & ((labels < 0) | (labels >= num_classes))
        if invalid.any():
            labels = labels.clone()
            labels[invalid] = self.ignore_index

        valid = self._valid_mask_from_labels(labels, self.ignore_index)
        if placed_alpha is not None and isinstance(placed_alpha, torch.Tensor):
            valid = self._maybe_restrict_region(valid, placed_alpha)

        ce_map = F.cross_entropy(logits, labels, ignore_index=self.ignore_index, reduction='none')  # [B,H,W]
        ce_topk = self._topk_mean(ce_map, valid, self.topk_ratio)

        probs = F.softmax(logits, dim=1) + 1e-12
        ent_map = -(probs * probs.log()).sum(dim=1)  # [B,H,W]
        ent_mean = ent_map[valid].mean() if valid.any() else ent_map.mean()

        return self.w_ce * ce_topk + self.w_ent * ent_mean

    def _topk_mean(self, x: torch.Tensor, mask: torch.Tensor, ratio: float) -> torch.Tensor:
        vals = x[mask]
        if vals.numel() == 0:
            return x.new_tensor(0.0, device=x.device)
        k = max(1, int(math.ceil(ratio * vals.numel())))
        topk = torch.topk(vals, k=k, largest=True).values
        return topk.mean()

    def reg_only(
        self,
        patch_rgb: torch.Tensor,      # [3,ph,pw] or [B,3,ph,pw]
        patch_mask: torch.Tensor      # [1,ph,pw] or [B,1,ph,pw]
    ) -> torch.Tensor:
        pr = patch_rgb.mean(dim=0) if patch_rgb.dim() == 4 else patch_rgb
        pm = patch_mask.mean(dim=0)  if patch_mask.dim() == 4 else patch_mask
        tv = total_variation(pr.unsqueeze(0))
        l2 = torch.sum(pr ** 2)
        mask_l1 = torch.sum(torch.abs(pm))
        return self.lambda_tv * tv + self.lambda_l2 * l2 + self.lambda_mask_l1 * mask_l1


# ----------------------------- Hook -----------------------------

@HOOKS.register_module()
class PatchAttackHook(Hook):
    """Register in `custom_hooks`.

    Extra cfg keys:
      attack_downsample: float in (0,1], default 1.0  (temporary downscale for attack forward)
      attack_microbatch: int >=1, default 1           (micro-batch size during attack)
    """
    def __init__(self, cfg: Dict[str, Any], device: Optional[Union[str, torch.device]] = None):
        super().__init__()
        cfg = cfg or {}
        self.enabled = cfg.get('enabled', True)
        if not self.enabled:
            return

        self.device = torch.device(device) if device is not None else (
            torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu'))
        self.patch_size = int(cfg.get('patch_size', 64))
        self.lr = float(cfg.get('lr', 0.3))
        self.steps_per_iter = int(cfg.get('steps_per_iter', 1))

        eot_cfg = cfg.get('eot', {})
        self.eot_rot = float(eot_cfg.get('rot', 20.0))
        self.eot_scale_min = float(eot_cfg.get('scale_min', 0.9))
        self.eot_scale_max = float(eot_cfg.get('scale_max', 1.1))
        self.eot_max_translate = float(eot_cfg.get('max_translate', 0.15))
        self.eot_color_jitter = float(eot_cfg.get('color_jitter', 0.1))
        self.eot_samples = int(eot_cfg.get('samples', 1))

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
        self.attack_downsample = float(cfg.get('attack_downsample', 1.0))
        self.attack_microbatch = int(cfg.get('attack_microbatch', 1))

        # modules
        self.patch_net = LearnablePatch(self.patch_size, init=cfg.get('init', 'rand')).to(self.device)
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
        """Support both legacy and new PackSegInputs-like structures.
        Return images[B,C,H,W] in [0,1], labels[B,H,W] long.
        """
        if isinstance(data_batch, dict) and 'inputs' in data_batch and 'data_samples' in data_batch:
            inputs = data_batch['inputs']
            # unify to [B,C,Hmax,Wmax]
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
                images = torch.as_tensor(inputs, dtype=torch.float32, device=self.device)

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
                lab_t = F.pad(lab_t.unsqueeze(0), (0, pw, 0, ph), mode='constant',
                              value=self.ignore_label).squeeze(0).long()
                labels_list.append(lab_t)
            labels = torch.stack(labels_list, dim=0).to(self.device)
            return images.clamp(0, 1), labels

        # Fallback legacy format
        if isinstance(data_batch, dict) and 'img' in data_batch and 'gt_semantic_seg' in data_batch:
            images = torch.as_tensor(data_batch['img'], dtype=torch.float32, device=self.device)
            labels = torch.as_tensor(data_batch['gt_semantic_seg'], device=self.device).long()
            if labels.dim() == 4 and labels.size(1) == 1:
                labels = labels.squeeze(1)
            return images.clamp(0, 1), labels

        raise KeyError(f'Unsupported data_batch format. Keys: {list(data_batch.keys())}')

    def before_run(self, runner):
        if not self.enabled:
            return
        try:
            runner.model.to(self.device)
        except Exception:
            pass

    # ---------------- helper to extract logits robustly ----------------
    def _extract_logits_from_model_output(self, outputs):
        if isinstance(outputs, torch.Tensor):
            return outputs
        if isinstance(outputs, (list, tuple)):
            for o in outputs:
                if isinstance(o, torch.Tensor) and o.dim() == 4:
                    return o
            for o in outputs:
                if isinstance(o, torch.Tensor):
                    return o
            return outputs[0]
        if isinstance(outputs, dict):
            for k in ['seg_logits', 'logits', 'pred_logits', 'pred_sem_seg', 'out', 'outputs', 'seg_pred']:
                if k in outputs and isinstance(outputs[k], torch.Tensor):
                    return outputs[k]
            for v in outputs.values():
                if isinstance(v, torch.Tensor) and v.dim() == 4:
                    return v
        raise RuntimeError('Unable to parse model outputs into logits tensor.')

    def _forward_get_outputs(self, model, fwd_input):
        """Try several signatures; never pass 'mode=' kw to keep wide compatibility."""
        try:
            return model(return_loss=False, img=fwd_input)
        except Exception:
            pass
        try:
            return model.module(return_loss=False, img=fwd_input)
        except Exception:
            pass
        try:
            return model.predict(fwd_input, data_samples=None)
        except Exception:
            pass
        try:
            return model(fwd_input)  # usually BaseSegmentor.forward -> _forward (tensor mode)
        except Exception:
            pass
        try:
            return model._forward(fwd_input, data_samples=None)
        except Exception:
            pass
        raise RuntimeError('Model forward path not found.')

    # ---------------- Main step (after each train iter) ----------------

    def after_train_iter(self, runner, batch_idx: int, data_batch=None, **kwargs):
        if not self.enabled:
            return
        if data_batch is None:
            data_batch = getattr(runner, 'data_batch', None)
            if data_batch is None:
                return

        images, labels = self._prepare_images_labels(data_batch)
        self._attack_step_on_batch(runner, images, labels, batch_idx)

        # lightweight placeholders (true values updated inside attack step)
        try:
            runner.log_buffer.update({'apt_attack_obj': 0.0, 'apt_reg': 0.0})
        except Exception:
            pass

    # ---------------- Attack core (micro-batched, streaming backward) ----------------

    def _attack_step_on_batch(self, runner, images, labels, batch_idx: int):
        B, C, H, W = images.shape

        # normalization config
        normalize_mean = None
        normalize_std = None
        scale_255 = 1.0
        if self.normalize_cfg:
            normalize_mean = torch.tensor(self.normalize_cfg.get('mean'),
                                          dtype=images.dtype, device=self.device).view(1, C, 1, 1)
            normalize_std = torch.tensor(self.normalize_cfg.get('std'),
                                         dtype=images.dtype, device=self.device).view(1, C, 1, 1)
            if float(normalize_mean.max().item()) > 2.0 or float(normalize_std.max().item()) > 2.0:
                scale_255 = 255.0

        model = runner.model
        try:
            was_training = model.training
        except Exception:
            was_training = True

        model_flags = []
        if not self.update_model:
            try:
                for p in model.parameters():
                    model_flags.append(p.requires_grad)
                    p.requires_grad = False
            except Exception:
                model_flags = []

        # === inner steps ===
        for _ in range(self.steps_per_iter):
            rgb_patch, mask_patch = self.patch_net()     # [3,ph,pw], [1,ph,pw]
            self.patch_optimizer.zero_grad(set_to_none=True)

            use_amp = torch.cuda.is_available()
            amp_dtype = torch.float16  # force fp16 to avoid bf16 interpolate issues

            num_chunks = max(1, (B + self.attack_microbatch - 1) // self.attack_microbatch)
            attack_obj_meter = 0.0

            for _e in range(max(1, self.eot_samples)):
                for ci in range(num_chunks):
                    s = ci * self.attack_microbatch
                    e = min(B, (ci + 1) * self.attack_microbatch)
                    if e <= s:
                        continue
                    mb = e - s

                    # EOT transforms per micro-batch
                    rgb_eot = random_eot_transform_batch(
                        patch=rgb_patch, batch_size=mb, out_size=rgb_patch.shape[-2:],
                        rot_degree=self.eot_rot, scale_min=self.eot_scale_min,
                        scale_max=self.eot_scale_max, max_translate=self.eot_max_translate,
                        color_jitter=self.eot_color_jitter, device=self.device
                    )
                    mask_eot = random_eot_transform_batch(
                        patch=mask_patch, batch_size=mb, out_size=mask_patch.shape[-2:],
                        rot_degree=self.eot_rot, scale_min=self.eot_scale_min,
                        scale_max=self.eot_scale_max, max_translate=self.eot_max_translate,
                        color_jitter=0.0, device=self.device
                    )

                    # apply to images/labels
                    imgs_mb, labs_mb, alpha_mb = self.applier.apply_to_image_batch(
                        images[s:e], labels[s:e], rgb_eot, mask_eot, location_mode=self.location
                    )
                    imgs_mb = imgs_mb.clamp(0.0, 1.0)

                    # normalize
                    fwd_input = imgs_mb
                    if normalize_mean is not None and normalize_std is not None:
                        fwd_input = (fwd_input * scale_255 - normalize_mean) / normalize_std

                    if self.attack_downsample < 1.0:
                        hh = max(1, int(fwd_input.shape[-2] * self.attack_downsample))
                        ww = max(1, int(fwd_input.shape[-1] * self.attack_downsample))
                        fwd_input = F.interpolate(fwd_input, size=(hh, ww), mode='bilinear', align_corners=False)

                    # forward (no grads for model)
                    model.eval()
                    with autocast(enabled=use_amp, dtype=amp_dtype):
                        outputs = self._forward_get_outputs(model, fwd_input)
                    logits = self._extract_logits_from_model_output(outputs)

                    # attack objective on this chunk
                    attack_obj_chunk = self.criterion.attack_obj_only(logits, labs_mb, alpha_mb)
                    loss_chunk = -(attack_obj_chunk) / float(max(1, self.eot_samples) * num_chunks)
                    loss_chunk.backward()  # grads accumulate on patch params

                    attack_obj_meter += float(attack_obj_chunk.detach().cpu().item()) * (mb / B)

                    # cleanup
                    del outputs, logits, imgs_mb, fwd_input, rgb_eot, mask_eot, labs_mb, alpha_mb
                    if (batch_idx % 25) == 0 and (ci % 4) == 0:
                        torch.cuda.empty_cache()

            # regularizers (single backward to keep graph light)
            reg = self.criterion.reg_only(rgb_patch, mask_patch)
            reg.backward()

            # patch update
            self.patch_optimizer.step()

            # clamp param logits to keep sigmoid in reasonable range (stability)
            with torch.no_grad():
                self.patch_net.rgb_param.data.clamp_(-6.0, 6.0)
                self.patch_net.mask_param.data.clamp_(-6.0, 6.0)

            if (batch_idx % 50) == 0:
                torch.cuda.empty_cache()

            # logging for this step
            try:
                runner.log_buffer.update({
                    'apt_attack_obj': float(attack_obj_meter) / max(1, self.eot_samples),
                    'apt_reg': float(reg.detach().cpu().item())
                })
            except Exception:
                pass

        # restore model flags
        try:
            if was_training is not None:
                model.train() if was_training else model.eval()
        except Exception:
            pass
        if not self.update_model and model_flags:
            try:
                for p, f in zip(model.parameters(), model_flags):
                    p.requires_grad = f
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
