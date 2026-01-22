# plugins/patch_attack_plugin.py
import os, math
from typing import ional, Tuple, Dict, Any, List, Union

import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from mmengine.hooks import Hook
from mmengine.registry import HOOKS
from torch.cuda.amp import autocast

print('[PatchAttackHook] loaded from:', __file__)

# ---------------- Utils ----------------

def total_variation(x: torch.Tensor) -> torch.Tensor:
    if x.dim() == 3:
        x = x.unsqueeze(0)
    tv_h = torch.abs(x[:, :, 1:, :] - x[:, :, :-1, :]).sum()
    tv_w = torch.abs(x[:, :, :, 1:] - x[:, :, :, :-1]).sum()
    return tv_h + tv_w

def make_affine_matrix(angle_deg: float, scale: float, tx: float, ty: float, device):
    theta = torch.zeros(2, 3, device=device, dtype=torch.float32)
    rad = torch.tensor(angle_deg, device=device, dtype=theta.dtype) * (math.pi / 180.0)
    c = torch.cos(rad) * scale
    s = torch.sin(rad) * scale
    theta[0, 0] = c;      theta[0, 1] = -s;     theta[0, 2] = float(tx)
    theta[1, 0] = s;      theta[1, 1] = c;      theta[1, 2] = float(ty)
    return theta

def random_eot_transform_batch(
    patch: torch.Tensor, batch_size: int, out_size: Tuple[int, int],
    rot_degree: float, scale_min: float, scale_max: float, max_translate: float,
    color_jitter: float, device
) -> torch.Tensor:
    """给补丁做 EOT 变换并返回 [B,C,H_out,W_out]"""
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
        if color_jitter > 0 and C == 3:
            b = warped.new_empty(1, 1, 1, 1).uniform_(1 - color_jitter, 1 + color_jitter)
            a = warped.new_empty(1, 1, 1, 1).uniform_(-color_jitter, color_jitter) * 0.05
            warped = (warped * b + a).clamp(0, 1)

        ph, pw = min(H_out, p), min(W_out, p)
        wr = F.interpolate(warped, size=(ph, pw), mode='bilinear', align_corners=False) if (ph != p or pw != p) else warped
        y0 = (H_out - ph) // 2; x0 = (W_out - pw) // 2
        out[i, :, y0:y0+ph, x0:x0+pw] = wr[0]
    return out

# -------------- Modules --------------

class LearnablePatch(nn.Module):
    """可学习补丁：RGB + Alpha（mask logits -> sigmoid）"""
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
        rgb = torch.sigmoid(self.rgb_param)    # [3,H,W] in (0,1)
        mask = torch.sigmoid(self.mask_param)  # [1,H,W] in (0,1)
        return rgb, mask

class PatchApplier:
    """把补丁贴到 batch 上，并把贴上的标签置为 ignore"""
    def __init__(self, ignore_label: int = 255):
        self.ignore_label = ignore_label

    def apply(
        self, images: torch.Tensor, labels: torch.Tensor,
        rgb_patch: torch.Tensor, mask: torch.Tensor, location: Union[str, tuple] = 'random'
    ):
        B, C, H, W = images.shape
        if rgb_patch.dim() == 3:
            ph, pw = rgb_patch.shape[-2:]
            rgb_b = rgb_patch.unsqueeze(0).repeat(B, 1, 1, 1)
        else:
            rgb_b = rgb_patch; ph, pw = rgb_patch.shape[-2:]
        mask_b = mask.unsqueeze(0).repeat(B, 1, 1, 1) if mask.dim() == 3 else mask

        imgs = images.clone()
        labs = labels.clone().long()
        if labs.dim() == 4 and labs.size(1) == 1:
            labs = labs.squeeze(1)
        placed_alpha = torch.zeros(B, 1, H, W, dtype=imgs.dtype, device=imgs.device)

        for i in range(B):
            if location == 'center':
                x = (W - pw) // 2; y = (H - ph) // 2
            elif isinstance(location, (tuple, list)):
                x, y = int(location[0]), int(location[1])
                x = max(0, min(W - pw, x)); y = max(0, min(H - ph, y))
            else:  # random
                x = int(torch.randint(0, max(1, W - pw + 1), (1,)).item())
                y = int(torch.randint(0, max(1, H - ph + 1), (1,)).item())

            alpha = mask_b[i]  # [1,ph,pw]
            rgb   = rgb_b[i]   # [3,ph,pw]
            region = imgs[i, :, y:y+ph, x:x+pw]
            ph2, pw2 = region.shape[-2:]
            if ph2 != ph or pw2 != pw:  # 边界裁剪
                alpha = alpha[:, :ph2, :pw2]; rgb = rgb[:, :ph2, :pw2]
            blended = alpha.repeat(3, 1, 1) * rgb + (1.0 - alpha.repeat(3, 1, 1)) * region
            imgs[i, :, y:y+ph2, x:x+pw2] = blended
            labs[i, y:y+ph2, x:x+pw2] = self.ignore_label
            placed_alpha[i, :, y:y+ph2, x:x+pw2] = alpha[:, :ph2, :pw2]

        return imgs.clamp(0, 1), labs, placed_alpha

# -------------- Loss (APT-style) --------------

class PatchLossAPT:
    """attack_obj = w_ce * TopK_CE + w_ent * Entropy；最终 loss = -attack_obj + 正则"""
    def __init__(
        self, ignore_index=255, topk_ratio=0.2, w_ce=0.5, w_ent=0.5,
        region='all', lambda_tv=1e-4, lambda_l2=1e-6, lambda_mask_l1=1e-3
    ):
        self.ignore_index = ignore_index
        self.topk_ratio = float(topk_ratio)
        self.w_ce = float(w_ce)
        self.w_ent = float(w_ent)
        self.region = region
        self.lambda_tv = lambda_tv
        self.lambda_l2 = lambda_l2
        self.lambda_mask_l1 = lambda_mask_l1

    def _valid_mask(self, labels):
        return (labels != self.ignore_index)

    def _restrict_region(self, valid, placed_alpha):
        if self.region == 'in_patch' and placed_alpha is not None:
            in_patch = placed_alpha.squeeze(1) > 1e-6
            return valid & in_patch
        return valid

    def _topk_mean(self, x: torch.Tensor, mask: torch.Tensor, ratio: float) -> torch.Tensor:
        vals = x[mask]
        if vals.numel() == 0:
            return x.new_tensor(0.0, device=x.device)
        k = max(1, int(math.ceil(ratio * vals.numel())))
        return torch.topk(vals, k=k, largest=True).values.mean()

    def attack_obj_only(self, logits, labels, placed_alpha=None):
        # 尺寸对齐
        if logits.shape[-2:] != labels.shape[-2:]:
            logits = F.interpolate(logits, size=labels.shape[-2:], mode='bilinear', align_corners=False)
        num_classes = logits.size(1)
        labels = labels.long().contiguous()
        invalid = (labels != self.ignore_index) & ((labels < 0) | (labels >= num_classes))
        if invalid.any():
            labels = labels.clone(); labels[invalid] = self.ignore_index

        valid = self._restrict_region(self._valid_mask(labels), placed_alpha)
        ce_map = F.cross_entropy(logits, labels, ignore_index=self.ignore_index, reduction='none')  # [B,H,W]
        ce_topk = self._topk_mean(ce_map, valid, self.topk_ratio)
        probs = F.softmax(logits, dim=1).clamp_min(1e-12)
        ent_map = -(probs * probs.log()).sum(dim=1)
        ent_mean = ent_map[valid].mean() if valid.any() else ent_map.mean()
        return self.w_ce * ce_topk + self.w_ent * ent_mean

    def reg_terms(self, patch_rgb, patch_mask):
        pr = patch_rgb.mean(dim=0) if patch_rgb.dim() == 4 else patch_rgb
        pm = patch_mask.mean(dim=0) if patch_mask.dim() == 4 else patch_mask
        tv = total_variation(pr.unsqueeze(0))
        l2 = torch.sum(pr ** 2)
        mask_l1 = torch.sum(torch.abs(pm))
        return self.lambda_tv * tv + self.lambda_l2 * l2 + self.lambda_mask_l1 * mask_l1

# -------------- Hook --------------

@HOOKS.register_module()
class PatchAttackHook(Hook):
    """
    在训练过程中每个 iter 追加一次“补丁更新”步骤，只更新补丁参数，不更新模型参数。
    """
    def __init__(self, cfg: Dict[str, Any]):
        super().__init__()
        self.enabled = cfg.get('enabled', True)
        if not self.enabled:
            return
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.patch_size = int(cfg.get('patch_size', 64))
        self.lr = float(cfg.get('lr', 0.3))
        self.steps_per_iter = int(cfg.get('steps_per_iter', 1))

        eot = cfg.get('eot', {})
        self.eot_rot = float(eot.get('rot', 20.0))
        self.eot_scale_min = float(eot.get('scale_min', 0.9))
        self.eot_scale_max = float(eot.get('scale_max', 1.1))
        self.eot_translate = float(eot.get('max_translate', 0.15))
        self.eot_color = float(eot.get('color_jitter', 0.1))
        self.eot_samples = int(eot.get('samples', 1))

        self.location = cfg.get('location', 'random')
        self.ignore_label = int(cfg.get('ignore_label', 255))
        self.normalize = cfg.get('normalize', None)  # 如果 mmseg pipeline 有 Normalize，传 mean/std（像素值）
        self.update_model = bool(cfg.get('update_model', False))
        self.attack_downsample = float(cfg.get('attack_downsample', 1.0))  # 前向临时降采样
        self.attack_microbatch = int(cfg.get('attack_microbatch', 1))      # 以更小子批次做前向/loss，省显存
        self.save_dir = cfg.get('save_dir', './work_dirs/patches')
        os.makedirs(self.save_dir, exist_ok=True)

        self.patch = LearnablePatch(self.patch_size, init=cfg.get('init', 'rand')).to(self.device)
        self.applier = PatchApplier(self.ignore_label)
        loss_cfg = cfg.get('loss', {})
        self.criterion = PatchLossAPT(
            ignore_index=self.ignore_label,
            topk_ratio=float(loss_cfg.get('topk_ratio', 0.2)),
            w_ce=float(loss_cfg.get('w_ce', 0.5)),
            w_ent=float(loss_cfg.get('w_ent', 0.5)),
            region=str(loss_cfg.get('region', 'all')),
            lambda_tv=float(cfg.get('reg', {}).get('lambda_tv', 1e-4)),
            lambda_l2=float(cfg.get('reg', {}).get('lambda_l2', 1e-6)),
            lambda_mask_l1=float(cfg.get('reg', {}).get('lambda_mask_l1', 1e-3)),
        )
        self.opt = torch.optim.Adam(self.patch.parameters(), lr=self.lr)

    def _prepare_batch(self, data_batch):
        """
        兼容 PackSegInputs：data_batch['inputs'] + data_batch['data_samples']
        返回 images[B,C,H,W] in [0,1]，labels[B,H,W] (long)
        """
        if isinstance(data_batch, dict) and 'inputs' in data_batch and 'data_samples' in data_batch:
            x = data_batch['inputs']
            if isinstance(x, (list, tuple)):  # [B, C,H,W] 或 [B, H,W,3]
                tensor_list = []
                Hmax, Wmax = 0, 0
                for item in x:
                    t = torch.as_tensor(item)
                    if t.dim() == 3 and t.shape[-1] == 3 and t.shape[0] != 3:
                        t = t.permute(2, 0, 1).contiguous()
                    t = t.float()
                    tensor_list.append(t)
                    Hmax, Wmax = max(Hmax, t.shape[1]), max(Wmax, t.shape[2])
                padded = [F.pad(t, (0, Wmax - t.shape[2], 0, Hmax - t.shape[1]), value=0.0) for t in tensor_list]
                images = torch.stack(padded, dim=0).to(self.device)
            else:
                images = torch.as_tensor(x).float().to(self.device)

            labels = []
            for ds in data_batch['data_samples']:
                if hasattr(ds, 'gt_sem_seg') and hasattr(ds.gt_sem_seg, 'data'):
                    lab = ds.gt_sem_seg.data
                elif hasattr(ds, 'gt_semantic_seg') and hasattr(ds.gt_semantic_seg, 'data'):
                    lab = ds.gt_semantic_seg.data
                else:
                    raise KeyError('gt_sem_seg not found')
                lab = torch.as_tensor(lab)
                if lab.dim() == 3 and lab.size(0) == 1:
                    lab = lab.squeeze(0)
                ph = images.shape[-2] - lab.shape[-2]
                pw = images.shape[-1] - lab.shape[-1]
                lab = F.pad(lab.unsqueeze(0), (0, pw, 0, ph), value=self.ignore_label).squeeze(0)
                labels.append(lab)
            labels = torch.stack(labels, dim=0).long().to(self.device)
            return images, labels

        if isinstance(data_batch, dict) and 'img' in data_batch and 'gt_semantic_seg' in data_batch:
            images = data_batch['img'].float().to(self.device)
            labels = torch.as_tensor(data_batch['gt_semantic_seg']).long().to(self.device)
            if labels.dim() == 4 and labels.size(1) == 1:
                labels = labels.squeeze(1)
            return images, labels

        raise KeyError(f'Unsupported data_batch keys: {list(data_batch.keys())}')

    def _forward_get_logits(self, model, fwd_input):
        """
        兼容多种 mmseg 前向形式，返回 [B,C,H,W] logits
        不用 bfloat16（很多算子不支持），只用 fp16 AMP（可用时）。
        """
        use_amp = torch.cuda.is_available()
        with autocast(enabled=use_amp, dtype=torch.float16):
            # 优先 mmseg 的 call 方式
            try:
                out = model(return_loss=False, img=fwd_input)
            except Exception:
                try:
                    out = model.module(return_loss=False, img=fwd_input)
                except Exception:
                    try:
                        # 一些 Segmentor 支持 model(img, mode='tensor')
                        out = model(img=fwd_input, mode='tensor')
                    except Exception:
                        out = model(fwd_input)  # 最后兜底

        # 提取 logits
        if isinstance(out, torch.Tensor) and out.dim() == 4:
            return out
        if isinstance(out, (list, tuple)):
            for o in out:
                if isinstance(o, torch.Tensor) and o.dim() == 4:
                    return o
            return out[0]
        if isinstance(out, dict):
            for k in ['seg_logits', 'logits', 'pred_logits', 'pred_sem_seg', 'out', 'outputs', 'seg_pred']:
                if k in out and isinstance(out[k], torch.Tensor):
                    v = out[k]
                    return v if v.dim() == 4 else v
            for v in out.values():
                if isinstance(v, torch.Tensor) and v.dim() == 4:
                    return v
        raise RuntimeError('Cannot extract logits from model output.')

    # ---- Hook entry points ----

    def after_train_iter(self, runner, batch_idx: int, data_batch=None, **kwargs):
        if not self.enabled:
            return
        if data_batch is None:
            data_batch = getattr(runner, 'data_batch', None)
            if data_batch is None:
                return

        images, labels = self._prepare_batch(data_batch)
        B, C, H, W = images.shape

        # 取出补丁
        rgb_patch, mask_patch = self.patch()
        ph, pw = rgb_patch.shape[-2:]

        # 统计
        attack_obj_acc = torch.tensor(0.0, device=self.device)
        reg_acc = torch.tensor(0.0, device=self.device)

        # 多次 EOT 样本
        for _ in range(max(1, self.eot_samples)):
            rgb_eot = random_eot_transform_batch(
                rgb_patch, B, (ph, pw), self.eot_rot, self.eot_scale_min, self.eot_scale_max,
                self.eot_translate, self.eot_color, self.device
            )
            mask_eot = random_eot_transform_batch(
                mask_patch, B, (ph, pw), self.eot_rot, self.eot_scale_min, self.eot_scale_max,
                self.eot_translate, 0.0, self.device
            )

            # 贴补丁 + 把补丁区域标签置为 ignore
            patched_imgs, patched_labs, placed_alpha = self.applier.apply(
                images, labels, rgb_eot, mask_eot, location=self.location
            )

            # 和 pipeline 的 Normalize 对齐（如果启用）
            fwd_imgs = patched_imgs
            if self.normalize is not None:
                mean = torch.tensor(self.normalize['mean'], dtype=fwd_imgs.dtype, device=self.device).view(1, C, 1, 1)
                std  = torch.tensor(self.normalize['std'],  dtype=fwd_imgs.dtype, device=self.device).view(1, C, 1, 1)
                scale_255 = 255.0 if (mean.max().item() > 2.0 or std.max().item() > 2.0) else 1.0
                fwd_imgs = (fwd_imgs * scale_255 - mean) / std

            # 可选：前向临时降采样（显著降显存）
            if self.attack_downsample < 1.0:
                nh = max(1, int(fwd_imgs.shape[-2] * self.attack_downsample))
                nw = max(1, int(fwd_imgs.shape[-1] * self.attack_downsample))
                fwd_imgs = F.interpolate(fwd_imgs, size=(nh, nw), mode='bilinear', align_corners=False)

            # 冻结模型参数，但不要 no_grad（需要对输入/补丁求导）
            model = runner.model
            req_flags = []
            if not self.update_model:
                try:
                    for p in model.parameters():
                        req_flags.append(p.requires_grad)
                        p.requires_grad = False
                except Exception:
                    req_flags = []
            was_training = getattr(model, 'training', True)
            model.eval()

            # —— 显存友好的 micro-batch 前向 —— #
            mb = max(1, self.attack_microbatch)
            attack_obj_eot = 0.0
            for s in range(0, B, mb):
                e = min(B, s + mb)
                fwd_input = fwd_imgs[s:e]
                labs_mb = patched_labs[s:e]
                alpha_mb = placed_alpha[s:e]
                logits = self._forward_get_logits(model, fwd_input)
                attack_obj_chunk = self.criterion.attack_obj_only(logits, labs_mb, alpha_mb)
                attack_obj_eot = attack_obj_eot + attack_obj_chunk

            attack_obj_eot = attack_obj_eot / math.ceil(B / mb)
            attack_obj_acc = attack_obj_acc + attack_obj_eot
            reg_acc = reg_acc + self.criterion.reg_terms(rgb_patch, mask_patch)

            # 恢复状态
            if not self.update_model and req_flags:
                try:
                    for p, f in zip(model.parameters(), req_flags):
                        p.requires_grad = f
                except Exception:
                    pass
            if was_training:
                model.train()
            else:
                model.eval()

            del patched_imgs, fwd_imgs, patched_labs, placed_alpha  # 释放局部显存

        # EOT 平均
        attack_obj_acc = attack_obj_acc / max(1, self.eot_samples)
        reg_acc = reg_acc / max(1, self.eot_samples)
        loss = -attack_obj_acc + reg_acc

        # 更新补丁
        self.opt.zero_grad(set_to_none=True)
        loss.backward()
        self.opt.step()

        # 保持 patch 参数稳定
        with torch.no_grad():
            self.patch.rgb_param.data.clamp_(-6.0, 6.0)
            self.patch.mask_param.data.clamp_(-6.0, 6.0)

        # 轻量日志（数值在 runner 日志里可见）
        try:
            runner.log_buffer.update({
                'apt_attack_obj': float(attack_obj_acc.detach().cpu().item()),
                'apt_reg': float(reg_acc.detach().cpu().item())
            })
        except Exception:
            pass

        if (batch_idx % 50) == 0 and torch.cuda.is_available():
            torch.cuda.empty_cache()

    # 保存补丁make_affine_matrix
    def after_train_epoch(self, runner):
        if not self.enabled:
            return
        rgb, mask = self.patch()
        rgb_img = (rgb.detach().cpu().permute(1, 2, 0).clamp(0, 1).numpy() * 255).astype('uint8')
        mask_img = (mask.detach().cpu().squeeze(0).clamp(0, 1).numpy() * 255).astype('uint8')
        epoch = getattr(runner, 'epoch', 'final')
        Image.fromarray(rgb_img).save(os.path.join(self.save_dir, f'patch_epoch_{epoch}.png'))
        Image.fromarray(mask_img).save(os.path.join(self.save_dir, f'patch_mask_epoch_{epoch}.png'))

    def after_run(self, runner):
        if not self.enabled:
            return
        rgb, mask = self.patch()
        rgb_img = (rgb.detach().cpu().permute(1, 2, 0).clamp(0, 1).numpy() * 255).astype('uint8')
        mask_img = (mask.detach().cpu().squeeze(0).clamp(0, 1).numpy() * 255).astype('uint8')
        Image.fromarray(rgb_img).save(os.path.join(self.save_dir, 'final_patch.png'))
        Image.fromarray(mask_img).save(os.path.join(self.save_dir, 'final_patch_mask.png'))
