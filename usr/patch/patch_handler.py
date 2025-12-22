import warnings
from typing import Union, List, Tuple, Optional
import os.path as osp
import pickle as pkl

import torch
from torch import Tensor
import torch.nn.functional as F



class Patch:
    # all values are from 0 to 1
    # RGB or Graysclae patch
    # Img' = transparency * Img + (I - transparency) * Mask
    def __init__(self, patch_size, patch_mode):
        """__init__ _summary_ this class contains a trainable patch
                    contains a mask (size x size), and mask

        Arguments:
            patch_size {int} -- _description_ 
            patch_mode {int} -- _description_ rgb or gray_scale, only have effect  
            on mask

        Keyword Arguments:
            device {str} -- _description_ (default: {"cuda"})
        """
        self.size = patch_size
        self.mode = patch_mode
        device = torch.device("cuda", 0)
        eps = 1e-3
        self.transparency = torch.nn.Parameter(eps * torch.rand(3, patch_size, patch_size,
                                             dtype=torch.float32,
                                             requires_grad=True,
                                             device=device))
        if self.mode == 'rgb':
            # optimizable
            self.mask = torch.nn.Parameter(eps * torch.rand(3, patch_size, patch_size,
                                         dtype=torch.float32,
                                         requires_grad=True,
                                         device=device))
        elif self.mode == "gray_scale":
            self.mask = torch.nn.Parameter(eps * torch.rand(patch_size, patch_size,
                                         dtype=torch.float32,
                                         requires_grad=True,
                                         device=device))
            assert self.mask.ndim == 2, "Patch init: wrong init!"
        else:
            raise "expected patch mode is rgb or gray_scale"

    @torch.enable_grad()
    def patch_mapping_to01(self):
        patch = self.mask ** 2 /( 1 + self.mask ** 2)
        trans = self.transparency ** 2 / (1 + self.transparency ** 2)
        return  patch, trans


class PatchHandler:
    """
    config : read "patch_config" segment and process it
    """
    def _preprocess_init(self, cfg):
        self.mean = torch.tensor(cfg.mean)
        self.std = torch.tensor(cfg.std)
        self.pad_val = cfg.pad_val
        self.seg_pad_val = cfg.seg_pad_val
        self.bgr_to_rgb = cfg.bgr_to_rgb
        if self.mean[0] > 1.0:
            self.mean = self.mean / 255.0
            self.std = self.std / 255.0
        self.mean = self.mean.view(3, 1, 1).cuda()
        self.std = self.std.view(3, 1, 1).cuda()
        
    def __init__(self, cfg) -> None:
        #**********************Config******************#
        self.cfg = cfg
        config = cfg.patch_config
        
        self.lr = config.lr
        self.batch_size = config.batch_size
        self.patch_path = config.patch_path
        self.patch_size = config.patch_size
        self.patch_mode = config.patch_mode
        # EOT configs
        self.enable_eot = config.enable_eot
        self.rot_deg = config.rot_deg
        self.scale = config.scale
        assert type(self.scale) == tuple, f"expected to get tuple but get {type(self.scale)}"
        self.max_translate = config.max_translate
        self.location = config.location
        self.patch_anchor: tuple # h_start, w_start
        self.ignore_label = config.ignore_label

        self._preprocess_init(self.cfg.model.data_preprocessor)

        if osp.exists(self.patch_path):
            self.patch = pkl.load(self.patch_path)
            print(f"load pkl instance Patch from {self.patch_path}")
        else:
            self.patch = Patch(self.patch_size, self.patch_mode)

        self.patch_optimizor = torch.optim.Adam(
            [self.patch.mask, self.patch.transparency], 
            lr=self.lr)
        
        #self.in_patch_mask = torch.zeros(self.cfg.batch_size, 1, self.cfg.crop_size[0], self.crop_size[1])

        #****************Operation****************#
        """
        _summary_
        """
    
    def get_01patch(self):
        """
        preprocess the patch. Just for classifier.
        Normalize, from patch [0, 1]. std and mean are for [0, 1] patch
        """
        if self.patch_mode == 'rgb':
            patch = self.patch.mask
            if self.bgr_to_rgb:
                # patch = patch[[2, 1, 0], ...]
                warnings.warn("check rgb or bgr here, we just want test gray scale")
        else:
            patch = torch.stack([self.patch.mask, 
                                         self.patch.mask,
                                         self.patch.mask])

        transparency = self.patch.transparency
        
        assert transparency.dim() == 3 and patch.dim() == 3, "wrong stack !"

        return transparency, patch
        
    def apply_patch(self, input_batch:Tensor, gt_batch: Tensor, classifier=False):
        """
        apply patch
        if for classifier, return patched batch and patched ground truth
        if for diffusion model, return clean patch and patched patch
        """
        assert input_batch.dim() == 4, f"Expected 4D tensor [B, C, H, W], got {input_batch.dim()}D"

        transparency, patch = self.get_01patch()
        if classifier:
            patch = (patch - self.mean) / self.std
        if self.enable_eot:
            batch_transparency, batch_patch = self.batch_eot_transform(
                patch,
                transparency,
                self.batch_size,
                self.rot_deg,
                self.scale,
                self.max_translate)
        
        h_start, w_start = self._get_location(self.cfg.crop_size)
        h_end = h_start + self.patch_size
        w_end = w_start + self.patch_size
        # Img' = patch * trans + (1-trans) * img
        patched_batch = input_batch.clone()
        patched_batch[:, :, h_start:h_end, w_start:w_end] = \
            input_batch[:, :, h_start:h_end, w_start:w_end] * (1 - transparency) + \
            transparency * patch

        if classifier:
            patched_gt = gt_batch.clone()
            patched_gt[:, h_start:h_end, w_start:w_end] = self.ignore_label
            return patched_batch, patched_gt
        else:
            patched_batch = patched_batch[:, :, h_start:h_end, w_start:w_end]
            clean_batch = input_batch[:, :, h_start:h_end, w_start:w_end]
            gt_ret = gt_batch[:, h_start:h_end, w_start:w_end]
            return clean_batch, patched_batch, gt_ret


    def patch_optim_step(self):
        self.patch_optimizor.step()
        self.patch_optimizor.zero_grad()

    def _get_location(self, size:Union[List, Tuple]) -> List:
        
        location = self.location
        h, w = size
        ret_val = [0, 0] # h_start, w_start
        if location == 'default':
            ret_val[0] = (w - self.patch_size) // 2
            ret_val[1] = (h - self.patch_size) // 2

        self.patch_anchor = ret_val
        return ret_val
    
    def batch_eot_transform(
            self,
            patch: Tensor, # [3, size, size]
            transparency: Tensor, #[1, size, size]
            batch_size: int,
            rot_deg: float, 
            scale: Tuple[float, float],
            max_translate: float,) -> Tuple[torch.Tensor, torch.Tensor]:
        """eot_transform_batch from 3 dim patch and transparency to 4 dim with eot

        Arguments:
            patch {Tensor} -- _description_
            rot_deg {float} -- _description_
            scale {Tuple[float, float]} -- _description_
            max_translate {float} -- _description_

        Returns:
            Tuple[torch.Tensor, torch.Tensor] -- _description_
        """        
        device = patch.device
        # if patch.dim() == 2:
        #     patch = patch.unsqueeze(0)
        assert patch.dim() == 3, "check the patch type, grayscale one is dismissed"
        assert transparency.dim() == 3 and transparency.shape[0] == 1

        C, p, _ = patch.shape

        angle = (torch.rand(batch_size, device=device) * 2 - 1) * rot_deg
        scale = scale[0] + (scale[1] - scale[0]) * torch.rand(batch_size, device=device)
        tx = (torch.rand(batch_size, device=device) * 2 - 1) * max_translate
        ty = (torch.rand(batch_size, device=device) * 2 - 1) * max_translate
        # affine [B, H, W]
        affine = self._make_affine_matrix(batch_size, angle, scale, tx, ty)
        grid = F.affine_grid(theta=affine,
                             size=[batch_size, C, p, p],
                             align_corners=False)
        
        copied_patch = patch.unsqueeze(0).repeat(batch_size, 1, 1, 1)
        copied_transparency = transparency.unsqueeze(0).repeat(batch_size, 1, 1, 1)

        warped_patch = F.grid_sample(copied_patch,
                               grid, mode='bilinear', 
                               padding_mode='zeros', 
                               align_corners=False)
        
        warped_transparency = F.grid_sample(copied_transparency,
                                            grid, 
                                            mode='bilinear',
                                            padding_mode='zeros',
                                            align_corners=False)
        #assert H_out > p and W_out > p, "your patch is bigger than original img!"
        # ph, pw = min(H_out, p), min(W_out, p)
        
        return warped_patch, warped_transparency

    """
    make a affine matrix in NDT space
    """
    @staticmethod
    def _make_affine_matrix(batch_size,
                            angle_deg: Union[float, torch.Tensor],
                            scale: Union[float, torch.Tensor],
                            tx: Union[float, torch.Tensor],
                            ty: Union[float, torch.Tensor]) -> Tensor:
        """_make_affine_matrix return a batch of affine matrix[B, 1, H, W]

        Arguments:
            batch_size {int} 
            angle_deg {Union[float, torch.Tensor]} -- 
            scale {Union[float, torch.Tensor]} -- 
            tx {Union[float, torch.Tensor]} -- translation scale
            ty {Union[float, torch.Tensor]} -- 

        Returns:
            Tensor -- affine matrix [B, 1, H, W]
        """
        device = torch.device("cuda", 0)
        if type(angle_deg) == float:
            angle_deg = torch.tensor(angle_deg, dtype=torch.float32, device=device)

        # assert type(angle_deg) == torch.Tensor
        affine = torch.zeros(batch_size, 2, 3, device=device, dtype=torch.float32)
        rad = torch.deg2rad(angle_deg)

        cos = torch.cos(rad) * scale
        sin = torch.sin(rad) * scale

        affine[:, 0, 0] = cos; affine[:, 0, 1] = -sin; affine[:, 0, 2] = tx
        affine[:, 1, 0] = sin; affine[:, 1, 1] = cos;  affine[:, 1, 2] = ty

        assert affine.dim() == 3 and affine.shape[0] == batch_size, "wrong affine in EOT transformation !"

        return affine