from typing import Union, List, Tuple
import os.path as osp
import pickle as pkl

import torchvision.transforms as transforms
import torchvision
import torch
import torch.nn as nn
from torch import Tensor
import torch.nn.functional as F

from mmengine import Config
from mmengine.structures.pixel_data import PixelData


class Patch(nn.Module):
    # all values are from 0 to 1
    # RGB or Graysclae patch
    # Img' = transparency * Img + (I - transparency) * Mask
    def __init__(self, patch_size, patch_mode, device="cuda"):
        """__init__ _summary_ this class contains a trainable patch(nn.Module)
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
        self.transparency = torch.ones(patch_size, patch_size, dtype=torch.float32, requires_grad=True, device=device)
        if self.mode == 'rgb':
            # optimizable
            self.mask = torch.zeros(3, patch_size, patch_size, dtype=torch.float32, requires_grad=True, device=device)
        elif self.mode == "gray_scale":
            self.mask = torch.zeros(patch_size, patch_size, dtype=torch.float32, requires_grad=True, device=device)
        else:
            raise "expected patch mode is rgb or gray_scale"

class PatchHandler:
    '''
    config : read "patch_config" segment and process it 
    '''
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

        self.rot_deg = config.rot_deg
        self.scale = config.scale
        assert type(self.scale) == tuple, f"expected to get tuple but get {type(self.scale)}"
        self.max_translate = config.max_translate
        self.location = config.location
        self.patch_anchor = (0, 0) # h_start, w_start
        self.ignore_label = config.ignore_label

        self._preprocess_init(self.cfg.model.data_preprocessor)

        # self.eot_transforms = transforms.Compose([
        #     transforms.RandomRotation(degrees=(-self.rot_deg, self.rot_deg)),
        #     transforms.RandomAffine(degrees = 0, translate=self.translate),
        #     transforms.RandomResizedCrop(size=self.patch_size, scale=self.scaling)
        # ])

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
    
    def _patch_preprocess(self):
        """_patch_preprocess normalize the patch for input

        Returns:
            tensor -- transparency  [1, size, size]
                      batched_patch [3, size, size] 
        """
        if self.patch_mode == 'rgb':
            patch = self.patch.mask
            if self.bgr_to_rgb:
                patch = patch[[2, 1, 0], ...]
        else:
            patch = torch.stack([self.patch.mask, 
                                         self.patch.mask, 
                                         self.patch.mask])

        patch = (patch - self.mean) / self.std
        transparency = self.patch.transparency.unsqueeze(0)
        
        assert transparency.dim() == 3 and patch.dim() == 3, "wrong stack !"

        return transparency, patch
        
    def apply_patch(self, input_batch:Tensor, gt_batch: Tensor):
        assert input_batch.dim() == 4, f"Expected 4D tensor [B, C, H, W], got {input_batch.dim()}D"

        transparency, patch = self._patch_preprocess()
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
        
        ret_batch = input_batch.clone()

        ret_batch[:, :, h_start:h_end, w_start:w_end] = \
            input_batch[:, :, h_start:h_end, w_start:w_end] * transparency + \
            (torch.tensor(1) - batch_transparency) * batch_patch

        batched_gt = gt_batch.clone()
        batched_gt[:, h_start:h_end, w_start:w_end] = self.ignore_label

        return ret_batch, batched_gt

    #TODO here need refine
    def update_patch(self, loss):
        self.patch_optimizor.zero_grad()
        loss.backward()
        self.patch_optimizor.step()

    def _get_location(self, size:Union[List, Tuple]) -> dict:
        
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
        affine = self._make_affine_matrix(batch_size, angle, scale, tx, ty, device=device)
        grid = F.affine_grid(theta=affine,
                             size=(batch_size, C, p, p),
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
                            ty: Union[float, torch.Tensor],
                            device="cuda") -> Tensor:
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