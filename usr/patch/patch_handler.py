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
        self.size = patch_size
        self.mode = patch_mode
        self.transparency = torch.ones(patch_size, patch_size, dtype=torch.float32, requires_grad=True, device=device)
        if self.mode is 'rgb':
            # optimizable
            self.mask = torch.zeros(3, patch_size, patch_size, dtype=torch.float32, requires_grad = True, device=device)
        elif self.mode is "gray_scale":
            self.mask = torch.zeros(1, patch_size, patch_size, dtype=torch.float32, requires_grad=True, device=device)
        else:
            raise "expected patch mode is rgb or gray_scale"
        
class Preprocessor:
        def __init__(self, 
                     mean=[],
                     std=[],
                     bgr_to_rgb=True,
                     pad_val=0,
                     seg_pad_val=255):
            self.mean = torch.tensor(mean)
            self.std = torch.tensor(std)
            self.bgr_to_rgb = bgr_to_rgb
            self.pad_val = pad_val
            self.seg_pad_val = seg_pad_val
            if self.mean[0] > 1.0:
                self.mean = self.mean / 255.0
                self.std = self.std / 255.0

class PatchHandler():
    '''
    config : read "patch_config" segment and process it 
    '''
    def _preprocess_init(self, cfg):
        self.mean = cfg.mean
        self.std = cfg.std
        self.pad_val = cfg.pad_val
        self.seg_pad_val = cfg.seg_pad_val
        self.bgr_to_rgb = cfg.bgr_to_rgb
        

    def __init__(self, cfg) -> None:
        super().__init__()
        if(config == None):
            print("config for patch handler is essential")
            raise KeyError
        #**********************Config******************#
        self.cfg = cfg
        config = cfg.patch_config
        
        self.lr = config.lr
        self.patch_path = config.patch_path
        self.patch_size = config.patch_size
        self.patch_mode = config.patch_mode

        self.rot_deg = config.rot_deg
        self.scale = config.scale
        self.max_translate = config.max_translate
        self.color_jitter = config.color_jitter
        self.location = config.location
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

    def _patch_preprocess():
        pass

    def apply_patch(self, input_batch:Tensor, gt_batch: Tensor):
        assert input_batch.dim() == 4, f"Expected 4D tensor [B, C, H, W], got {input_batch.dim()}D"

        transformed_patch = self.eot_transform_batch(self.patch, self.cfg.batch_size)
        h_start, w_start = self._get_location(self.cfg.crop_size)
        
        input_batch[:, :, h_start:h_start + self.patch_size, w_start:w_start + self.patch_size] = transformed_patch
        gt_batch[:, :, h_start:h_start + self.patch_size, w_start:w_start + self.patch_size] = self.ignore_label

        return input_batch, gt_batch

    #TODO here need refine
    def update_patch(self, loss):
        self.optimizor.zero_grad()
        loss.backward()
        self.optimizor.step()

    def _get_location(self, size:Union[List, Tuple]) -> dict:
        location = self.location
        _, h, w = size
        ret_val = tuple(0, 0) # h_start, w_start
        if location == 'default':
            ret_val[0] = (w - self.patch_size) // 2
            ret_val[0] = (h - self.patch_size) // 2

        return ret_val
    
    # get a batch of patch size [B, C, patch_size, patch_size]
    # padding mode : zeros
    def eot_transform_batch(
            self,
            patch: Tensor,
            batch_size: int,
            rot_deg: float, 
            scale: Tuple[float, float],
            max_translate: float,
            color_jitter:float,
            device='cuda',
    ) -> Tensor:
        # if patch.dim() == 2:
        #     patch = patch.unsqueeze(0)
        assert patch.dim() == 3, "check the patch type, grayscale one is dismissed"

        C, p, _ = patch.shape
        out = torch.zeros(batch_size, C, p, p, device=device, dtype=patch.dtype)

        for i in range(batch_size):
            angle = (torch.rand(1, device=device) * 2 - 1) * rot_deg
            scale = scale[0] + (scale[1] - scale[0]) * torch.rand(1, device=device)
            tx = (torch.rand(1, device=device) * 2 - 1) * max_translate
            ty = (torch.rand(1, device=device) * 2 - 1) * max_translate
            affine2x3: Tensor
            affine1x2x3 = self._make_affine_matrix(angle, scale, tx, ty, device=device).unsqueeze(0)
            grid = F.affine_grid(theta=affine1x2x3, size=(1, C, p, p), align_corners=False)
            warped = F.grid_sample(patch.unsqueeze(0), grid, mode='bilinear', padding_mode='zeros', align_corners=False)

            if color_jitter > 0 and C == 3:
                b = warped.new_empty(1, 1, 1, 1).uniform_(1 - color_jitter, 1 + color_jitter)
                a = warped.new_empty(1, 1, 1, 1).uniform_(-color_jitter, color_jitter) * 0.05
                warped = (warped * b + a).clamp(0, 1)

            #assert H_out > p and W_out > p, "your patch is bigger than original img!"
            # ph, pw = min(H_out, p), min(W_out, p)
            out[i] = warped[0]
        return out

    """
    make a affine matrix in NDT space
    """
    @staticmethod
    def _make_affine_matrix(angle_deg: Union[float,torch.Tensor], 
                            scale: Union[float, torch.Tensor],
                            tx: Union[float, torch.Tensor],
                            ty: Union[float, torch.Tensor],
                            device='cuda') -> Tensor:
        if type(angle_deg) == float:
            angle_deg = torch.tensor(angle_deg, dtype=torch.float32, device=device)

        # assert type(angle_deg) == torch.Tensor

        affine2x3 = torch.zeros(2, 3, device=device, dtype=torch.float32)
        rad = torch.deg2rad(angle_deg)

        cos = torch.cos(rad) * scale
        sin = torch.sin(rad) * scale

        affine2x3[0][0] = cos; affine2x3[0][1] = -sin; affine2x3[0][2] = tx
        affine2x3[1][0] = sin; affine2x3[1][1] = cos;  affine2x3[1][2] = ty

        return affine2x3