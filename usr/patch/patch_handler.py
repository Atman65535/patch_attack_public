from typing import Union, List, Tuple

import torchvision.transforms as transforms
import torchvision
import torch
import torch.nn as nn

from mmengine import Config
from mmengine.structures.pixel_data import PixelData

class PatchHandler(nn.Module):
    '''
    config : read "patch_config" segment and process it 
    '''
    def __init__(self, config: dict) -> None:
        super().__init__()
        if(config == None):
            print("config for patch handler is essential")
            raise KeyError
        
        self.config = config
        self.patch_size = config['patch_size']
        self.patch_path = config['patch_path']
        self.rot_deg = config['rot_deg']
        self.translate = config['translate']
        self.scaling = config['scaling']
        self.location = config['location']
        self.ignore_label = config['ignore_label']
        self.lr = config['lr']

        self.eot_transforms = transforms.Compose([
            transforms.RandomRotation(degrees=(-self.rot_deg, self.rot_deg)),
            transforms.RandomAffine(degrees = 0, translate=self.translate),
            transforms.RandomResizedCrop(size=self.patch_size, scale=self.scaling)
        ])
        #TODO verify the type here and finish gradient requirement
        #self.patch = torchvision.io.read_image(self.patch_path, torchvision.io.ImageReadMode.UNCHANGED)

        self.patch : nn.Parameter
        patch_tensor = None
        self.patch = nn.Parameter()
        #self.patch.requires_grad_(True)
        self.optimizor = torch.optim.Adam(self.patch.parameters(), lr=self.lr)




    def apply_patch(self, batch:dict):
        inputs : list    # contains tensor, img
        data_samples : list # contains mmseg.structures.seg_data_sample.SegDataSample
        inputs = batch['inputs'] 
        data_samples = batch['data_samples']

        size : torch.Size
        size = inputs[0].shape # torch.Size([3, 1024, 1024])
        loc = self._get_location(size)

        transformed_patch = self._transform_patch()

        for img in inputs:
            img[:, 
                loc['h_start']:loc['h_start'] + self.patch_size,
                loc['w_start']:loc['w_start'] + self.patch_size] = transformed_patch
            
        seg_map : PixelData
        for tmp in data_samples:
            seg_map = tmp.gt_sem_seg
            data = seg_map.data
            data[:, 
                loc['h_start']:loc['h_start'] + self.patch_size,
                loc['w_start']:loc['w_start'] + self.patch_size] = self.ignore_label
            
        return batch


    #TODO here need refine
    def update_patch(self, loss):
        self.optimizor.zero_grad()
        loss.backward()
        self.optimizor.step()

    def _get_location(self, size:Union[List, Tuple]) -> dict:
        location = self.location
        _, h, w = size
        ret_val = dict(
            h_start = 0,
            w_start = 0
        )
        if location == 'default':
            ret_val['h_start'] = (w - self.patch_size) // 2
            ret_val['w_start'] = (h - self.patch_size) // 2

        return ret_val
    
    def _transform_patch(self):
        transformed = self.eot_transforms(self.patch)
        return transformed
    
    def _make_affine_matrix(angle_deg: float, scale: float,
                            tx: float, ty: float, device='cuda'):
        theta = torch.zeros(2, 3, device=device)
