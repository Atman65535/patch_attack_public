from typing import List, Tuple, Union

import torch
from torch import Tensor

from mmengine.config import Config
from mmseg.structures import SegDataSample

class Utils:
    
    @staticmethod
    def parse_data_samples(data: List[SegDataSample],
                              gt_sem=False,
                              pred=False,
                              logits=False,):
        '''
        usage inject the 'data_smaples' part of data, 
        then enable return of the part you want
        '''
        gt_sem_seg:Tensor
        pred_sem_seg:Tensor
        seg_logits:Tensor
        gt_sem_seg = None
        pred_sem_seg = None
        seg_logits = None
        #[B, C, H, W]
        if gt_sem:
            device = data[0].gt_sem_seg.data.device
            gt_sem_seg = torch.stack(
                [i.gt_sem_seg.data for i in data], 
                device=data[0].gt_sem_seg.data.device)
        if pred:
            pred_sem_seg = torch.stack(
                [i.pred_sem_seg for i in data],
                device=data[0].pred_sem_seg.data.device)
        if logits:
            seg_logits = torch.stack(
                [i.seg_logits.data for i in data],
                device=data[0].seg_logits.data.device)

        return gt_sem_seg, pred_sem_seg, seg_logits
    
    @staticmethod
    def parse_inputs(data:SegDataSample, device='cuda'):
        inputs = data['input']
        data_samples = data['data_sample']

        assert inputs != None, "Utils: inputs is none, plz check data "
        assert data_samples != None, "Utils: data sample is none, plz check data"

        input_batch = torch.stack(
            [i for i in inputs], dim=0, device=device)
        gt_sem_seg = torch.stack(
            [i.gt_sem_seg.data for i in data_samples], dim=0, device=device) 
        
        return input_batch, gt_sem_seg
    
    @staticmethod
    def parse_model_output(data:List[SegDataSample], device='cuda'):
        pred_sem_seg = torch.stack(
            [i.pred_seg_seg for i in data], dim=0, device=device)
        seg_logits = torch.stack(
            [i.seg_logits.data for i in data], dim=0, device=device)
        
        return pred_sem_seg, seg_logits
    
    @staticmethod
    def config_preprocess(cfg_file:str):
        cfg = Config.fromfile(cfg_file)
        return cfg