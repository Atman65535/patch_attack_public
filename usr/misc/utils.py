import warnings
from typing import List, Tuple, Union

import torch
from torch import Tensor

from mmengine.config import Config
from mmseg.structures import SegDataSample

class Utils:
    # 这几个类都是为mmlab的API服务，把mmlab的各种数据结构换成tensor
    @staticmethod
    def parse_data_samples(data: List[SegDataSample],
                              gt_sem=False,
                              pred=False,
                              logits=False,):
        """parse_data_samples from List[SegDataSample] to Tensor

        Usage:    _, pred, _ = Utils.parse_data_samples(data, pred=True)

        Arguments:
            data {List[SegDataSample]} -- _description_

        Keyword Arguments:
            gt_sem {bool} -- _description_ (default: {False})
            pred {bool} -- _description_ (default: {False})
            logits {bool} -- _description_ (default: {False})

        Returns:
            _type_ -- _description_
        """        '''
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
            gt_sem_seg = torch.cat(
                [i.gt_sem_seg.data for i in data])
        if pred:
            pred_sem_seg = torch.cat(
                [i.pred_sem_seg.data for i in data])
        if logits:
            seg_logits = torch.cat(
                [i.seg_logits.data for i in data])

        return gt_sem_seg, pred_sem_seg, seg_logits
    
    @staticmethod
    def parse_inputs(data:SegDataSample):
        inputs = data['input']
        data_samples = data['data_sample']

        assert inputs != None, "Utils: inputs is none, plz check data "
        assert data_samples != None, "Utils: data sample is none, plz check data"

        input_batch = torch.stack(
            [i for i in inputs], dim=0)
        gt_sem_seg = torch.stack(
            [i.gt_sem_seg.data for i in data_samples], dim=0) 
        
        return input_batch, gt_sem_seg

    # Aborted
    # @staticmethod
    # def parse_model_output(data:List[SegDataSample]):
    #     """parse_model_output from mmlab type to universal tensor type
    #
    #     Usage:
    #         pred, logits = Utils.parse_model_output(res)
    #
    #     here all tensors are on cuda and the output will on cuda as well
    #     Arguments:
    #         data {List[SegDataSample]} -- data after model.predict
    #
    #     Returns:
    #         tensor -- literally
    #     """
    #     warnings.warn("aborted. Use classifier pipeline instead")
    #     pred_sem_seg = torch.stack(
    #         [i.pred_sem_seg.data for i in data], dim=0)
    #     seg_logits = torch.stack(
    #         [i.seg_logits.data for i in data], dim=0)
    #
    #     return pred_sem_seg, seg_logits

    # Aborted
    # @staticmethod
    # def config_preprocess(cfg_file:str):
    #     cfg = Config.fromfile(cfg_file)
    #     return cfg