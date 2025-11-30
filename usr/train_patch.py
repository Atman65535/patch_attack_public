import argparse
import logging
import os
import os.path as osp
from typing import List, Dict, Tuple, Optional, Union

import torch
from torch.utils.data import DataLoader
from torch.optim import Adam
from torch import Tensor

from mmengine.config import Config, DictAction
from mmengine.logging import print_log
from mmengine.runner import Runner

from mmseg.registry import RUNNERS, DATASETS, MODELS, HOOKS
from mmseg.models import EncoderDecoder
from mmseg.structures import SegDataSample

from usr.patch import PatchHandler
from usr.metrics import PatchMetrics
from usr.utils import Utils

def build_model(cfg) -> EncoderDecoder:
    """build_model build from MMLab APIs

    Arguments:
        cfg {ConfigDict} -- 

    Returns:
        EncoderDecoder -- just type, refer to mmseg.model.encoderdecoder
    """    
    model = MODELS.build(cfg).cuda()
    model.eval()
    for param in model.parameters():
        param.requires_grad_(False)
    preprocessor = model.data_preprocessor
    return model, preprocessor

def predict(model, data):
    """predict predict using model

    Arguments:
        model {encoderdecoder}
        data {tensor} -- 

    Returns:
        tensor -- tensor parsed from mmlab data sample
    """    
    res : List[SegDataSample]
    res = model.predict(data) 

    pred, logits = Utils.parse_model_output(res)
    return pred, logits

def main():
    torch.autograd.set_detect_anomaly(True)
    # cfg init
    config_file = "usr/configs/exp/patch_config.py"
    cfg = Utils.config_preprocess(config_file)
    # patch associated
    patch_handler = PatchHandler(cfg)
    patch_metrics = PatchMetrics(cfg)
    # data associated
    data_loader = Runner.build_dataloader(cfg.train_dataloader)
    # model associated
    model, preprocessor = build_model(cfg.model)
    
    
    # train iter
    for _ in range(cfg.epochs):
        # contains ['pred_sem_seg', 'seg_logits']
        for _, batch in enumerate(data_loader, 0):
            # preprocess: normalize and apply patch
            preprocessed = preprocessor(batch)
            data = preprocessed['inputs']
            data_gt, _, _ = Utils.parse_data_samples(preprocessed['data_samples'], gt_sem=True)
            #**********Attack***********#
            data_patched, gt_patched = patch_handler.apply_patch(data, data_gt)
            pred, logits = predict(model, data_patched)
            
            # gt_patched or data_gt? if the patch is invisible, we should use data_gt. other wise use gt_patched.
            classify_loss = patch_metrics.classify_loss(logits, data_gt, patch_handler.patch_anchor)
            patch_handler.update_patch(loss=classify_loss)

if __name__ == "__main__":

    from mmseg.utils import register_all_modules
    register_all_modules()

    main()