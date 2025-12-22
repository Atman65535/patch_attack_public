import argparse
import logging
import os
import os.path as osp
from typing import List, Dict, Tuple, Optional, Union

from mmengine.runner import Runner

from mmseg.registry import  MODELS
from mmseg.structures import SegDataSample
from usr.diffusion_loss_neat.diff_loss_pipeline import DiffLossTools

from usr.patch import PatchHandler
from usr.metrics import PatchMetrics
from usr.utils import Utils, LossHandler

def build_model(cfg):
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

def classifier_pipeline(patch_handler, model, patch_metrics, preprocessor, batch):
    preprocessed = preprocessor(batch)
    data = preprocessed['inputs']
    data_gt, _, _ = Utils.parse_data_samples(preprocessed['data_samples'], gt_sem=True)
    gt_ret = data_gt.clone()
    #**********Attack***********#
    data_patched, gt_patched = patch_handler.apply_patch(data, data_gt, classifier=True)
    pred, logits = predict(model, data_patched)

    # gt_patched or data_gt? if the patch is invisible, we should use data_gt. otherwise use gt_patched.
    classify_loss = patch_metrics.classify_loss(logits, data_gt, patch_handler.patch_anchor)
    return classify_loss, gt_ret

def main():
    # cfg init
    config_file = "usr/configs/exp/patch_config.py"
    cfg = Utils.config_preprocess(config_file)
    model, preprocessor = build_model(cfg.model)
    # patch associated
    patch_handler = PatchHandler(cfg)
    patch_metrics = PatchMetrics(cfg)
    # data associated
    data_loader = Runner.build_dataloader(cfg.train_dataloader)
    diffusion_loss_pipeline = DiffLossTools(cfg.diffusion_config)
    # train iter
    loss_iter = cfg.loss_back_iter
    loss_iter_cnt = 0
    loss = LossHandler(cfg.weight_config)
    for e in range(cfg.epochs):
        # contains ['pred_sem_seg', 'seg_logits']
        for _, batch in enumerate(data_loader, 0):
            loss_iter_cnt += 1
            # preprocess: normalize and apply patch
            classify_loss, gt_batch = classifier_pipeline(patch_handler, model, patch_metrics, preprocessor, batch)
            diff_batch = diffusion_loss_pipeline.image_preprocessor01(batch['inputs'], 1024, 1024)
            clean_batch, adv_batch, gt = patch_handler.apply_patch(diff_batch, gt_batch, False)
            clean_batch = clean_batch * 2.0 - 1.0
            adv_batch = adv_batch * 2.0 - 1.0
            loss.update(classifier=classify_loss)
            for clean, adv, gt in zip(clean_batch, adv_batch, gt):
                clean = clean.unsqueeze(0)
                adv = adv.unsqueeze(0)
                self_loss, cross_loss = diffusion_loss_pipeline.get_loss(clean, adv, gt)
                loss.update(self_attn=self_loss, cross=cross_loss)
            if loss_iter_cnt == loss_iter:
                patch_handler.patch_optim_step()
                loss.log(e)
                loss.reset()

if __name__ == "__main__":

    from mmseg.utils import register_all_modules
    register_all_modules()

    main()