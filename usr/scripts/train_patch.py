import argparse
import logging
import os
import os.path as osp
from typing import List, Dict, Tuple, Optional, Union

import torch
from torch.utils.data import DataLoader
from torch.optim import Adam

from mmengine.config import Config, DictAction
from mmengine.logging import print_log
from mmengine.runner import Runner

from mmseg.registry import RUNNERS, DATASETS, MODELS, HOOKS
from mmseg.models import EncoderDecoder
from mmseg.structures import SegDataSample

from usr.patch import PatchHandler
from usr.metrics import PatchMetrics

def parse_config():
    pass

config_file = "usr/configs/exp/patch_config.py"
def main():
    cfg = Config.fromfile(config_file)
    epochs = cfg.get("epochs")
    patch_handler = PatchHandler(cfg.get("patch_config"))
    patch_metrics = PatchMetrics(cfg.get("patch_metrics"))
    

    data_loader = cfg.get("train_dataloader")
    data_loader = Runner.build_dataloader(cfg.get("train_dataloader"))

    model:EncoderDecoder
    model = MODELS.build(cfg.get('model')).cuda()
    model.eval()
    for param in model.parameters():
        param.requires_grad_(False)

    for _ in range(epochs):
        for _, batch in enumerate(data_loader, 0):
            data = model.data_preprocessor(batch)

            ins : torch.Tensor
            ins = data['inputs']
            ins.cuda()
            # return a list len(batch_size)
            # contains ['pred_sem_seg', 'seg_logits']
            res : List[SegDataSample]
            res = model.predict(ins) 

            patch_metrics.update_prediction(res, data)

            # merged_logits = torch.stack([logits.seg_logits.data for logits in res], dim=0)
            # merged_labels = torch.stack([label.gt_sem_seg.data for label in data['data_samples']], dim=0)
            # merged_labels_sqeezed = merged_labels.squeeze()
            loss = patch_metrics.get_loss()
            patch_handler.update_patch(loss)
    return

if __name__ == "__main__":

    from mmseg.utils import register_all_modules
    register_all_modules()

    main()