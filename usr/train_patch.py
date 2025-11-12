import argparse
import logging
import os
import os.path as osp

import torch
from torch.utils.data import DataLoader

from mmengine.config import Config, DictAction
from mmengine.logging import print_log
from mmengine.runner import Runner

from mmseg.registry import RUNNERS, DATASETS, MODELS, HOOKS

from usr.patch import PatchHandler
from usr.metrics import PatchMetrics


config_file = "usr/configs/exp/patch_config.py"
def main():

    cfg = Config.fromfile(config_file)
    data_loader = cfg.get("train_dataloader")
    data_loader = Runner.build_dataloader(cfg.get("train_dataloader"))
    model = MODELS.build(cfg.get('model'))
    patch_handler = PatchHandler(cfg.get("patch_config"))
    for iter in range(1):
        for idx, batch in enumerate(data_loader, 0):
            print(batch)
    return

if __name__ == "__main__":

    from mmseg.utils import register_all_modules
    register_all_modules()

    main()