import torch
import torch.nn as nn

from mmengine.evaluator import BaseMetric
from mmseg.registry import METRICS
from mmseg.evaluation.metrics import IoUMetric

# TODO implement of fw IoU