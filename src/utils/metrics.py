"""
File: metrics.py
Author: Atman
Date: 1/25/26
Description:
    
"""
import torch
from torchmetrics.classification import MulticlassAccuracy
from torchmetrics.classification import MulticlassJaccardIndex
import lpips
class MetricsKit:
    def __init__(self, cfg):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        if not torch.cuda.is_available():
            raise RuntimeError("Please Use GPU For Evaluation(metrics_cfg.device)")
        self.acc = MulticlassAccuracy(num_classes=cfg.num_classes,
                                      average=cfg.average,
                                      ignore_index=cfg.ignore_index)
        self.miou = MulticlassJaccardIndex(num_classes=cfg.num_classes,
                                           average=cfg.average,
                                           ignore_index=cfg.ignore_index)
        self.lpips = lpips.LPIPS(net=cfg.lpips_net)
        # send to cuda
        self.acc.to(self.device)
        self.miou.to(self.device)
        self.lpips.to(self.device)

    @torch.no_grad()
    def asr_score(self, pred, gt):
        return 1. - self.acc(pred, gt).item()

    @torch.no_grad()
    def miou_score(self, pred, gt):
        return self.miou(pred, gt).item()

    @torch.no_grad()
    def lpips_score(self, clean, adv):
        assert torch.max(clean) < 1.001 and torch.min(clean) > -0.001, \
            "LPIPS only accept tensor in range [0, 1]"
        return self.lpips((clean * 2. - 1.), (adv * 2. - 1.)).mean().item()

    # TODO Optional: Finish Batch Update logit, save GPU power.

#
# if __name__ == "__main__":
#     pred = torch.randint(0, 19, [3, 255, 255]).to("cuda")
#     gt = pred.clone().to("cuda")
#     clean = torch.rand(3, 255, 255).to("cuda")
#     import omegaconf
#     cfg = omegaconf.OmegaConf.load("/home/atman/a_workspace/mmlab/mmsegmentation/src/configs/D4A_config_local.yaml")
#     metrics = MetricsKit(cfg.metrics_cfg)
#     print(metrics.lpips_score(clean, clean))
#     print(metrics.asr_score(pred, gt))
#     print(metrics.miou_score(pred, gt))
#
