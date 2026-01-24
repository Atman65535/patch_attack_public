"""
File: loss_handler.py
Author: Atman
Date: 12/22/25
Description:
    
"""
import logging
import torch

class LossHandler:
    """ 
    这就是个log类，输入loss，会在每次调用它的时候自动打log，其余暂时没有任何作用
    """
    def __init__(self, cfg):
        """
        Args:
            cfg:config.weight_config
        """
        self.loss = 0
        self.classifier_loss = 0
        self.self_loss = 0
        self.cross_loss = 0
        self.classifier_weight = cfg.classifier
        self.self_attn_weight = cfg.self
        self.cross_attn_weight = cfg.cross
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )

    def reset(self):
        self.self_loss = 0
        self.cross_loss = 0
        self.classifier_loss = 0

    def update(self, classifier=None, self_attn=None, cross=None):
        device = torch.device("cuda", 0)
        current_loss = torch.tensor(0.0, device=device)
        if classifier is not None:
            self.classifier_loss += self.classifier_weight * classifier.item()
        if self_attn is not None:
            self.self_loss += self.self_attn_weight * self_attn.item()
        if cross is not None:
            self.cross_loss += self.cross_attn_weight * cross.item()

    def log(self, epoch):
        logging.info(f"epoch : {epoch} \nclassifier_loss : {self.classifier_loss}\nself_attn_loss : {self.self_loss}\ncross_attn_loss : {self.cross_loss}")

if __name__ == "__main__":
    print("pass validation")
