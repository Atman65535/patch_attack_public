"""
File: loss_handler.py
Author: Atman
Date: 12/22/25
Description:
    
"""
import logging

class LossHandler:
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
        self.loss = 0
        self.self_loss = 0
        self.cross_loss = 0
        self.classifier_loss = 0

    def update(self, classifier=0, self_attn=0, cross=0):
        self.loss += self.classifier_weight * classifier + self.self_attn_weight * self_attn + self.cross_attn_weight * cross
        self.classifier_loss += self.classifier_weight * classifier
        self.self_loss += self.self_attn_weight * self_attn
        self.cross_loss += self.cross_attn_weight * cross
        self.loss.backward()
        self.loss = 0
    def log(self, epoch):
        logging.info(f"epoch : {epoch}"
                     f"classifier_loss : {self.classifier_loss}"
                     f"self_attn_loss : {self.self_loss}"
                     f"cross_attn_loss : {self.cross_loss}")

if __name__ == "__main__":
    print("pass validation")
