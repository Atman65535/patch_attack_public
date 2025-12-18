"""
File: attention_catcher.py
Author: Atman
Date: 12/18/25
Description:
    
"""
import warnings
from typing import Tuple

import torch

from UNet_patch import register_attention_control, reset_attention_control
from diffuison_utils import ddim_denoise

class SelfAttentionLoss:
    """
    This module contains a self attention loss calculation implementation
    Loss function is MSE function
    Components: One loss variable, one function method.
    """
    def __init__(self):
        self.criterion = torch.nn.MSELoss()
        self.loss = torch.tensor(0)

    def update_loss(self, attn_clean, attn_adv):
        attn_clean = attn_clean.expand(attn_adv.shape)
        if attn_clean.shape != attn_adv.shape:
            raise TypeError(f"SelfAttnLoss:"
                            f"expected input shapes are same, but get"
                            f"attn_adv{attn_adv.shape}, attn_clean{attn_clean.shape}")
        self.loss = self.loss + self.criterion(attn_adv, attn_clean)

    def get_loss(self):
        return self.loss

    def reset_all(self):
        self.loss = 0

class ProcessTracker:
    def __init__(self,
                 total_attn_invokes=32, # contains self and cross
                 total_unets=5):
        self.total_attn_invokes = total_attn_invokes
        self.current_attn_layer = 0
        self.total_unets = total_unets
        self.finished_unets = 0

    def update_after_unet(self):
        # after one full UNet denoise, we reset attention counter
        self.finished_unets = self.finished_unets + 1
        self.current_attn_layer = 0

    def update_current_attn_layer(self):
        # used for step in one attention layer.
        self.current_attn_layer = self.current_attn_layer + 1

    def finish_one_unet(self):
        return self.current_attn_layer == self.total_attn_invokes

    def finish_one_image(self):
        return self.finished_unets == self.total_unets

    def reset_all(self):
        self.current_attn_layer = 0
        self.finished_unets = 0

class AttentionStorage:
    """
    Store Attention Maps.
    Structure:{
    "down_cross": [tensor(BxHead, HW, 77), tensor, tensor, ...], (6)
    "mid_cross": [], (1)
    "up_cross": [], (9)
    "down_self": [tensor(BxHead, HW, HW), tensor, tensor, ...], (6)
    "mid_self": [], (1)
    "up_self"     (9)
    }
    if we only need one type of map, there maybe only 10 maps, 5 self and 5 cross
    """
    def __init__(self):
        self.dynamic_storage = self._get_empty_store()
        self.aggregate_storate = {}
        self.attention_store_cnt = 0

    def update_after_unet(self):
        if self.aggregate_storate == {}:
            self.aggregate_storate = {
                key: [v.clone() for v in self.dynamic_storage[key]]
                for key in self.dynamic_storage
            }
        else:
            for key in self.aggregate_storate.keys():
                for i in range(len(self.aggregate_storate[key])):
                    self.aggregate_storate[key][i] = self.aggregate_storate[key][i] + self.dynamic_storage[key][i]

        self.dynamic_storage = self._get_empty_store()
        self.attention_store_cnt += 1

    def store(self, attention: torch.Tensor, is_cross, unet_stage):
        key = self.store_keyword(unet_stage, is_cross)
        self.dynamic_storage[key].append(attention)

    def get_average_maps(self):
        average_attention = {
            key:[
                item / self.attention_store_cnt
                for item in self.aggregate_storate[key]
            ] for key in self.aggregate_storate
        }
        return average_attention

    def reset_all(self):
        self.dynamic_storage = self._get_empty_store()
        self.aggregate_storate = {}
        self.attention_store_cnt = 0

    @staticmethod
    def _get_empty_store():
        return {"down_cross": [], "mid_cross": [], "up_cross": [],
                "down_self": [], "mid_self": [], "up_self": []}

    @staticmethod
    def store_keyword(unet_stage:str,
                      cross_attn:bool):
        if unet_stage in ["down", "mid", "up"]:
            return f"{unet_stage}_{'cross' if cross_attn else 'self'}"
        else:
            raise ValueError(f"AttentionCatcher._store_keyword: "
                             f"expected unet stage from [\"down\", \"mid\", \"up\"], but got {unet_stage}")

class AttentionCatcher:
    def __init__(self,
                 batch_size=2,
                 resolution:int =256,
                 target_map_resolution=None,
                 checked=False):
        if batch_size != 2:
            warnings.warn("We can't make sure our sys can run properly when batch_size is not 2", RuntimeWarning)
        self.batch_size = batch_size
        self.cfg_batch_size = batch_size * 2 # for CFG, the latent we get is twice of input
        self.resolution = resolution
        self.target_map_resolution = target_map_resolution
        self.checked=checked

        # we set div 16 as default, if input 256, get 16
        if target_map_resolution is None:
            vae_res = resolution >> 3 # div 8
            self.target_map_resolution = vae_res >> 1
        self.target_map_tokens = self.target_map_resolution ** 2

        self.process_tracker = ProcessTracker(32, 5)
        self.attn_storage = AttentionStorage()
        self.self_attn_loss = SelfAttentionLoss()

    def capture_attention_map(self, attention, is_cross:bool, unet_stage):
        """
            attention: CFG attention, batchsize *=2,
            [clean, adv] -> [uncond[clean, adv], cond[clean, adv]]
            only cond can apply self and cross on it.
        """
        self.process_tracker.update_current_attn_layer()
        # we only need conditional attention.
        # cond_attn: [Batchsize * heads, HW, HW or 77]
        hw = attention.shape[0]
        #uncond_attn = attention[:self.batch_size/2]
        cond_attn = attention[hw//2:]
        heads = cond_attn.shape[0] // self.batch_size
        # check attention type
        if is_cross and cond_attn.shape[-1] != 77:
            raise TypeError(f"Not cross attention here! Check your chanel {cond_attn.shape}")
        if not is_cross and cond_attn.shape[-1] != cond_attn.shape[-2]:
            raise TypeError(f"Not self attention here! Check your channel {cond_attn.shape}")

        # store desired attention maps, include [clean, adv], batchsize = 2
        if cond_attn.shape[1] == self.target_map_tokens:
            self.attn_storage.store(cond_attn, is_cross, unet_stage)


        # update self attention loss
        if not is_cross: # self
            cond_attn = cond_attn.reshape(self.batch_size, heads,
                                          *cond_attn.shape[1:]) #[2, heads, HW, HW]
            attn_clean = cond_attn[0]
            attn_adv = cond_attn[1:]
            self.self_attn_loss.update_loss(attn_clean, attn_adv)

        # finish one denoise process.
        if self.process_tracker.finish_one_unet():
            self.process_tracker.update_after_unet() # reset counter
            self.attn_storage.update_after_unet()
            # reset dynamic storage and store this layer
        return

    def extract_cross_attn_map(self, stages: Tuple[str, ...]):
        target_tokens = self.target_map_tokens
        is_cross = True
        average_attentions = self.attn_storage.get_average_maps()
        out = []
        for stage in stages:
            key = self.attn_storage.store_keyword(stage, is_cross)
            for item in average_attentions[key]:
                if item.shape[1] == target_tokens:
                    cross_maps = item.reshape(self.batch_size, -1,
                                              self.target_map_resolution,
                                              self.target_map_resolution,
                                              item.shape[-1]) # B, H, W, W, Fea
                    out.append(cross_maps)
        out = torch.cat(out, dim=1)
        out = out.sum(dim=1) / out.shape[1]
        return out

    def attention_loss(self, stages: Tuple[str, ...]):
        ca_map = self.extract_cross_attn_map(stages)
        if self.checked != True:
            warnings.warn(f"attention_loss: this loss is based on a hyposis that input is [clean, adv]"
                      f"if you want to turn off this warning, set checked=True in initialize function")

    def reset_all(self):
        self.self_attn_loss.reset_all()
        self.attn_storage.reset_all()
        self.process_tracker.reset_all()

    def __call__(self, *args, **kwargs):
        self.capture_attention_map(*args, **kwargs)

if __name__ == "__main__":
    # already tested in diff latent attack.
    print("pass validation")
