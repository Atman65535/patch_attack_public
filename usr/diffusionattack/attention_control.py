from typing import Union, Tuple

import torch
from diffusers.models.attention_processor import Attention, AttnProcessor
import abc

class AttentionControl(abc.ABC):
    def __init__(self):
        pass

    def between_steps(self):
        return

    def reset(self):
        pass

    @abc.abstractmethod
    def forward(self, 
                attn: Attention, 
                is_cross: bool, 
                place_in_unet):
        raise NotImplementedError

    def __call__(self,
                 attn,
                 is_cross: bool,
                 place_in_unet: str):
        pass
    
class AttentionStore(AttentionControl):
    def __init__(self, res):


class AttentionControlEdit(AttentionStore, abc.ABC):
    def __init__(self, num_steps,
                 self_replace_steps,
                 res):
        """

        Args:
            num_steps:
            self_replace_steps: [start, end]
            res:
        """
        super().__init__(res)
        self.batch_size = 2
        if type(self_replace_steps) is float:
            self_replace_steps = (0, self_replace_steps)
            self.replace_section = (int(num_steps * self_replace_steps[0]),
                                     int(num_steps * self_replace_steps[1]))
            self.loss = 0
            self.criterion = torch.nn.MSELoss()