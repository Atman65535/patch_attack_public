from typing import Union, Tuple, List
import warnings

import torch

def _empty_attention_storage():
    return dict({
        "down_cross": [], "mid_cross": [], "up_cross": [],
        "down_self": [], "mid_self": [], "up_self": []
    })

def _get_attention_storage_key(phase_of_unet, is_cross_attention):
    """
    Input UNet stage and attention type, return key of the attention storage dict.
    Args:
        phase_of_unet: "down", "mid", "up"
        is_cross_attention: true -> cross attention , else self attention

    Returns: "down_cross", "mid_cross", "up_self"....
            coordinate with _empty_attention_storage keys.
    """
    return f"{phase_of_unet}_{'cross' if is_cross_attention else 'self'}"


def _expand_attention_clean_to_adversarial(attention_clean,
                                           attention_adversarial):
    return attention_clean.unsqueeze(0).expand(attention_adversarial.shape[0],
                                               *attention_clean.shape)


class AttentionController:
    def __init__(self,
                 diffusion_batch_size:int,
                 total_steps_for_diffusion,  # eg: 20
                 self_attention_control_steps_section,  # (0, 20) -> [0, 20]
                 original_resolution: int,  # eg: 256 * 256, input 256
                 max_acceptable_latent_resolution=None):
        """

        Args:
            diffusion_batch_size: batch size of diffusion step. eg [clean, latent] -> B = 2
            total_steps_for_diffusion: slice 1000 into T steps. we usually take the last several steps
            self_attention_control_steps_section: start to end
            original_resolution: resolution of original image
            max_acceptable_latent_resolution: eg: ori=256, VAE-> 256/8 = 32,
                                            UNet down sample:16, 8, 4
        """
        # macro information, stage and process control
        self.current_diffusion_step = 0
        self.attention_layer_counter = None
        self.current_attention_layer_index = 0
        # attention map storage
        self.single_diffusion_step_attention_storage = _empty_attention_storage()
        self.original_resolution = original_resolution
        self.attention_sum_storage = {}
        if max_acceptable_latent_resolution:
            self.max_acceptable_latent_resolution = max_acceptable_latent_resolution
        else:
            self.max_acceptable_latent_resolution = self.original_resolution // 16
            if self.original_resolution % 16:
                warnings.warn("the resolution may illegal!", RuntimeWarning)
        # self_attention_loss processor
        self.batch_size = 2  # [attention_of_clean_latent, attn_adversarial_latent]
        self.total_steps_for_diffusion = total_steps_for_diffusion
        self.self_attention_control_steps_section = self_attention_control_steps_section
        self.self_attention_difference_loss = 0
        self.criterion = torch.nn.MSELoss()

    def __call__(self,
                 attention_map: torch.Tensor, # QK^T -> softmax -> attention_map. size[
                 is_cross_attention,
                 phase_of_unet):
        if self.current_attention_layer_index >= 0:
            batch_times_heads = attention_map.shape[0]
            # CFG: first 1/2 are unconditional, last 1/2 are conditional
            self.forward(attention_map[batch_times_heads // 2:], is_cross_attention, phase_of_unet)

        self.current_attention_layer_index += 1
        if self.current_attention_layer_index == self.attention_layer_counter:
            self.current_attention_layer_index = 0
            self.current_diffusion_step += 1
            self.update_storage_between_diffusion_steps()
        return attention_map

    def forward(self,
                attention_map: torch.Tensor,
                is_cross_attention: bool,
                phase_of_unet: str  # ["down", "mid", "up"]
                ):
        # store the attention map
        attention_storage_key = _get_attention_storage_key(phase_of_unet, is_cross_attention)
        attention_map_sequence_length = attention_map.shape[1]
        if attention_map_sequence_length <= self.max_acceptable_latent_resolution ** 2:
            self.single_diffusion_step_attention_storage[attention_storage_key].append(attention_map)
        # loss decision logic
        if is_cross_attention or self._in_self_attention_section():
            head_count = attention_map.shape[0] // self.batch_size
            attention_map = attention_map.reshape(self.batch_size, head_count, *attention_map.shape[1:])
            attention_clean, attention_adversarial = attention_map[0], attention_map[1:]
            if not is_cross_attention:
                _expanded_gt = _expand_attention_clean_to_adversarial(attention_clean, attention_adversarial)
                loss_per_pix = self.criterion(attention_adversarial, _expanded_gt) / (attention_map.shape[-1] ** 2)
                self.self_attention_difference_loss += loss_per_pix
            attention_map = attention_map.reshape(self.batch_size * head_count, *attention_map.shape[2:])

        return attention_map

    def update_storage_between_diffusion_steps(self):
        if len(self.attention_sum_storage) == 0:
            self.attention_sum_storage = self.single_diffusion_step_attention_storage
        else:
            for key in self.single_diffusion_step_attention_storage.keys():
                for i in range(len(self.single_diffusion_step_attention_storage[key])):
                    self.attention_sum_storage[key][i] = self.single_diffusion_step_attention_storage[key][i] + \
                                                         self.attention_sum_storage[key][i]
        self.single_diffusion_step_attention_storage = _empty_attention_storage()

    def controller_reset(self):
        self.current_diffusion_step = 0  # diffusion step counter. one for one
        self.current_attention_layer_index = 0

        self.single_diffusion_step_attention_storage = _empty_attention_storage()
        self.attention_sum_storage = {}

    def _in_self_attention_section(self):
        if self.self_attention_control_steps_section[0] <= self.current_diffusion_step \
                <= self.self_attention_control_steps_section[1]:
            return True
        return False

    def _get_average_attention(self):
        average_attention = {
            key: [item / self.current_diffusion_step for item in self.attention_sum_storage[key]]
            for key in self.attention_sum_storage.keys()
        }
        return average_attention

    def aggregate_attention_map(self,
                                prompts_of_latent: List,
                                target_map_resolution: int,
                                expected_unet_phases: Tuple[str, ...],
                                is_cross_attention: bool,
                                # select:int
                                # is_cpu=True
                                ):
        """

        Args:
            prompts_of_latent: eg: [prompt_clean, prompt_adv, prompt_adv2]
            target_map_resolution: we have different maps, 64x64,
                                   16x16 represents different scale of features
            expected_unet_phases: down, up, mid
            is_cross_attention:

        Returns: a total map represents the selected stage in whole diffusion progress
        shape: [Batch, res, res, feature] eg: [2, 16, 16, 77]

        """
        out = []
        average_attention = self._get_average_attention()
        latent_length = target_map_resolution ** 2
        for phase in expected_unet_phases:
            for item in average_attention[_get_attention_storage_key(phase, is_cross_attention)]:
                if item.shape[1] == latent_length:
                    cross_maps = item.reshape(len(prompts_of_latent), -1, target_map_resolution, target_map_resolution,
                                              item.shape[-1])
                    out.append(cross_maps)
        out = torch.cat(out, dim=1)  # B, count, res, res, feature
        out = out.sum(1) / out.shape[1]
        return out
