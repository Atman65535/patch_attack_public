"""
File: hijack_decorator.py
Author: Atman
Date: 12/7/25
Description: A wrapped method for attention control and Unet component replacement.
            But it is too hard to read and interpret.
    
"""
from typing import Optional
import torch

from diffusers.models.attention_processor import Attention
from diffusers.pipelines.stable_diffusion.pipeline_stable_diffusion import StableDiffusionPipeline

from attention_control import AttentionController

def hijack_recorder(net_,
                    target:str,
                    hijack_counter,
                    hijack_callable,
                    *args):
    """

    Args:
        net_: module to attack
        target: name of module
        hijack_counter: counter for hijack, set zero when calling in outer scope
        hijack_callable: some function, receive net_ and args
        *args: for hijack

    Returns: counter, indicates the invoking time of hijack_callable

    """
    if net_.__class__.__name__ is target:
        net_.forward = hijack_callable(net_, args)
        return hijack_counter + 1
    elif hasattr(net_, "children"):
        for net__ in net_.children():
            hijack_counter = hijack_recorder(net__, target, hijack_counter, hijack_callable, args)
    return hijack_counter

def replace_module_with_name(model:StableDiffusionPipeline,
                             controller:AttentionController,
                             target_name: str,
                             hijack_callable,
                             *args):
    attention_layer_counter = 0
    sub_nets = model.unet.named_children()
    for net in sub_nets:
        if "down" in net[0]:
            attention_layer_counter += hijack_recorder(net[1], target_name, 0, hijack_callable, controller,"down")
        elif "mid" in net[0]:
            attention_layer_counter += hijack_recorder(net[1], target_name, 0, hijack_callable, controller,"mid")
        elif "up" in net[0]:
            attention_layer_counter += hijack_recorder(net[1], target_name, 0, hijack_callable, controller,"up")
    controller.attention_layer_counter = attention_layer_counter

class AttentionContextHolder:
    def __init__(self,
                 context: Attention,
                 phase_of_unet: str):
        self.residual = None
        self.time_embedding = None
        self.context = context
        self.phase_of_unet = phase_of_unet
    def __call__(self, user_forward):
        def _attention(hidden_states: torch.Tensor,
                       encoder_hidden_states: Optional[torch.Tensor] = None,
                       attention_mask: Optional[torch.Tensor] = None,
                       **cross_attention_kwargs,):
            r"""
            The forward method of the `Attention` class.
            Args:
                hidden_states (`torch.Tensor`):
                    The hidden states of the query.
                encoder_hidden_states (`torch.Tensor`, *optional*):
                    The hidden states of the encoder.
                attention_mask (`torch.Tensor`, *optional*):
                    The attention mask to use. If `None`, no mask is applied.
                **cross_attention_kwargs:
                    Additional keyword arguments to pass along to the cross attention.

            Returns:
                `torch.Tensor`: The output of the attention layer.
            """
            self.time_embedding = cross_attention_kwargs.get("temb")
            if self.context.residual_connection:
                self.residual = hidden_states

            self.is_cross_attention = encoder_hidden_states is not None

            self._shape_process_with_norm(hidden_states, encoder_hidden_states)
            self.attention_mask = self.context.prepare_attention_mask(attention_mask,
                                                                      self.sequence_length,
                                                                      self.batch_size)
            hidden_states = self._group_norm_layer(hidden_states)
            query, key, value = self._get_qkv(hidden_states, encoder_hidden_states)

            attention_probs = self.context.get_attention_scores(query, key, attention_mask)

            attention_probs = user_forward(attention_probs, self.is_cross_attention, self.phase_of_unet)
            hidden_states = self._post_process(attention_probs, value)
            return hidden_states

        return _attention

    def _shape_process_with_norm(self, hidden_states, encoder_hidden_states):
        if self.context.spatial_norm is not None:
            hidden_states = self.context.spatial_norm(hidden_states, self.time_embedding)

        self.input_ndim = hidden_states.ndim
        if self.input_ndim == 4:
            self.batch_size, self.channel, self.height, self.width = hidden_states.shape
            hidden_states = hidden_states.view(self.batch_size, self.channel,
                                               self.height * self.width).transpose(1, 2)
        self.batch_size, self.sequence_length, _ = (
            hidden_states.shape if encoder_hidden_states is None else encoder_hidden_states.shape
        )

    def _group_norm_layer(self, hidden_states):
        if self.context.group_norm is not None:
            return self.context.group_norm(hidden_states.transpose(1, 2)).transpose(1, 2)
        else:
            return hidden_states

    def _get_qkv(self, hidden_states, encoder_hidden_states):
        query = self.context.to_q(hidden_states)
        if encoder_hidden_states is None:
            encoder_hidden_states = hidden_states
        elif self.context.norm_cross:
            encoder_hidden_states = self.context.norm_encoder_hidden_states(encoder_hidden_states)

        key = self.context.to_k(encoder_hidden_states)
        value = self.context.to_v(encoder_hidden_states)

        query = self.context.head_to_batch_dim(query)
        key = self.context.head_to_batch_dim(key)
        value = self.context.head_to_batch_dim(value)

        return query, key, value
    def _post_process(self, attention_probs, value):
        hidden_states = torch.bmm(attention_probs, value)
        hidden_states = self.context.batch_to_head_dim(hidden_states)
        hidden_states = self._out_proj(hidden_states)
        if self.input_ndim  == 4:
            hidden_states = hidden_states.transpose(-1, -2).reshape(self.batch_size,
                                                                    self.channel,
                                                                    self.height,
                                                                    self.width)
        if self.context.residual_connection:
            hidden_states = hidden_states + self.residual
        hidden_states = hidden_states / self.context.rescale_output_factor

        return hidden_states

    def _out_proj(self, hidden_states):
        hidden_states = self.context.to_out[0](hidden_states)
        hidden_states = self.context.to_out[1](hidden_states)
        return hidden_states


def hijack_wrapper(context, controller, phase_of_unet):
    @AttentionContextHolder(context, phase_of_unet)
    def inject_controller(attention_probs, is_cross_attention, phase_of_unet):
        attention_probs = controller(attention_probs, is_cross_attention, phase_of_unet)
        return attention_probs
    return inject_controller

if __name__ == "__main__":
    pass