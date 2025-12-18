from typing import Optional
import torch

from attention_control import AttentionController


def monkey_head_to_batch_dim(head_size, tensor: torch.Tensor, out_dim: int = 3):
    if tensor.ndim == 3:
        batch_size, seq_len, dim = tensor.shape
        extra_dim = 1
    else:
        batch_size, extra_dim, seq_len, dim = tensor.shape
    tensor = tensor.reshape(batch_size, seq_len * extra_dim, head_size, dim // head_size)
    tensor = tensor.permute(0, 2, 1, 3)

    if out_dim == 3:
        tensor = tensor.reshape(batch_size * head_size, seq_len * extra_dim, dim // head_size)

    return tensor


def monkey_batch_to_head_dim(head_size, tensor: torch.Tensor) -> torch.Tensor:
    r"""
    Reshape the tensor from `[batch_size, seq_len, dim]` to `[batch_size // heads, seq_len, dim * heads]`. `heads`
    is the number of heads initialized while constructing the `Attention` class.

    Args:
        head_size:
        tensor (`torch.Tensor`): The tensor to reshape.

    Returns:
        `torch.Tensor`: The reshaped tensor.
    """
    batch_size, seq_len, dim = tensor.shape
    tensor = tensor.reshape(batch_size // head_size, head_size, seq_len, dim)
    tensor = tensor.permute(0, 2, 1, 3).reshape(batch_size // head_size, seq_len, dim * head_size)
    return tensor


def register_attention_control(model,
                               controller: AttentionController):
    def ca_forward(self, place_in_unet):
        def forward(
                hidden_states: torch.Tensor,
                encoder_hidden_states: Optional[torch.Tensor] = None,
                attention_mask: Optional[torch.Tensor] = None,
                **cross_attention_kwargs,
        ) -> torch.Tensor:
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
            temb = cross_attention_kwargs.get("temb")
            if temb is None:
                raise ValueError(f"time embedding is none!")

            residual = hidden_states
            if self.spatial_norm is not None:
                hidden_states = self.spatial_norm(hidden_states, temb)

            input_ndim = hidden_states.ndim

            if input_ndim == 4:
                # turn to [B, len, feature]
                batch_size, channel, height, width = hidden_states.shape
                hidden_states = hidden_states.view(batch_size, channel, height * width).transpose(1, 2)

            batch_size, sequence_length, _ = (
                hidden_states.shape if encoder_hidden_states is None else encoder_hidden_states.shape
            )
            attention_mask = self.prepare_attention_mask(attention_mask, sequence_length, batch_size)

            if self.group_norm is not None:
                hidden_states = self.group_norm(hidden_states.transpose(1, 2)).transpose(1, 2)

            query = self.to_q(hidden_states)

            is_cross_attention = encoder_hidden_states is not None

            if encoder_hidden_states is None:
                encoder_hidden_states = hidden_states
            elif self.norm_cross:
                encoder_hidden_states = self.norm_encoder_hidden_states(
                    encoder_hidden_states)
            key = self.to_k(encoder_hidden_states)
            value = self.to_v(encoder_hidden_states)

            query = monkey_head_to_batch_dim(self.heads, query)
            key = monkey_head_to_batch_dim(self.heads, key)
            value = monkey_head_to_batch_dim(self.heads, value)

            attention_probs = self.get_attention_scores(query, key, attention_mask)
            attention_probs = controller(attention_probs, is_cross_attention, place_in_unet)
            hidden_states = torch.bmm(attention_probs, value)
            hidden_states = monkey_batch_to_head_dim(self.heads, hidden_states)

            # linear proj
            hidden_states = self.to_out[0](hidden_states)
            # dropout
            hidden_states = self.to_out[1](hidden_states)

            if input_ndim == 4:
                hidden_states = hidden_states.transpose(-1, -2).reshape(batch_size, channel, height, width)

            if self.residual_connection:
                hidden_states = hidden_states + residual

            hidden_states = hidden_states / self.rescale_output_factor

            return hidden_states

        return forward

    def register_recorder(net_, count, place_in_unet):
        if net_.__class__.__name__ == "Attention":
            net_.forward = ca_forward(net_, place_in_unet)
            return count + 1
        elif hasattr(net_, "children"):
            for net__ in net_.children():
                count = register_recorder(net__, count, place_in_unet)
        return count

    attention_layer_counter = 0
    sub_nets = model.unet.named_children()
    for net in sub_nets:
        if "down" in net[0]:
            attention_layer_counter += register_recorder(net[1], 0, "down")
        elif "up" in net[0]:
            attention_layer_counter += register_recorder(net[1], 0, "up")
        elif "mid" in net[0]:
            attention_layer_counter += register_recorder(net[1], 0, "mid")
    controller.attention_layer_counter = attention_layer_counter


def release_attention_control(model,
                              controller: AttentionController):
    def ca_forward(self, place_in_unet):
        def forward(
                hidden_states: torch.Tensor,
                encoder_hidden_states: Optional[torch.Tensor] = None,
                attention_mask: Optional[torch.Tensor] = None,
                **cross_attention_kwargs,
        ) -> torch.Tensor:
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
            temb = cross_attention_kwargs.get("temb")
            if temb is None:
                raise ValueError(f"time embedding is none!")

            residual = hidden_states
            if self.spatial_norm is not None:
                hidden_states = self.spatial_norm(hidden_states, temb)

            input_ndim = hidden_states.ndim

            if input_ndim == 4:
                # turn to [B, len, feature]
                batch_size, channel, height, width = hidden_states.shape
                hidden_states = hidden_states.view(batch_size, channel, height * width).transpose(1, 2)

            batch_size, sequence_length, _ = (
                hidden_states.shape if encoder_hidden_states is None else encoder_hidden_states.shape
            )
            attention_mask = self.prepare_attention_mask(attention_mask, sequence_length, batch_size)

            if self.group_norm is not None:
                hidden_states = self.group_norm(hidden_states.transpose(1, 2)).transpose(1, 2)

            query = self.to_q(hidden_states)

            if encoder_hidden_states is None:
                encoder_hidden_states = hidden_states
            elif self.norm_cross:
                encoder_hidden_states = self.norm_encoder_hidden_states(
                    encoder_hidden_states)
            key = self.to_k(encoder_hidden_states)
            value = self.to_v(encoder_hidden_states)

            query = monkey_head_to_batch_dim(self.heads, query)
            key = monkey_head_to_batch_dim(self.heads, key)
            value = monkey_head_to_batch_dim(self.heads, value)

            attention_probs = self.get_attention_scores(query, key, attention_mask)
            hidden_states = torch.bmm(attention_probs, value)
            hidden_states = monkey_batch_to_head_dim(self.heads, hidden_states)

            # linear proj
            hidden_states = self.to_out[0](hidden_states)
            # dropout
            hidden_states = self.to_out[1](hidden_states)

            if input_ndim == 4:
                hidden_states = hidden_states.transpose(-1, -2).reshape(batch_size, channel, height, width)

            if self.residual_connection:
                hidden_states = hidden_states + residual

            hidden_states = hidden_states / self.rescale_output_factor

            return hidden_states

        return forward

    def register_recorder(net_, count, place_in_unet):
        if net_.__class__.__name__ == "Attention":
            net_.forward = ca_forward(net_, place_in_unet)
            return count + 1
        elif hasattr(net_, "children"):
            for net__ in net_.children():
                count = register_recorder(net__, count, place_in_unet)
        return count

    attention_layer_counter = 0
    sub_nets = model.unet.named_children()
    for net in sub_nets:
        if "down" in net[0]:
            attention_layer_counter += register_recorder(net[1], 0, "down")
        elif "up" in net[0]:
            attention_layer_counter += register_recorder(net[1], 0, "up")
        elif "mid" in net[0]:
            attention_layer_counter += register_recorder(net[1], 0, "mid")
    controller.attention_layer_counter = attention_layer_counter
