"""
File: UAP.py
Author: Atman
Date: 1/28/26
Description:
    
"""
import torch

class LMAGProcessor(torch.nn.Module):
    def __init__(self, prev_processor, name, collector):
        super().__init__()
        if prev_processor.__class__.__name__ == "AttnProcessor2_0":
            self.is_cross = False
        elif prev_processor.__class__.__name__ == "IPAdapterAttnProcessor2_0":
            self.is_cross = True
        else:
            raise NotImplementedError(f"Expected AttnProcessor2_0 or IPAdapterAttnProcessor2_0 but get {type(prev_processor)}")
        self.processor = prev_processor
        if "DownBlock" in name:
            self.stage = "down"
        elif "UpBlock" in name:
            self.stage = "up"
        elif "MidBlock" in name:
            self.stage = "mid"
        else:
            raise ValueError(f"No storage correspond to {name}")
        self.collector = collector

    def __call__(self,
                 attn,
                 hidden_states,
                 encoder_hidden_states=None,
                 attention_mask=None,
                 temb=None,
                 *args, **kwargs):
        ip_hidden_states = ip_adapter_masks = channel = height = width = None
        if self.is_cross:
            scale = kwargs.get("scale", float(1.0))
            ip_adapter_masks = kwargs.get("ip_adapter_masks", None)
        residual = hidden_states
        if isinstance(encoder_hidden_states, tuple):
            encoder_hidden_states, ip_hidden_states = encoder_hidden_states

        if attn.spatial_norm is not None:
            hidden_states = attn.spatial_norm(hidden_states, temb)

        input_ndim = hidden_states.ndim

        if input_ndim == 4:
            batch_size, channel, height, width = hidden_states.shape
            hidden_states = hidden_states.view(batch_size, channel, height * width).transpose(1, 2)

        batch_size, sequence_length, _ = (
            hidden_states.shape if encoder_hidden_states is None else encoder_hidden_states.shape
        )

        if attention_mask is not None:
            attention_mask = attn.prepare_attention_mask(attention_mask, sequence_length, batch_size)
            #attention_mask = attention_mask.view(batch_size, attn.heads, -1, attention_mask.shape[-1])

        if attn.group_norm is not None:
            hidden_states = attn.group_norm(hidden_states.transpose(1, 2)).transpose(1, 2)

        query = attn.to_q(hidden_states)

        if encoder_hidden_states is None:
            encoder_hidden_states = hidden_states
        elif attn.norm_cross:
            encoder_hidden_states = attn.norm_encoder_hidden_states(encoder_hidden_states)

        key = attn.to_k(encoder_hidden_states)
        value = attn.to_v(encoder_hidden_states)


        query = attn.head_to_batch_dim(query)
        key = attn.head_to_batch_dim(key)
        value = attn.head_to_batch_dim(value)

        attention_probs = attn.get_attention_scores(query, key, attention_mask)
        self.collector.save_attention(attention_probs, self.stage,
                                      "txt" if self.is_cross
                                      else "self")
        hidden_states = torch.bmm(attention_probs, value)
        hidden_states = attn.batch_to_head_dim(hidden_states)

        if self.is_cross:
            assert len(ip_hidden_states) == 1
            assert ip_adapter_masks is None
            ip_adapter_masks = [None] * 1

            for current_ip_hidden_states, scale, to_k_ip, to_v_ip, mask in zip(
                    ip_hidden_states, self.processor.scale,
                    self.processor.to_k_ip, self.processor.to_v_ip, ip_adapter_masks
            ):
                ip_key = to_k_ip(current_ip_hidden_states)
                ip_value = to_v_ip(current_ip_hidden_states)

                ip_key = attn.head_to_batch_dim(ip_key)
                ip_value = attn.head_to_batch_dim(ip_value)

                ip_hidden_probs = attn.get_attention_scores(query, ip_key, attention_mask=None)
                self.collector.save_attention(ip_hidden_probs, self.stage, "ip")
                current_ip_hidden_states = torch.bmm(ip_hidden_probs, ip_value)
                current_ip_hidden_states = attn.batch_to_head_dim(current_ip_hidden_states)
                hidden_states = hidden_states + scale * current_ip_hidden_states

        hidden_states = attn.to_out[0](hidden_states)
        hidden_states = attn.to_out[1](hidden_states)

        if input_ndim == 4:
            hidden_states = hidden_states.transpose(-1, -2).reshape(batch_size, channel, height, width)

        if attn.residual_connection:
            hidden_states = hidden_states + residual

        hidden_states = hidden_states / attn.rescale_output_factor

        return hidden_states


if __name__ == "__main__":
    print("pass validation")
