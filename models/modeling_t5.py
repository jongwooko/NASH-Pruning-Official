import copy
import math
import os
import warnings
from typing import Optional, Tuple, Union

import torch
from torch import nn
from torch.nn import CrossEntropyLoss
from torch.nn import functional as F

# from transformers.models.t5.modeling_t5 import T5Stack 
from transformers import AutoConfig, AutoModelForSequenceClassification, AutoModelForSeq2SeqLM, T5EncoderModel, T5PreTrainedModel, T5ForConditionalGeneration, T5Config
from transformers.modeling_outputs import BaseModelOutput, BaseModelOutputWithPastAndCrossAttentions, Seq2SeqLMOutput
from transformers.utils.model_parallel_utils import assert_device_map, get_device_map
from transformers.activations import ACT2FN
from transformers.modeling_utils import find_pruneable_heads_and_indices, prune_linear_layer, PreTrainedModel
from torch.utils.checkpoint import checkpoint

from utils.nash_utils import *

__HEAD_MASK_WARNING_MSG = """
The input argument `head_mask` was split into two arguments `head_mask` and `decoder_head_mask`. Currently,
`decoder_head_mask` is set to copy `head_mask`, but this feature is deprecated and will be removed in future versions.
If you do not want to use any `decoder_head_mask` now, please set `decoder_head_mask = torch.ones(num_layers,
num_heads)`.
"""

class NashT5LayerNorm(nn.Module):
    def __init__(self, hidden_size, eps=1e-6):
        """
        Construct a layernorm module in the T5 style No bias and no subtraction of mean.
        """
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.variance_epsilon = eps

    def forward(self, hidden_states, hidden_z=None):
        if hidden_z is not None:
            remaining_index = torch.where(~hidden_z.eq(0))[0]
            compressed_hidden_states = torch.index_select(
            hidden_states, dim=-1, index=remaining_index)
            compressed_weight = self.weight[remaining_index]
            variance = compressed_hidden_states.to(self.weight.dtype).pow(2).mean(-1, keepdim=True)
            compressed_hidden_states = compressed_hidden_states * torch.rsqrt(variance + self.variance_epsilon)
            compressed_hidden_states = compressed_weight * compressed_hidden_states
            out = hidden_states.clone()
            out[:, :, remaining_index] = compressed_hidden_states
        else:
            variance = hidden_states.to(torch.float32).pow(2).mean(-1, keepdim=True)
            hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)

        # convert into half-precision if necessary
            if self.weight.dtype in [torch.float16, torch.bfloat16]:
                hidden_states = hidden_states.to(self.weight.dtype)
            out = self.weight * hidden_states
        
        return out

__HEAD_MASK_WARNING_MSG = """
The input argument `head_mask` was split into two arguments `head_mask` and `decoder_head_mask`. Currently,
`decoder_head_mask` is set to copy `head_mask`, but this feature is deprecated and will be removed in future versions.
If you do not want to use any `decoder_head_mask` now, please set `decoder_head_mask = torch.ones(num_layers,
num_heads)`.
"""

class NashT5Block(nn.Module):
    def __init__(self, config, has_relative_attention_bias=False):
        super().__init__()
        self.is_decoder = config.is_decoder
        self.layer = nn.ModuleList()
        self.layer.append(NashT5LayerSelfAttention(config, has_relative_attention_bias=has_relative_attention_bias))
        
        if self.is_decoder:
            self.layer.append(NashT5LayerCrossAttention(config))

        self.layer.append(NashT5LayerFF(config))

    def forward(
        self,
        hidden_states,
        attention_mask=None,
        position_bias=None,
        encoder_hidden_states=None,
        encoder_attention_mask=None,
        encoder_decoder_position_bias=None,
        layer_head_mask=None,
        cross_attn_layer_head_mask=None,
        past_key_value=None,
        use_cache=False,
        output_attentions=False,
        return_dict=True,
        head_z=None,
        head_layer_z=None,
        intermediate_z=None,
        mlp_z=None,
        hidden_z=None,
        cross_head_z=None,
        cross_head_layer_z=None,
        self_index=None,
        cross_index=None,
    ):
        if past_key_value is not None:
            assert self.is_decoder, "Only decoder can use `past_key_values`"
            expected_num_past_key_values = 2 if encoder_hidden_states is None else 4

            if len(past_key_value) != expected_num_past_key_values:
                raise ValueError(
                    f"There should be {expected_num_past_key_values} past states. "
                    f"{'2 (past / key) for cross attention. ' if expected_num_past_key_values == 4 else ''}"
                    f"Got {len(past_key_value)} past key / value states"
                )

            self_attn_past_key_value = past_key_value[:2]
            cross_attn_past_key_value = past_key_value[2:]
        else:
            self_attn_past_key_value, cross_attn_past_key_value = None, None
        
        self_attention_outputs = self.layer[0](
            hidden_states,
            attention_mask=attention_mask,
            position_bias=position_bias,
            layer_head_mask=layer_head_mask,
            past_key_value=self_attn_past_key_value,
            use_cache=use_cache,
            output_attentions=output_attentions,
            head_z=head_z,
            head_layer_z=head_layer_z,
            hidden_z=hidden_z,
            heads_idx=self_index
        )
        hidden_states, present_key_value_state = self_attention_outputs[:2]
        attention_outputs = self_attention_outputs[2:] # keep self_att outputs and relative position bias
        
        # clamp inf values to enable fp16 training
        if hidden_states.dtype == torch.float16 and torch.isinf(hidden_states).any():
            clamp_value = torch.finfo(hidden_states.dtype).max - 1000
            hidden_states = torch.clamp(hidden_states, min=-clamp_value, max=clamp_value)

        do_cross_attention = self.is_decoder and encoder_hidden_states is not None
        if do_cross_attention:
            # the actual query length is unknown for cross attention
            # if using past key value states. Need to inject it here
            if present_key_value_state is not None:
                query_length = present_key_value_state[0].shape[2]
            else:
                query_length = None
                
            cross_attention_outputs = self.layer[1](
                hidden_states,
                key_value_states=encoder_hidden_states,
                attention_mask=encoder_attention_mask,
                position_bias=encoder_decoder_position_bias,
                layer_head_mask=cross_attn_layer_head_mask,
                past_key_value=cross_attn_past_key_value,
                query_length=query_length,
                use_cache=use_cache,
                output_attentions=output_attentions,
                head_z=cross_head_z,
                head_layer_z=cross_head_layer_z,
                hidden_z=hidden_z,
                heads_idx=cross_index
            )
            hidden_states = cross_attention_outputs[0]
            
            # clamp inf values to enable fp16 training
            if hidden_states.dtype == torch.float16 and torch.isinf(hidden_states).any():
                clamp_value = torch.finfo(hidden_states.dtype).max - 1000
                hidden_states = torch.clamp(hidden_states, min=-clamp_value, max=clamp_value)

            # Combine self attn and cross attn key value states
            if present_key_value_state is not None:
                if cross_attention_outputs[1] is None:
                    present_key_value_state = present_key_value_state + (None, None)
                else:
                    present_key_value_state = present_key_value_state + cross_attention_outputs[1]
            
            attention_outputs = attention_outputs + cross_attention_outputs[2:] # attention weights + enc_dec relative position_bias

        # Apply Feed Forward layer
        hidden_states = self.layer[-1](hidden_states, intermediate_z=intermediate_z, mlp_z=mlp_z, hidden_z=hidden_z, inference=False)
        # clamp inf values to enable fp16 training
        if hidden_states.dtype == torch.float16 and torch.isinf(hidden_states).any():
            clamp_value = torch.finfo(hidden_states.dtype).max - 1000
            hidden_states = torch.clamp(hidden_states, min=-clamp_value, max=clamp_value)

        outputs = (hidden_states,)
        if use_cache:
            outputs = outputs + (present_key_value_state,) + attention_outputs
        else:
            outputs = outputs + attention_outputs

        return outputs  # hidden-states, present_key_value_states, (self-attention position bias), (self-attention weights), (cross-attention position bias), (cross-attention weights)


class NashT5DenseReluDense(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.wi = nn.Linear(config.d_model, config.d_ff, bias=False)
        self.wo = nn.Linear(config.d_ff, config.d_model, bias=False)
        self.dropout = nn.Dropout(config.dropout_rate)

    def forward(self, hidden_states, intermediate_z):
        if self.wi is None:
            return None
        else:
            hidden_states = self.wi(hidden_states)
            hidden_states = nn.functional.relu(hidden_states)
            hidden_states = self.dropout(hidden_states)
            if intermediate_z is not None:
                hidden_states = hidden_states.mul(intermediate_z)
            hidden_states = self.wo(hidden_states)
        return hidden_states


class NashT5DenseGatedGeluDense(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.wi_0 = nn.Linear(config.d_model, config.d_ff, bias=False)
        self.wi_1 = nn.Linear(config.d_model, config.d_ff, bias=False)
        self.wo = nn.Linear(config.d_ff, config.d_model, bias=False)
        self.dropout = nn.Dropout(config.dropout_rate)
        self.gelu_act = ACT2FN["gelu_new"]

    def forward(self, hidden_states, intermediate_z):
        if self.wi_0 is None:
            return None
        else:
            hidden_gelu = self.gelu_act(self.wi_0(hidden_states))
            hidden_linear = self.wi_1(hidden_states)
            hidden_states = hidden_gelu * hidden_linear
            hidden_states = self.dropout(hidden_states)
            if intermediate_z is not None:
                hidden_states = hidden_states.mul(intermediate_z)
            hidden_states = self.wo(hidden_states)
        return hidden_states


class NashT5LayerFF(nn.Module):
    def __init__(self, config):
        super().__init__()
        if config.feed_forward_proj == "relu":
            self.DenseReluDense = NashT5DenseReluDense(config)
        elif config.feed_forward_proj == "gated-gelu":
            self.DenseReluDense = NashT5DenseGatedGeluDense(config)
        else:
            raise ValueError(
                f"{self.config.feed_forward_proj} is not supported. Choose between `relu` and `gated-gelu`"
            )

        self.layer_norm = NashT5LayerNorm(config.d_model, eps=config.layer_norm_epsilon)
        self.dropout = nn.Dropout(config.dropout_rate)

    def forward(self, hidden_states, intermediate_z=None, mlp_z=None, hidden_z=None, inference=False):
        forwarded_states = self.layer_norm(hidden_states, hidden_z)
        forwarded_states = self.DenseReluDense(forwarded_states, intermediate_z)
        if forwarded_states is None: # maybe works.
            return hidden_states
        if mlp_z is not None:
            forwarded_states *= mlp_z
        if not inference and forwarded_states.sum().eq(0).item():
            return hidden_states + forwarded_states
        else:
            if hidden_z is not None:
                forwarded_states = forwarded_states.mul(hidden_z)
            hidden_states = hidden_states + self.dropout(forwarded_states)
            if hidden_z is not None:
                hidden_states = hidden_states.mul(hidden_z)
        return hidden_states
###

class NashT5Attention(nn.Module):
    def __init__(self, config: T5Config, has_relative_attention_bias=False):
        super().__init__()
        self.is_decoder = config.is_decoder
        self.has_relative_attention_bias = has_relative_attention_bias

        self.relative_attention_num_buckets = config.relative_attention_num_buckets
        self.d_model = config.d_model
        self.key_value_proj_dim = config.d_kv
        self.n_heads = config.num_heads
        self.dropout = config.dropout_rate
        self.inner_dim = self.n_heads * self.key_value_proj_dim

        self.config = config

        # Mesh TensorFlow initialization to avoid scaling before softmax
        self.q = nn.Linear(self.d_model, self.inner_dim, bias=False)
        self.k = nn.Linear(self.d_model, self.inner_dim, bias=False)
        self.v = nn.Linear(self.d_model, self.inner_dim, bias=False)
        self.o = nn.Linear(self.inner_dim, self.d_model, bias=False)

        if self.has_relative_attention_bias:
            self.relative_attention_bias = nn.Embedding(self.relative_attention_num_buckets, self.n_heads)
        self.pruned_heads = set()
        self.gradient_checkpointing = False

    def reorder_heads(self, idx):
        n, a = self.n_heads, self.key_value_proj_dim
        index = torch.arange(n*a).reshape(n, a)[idx].view(-1).contiguous().long()

        def reorder_head_matrix(linearLayer, index, dim=0):
            index = index.to(linearLayer.weight.device)
            W = linearLayer.weight.index_select(dim, index).clone().detach()
            linearLayer.weight.requires_grad = False
            linearLayer.weight.copy_(W.contiguous())
            linearLayer.weight.requires_grad = True

        reorder_head_matrix(self.q, index)
        reorder_head_matrix(self.k, index)
        reorder_head_matrix(self.v, index)
        reorder_head_matrix(self.o, index, dim=1) 

    def reorder_PE_bias(self, index):
        index = index.to(self.relative_attention_bias.weight.device)
        W = self.relative_attention_bias.weight.index_select(1, index).clone().detach() # NOTE: nn.Linear, nn.Embedding are transposed 
        self.relative_attention_bias.weight.requires_grad = False
        self.relative_attention_bias.weight.copy_(W.contiguous())
        self.relative_attention_bias.weight.requires_grad = True

    def prune_heads(self, heads):
        if len(heads) == 0:
            return
        heads, index = find_pruneable_heads_and_indices(
            heads, self.n_heads, self.key_value_proj_dim, self.pruned_heads
        )

        if len(index) == 0:
            self.q = None
            self.k = None
            self.v = None
            self.o = None
        else:
            # Prune linear layers
            self.q = prune_linear_layer(self.q, index)
            self.k = prune_linear_layer(self.k, index)
            self.v = prune_linear_layer(self.v, index)
            self.o = prune_linear_layer(self.o, index, dim=1)
        # Update hyper params
        self.n_heads = self.n_heads - len(heads)
        self.inner_dim = self.key_value_proj_dim * self.n_heads
        self.pruned_heads = self.pruned_heads.union(heads)

    @staticmethod
    def _relative_position_bucket(relative_position, bidirectional=True, num_buckets=32, max_distance=128):
        """
        Adapted from Mesh Tensorflow:
        https://github.com/tensorflow/mesh/blob/0cb87fe07da627bf0b7e60475d59f95ed6b5be3d/mesh_tensorflow/transformer/transformer_layers.py#L593
        Translate relative position to a bucket number for relative attention. The relative position is defined as
        memory_position - query_position, i.e. the distance in tokens from the attending position to the attended-to
        position. If bidirectional=False, then positive relative positions are invalid. We use smaller buckets for
        small absolute relative_position and larger buckets for larger absolute relative_positions. All relative
        positions >=max_distance map to the same bucket. All relative positions <=-max_distance map to the same bucket.
        This should allow for more graceful generalization to longer sequences than the model has been trained on
        Args:
            relative_position: an int32 Tensor
            bidirectional: a boolean - whether the attention is bidirectional
            num_buckets: an integer
            max_distance: an integer
        Returns:
            a Tensor with the same shape as relative_position, containing int32 values in the range [0, num_buckets)
        """
        relative_buckets = 0
        if bidirectional:
            num_buckets //= 2
            relative_buckets += (relative_position > 0).to(torch.long) * num_buckets
            relative_position = torch.abs(relative_position)
        else:
            relative_position = -torch.min(relative_position, torch.zeros_like(relative_position))
        # now relative_position is in the range [0, inf)

        # half of the buckets are for exact increments in positions
        max_exact = num_buckets // 2
        is_small = relative_position < max_exact

        # The other half of the buckets are for logarithmically bigger bins in positions up to max_distance
        relative_postion_if_large = max_exact + (
            torch.log(relative_position.float() / max_exact)
            / math.log(max_distance / max_exact)
            * (num_buckets - max_exact)
        ).to(torch.long)
        relative_postion_if_large = torch.min(
            relative_postion_if_large, torch.full_like(relative_postion_if_large, num_buckets - 1)
        )

        relative_buckets += torch.where(is_small, relative_position, relative_postion_if_large)
        return relative_buckets

    def compute_bias(self, query_length, key_length): # 128, 128
        """Compute binned relative position bias"""
        context_position = torch.arange(
            query_length, dtype=torch.long, device=self.relative_attention_bias.weight.device
        )[:, None] # [128,1]
        memory_position = torch.arange(
            key_length, dtype=torch.long, device=self.relative_attention_bias.weight.device
        )[None, :] # [1,128]
        relative_position = memory_position - context_position  # shape (query_length, key_length)
        relative_position_bucket = self._relative_position_bucket(
            relative_position,  # shape (query_length, key_length)
            bidirectional=(not self.is_decoder),
            num_buckets=self.relative_attention_num_buckets,
        )
        values = self.relative_attention_bias(relative_position_bucket)  # shape (query_length, key_length, num_heads)
        values = values.permute([2, 0, 1]).unsqueeze(0)  # shape (1, num_heads, query_length, key_length)
        return values

    def forward(
        self,
        hidden_states,
        mask=None,
        key_value_states=None,
        position_bias=None,
        past_key_value=None,
        layer_head_mask=None,
        query_length=None,
        use_cache=False,
        output_attentions=False,
        head_z=None,
        heads_idx=None
    ):
        """
        Self-attention (if key_value_states is None) or attention over source sentence (provided by key_value_states).
        """
        # Input is (batch_size, seq_length, dim)
        # Mask is (batch_size, key_length) (non-causal) or (batch_size, key_length, key_length)
        # past_key_value[0] is (batch_size, n_heads, q_len - 1, dim_per_head)
        # outputs = (attn_output,) + (present_key_value_state,) + (position_bias,) if output_attentions (attn_weights,)
        if self.v is None:
            hidden_states = torch.zeros_like(hidden_states).to(hidden_states.device) #
            if self.has_relative_attention_bias:
                batch_size, seq_length = hidden_states.shape[:2]
                real_seq_length = seq_length
                key_length = real_seq_length if key_value_states is None else key_value_states.shape[1]
                position_bias = self.compute_bias(real_seq_length, key_length) # Stopgap.
                if mask is not None:
                    position_bias = position_bias + mask
            
            return (hidden_states, None, position_bias, None) if output_attentions else (hidden_states, None, position_bias)
        
        batch_size, seq_length = hidden_states.shape[:2]
        
        real_seq_length = seq_length

        if past_key_value is not None:
            assert (
                len(past_key_value) == 2
            ), f"past_key_value should have 2 past states: keys and values. Got { len(past_key_value)} past states"
            real_seq_length += past_key_value[0].shape[2] if query_length is None else query_length

        key_length = real_seq_length if key_value_states is None else key_value_states.shape[1]

        def shape(states):
            """projection"""
            return states.view(batch_size, -1, self.n_heads, self.key_value_proj_dim).transpose(1, 2)

        def unshape(states):
            """reshape"""
            return states.transpose(1, 2).contiguous().view(batch_size, -1, self.inner_dim)

        def project(hidden_states, proj_layer, key_value_states, past_key_value):
            """projects hidden states correctly to key/query states"""
            if key_value_states is None:
                # self-attn
                # (batch_size, n_heads, seq_length, dim_per_head)
                hidden_states = shape(proj_layer(hidden_states))
            elif past_key_value is None:
                # cross-attn
                # (batch_size, n_heads, seq_length, dim_per_head)
                hidden_states = shape(proj_layer(key_value_states))

            if past_key_value is not None:
                if key_value_states is None:
                    # self-attn
                    # (batch_size, n_heads, key_length, dim_per_head)
                    hidden_states = torch.cat([past_key_value, hidden_states], dim=2)
                else:
                    # cross-attn
                    hidden_states = past_key_value
            return hidden_states

        # get query states
        query_states = shape(self.q(hidden_states))  # (batch_size, n_heads, seq_length, dim_per_head)

        # get key/value states
        key_states = project(
            hidden_states, self.k, key_value_states, past_key_value[0] if past_key_value is not None else None
        )
        value_states = project(
            hidden_states, self.v, key_value_states, past_key_value[1] if past_key_value is not None else None
        )
        
        # compute scores
        scores = torch.matmul(
            query_states, key_states.transpose(3, 2)
        )  # equivalent of torch.einsum("bnqd,bnkd->bnqk", query_states, key_states), compatible with onnx op>9
        
        revised_position_bias = None 
        if position_bias is None:
            if not self.has_relative_attention_bias:
                position_bias = torch.zeros(
                    (1, self.config.num_heads, real_seq_length, key_length), device=scores.device, dtype=scores.dtype
                ) 
                if self.gradient_checkpointing and self.training:
                    position_bias.requires_grad = True
                    
            else: # True
                position_bias = self.compute_bias(real_seq_length, key_length) # [64, 12, 128, 128] # 
            # if key and values are already calculated
            # we want only the last query position bias
            if past_key_value is not None:
                position_bias = position_bias[:, :, -hidden_states.size(1) :, :]
            if mask is not None:
                position_bias = position_bias + mask  # (batch_size, n_heads, seq_length, key_length)
        
        if heads_idx is not None:
            revised_position_bias = position_bias[:, heads_idx, :, :]
        else:
            revised_position_bias = position_bias
        
        scores += revised_position_bias
        attn_weights = nn.functional.softmax(scores.float(), dim=-1).type_as(
            scores
        )  # (batch_size, n_heads, seq_length, key_length)
        attn_weights = nn.functional.dropout(
            attn_weights, p=self.dropout, training=self.training
        )  # (batch_size, n_heads, seq_length, key_length)

        # Mask heads if we want to
        if layer_head_mask is not None:
            attn_weights = attn_weights * layer_head_mask
        
        attn_output = torch.matmul(attn_weights, value_states) # Cofi edited
        if head_z is not None:
            attn_output *= head_z
        attn_output = unshape(attn_output)  # (batch_size, seq_length, dim)
        attn_output = self.o(attn_output)
        
        present_key_value_state = (key_states, value_states) if (self.is_decoder and use_cache) else None
        outputs = (attn_output,) + (present_key_value_state,) + (position_bias,)
        
        if output_attentions:
            outputs = outputs + (attn_weights,)
        return outputs

class NashT5LayerSelfAttention(nn.Module):
    def __init__(self, config, has_relative_attention_bias=False):
        super().__init__()
        self.SelfAttention = NashT5Attention(config, has_relative_attention_bias=has_relative_attention_bias)
        self.layer_norm = NashT5LayerNorm(config.d_model, eps=config.layer_norm_epsilon)
        self.dropout = nn.Dropout(config.dropout_rate)
        self.config = config
    def forward(
        self,
        hidden_states,
        attention_mask=None,
        position_bias=None,
        layer_head_mask=None,
        past_key_value=None,
        use_cache=False,
        output_attentions=False,
        head_z=None,
        head_layer_z=None,
        hidden_z=None,
        inference=False,
        heads_idx=None
    ):
        normed_hidden_states = self.layer_norm(hidden_states, hidden_z)
        attention_output = self.SelfAttention(
            normed_hidden_states,
            mask=attention_mask,
            position_bias=position_bias,
            layer_head_mask=layer_head_mask,
            past_key_value=past_key_value,
            use_cache=use_cache,
            output_attentions=output_attentions,
            head_z=head_z,
            heads_idx=heads_idx
        )

        att_output=attention_output[0]
        if head_layer_z is not None:
            att_output = att_output.mul(head_layer_z) # stopgap
        if not inference and att_output.sum().eq(0).item():
            hidden_states = hidden_states + att_output
        else:
            if hidden_z is not None:
                att_output = att_output.mul(hidden_z)
            hidden_states = hidden_states + self.dropout(att_output)
            if hidden_z is not None:
                hidden_states = hidden_states.mul(hidden_z)

        outputs = (hidden_states,) + attention_output[1:]  # add attentions if we output them
        return outputs

class NashT5LayerCrossAttention(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.EncDecAttention = NashT5Attention(config, has_relative_attention_bias=False)
        self.layer_norm = NashT5LayerNorm(config.d_model, eps=config.layer_norm_epsilon)
        self.dropout = nn.Dropout(config.dropout_rate)

    def forward(
        self,
        hidden_states,
        key_value_states,
        attention_mask=None,
        position_bias=None,
        layer_head_mask=None,
        past_key_value=None,
        use_cache=False,
        query_length=None,
        output_attentions=False,
        head_z=None, # added
        head_layer_z=None,
        hidden_z=None,
        inference=False,
        heads_idx=None
    ):

        normed_hidden_states = self.layer_norm(hidden_states, hidden_z)
        attention_output = self.EncDecAttention(
            normed_hidden_states,
            mask=attention_mask,
            key_value_states=key_value_states,
            position_bias=position_bias,
            layer_head_mask=layer_head_mask,
            past_key_value=past_key_value,
            use_cache=use_cache,
            query_length=query_length,
            output_attentions=output_attentions,
            head_z=head_z,
            heads_idx=heads_idx
        )
        att_output = attention_output[0]
        if head_layer_z is not None:
            att_output = att_output.mul(head_layer_z)
        if not inference and att_output.sum().eq(0).item():
            hidden_states = hidden_states + att_output
        else:
            if hidden_z is not None:
                att_output = att_output.mul(hidden_z)
            hidden_states = hidden_states + self.dropout(att_output)
            if hidden_z is not None:
                hidden_states = hidden_states.mul(hidden_z)
        
        outputs = (hidden_states,) + attention_output[1:] #+ (att_output,) # add attentions if we output them
        
        return outputs

class NashT5Stack(T5PreTrainedModel):
    def __init__(self, config, embed_tokens=None):
        super().__init__(config)

        self.embed_tokens = embed_tokens
        self.is_decoder = config.is_decoder
        
        if self.is_decoder:
            self.block = nn.ModuleList(
                [NashT5Block(config, has_relative_attention_bias=bool(i == 0)) for i in range(config.num_decoder_layers)]
            )
        else:
            self.block = nn.ModuleList(
                [NashT5Block(config, has_relative_attention_bias=bool(i == 0)) for i in range(config.num_layers)]
            )
        self.final_layer_norm = NashT5LayerNorm(config.d_model, eps=config.layer_norm_epsilon)
        self.dropout = nn.Dropout(config.dropout_rate)

        # Model parallel
        self.model_parallel = False
        self.device_map = None
        self.gradient_checkpointing = False
        # self.enc_head_idx = self.config.enc_pruned_head

    def parallelize(self, device_map=None):
        # Check validity of device_map
        self.device_map = (
            get_device_map(len(self.block), range(torch.cuda.device_count())) if device_map is None else device_map
        )
        assert_device_map(self.device_map, len(self.block))
        self.model_parallel = True
        self.first_device = "cpu" if "cpu" in self.device_map.keys() else "cuda:" + str(min(self.device_map.keys()))
        self.last_device = "cuda:" + str(max(self.device_map.keys()))
        # Load onto devices
        for k, v in self.device_map.items():
            for layer in v:
                cuda_device = "cuda:" + str(k)
                self.block[layer] = self.block[layer].to(cuda_device)

        # Set embed_tokens to first layer
        self.embed_tokens = self.embed_tokens.to(self.first_device)
        # Set final layer norm to last device
        self.final_layer_norm = self.final_layer_norm.to(self.last_device)


    def deparallelize(self):
        self.model_parallel = False
        self.device_map = None
        self.first_device = "cpu"
        self.last_device = "cpu"
        for i in range(len(self.block)):
            self.block[i] = self.block[i].to("cpu")
        self.embed_tokens = self.embed_tokens.to("cpu")
        self.final_layer_norm = self.final_layer_norm.to("cpu")
        torch.cuda.empty_cache()

    def get_input_embeddings(self):
        return self.embed_tokens

    def set_input_embeddings(self, new_embeddings):
        self.embed_tokens = new_embeddings

    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        encoder_hidden_states=None,
        encoder_attention_mask=None,
        inputs_embeds=None,
        head_mask=None,
        cross_attn_head_mask=None,
        past_key_values=None,
        use_cache=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
        head_z=None,
        head_layer_z=None,
        intermediate_z=None,
        mlp_z=None,
        hidden_z=None,
        cross_head_z=None,
        cross_head_layer_z=None,
    ):
        # Model parallel
        if self.model_parallel:
            torch.cuda.set_device(self.first_device)
            self.embed_tokens = self.embed_tokens.to(self.first_device)
        use_cache = use_cache if use_cache is not None else self.config.use_cache
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        if input_ids is not None and inputs_embeds is not None:
            err_msg_prefix = "decoder_" if self.is_decoder else ""
            raise ValueError(
                f"You cannot specify both {err_msg_prefix}input_ids and {err_msg_prefix}inputs_embeds at the same time"
            )
        elif input_ids is not None:
            input_shape = input_ids.size()
            input_ids = input_ids.view(-1, input_shape[-1])
        elif inputs_embeds is not None:
            input_shape = inputs_embeds.size()[:-1]
        else:
            err_msg_prefix = "decoder_" if self.is_decoder else ""
            raise ValueError(f"You have to specify either {err_msg_prefix}input_ids or {err_msg_prefix}inputs_embeds")

        if inputs_embeds is None:
            assert self.embed_tokens is not None, "You have to initialize the model with valid token embeddings"
            inputs_embeds = self.embed_tokens(input_ids)
            if hidden_z is not None:
                inputs_embeds = inputs_embeds.mul(hidden_z)

        batch_size, seq_length = input_shape

        # required mask seq length can be calculated via length of past
        # past_key_values[0][0] shape: (batch_size, n_heads, seq_length, dim_per_head)
        # past_key_values[0][0] may not be existed.
        # mask_seq_length = past_key_values[0][0].shape[2] + seq_length if past_key_values is not None else seq_length
        
        if past_key_values is not None:
            for ii in range(len(past_key_values)):
                if past_key_values[ii] is not None:
                    mask_seq_length = past_key_values[ii][0].shape[2] + seq_length
                    break
        else:
            mask_seq_length = seq_length
            
        if 'mask_seq_length' not in locals():
            mask_seq_length = seq_length 
        
        if use_cache is True:
            assert self.is_decoder, f"`use_cache` can only be set to `True` if {self} is used as a decoder"

        if attention_mask is None:
            attention_mask = torch.ones(batch_size, mask_seq_length).to(inputs_embeds.device)
        if self.is_decoder and encoder_attention_mask is None and encoder_hidden_states is not None:
            encoder_seq_length = encoder_hidden_states.shape[1]
            encoder_attention_mask = torch.ones(
                batch_size, encoder_seq_length, device=inputs_embeds.device, dtype=torch.long
            )

        # initialize past_key_values with `None` if past does not exist
        if past_key_values is None:
            past_key_values = [None] * len(self.block)

        # We can provide a self-attention mask of dimensions [batch_size, from_seq_length, to_seq_length]
        # ourselves in which case we just need to make it broadcastable to all heads.
        extended_attention_mask = self.get_extended_attention_mask(attention_mask, input_shape, inputs_embeds.device)

        # If a 2D or 3D attention mask is provided for the cross-attention
        # we need to make broadcastable to [batch_size, num_heads, seq_length, seq_length]
        if self.is_decoder and encoder_hidden_states is not None:
            encoder_batch_size, encoder_sequence_length, _ = encoder_hidden_states.size()
            encoder_hidden_shape = (encoder_batch_size, encoder_sequence_length)
            if encoder_attention_mask is None:
                encoder_attention_mask = torch.ones(encoder_hidden_shape, device=inputs_embeds.device)
            encoder_extended_attention_mask = self.invert_attention_mask(encoder_attention_mask)
        else:
            encoder_extended_attention_mask = None

        # Prepare head mask if needed
        head_mask = self.get_head_mask(head_mask, self.config.num_layers)
        cross_attn_head_mask = self.get_head_mask(cross_attn_head_mask, self.config.num_layers)
        present_key_value_states = () if use_cache else None
        all_hidden_states = () if output_hidden_states else None
        all_attentions = () if output_attentions else None
        all_cross_attentions = () if (output_attentions and self.is_decoder) else None
        position_bias = None
        encoder_decoder_position_bias = None

        hidden_states = self.dropout(inputs_embeds)

        origin_position_bias = None
        origin_encoder_decoder_position_bias = None


        
        for i, (layer_module, past_key_value) in enumerate(zip(self.block, past_key_values)):
            indices = [i for i in range(self.config.num_heads)]
            if hasattr(self.config, "enc_pruned_heads") and not self.is_decoder:
                try:
                    pruned = self.config.enc_pruned_heads[i]
                except KeyError:
                    pruned = self.config.enc_pruned_heads[str(i)]
                self_index = list(set(indices)-set(pruned))
                
            elif hasattr(self.config, "dec_self_pruned_heads") and self.is_decoder:
                try:
                    pruned = self.config.dec_self_pruned_heads[i]
                except KeyError:
                    pruned = self.config.dec_self_pruned_heads[str(i)]
                self_index = list(set(indices)-set(pruned))
                
            else:
                self_index = None
            
            if self.is_decoder and hasattr(self.config, "dec_cross_pruned_heads"):
                try:
                    pruned = self.config.dec_cross_pruned_heads[i]
                except KeyError:
                    pruned = self.config.dec_cross_pruned_heads[str(i)]
                cross_index = list(set(indices)-set(pruned))
            else:
                cross_index = None


            layer_head_mask = head_mask[i]
            cross_attn_layer_head_mask = cross_attn_head_mask[i]
            # Model parallel
            if self.model_parallel:
                torch.cuda.set_device(hidden_states.device)
                # Ensure that attention_mask is always on the same device as hidden_states
                if attention_mask is not None:
                    attention_mask = attention_mask.to(hidden_states.device)
                if position_bias is not None:
                    position_bias = position_bias.to(hidden_states.device)
                if encoder_hidden_states is not None:
                    encoder_hidden_states = encoder_hidden_states.to(hidden_states.device)
                if encoder_extended_attention_mask is not None:
                    encoder_extended_attention_mask = encoder_extended_attention_mask.to(hidden_states.device)
                if encoder_decoder_position_bias is not None:
                    encoder_decoder_position_bias = encoder_decoder_position_bias.to(hidden_states.device)
                if layer_head_mask is not None:
                    layer_head_mask = layer_head_mask.to(hidden_states.device)
                if cross_attn_layer_head_mask is not None:
                    cross_attn_layer_head_mask = cross_attn_layer_head_mask.to(hidden_states.device)
            if output_hidden_states and self.is_decoder:
                all_hidden_states = all_hidden_states + (self.dropout(self.final_layer_norm(hidden_states, hidden_z)),)
            elif output_hidden_states:
                all_hidden_states = all_hidden_states + (hidden_states,)
                                                         
            if self.gradient_checkpointing and self.training:
                if use_cache:
                    logger.warn(
                        "`use_cache=True` is incompatible with gradient checkpointing. Setting `use_cache=False`..."
                    )
                    use_cache = False

                def create_custom_forward(module):
                    def custom_forward(*inputs):
                        return tuple(module(*inputs, use_cache, output_attentions))

                    return custom_forward

                layer_outputs = checkpoint(
                    create_custom_forward(layer_module),
                    hidden_states,
                    extended_attention_mask,
                    position_bias,
                    encoder_hidden_states,
                    encoder_extended_attention_mask,
                    encoder_decoder_position_bias,
                    layer_head_mask,
                    cross_attn_layer_head_mask,
                    None,  # past_key_value is always None with gradient checkpointing
                )
            else:
                layer_outputs = layer_module(
                    hidden_states,
                    attention_mask=extended_attention_mask,
                    position_bias=position_bias,
                    encoder_hidden_states=encoder_hidden_states,
                    encoder_attention_mask=encoder_extended_attention_mask,
                    encoder_decoder_position_bias=encoder_decoder_position_bias,
                    layer_head_mask=layer_head_mask,
                    cross_attn_layer_head_mask=cross_attn_layer_head_mask,
                    past_key_value=past_key_value,
                    use_cache=use_cache,
                    output_attentions=output_attentions,
                    head_z=head_z[i] if head_z is not None else None,
                    head_layer_z=head_layer_z[i] if head_layer_z is not None else None,
                    intermediate_z=intermediate_z[i] if intermediate_z is not None else None,
                    mlp_z=mlp_z[i] if mlp_z is not None else None,
                    hidden_z=hidden_z,
                    cross_head_z=cross_head_z[i] if cross_head_z is not None else None,
                    cross_head_layer_z=cross_head_layer_z[i] if cross_head_layer_z is not None else None,
                    self_index=self_index if self_index is not None else None,
                    cross_index=cross_index if cross_index is not None else None
                )


            # layer_outputs is a tuple with:
            # hidden-states, key-value-states, (self-attention position bias), (self-attention weights), (cross-attention position bias), (cross-attention weights)
            if use_cache is False:
                layer_outputs = layer_outputs[:1] + (None,) + layer_outputs[1:]

            hidden_states, present_key_value_state = layer_outputs[:2]

            # We share the position biases between the layers - the first layer store them
            # layer_outputs = hidden-states, key-value-states (self-attention position bias), (self-attention weights),
            # (cross-attention position bias), (cross-attention weights)
            
            if origin_position_bias is None: 
                origin_position_bias = layer_outputs[2]
                position_bias = layer_outputs[2]
            if self.is_decoder and encoder_hidden_states is not None:
                if origin_encoder_decoder_position_bias is None: 
                    origin_encoder_decoder_position_bias = layer_outputs[4 if output_attentions else 3]

            # append next layer key value states
            if use_cache:
                present_key_value_states = present_key_value_states + (present_key_value_state,)
                
            if output_attentions:
                try:
                    all_attentions = all_attentions + (layer_outputs[3],)
                    if self.is_decoder:
                        all_cross_attentions = all_cross_attentions + (layer_outputs[5],)
                except:
                    # print(f"{i}_th {'decoder' if self.is_decoder else 'encoder'}layer: No attentions since all the weights are pruned")
                    all_attentions = (None)
                    if self.is_decoder:
                        all_cross_attentions = (None)

            # Model Parallel: If it's the last layer for that device, put things on the next device
            if self.model_parallel:
                for k, v in self.device_map.items():
                    if i == v[-1] and "cuda:" + str(k) != self.last_device:
                        hidden_states = hidden_states.to("cuda:" + str(k + 1))

        hidden_states = self.final_layer_norm(hidden_states, hidden_z)
        hidden_states = self.dropout(hidden_states)

        # Add last layer which is the output of final layer norm 
        if output_hidden_states:
            all_hidden_states = all_hidden_states + (hidden_states,)

        if not return_dict:
            return tuple(
                v
                for v in [
                    hidden_states,
                    present_key_value_states,
                    all_hidden_states,
                    all_attentions,
                    all_cross_attentions,
                ]
                if v is not None
            )
        return BaseModelOutputWithPastAndCrossAttentions(
            last_hidden_state=hidden_states,
            past_key_values=present_key_value_states,
            hidden_states=all_hidden_states,
            attentions=all_attentions,
            cross_attentions=all_cross_attentions,
        )

class NashT5ForConditionalGeneration(T5PreTrainedModel):
    _keys_to_ignore_on_load_missing = [
        r"encoder\.embed_tokens\.weight",
        r"decoder\.embed_tokens\.weight",
        r"lm_head\.weight",
    ]
    _keys_to_ignore_on_load_unexpected = [
        r"decoder\.block\.0\.layer\.1\.EncDecAttention\.relative_attention_bias\.weight",
    ]

    def __init__(self, config):
        super().__init__(config)

        self.do_layer_distill = getattr(config, "do_layer_distill", False)

        if self.do_layer_distill:
            self.layer_transformation = nn.Linear(
                config.hidden_size, config.hidden_size)
        else:
            self.layer_transformation = None  #added for cofi pruning.

        self.model_dim = config.d_model
        self.shared = nn.Embedding(config.vocab_size, config.d_model)

        encoder_config = copy.deepcopy(config)
        encoder_config.is_decoder = False
        encoder_config.use_cache = False
        encoder_config.is_encoder_decoder = False
        self.encoder = NashT5Stack(encoder_config, self.shared)

        decoder_config = copy.deepcopy(config)
        decoder_config.is_decoder = True
        decoder_config.is_encoder_decoder = False
        decoder_config.num_layers = config.num_decoder_layers
        self.decoder = NashT5Stack(decoder_config, self.shared)

        self.lm_head = nn.Linear(config.d_model, config.vocab_size, bias=False)

        # Model parallel
        self.model_parallel = False
        self.device_map = None

    def parallelize(self, device_map=None):
        self.device_map = (
            get_device_map(len(self.encoder.block), range(torch.cuda.device_count()))
            if device_map is None
            else device_map
        )
        assert_device_map(self.device_map, len(self.encoder.block))
        self.encoder.parallelize(self.device_map)
        self.decoder.parallelize(self.device_map)
        self.lm_head = self.lm_head.to(self.decoder.first_device)
        self.model_parallel = True

    def deparallelize(self):
        self.encoder.deparallelize()
        self.decoder.deparallelize()
        self.encoder = self.encoder.to("cpu")
        self.decoder = self.decoder.to("cpu")
        self.lm_head = self.lm_head.to("cpu")
        self.model_parallel = False
        self.device_map = None
        torch.cuda.empty_cache()

    def get_input_embeddings(self):
        return self.shared

    def set_input_embeddings(self, new_embeddings):
        self.shared = new_embeddings
        self.encoder.set_input_embeddings(new_embeddings)
        self.decoder.set_input_embeddings(new_embeddings)

    def set_output_embeddings(self, new_embeddings):
        self.lm_head = new_embeddings

    def get_output_embeddings(self):
        return self.lm_head

    def get_encoder(self):
        return self.encoder

    def get_decoder(self):
        return self.decoder

    def prune_heads(self, heads_to_prune):
        """
        Prunes heads of the base model.

        Arguments:
            heads_to_prune (:obj:`Dict[int, List[int]]`):
                Dictionary with keys being selected layer indices (:obj:`int`) and associated values being the list of
                heads to prune in said layer (list of :obj:`int`). For instance {1: [0, 2], 2: [2, 3]} will prune heads
                0 and 2 on layer 1 and heads 2 and 3 on layer 2.
        """
        # save new sets of pruned heads as union of previously stored pruned heads and newly pruned heads
        att_loc = heads_to_prune['att_loc']
        del heads_to_prune['att_loc']
        for layer, heads in heads_to_prune.items():
            union_heads = set(self.config.pruned_heads.get(layer, [])) | set(heads)
            # dec_self_union_heads = set(self.config.dec_self_pruned_heads.get(layer, [])) | set(heads)
            # dec_cross_union_heads = set(self.config.dec_cross_pruned_heads.get(layer, [])) | set(heads)
            if att_loc == 'enc_self':
                if layer == 0:
                    self.config.enc_pruned_heads = {}
                    self.encoder.config.enc_pruned_heads = {}
                self.config.enc_pruned_heads[layer] = list(union_heads)  # Unfortunately we have to store it as list for JSON
                self.encoder.config.enc_pruned_heads[layer] = list(union_heads)
            elif att_loc == 'dec_self':
                if layer == 0:
                    self.config.dec_self_pruned_heads = {}
                    self.decoder.config.dec_self_pruned_heads = {}
                self.config.dec_self_pruned_heads[layer] = list(union_heads)  # Unfortunately we have to store it as list for JSON
                self.decoder.config.dec_self_pruned_heads[layer] = list(union_heads)
            elif att_loc == 'dec_cross':
                if layer == 0:
                    self.config.dec_cross_pruned_heads = {}
                    self.decoder.config.dec_cross_pruned_heads = {}
                self.config.dec_cross_pruned_heads[layer] = list(union_heads)  # Unfortunately we have to store it as list for JSON
                self.decoder.config.dec_cross_pruned_heads[layer] = list(union_heads)
        self._prune_heads(heads_to_prune, att_loc)

    def _prune_heads(self, heads_to_prune, att_loc):
        """
        Prunes heads of the model. heads_to_prune: dict of {layer_num: list of heads to prune in this layer} See base
        class PreTrainedModel
        """        

        for layer, heads in heads_to_prune.items():
            if att_loc == "enc_self":
                self.encoder.block[layer].layer[0].SelfAttention.prune_heads(heads)
            elif att_loc == "dec_self":
                self.decoder.block[layer].layer[0].SelfAttention.prune_heads(heads)
            elif att_loc == "dec_cross":
                self.decoder.block[layer].layer[1].EncDecAttention.prune_heads(heads)
            else:
                raise NotImplementedError()

    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        decoder_input_ids=None,
        decoder_attention_mask=None,
        head_mask=None,
        decoder_head_mask=None,
        cross_attn_head_mask=None,
        encoder_outputs=None,
        past_key_values=None,
        inputs_embeds=None,
        decoder_inputs_embeds=None,
        labels=None,
        use_cache=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
        head_z=None,
        head_layer_z=None,
        intermediate_z=None,
        mlp_z=None,
        hidden_z=None,
        dec_self_head_z=None,
        dec_cross_head_z=None,
        dec_self_head_layer_z=None,
        dec_cross_head_layer_z=None,
        dec_intermediate_z=None,
        dec_mlp_z=None,
        dec_hidden_z=None
    ):

        use_cache = use_cache if use_cache is not None else self.config.use_cache
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        # FutureWarning: head_mask was separated into two input args - head_mask, decoder_head_mask
        if head_mask is not None and decoder_head_mask is None:
            if self.config.num_layers == self.config.num_decoder_layers:
                warnings.warn(__HEAD_MASK_WARNING_MSG, FutureWarning)
                decoder_head_mask = head_mask

        # Encode if needed (training, first prediction pass)
        if encoder_outputs is None:
            # Convert encoder inputs in embeddings if needed
            encoder_outputs = self.encoder(
                input_ids=input_ids,
                attention_mask=attention_mask,
                inputs_embeds=inputs_embeds,
                head_mask=head_mask,
                output_attentions=output_attentions,
                output_hidden_states=output_hidden_states,
                return_dict=return_dict,
                head_z=head_z,
                head_layer_z=head_layer_z,
                intermediate_z=intermediate_z,
                mlp_z=mlp_z,
                hidden_z=hidden_z
            )

        elif return_dict and not isinstance(encoder_outputs, BaseModelOutputWithPastAndCrossAttentions):
            encoder_outputs = BaseModelOutput(
                last_hidden_state=encoder_outputs[0],
                hidden_states=encoder_outputs[1] if len(encoder_outputs) > 1 else None,
                attentions=encoder_outputs[2] if len(encoder_outputs) > 2 else None,
            )

        hidden_states = encoder_outputs[0]

        if self.model_parallel:
            torch.cuda.set_device(self.decoder.first_device)

        if labels is not None and decoder_input_ids is None and decoder_inputs_embeds is None:
            labels = labels.long() # stopgap, need to figure out
            # get decoder inputs from shifting lm labels to the right
            decoder_input_ids = self._shift_right(labels)

        # Set device for model parallelism
        if self.model_parallel:
            torch.cuda.set_device(self.decoder.first_device)
            hidden_states = hidden_states.to(self.decoder.first_device)
            if decoder_input_ids is not None:
                decoder_input_ids = decoder_input_ids.to(self.decoder.first_device)
            if attention_mask is not None:
                attention_mask = attention_mask.to(self.decoder.first_device)
            if decoder_attention_mask is not None:
                decoder_attention_mask = decoder_attention_mask.to(self.decoder.first_device)

        # Decode
        decoder_outputs = self.decoder(
            input_ids=decoder_input_ids,
            attention_mask=decoder_attention_mask,
            inputs_embeds=decoder_inputs_embeds,
            past_key_values=past_key_values,
            encoder_hidden_states=hidden_states,
            encoder_attention_mask=attention_mask,
            head_mask=decoder_head_mask,
            cross_attn_head_mask=cross_attn_head_mask,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
            head_z=dec_self_head_z,
            head_layer_z=dec_self_head_layer_z,
            intermediate_z=dec_intermediate_z,
            mlp_z=dec_mlp_z,
            hidden_z=hidden_z,
            cross_head_z=dec_cross_head_z,
            cross_head_layer_z=dec_cross_head_layer_z
        )
        sequence_output = decoder_outputs[0]
        
        # Set device for model parallelism
        if self.model_parallel:
            torch.cuda.set_device(self.encoder.first_device)
            self.lm_head = self.lm_head.to(self.encoder.first_device)
            sequence_output = sequence_output.to(self.lm_head.weight.device)

        if self.config.tie_word_embeddings:
            # Rescale output before projecting on vocab
            # See https://github.com/tensorflow/mesh/blob/fa19d69eafc9a482aff0b59ddd96b025c0cb207d/mesh_tensorflow/transformer/transformer.py#L586
            sequence_output = sequence_output * (self.model_dim ** -0.5)

        lm_logits = self.lm_head(sequence_output)

        loss = None
        if labels is not None:
            loss_fct = CrossEntropyLoss(ignore_index=-100) #original ignore_index=-100
            loss = loss_fct(lm_logits.view(-1, lm_logits.size(-1)), labels.view(-1))
                
        if not return_dict:
            output = (lm_logits,) + decoder_outputs[1:] + encoder_outputs
            return ((loss,) + output) if loss is not None else output

        return Seq2SeqLMOutput(
            loss=loss,
            logits=lm_logits,
            past_key_values=decoder_outputs.past_key_values,
            decoder_hidden_states=decoder_outputs.hidden_states,
            decoder_attentions=decoder_outputs.attentions,
            cross_attentions=decoder_outputs.cross_attentions,
            encoder_last_hidden_state=encoder_outputs.last_hidden_state,
            encoder_hidden_states=encoder_outputs.hidden_states,
            encoder_attentions=encoder_outputs.attentions,
        )

    def prepare_inputs_for_generation(
        self,
        input_ids,
        past=None,
        attention_mask=None,
        head_mask=None,
        decoder_head_mask=None,
        cross_attn_head_mask=None,
        use_cache=None,
        encoder_outputs=None,
        **kwargs
    ):

        # cut decoder_input_ids if past is used
        if past is not None:
            input_ids = input_ids[:, -1:]

        return {
            "decoder_input_ids": input_ids,
            "past_key_values": past,
            "encoder_outputs": encoder_outputs,
            "attention_mask": attention_mask,
            "head_mask": head_mask,
            "decoder_head_mask": decoder_head_mask,
            "cross_attn_head_mask": cross_attn_head_mask,
            "use_cache": use_cache,
        }

    def prepare_decoder_input_ids_from_labels(self, labels: torch.Tensor):
        return self._shift_right(labels)

    def _reorder_cache(self, past, beam_idx):
        # if decoder past is not included in output
        # speedy decoding is disabled and no need to reorder
        if past is None:
            logger.warning("You might want to consider setting `use_cache=True` to speed up decoding")
            return past

        reordered_decoder_past = ()
        for layer_past_states in past:
            # get the correct batch idx from layer past batch dim
            # batch dim of `past` is at 2nd position
            reordered_layer_past_states = ()
            for layer_past_state in layer_past_states:
                # need to set correct `past` for each of the four key / value states
                reordered_layer_past_states = reordered_layer_past_states + (
                    layer_past_state.index_select(0, beam_idx.to(layer_past_state.device)),
                )

            assert reordered_layer_past_states[0].shape == layer_past_states[0].shape
            assert len(reordered_layer_past_states) == len(layer_past_states)

            reordered_decoder_past = reordered_decoder_past + (reordered_layer_past_states,)
        return reordered_decoder_past


