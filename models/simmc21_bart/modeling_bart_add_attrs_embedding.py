# coding=utf-8
# Copyright 2020 The HuggingFace Team. All rights reserved.
# Copyright 2022 iFLYTEK, The State Key Laboratory of Cognitive Intelligence.

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.


""" PyTorch BART model for DSTC-11 SIMMC 2.1

Updated by Yirong Chen 
Used for [SIMMC 2.1](https://github.com/facebookresearch/simmc2)
Mail: [yrchen5@iflytek.com](mailto:yrchen5@iflytek.com) or [eeyirongchen@mail.scut.edu.cn](mailto:eeyirongchen@mail.scut.edu.cn)
Date: 2022/08/08


Usage: 

说明：本模型在原模型的基础上增加了歧义候选识别任务
更新记录：


"""
import copy
import math
import random
import warnings
from typing import List, Optional, Tuple, Union, Dict

import torch
import torch.utils.checkpoint
from torch import nn
from torch.nn import BCEWithLogitsLoss, CrossEntropyLoss, MSELoss

from transformers.activations import ACT2FN
from transformers.modeling_outputs import (
    BaseModelOutput,
    BaseModelOutputWithPastAndCrossAttentions,
    CausalLMOutputWithCrossAttentions,
    Seq2SeqLMOutput,
    Seq2SeqModelOutput,
    Seq2SeqQuestionAnsweringModelOutput,
    Seq2SeqSequenceClassifierOutput,
)
from transformers.modeling_utils import PreTrainedModel
from transformers.utils import (
    add_code_sample_docstrings,
    add_end_docstrings,
    add_start_docstrings,
    add_start_docstrings_to_model_forward,
    logging,
    replace_return_docstrings,
)

from .configuration_bart import BartConfig
from .modeling_bart_outputs import Seq2SeqLMOutputForSIMMC
from ..focal_loss import FocalLoss


logger = logging.get_logger(__name__)

_CHECKPOINT_FOR_DOC = "facebook/bart-base"
_CONFIG_FOR_DOC = "BartConfig"
_TOKENIZER_FOR_DOC = "BartTokenizer"

# Base model docstring
_EXPECTED_OUTPUT_SHAPE = [1, 8, 768]

# SequenceClassification docstring
_CHECKPOINT_FOR_SEQUENCE_CLASSIFICATION = "valhalla/bart-large-sst2"
_SEQ_CLASS_EXPECTED_LOSS = 0.0
_SEQ_CLASS_EXPECTED_OUTPUT = "'POSITIVE'"

# QuestionAsnwering docstring
_CHECKPOINT_FOR_QA = "valhalla/bart-large-finetuned-squadv1"
_QA_EXPECTED_LOSS = 0.59
_QA_EXPECTED_OUTPUT = "' nice puppet'"


BART_PRETRAINED_MODEL_ARCHIVE_LIST = [
    "facebook/bart-large",
    # see all BART models at https://huggingface.co/models?filter=bart
]


def shift_tokens_right(input_ids: torch.Tensor, pad_token_id: int, decoder_start_token_id: int):
    """
    Shift input ids one token to the right.
    """
    shifted_input_ids = input_ids.new_zeros(input_ids.shape)
    shifted_input_ids[:, 1:] = input_ids[:, :-1].clone()
    shifted_input_ids[:, 0] = decoder_start_token_id

    if pad_token_id is None:
        raise ValueError("self.model.config.pad_token_id has to be defined.")
    # replace possible -100 values in labels by `pad_token_id`
    shifted_input_ids.masked_fill_(shifted_input_ids == -100, pad_token_id)

    return shifted_input_ids


def _make_causal_mask(input_ids_shape: torch.Size, dtype: torch.dtype, past_key_values_length: int = 0):
    """
    Make causal mask used for bi-directional self-attention.
    """
    bsz, tgt_len = input_ids_shape
    mask = torch.full((tgt_len, tgt_len), torch.tensor(float("-inf")))
    mask_cond = torch.arange(mask.size(-1))
    mask.masked_fill_(mask_cond < (mask_cond + 1).view(mask.size(-1), 1), 0)
    mask = mask.to(dtype)

    if past_key_values_length > 0:
        mask = torch.cat([torch.zeros(tgt_len, past_key_values_length, dtype=dtype), mask], dim=-1)
    return mask[None, None, :, :].expand(bsz, 1, tgt_len, tgt_len + past_key_values_length)


def _expand_mask(mask: torch.Tensor, dtype: torch.dtype, tgt_len: Optional[int] = None):
    """
    Expands attention_mask from `[bsz, seq_len]` to `[bsz, 1, tgt_seq_len, src_seq_len]`.
    """
    bsz, src_len = mask.size()
    tgt_len = tgt_len if tgt_len is not None else src_len

    expanded_mask = mask[:, None, None, :].expand(bsz, 1, tgt_len, src_len).to(dtype)

    inverted_mask = 1.0 - expanded_mask

    return inverted_mask.masked_fill(inverted_mask.to(torch.bool), torch.finfo(dtype).min)


class BartLearnedPositionalEmbedding(nn.Embedding):
    """
    This module learns positional embeddings up to a fixed maximum size.
    """

    def __init__(self, num_embeddings: int, embedding_dim: int):
        # Bart is set up so that if padding_idx is specified then offset the embedding ids by 2
        # and adjust num_embeddings appropriately. Other models don't have this hack
        self.offset = 2
        super().__init__(num_embeddings + self.offset, embedding_dim)

    def forward(self, input_ids_shape: torch.Size, past_key_values_length: int = 0):
        """`input_ids_shape` is expected to be [bsz x seqlen]."""
        bsz, seq_len = input_ids_shape[:2]
        positions = torch.arange(
            past_key_values_length, past_key_values_length + seq_len, dtype=torch.long, device=self.weight.device
        )
        return super().forward(positions + self.offset)


class BartAttention(nn.Module):
    """Multi-headed attention from 'Attention Is All You Need' paper"""

    def __init__(
        self,
        embed_dim: int,
        num_heads: int,
        dropout: float = 0.0,
        is_decoder: bool = False,
        bias: bool = True,
    ):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.dropout = dropout
        self.head_dim = embed_dim // num_heads

        if (self.head_dim * num_heads) != self.embed_dim:
            raise ValueError(
                f"embed_dim must be divisible by num_heads (got `embed_dim`: {self.embed_dim}"
                f" and `num_heads`: {num_heads})."
            )
        self.scaling = self.head_dim**-0.5
        self.is_decoder = is_decoder

        self.k_proj = nn.Linear(embed_dim, embed_dim, bias=bias)
        self.v_proj = nn.Linear(embed_dim, embed_dim, bias=bias)
        self.q_proj = nn.Linear(embed_dim, embed_dim, bias=bias)
        self.out_proj = nn.Linear(embed_dim, embed_dim, bias=bias)

    def _shape(self, tensor: torch.Tensor, seq_len: int, bsz: int):
        return tensor.view(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2).contiguous()

    def forward(
        self,
        hidden_states: torch.Tensor,
        key_value_states: Optional[torch.Tensor] = None,
        past_key_value: Optional[Tuple[torch.Tensor]] = None,
        attention_mask: Optional[torch.Tensor] = None,
        layer_head_mask: Optional[torch.Tensor] = None,
        output_attentions: bool = False,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Tuple[torch.Tensor]]]:
        """Input shape: Batch x Time x Channel"""

        # if key_value_states are provided this layer is used as a cross-attention layer
        # for the decoder
        is_cross_attention = key_value_states is not None

        bsz, tgt_len, _ = hidden_states.size()

        # get query proj
        query_states = self.q_proj(hidden_states) * self.scaling
        # get key, value proj
        if is_cross_attention and past_key_value is not None:
            # reuse k,v, cross_attentions
            key_states = past_key_value[0]
            value_states = past_key_value[1]
        elif is_cross_attention:
            # cross_attentions
            key_states = self._shape(self.k_proj(key_value_states), -1, bsz)
            value_states = self._shape(self.v_proj(key_value_states), -1, bsz)
        elif past_key_value is not None:
            # reuse k, v, self_attention
            key_states = self._shape(self.k_proj(hidden_states), -1, bsz)
            value_states = self._shape(self.v_proj(hidden_states), -1, bsz)
            key_states = torch.cat([past_key_value[0], key_states], dim=2)
            value_states = torch.cat([past_key_value[1], value_states], dim=2)
        else:
            # self_attention
            key_states = self._shape(self.k_proj(hidden_states), -1, bsz)
            value_states = self._shape(self.v_proj(hidden_states), -1, bsz)

        if self.is_decoder:
            # if cross_attention save Tuple(torch.Tensor, torch.Tensor) of all cross attention key/value_states.
            # Further calls to cross_attention layer can then reuse all cross-attention
            # key/value_states (first "if" case)
            # if uni-directional self-attention (decoder) save Tuple(torch.Tensor, torch.Tensor) of
            # all previous decoder key/value_states. Further calls to uni-directional self-attention
            # can concat previous decoder key/value_states to current projected key/value_states (third "elif" case)
            # if encoder bi-directional self-attention `past_key_value` is always `None`
            past_key_value = (key_states, value_states)

        proj_shape = (bsz * self.num_heads, -1, self.head_dim)
        query_states = self._shape(query_states, tgt_len, bsz).view(*proj_shape)
        key_states = key_states.view(*proj_shape)
        value_states = value_states.view(*proj_shape)

        src_len = key_states.size(1)
        attn_weights = torch.bmm(query_states, key_states.transpose(1, 2))

        if attn_weights.size() != (bsz * self.num_heads, tgt_len, src_len):
            raise ValueError(
                f"Attention weights should be of size {(bsz * self.num_heads, tgt_len, src_len)}, but is"
                f" {attn_weights.size()}"
            )

        if attention_mask is not None:
            if attention_mask.size() != (bsz, 1, tgt_len, src_len):
                raise ValueError(
                    f"Attention mask should be of size {(bsz, 1, tgt_len, src_len)}, but is {attention_mask.size()}"
                )
            attn_weights = attn_weights.view(bsz, self.num_heads, tgt_len, src_len) + attention_mask
            attn_weights = attn_weights.view(bsz * self.num_heads, tgt_len, src_len)

        attn_weights = nn.functional.softmax(attn_weights, dim=-1)

        if layer_head_mask is not None:
            if layer_head_mask.size() != (self.num_heads,):
                raise ValueError(
                    f"Head mask for a single layer should be of size {(self.num_heads,)}, but is"
                    f" {layer_head_mask.size()}"
                )
            attn_weights = layer_head_mask.view(1, -1, 1, 1) * attn_weights.view(bsz, self.num_heads, tgt_len, src_len)
            attn_weights = attn_weights.view(bsz * self.num_heads, tgt_len, src_len)

        if output_attentions:
            # this operation is a bit awkward, but it's required to
            # make sure that attn_weights keeps its gradient.
            # In order to do so, attn_weights have to be reshaped
            # twice and have to be reused in the following
            attn_weights_reshaped = attn_weights.view(bsz, self.num_heads, tgt_len, src_len)
            attn_weights = attn_weights_reshaped.view(bsz * self.num_heads, tgt_len, src_len)
        else:
            attn_weights_reshaped = None

        attn_probs = nn.functional.dropout(attn_weights, p=self.dropout, training=self.training)

        attn_output = torch.bmm(attn_probs, value_states)

        if attn_output.size() != (bsz * self.num_heads, tgt_len, self.head_dim):
            raise ValueError(
                f"`attn_output` should be of size {(bsz, self.num_heads, tgt_len, self.head_dim)}, but is"
                f" {attn_output.size()}"
            )

        attn_output = attn_output.view(bsz, self.num_heads, tgt_len, self.head_dim)
        attn_output = attn_output.transpose(1, 2)

        # Use the `embed_dim` from the config (stored in the class) rather than `hidden_state` because `attn_output` can be
        # partitioned aross GPUs when using tensor-parallelism.
        attn_output = attn_output.reshape(bsz, tgt_len, self.embed_dim)

        attn_output = self.out_proj(attn_output)

        return attn_output, attn_weights_reshaped, past_key_value


class BartEncoderLayer(nn.Module):
    def __init__(self, config: BartConfig):
        super().__init__()
        self.embed_dim = config.d_model
        self.self_attn = BartAttention(
            embed_dim=self.embed_dim,
            num_heads=config.encoder_attention_heads,
            dropout=config.attention_dropout,
        )
        self.self_attn_layer_norm = nn.LayerNorm(self.embed_dim)
        self.dropout = config.dropout
        self.activation_fn = ACT2FN[config.activation_function]
        self.activation_dropout = config.activation_dropout
        self.fc1 = nn.Linear(self.embed_dim, config.encoder_ffn_dim)
        self.fc2 = nn.Linear(config.encoder_ffn_dim, self.embed_dim)
        self.final_layer_norm = nn.LayerNorm(self.embed_dim)

    def forward(
        self,
        hidden_states: torch.FloatTensor,
        attention_mask: torch.FloatTensor,
        layer_head_mask: torch.FloatTensor,
        output_attentions: Optional[bool] = False,
    ) -> Tuple[torch.FloatTensor, Optional[torch.FloatTensor]]:
        """
        Args:
            hidden_states (`torch.FloatTensor`): input to the layer of shape `(seq_len, batch, embed_dim)`
            attention_mask (`torch.FloatTensor`): attention mask of size
                `(batch, 1, tgt_len, src_len)` where padding elements are indicated by very large negative values.
            layer_head_mask (`torch.FloatTensor`): mask for attention heads in a given layer of size
                `(encoder_attention_heads,)`.
            output_attentions (`bool`, *optional*):
                Whether or not to return the attentions tensors of all attention layers. See `attentions` under
                returned tensors for more detail.
        """
        residual = hidden_states
        hidden_states, attn_weights, _ = self.self_attn(
            hidden_states=hidden_states,
            attention_mask=attention_mask,
            layer_head_mask=layer_head_mask,
            output_attentions=output_attentions,
        )
        hidden_states = nn.functional.dropout(hidden_states, p=self.dropout, training=self.training)
        hidden_states = residual + hidden_states
        hidden_states = self.self_attn_layer_norm(hidden_states)

        residual = hidden_states
        hidden_states = self.activation_fn(self.fc1(hidden_states))
        hidden_states = nn.functional.dropout(hidden_states, p=self.activation_dropout, training=self.training)
        hidden_states = self.fc2(hidden_states)
        hidden_states = nn.functional.dropout(hidden_states, p=self.dropout, training=self.training)
        hidden_states = residual + hidden_states
        hidden_states = self.final_layer_norm(hidden_states)

        if hidden_states.dtype == torch.float16 and (
            torch.isinf(hidden_states).any() or torch.isnan(hidden_states).any()
        ):
            clamp_value = torch.finfo(hidden_states.dtype).max - 1000
            hidden_states = torch.clamp(hidden_states, min=-clamp_value, max=clamp_value)

        outputs = (hidden_states,)

        if output_attentions:
            outputs += (attn_weights,)

        return outputs


class BartDecoderLayer(nn.Module):
    def __init__(self, config: BartConfig):
        super().__init__()
        self.embed_dim = config.d_model

        self.self_attn = BartAttention(
            embed_dim=self.embed_dim,
            num_heads=config.decoder_attention_heads,
            dropout=config.attention_dropout,
            is_decoder=True,
        )
        self.dropout = config.dropout
        self.activation_fn = ACT2FN[config.activation_function]
        self.activation_dropout = config.activation_dropout

        self.self_attn_layer_norm = nn.LayerNorm(self.embed_dim)
        self.encoder_attn = BartAttention(
            self.embed_dim,
            config.decoder_attention_heads,
            dropout=config.attention_dropout,
            is_decoder=True,
        )
        self.encoder_attn_layer_norm = nn.LayerNorm(self.embed_dim)
        self.fc1 = nn.Linear(self.embed_dim, config.decoder_ffn_dim)
        self.fc2 = nn.Linear(config.decoder_ffn_dim, self.embed_dim)
        self.final_layer_norm = nn.LayerNorm(self.embed_dim)

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        encoder_hidden_states: Optional[torch.Tensor] = None,
        encoder_attention_mask: Optional[torch.Tensor] = None,
        layer_head_mask: Optional[torch.Tensor] = None,
        cross_attn_layer_head_mask: Optional[torch.Tensor] = None,
        past_key_value: Optional[Tuple[torch.Tensor]] = None,
        output_attentions: Optional[bool] = False,
        use_cache: Optional[bool] = True,
    ) -> Tuple[torch.FloatTensor, Optional[Tuple[torch.FloatTensor, torch.FloatTensor]]]:
        """
        Args:
            hidden_states (`torch.FloatTensor`): input to the layer of shape `(batch, seq_len, embed_dim)`
            attention_mask (`torch.FloatTensor`): attention mask of size
                `(batch, 1, tgt_len, src_len)` where padding elements are indicated by very large negative values.
            encoder_hidden_states (`torch.FloatTensor`):
                cross attention input to the layer of shape `(batch, seq_len, embed_dim)`
            encoder_attention_mask (`torch.FloatTensor`): encoder attention mask of size
                `(batch, 1, tgt_len, src_len)` where padding elements are indicated by very large negative values.
            layer_head_mask (`torch.FloatTensor`): mask for attention heads in a given layer of size
                `(encoder_attention_heads,)`.
            cross_attn_layer_head_mask (`torch.FloatTensor`): mask for cross-attention heads in a given layer of
                size `(decoder_attention_heads,)`.
            past_key_value (`Tuple(torch.FloatTensor)`): cached past key and value projection states
            output_attentions (`bool`, *optional*):
                Whether or not to return the attentions tensors of all attention layers. See `attentions` under
                returned tensors for more detail.
        """
        residual = hidden_states

        # Self Attention
        # decoder uni-directional self-attention cached key/values tuple is at positions 1,2
        self_attn_past_key_value = past_key_value[:2] if past_key_value is not None else None
        # add present self-attn cache to positions 1,2 of present_key_value tuple
        hidden_states, self_attn_weights, present_key_value = self.self_attn(
            hidden_states=hidden_states,
            past_key_value=self_attn_past_key_value,
            attention_mask=attention_mask,
            layer_head_mask=layer_head_mask,
            output_attentions=output_attentions,
        )
        hidden_states = nn.functional.dropout(hidden_states, p=self.dropout, training=self.training)
        hidden_states = residual + hidden_states
        hidden_states = self.self_attn_layer_norm(hidden_states)

        # Cross-Attention Block
        cross_attn_present_key_value = None
        cross_attn_weights = None
        if encoder_hidden_states is not None:
            residual = hidden_states

            # cross_attn cached key/values tuple is at positions 3,4 of present_key_value tuple
            cross_attn_past_key_value = past_key_value[-2:] if past_key_value is not None else None
            hidden_states, cross_attn_weights, cross_attn_present_key_value = self.encoder_attn(
                hidden_states=hidden_states,
                key_value_states=encoder_hidden_states,
                attention_mask=encoder_attention_mask,
                layer_head_mask=cross_attn_layer_head_mask,
                past_key_value=cross_attn_past_key_value,
                output_attentions=output_attentions,
            )
            hidden_states = nn.functional.dropout(hidden_states, p=self.dropout, training=self.training)
            hidden_states = residual + hidden_states
            hidden_states = self.encoder_attn_layer_norm(hidden_states)

            # add cross-attn to positions 3,4 of present_key_value tuple
            present_key_value = present_key_value + cross_attn_present_key_value

        # Fully Connected
        residual = hidden_states
        hidden_states = self.activation_fn(self.fc1(hidden_states))
        hidden_states = nn.functional.dropout(hidden_states, p=self.activation_dropout, training=self.training)
        hidden_states = self.fc2(hidden_states)
        hidden_states = nn.functional.dropout(hidden_states, p=self.dropout, training=self.training)
        hidden_states = residual + hidden_states
        hidden_states = self.final_layer_norm(hidden_states)

        outputs = (hidden_states,)

        if output_attentions:
            outputs += (self_attn_weights, cross_attn_weights)

        if use_cache:
            outputs += (present_key_value,)

        return outputs


class BartClassificationHead(nn.Module):
    """Head for sentence-level classification tasks."""

    def __init__(
        self,
        input_dim: int,
        inner_dim: int,
        num_classes: int,
        pooler_dropout: float,
    ):
        super().__init__()
        self.dense = nn.Linear(input_dim, inner_dim)
        self.dropout = nn.Dropout(p=pooler_dropout)
        self.out_proj = nn.Linear(inner_dim, num_classes)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        hidden_states = self.dropout(hidden_states)
        hidden_states = self.dense(hidden_states)
        hidden_states = torch.tanh(hidden_states)
        hidden_states = self.dropout(hidden_states)
        hidden_states = self.out_proj(hidden_states)
        return hidden_states


class BartPretrainedModel(PreTrainedModel):
    config_class = BartConfig
    base_model_prefix = "model"
    supports_gradient_checkpointing = True
    _keys_to_ignore_on_load_unexpected = [r"encoder.version", r"decoder.version"]

    def _init_weights(self, module):
        std = self.config.init_std
        if isinstance(module, nn.Linear):
            module.weight.data.normal_(mean=0.0, std=std)
            if module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.Embedding):
            module.weight.data.normal_(mean=0.0, std=std)
            if module.padding_idx is not None:
                module.weight.data[module.padding_idx].zero_()

    def _set_gradient_checkpointing(self, module, value=False):
        if isinstance(module, (BartDecoder, BartEncoder)):
            module.gradient_checkpointing = value

    @property
    def dummy_inputs(self):
        pad_token = self.config.pad_token_id
        input_ids = torch.tensor([[0, 6, 10, 4, 2], [0, 8, 12, 2, pad_token]], device=self.device)
        dummy_inputs = {
            "attention_mask": input_ids.ne(pad_token),
            "input_ids": input_ids,
        }
        return dummy_inputs


class PretrainedBartModel(BartPretrainedModel):
    def __init_subclass__(self):
        warnings.warn(
            "The class `PretrainedBartModel` has been depreciated, please use `BartPretrainedModel` instead.",
            FutureWarning,
        )


BART_START_DOCSTRING = r"""
    This model inherits from [`PreTrainedModel`]. Check the superclass documentation for the generic methods the
    library implements for all its model (such as downloading or saving, resizing the input embeddings, pruning heads
    etc.)

    This model is also a PyTorch [torch.nn.Module](https://pytorch.org/docs/stable/nn.html#torch.nn.Module) subclass.
    Use it as a regular PyTorch Module and refer to the PyTorch documentation for all matter related to general usage
    and behavior.

    Parameters:
        config ([`BartConfig`]):
            Model configuration class with all the parameters of the model. Initializing with a config file does not
            load the weights associated with the model, only the configuration. Check out the
            [`~PreTrainedModel.from_pretrained`] method to load the model weights.
"""

BART_GENERATION_EXAMPLE = r"""
    Summarization example:

    ```python
    >>> from transformers import BartTokenizer, BartForConditionalGeneration

    >>> model = BartForConditionalGeneration.from_pretrained("facebook/bart-large-cnn")
    >>> tokenizer = BartTokenizer.from_pretrained("facebook/bart-large-cnn")

    >>> ARTICLE_TO_SUMMARIZE = (
    ...     "PG&E stated it scheduled the blackouts in response to forecasts for high winds "
    ...     "amid dry conditions. The aim is to reduce the risk of wildfires. Nearly 800 thousand customers were "
    ...     "scheduled to be affected by the shutoffs which were expected to last through at least midday tomorrow."
    ... )
    >>> inputs = tokenizer([ARTICLE_TO_SUMMARIZE], max_length=1024, return_tensors="pt")

    >>> # Generate Summary
    >>> summary_ids = model.generate(inputs["input_ids"], num_beams=2, min_length=0, max_length=20)
    >>> tokenizer.batch_decode(summary_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]
    'PG&E scheduled the blackouts in response to forecasts for high winds amid dry conditions'
    ```

    Mask filling example:

    ```python
    >>> from transformers import BartTokenizer, BartForConditionalGeneration

    >>> tokenizer = BartTokenizer.from_pretrained("facebook/bart-base")
    >>> model = BartForConditionalGeneration.from_pretrained("facebook/bart-base")

    >>> TXT = "My friends are <mask> but they eat too many carbs."
    >>> input_ids = tokenizer([TXT], return_tensors="pt")["input_ids"]
    >>> logits = model(input_ids).logits

    >>> masked_index = (input_ids[0] == tokenizer.mask_token_id).nonzero().item()
    >>> probs = logits[0, masked_index].softmax(dim=0)
    >>> values, predictions = probs.topk(5)

    >>> tokenizer.decode(predictions).split()
    ['not', 'good', 'healthy', 'great', 'very']
    ```
"""

BART_INPUTS_DOCSTRING = r"""
    Args:
        input_ids (`torch.LongTensor` of shape `(batch_size, sequence_length)`):
            Indices of input sequence tokens in the vocabulary. Padding will be ignored by default should you provide
            it.

            Indices can be obtained using [`BartTokenizer`]. See [`PreTrainedTokenizer.encode`] and
            [`PreTrainedTokenizer.__call__`] for details.

            [What are input IDs?](../glossary#input-ids)
        attention_mask (`torch.Tensor` of shape `(batch_size, sequence_length)`, *optional*):
            Mask to avoid performing attention on padding token indices. Mask values selected in `[0, 1]`:

            - 1 for tokens that are **not masked**,
            - 0 for tokens that are **masked**.

            [What are attention masks?](../glossary#attention-mask)
        decoder_input_ids (`torch.LongTensor` of shape `(batch_size, target_sequence_length)`, *optional*):
            Indices of decoder input sequence tokens in the vocabulary.

            Indices can be obtained using [`BartTokenizer`]. See [`PreTrainedTokenizer.encode`] and
            [`PreTrainedTokenizer.__call__`] for details.

            [What are decoder input IDs?](../glossary#decoder-input-ids)

            Bart uses the `eos_token_id` as the starting token for `decoder_input_ids` generation. If `past_key_values`
            is used, optionally only the last `decoder_input_ids` have to be input (see `past_key_values`).

            For translation and summarization training, `decoder_input_ids` should be provided. If no
            `decoder_input_ids` is provided, the model will create this tensor by shifting the `input_ids` to the right
            for denoising pre-training following the paper.
        decoder_attention_mask (`torch.LongTensor` of shape `(batch_size, target_sequence_length)`, *optional*):
            Default behavior: generate a tensor that ignores pad tokens in `decoder_input_ids`. Causal mask will also
            be used by default.

            If you want to change padding behavior, you should read [`modeling_bart._prepare_decoder_attention_mask`]
            and modify to your needs. See diagram 1 in [the paper](https://arxiv.org/abs/1910.13461) for more
            information on the default strategy.
        head_mask (`torch.Tensor` of shape `(encoder_layers, encoder_attention_heads)`, *optional*):
            Mask to nullify selected heads of the attention modules in the encoder. Mask values selected in `[0, 1]`:

            - 1 indicates the head is **not masked**,
            - 0 indicates the head is **masked**.

        decoder_head_mask (`torch.Tensor` of shape `(decoder_layers, decoder_attention_heads)`, *optional*):
            Mask to nullify selected heads of the attention modules in the decoder. Mask values selected in `[0, 1]`:

            - 1 indicates the head is **not masked**,
            - 0 indicates the head is **masked**.

        cross_attn_head_mask (`torch.Tensor` of shape `(decoder_layers, decoder_attention_heads)`, *optional*):
            Mask to nullify selected heads of the cross-attention modules in the decoder. Mask values selected in `[0,
            1]`:

            - 1 indicates the head is **not masked**,
            - 0 indicates the head is **masked**.

        encoder_outputs (`tuple(tuple(torch.FloatTensor)`, *optional*):
            Tuple consists of (`last_hidden_state`, *optional*: `hidden_states`, *optional*: `attentions`)
            `last_hidden_state` of shape `(batch_size, sequence_length, hidden_size)`, *optional*) is a sequence of
            hidden-states at the output of the last layer of the encoder. Used in the cross-attention of the decoder.
        past_key_values (`tuple(tuple(torch.FloatTensor))`, *optional*, returned when `use_cache=True` is passed or when `config.use_cache=True`):
            Tuple of `tuple(torch.FloatTensor)` of length `config.n_layers`, with each tuple having 2 tensors of shape
            `(batch_size, num_heads, sequence_length, embed_size_per_head)`) and 2 additional tensors of shape
            `(batch_size, num_heads, encoder_sequence_length, embed_size_per_head)`.

            Contains pre-computed hidden-states (key and values in the self-attention blocks and in the cross-attention
            blocks) that can be used (see `past_key_values` input) to speed up sequential decoding.

            If `past_key_values` are used, the user can optionally input only the last `decoder_input_ids` (those that
            don't have their past key value states given to this model) of shape `(batch_size, 1)` instead of all
            `decoder_input_ids` of shape `(batch_size, sequence_length)`. inputs_embeds (`torch.FloatTensor` of shape
            `(batch_size, sequence_length, hidden_size)`, *optional*): Optionally, instead of passing `input_ids` you
            can choose to directly pass an embedded representation. This is useful if you want more control over how to
            convert `input_ids` indices into associated vectors than the model's internal embedding lookup matrix.
        decoder_inputs_embeds (`torch.FloatTensor` of shape `(batch_size, target_sequence_length, hidden_size)`, *optional*):
            Optionally, instead of passing `decoder_input_ids` you can choose to directly pass an embedded
            representation. If `past_key_values` is used, optionally only the last `decoder_inputs_embeds` have to be
            input (see `past_key_values`). This is useful if you want more control over how to convert
            `decoder_input_ids` indices into associated vectors than the model's internal embedding lookup matrix.

            If `decoder_input_ids` and `decoder_inputs_embeds` are both unset, `decoder_inputs_embeds` takes the value
            of `inputs_embeds`.
        use_cache (`bool`, *optional*):
            If set to `True`, `past_key_values` key value states are returned and can be used to speed up decoding (see
            `past_key_values`).
        output_attentions (`bool`, *optional*):
            Whether or not to return the attentions tensors of all attention layers. See `attentions` under returned
            tensors for more detail.
        output_hidden_states (`bool`, *optional*):
            Whether or not to return the hidden states of all layers. See `hidden_states` under returned tensors for
            more detail.
        return_dict (`bool`, *optional*):
            Whether or not to return a [`~utils.ModelOutput`] instead of a plain tuple.
"""


class BartEncoder(BartPretrainedModel):
    """
    Transformer encoder consisting of *config.encoder_layers* self attention layers. Each layer is a
    [`BartEncoderLayer`].

    Args:
        config: BartConfig
        embed_tokens (nn.Embedding): output embedding
    """

    def __init__(self, config: BartConfig, embed_tokens: Optional[nn.Embedding] = None):
        super().__init__(config)

        self.dropout = config.dropout
        self.layerdrop = config.encoder_layerdrop

        embed_dim = config.d_model
        self.padding_idx = config.pad_token_id
        self.max_source_positions = config.max_position_embeddings
        self.embed_scale = math.sqrt(embed_dim) if config.scale_embedding else 1.0

        if embed_tokens is not None:
            self.embed_tokens = embed_tokens
        else:
            self.embed_tokens = nn.Embedding(config.vocab_size, embed_dim, self.padding_idx)

        self.embed_positions = BartLearnedPositionalEmbedding(
            config.max_position_embeddings,
            embed_dim,
        )
        self.layers = nn.ModuleList([BartEncoderLayer(config) for _ in range(config.encoder_layers)])
        self.layernorm_embedding = nn.LayerNorm(embed_dim)

        self.gradient_checkpointing = False
        # Initialize weights and apply final processing
        self.post_init()

    def get_input_embeddings(self):
        return self.embed_tokens

    def set_input_embeddings(self, value):
        self.embed_tokens = value

    def forward(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        head_mask: Optional[torch.Tensor] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple, BaseModelOutput]:
        r"""
        Args:
            input_ids (`torch.LongTensor` of shape `(batch_size, sequence_length)`):
                Indices of input sequence tokens in the vocabulary. Padding will be ignored by default should you
                provide it.

                Indices can be obtained using [`BartTokenizer`]. See [`PreTrainedTokenizer.encode`] and
                [`PreTrainedTokenizer.__call__`] for details.

                [What are input IDs?](../glossary#input-ids)
            attention_mask (`torch.Tensor` of shape `(batch_size, sequence_length)`, *optional*):
                Mask to avoid performing attention on padding token indices. Mask values selected in `[0, 1]`:

                - 1 for tokens that are **not masked**,
                - 0 for tokens that are **masked**.

                [What are attention masks?](../glossary#attention-mask)
            head_mask (`torch.Tensor` of shape `(encoder_layers, encoder_attention_heads)`, *optional*):
                Mask to nullify selected heads of the attention modules. Mask values selected in `[0, 1]`:

                - 1 indicates the head is **not masked**,
                - 0 indicates the head is **masked**.

            inputs_embeds (`torch.FloatTensor` of shape `(batch_size, sequence_length, hidden_size)`, *optional*):
                Optionally, instead of passing `input_ids` you can choose to directly pass an embedded representation.
                This is useful if you want more control over how to convert `input_ids` indices into associated vectors
                than the model's internal embedding lookup matrix.
            output_attentions (`bool`, *optional*):
                Whether or not to return the attentions tensors of all attention layers. See `attentions` under
                returned tensors for more detail.
            output_hidden_states (`bool`, *optional*):
                Whether or not to return the hidden states of all layers. See `hidden_states` under returned tensors
                for more detail.
            return_dict (`bool`, *optional*):
                Whether or not to return a [`~utils.ModelOutput`] instead of a plain tuple.
        """
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        # retrieve input_ids and inputs_embeds
        if input_ids is not None and inputs_embeds is not None:
            raise ValueError("You cannot specify both input_ids and inputs_embeds at the same time")
        elif input_ids is not None:
            input_shape = input_ids.size()
            input_ids = input_ids.view(-1, input_shape[-1])
        elif inputs_embeds is not None:
            input_shape = inputs_embeds.size()[:-1]
        else:
            raise ValueError("You have to specify either input_ids or inputs_embeds")

        if inputs_embeds is None:
            inputs_embeds = self.embed_tokens(input_ids) * self.embed_scale

        embed_pos = self.embed_positions(input_shape)

        hidden_states = inputs_embeds + embed_pos
        hidden_states = self.layernorm_embedding(hidden_states)
        hidden_states = nn.functional.dropout(hidden_states, p=self.dropout, training=self.training)

        # expand attention_mask
        if attention_mask is not None:
            # [bsz, seq_len] -> [bsz, 1, tgt_seq_len, src_seq_len]
            attention_mask = _expand_mask(attention_mask, inputs_embeds.dtype)

        encoder_states = () if output_hidden_states else None
        all_attentions = () if output_attentions else None

        # check if head_mask has a correct number of layers specified if desired
        if head_mask is not None:
            if head_mask.size()[0] != (len(self.layers)):
                raise ValueError(
                    f"The head_mask should be specified for {len(self.layers)} layers, but it is for"
                    f" {head_mask.size()[0]}."
                )

        for idx, encoder_layer in enumerate(self.layers):
            if output_hidden_states:
                encoder_states = encoder_states + (hidden_states,)
            # add LayerDrop (see https://arxiv.org/abs/1909.11556 for description)
            dropout_probability = random.uniform(0, 1)
            if self.training and (dropout_probability < self.layerdrop):  # skip the layer
                layer_outputs = (None, None)
            else:
                if self.gradient_checkpointing and self.training:

                    def create_custom_forward(module):
                        def custom_forward(*inputs):
                            return module(*inputs, output_attentions)

                        return custom_forward

                    layer_outputs = torch.utils.checkpoint.checkpoint(
                        create_custom_forward(encoder_layer),
                        hidden_states,
                        attention_mask,
                        (head_mask[idx] if head_mask is not None else None),
                    )
                else:
                    layer_outputs = encoder_layer(
                        hidden_states,
                        attention_mask,
                        layer_head_mask=(head_mask[idx] if head_mask is not None else None),
                        output_attentions=output_attentions,
                    )

                hidden_states = layer_outputs[0]

            if output_attentions:
                all_attentions = all_attentions + (layer_outputs[1],)

        if output_hidden_states:
            encoder_states = encoder_states + (hidden_states,)

        if not return_dict:
            return tuple(v for v in [hidden_states, encoder_states, all_attentions] if v is not None)
        return BaseModelOutput(
            last_hidden_state=hidden_states, hidden_states=encoder_states, attentions=all_attentions
        )


class BartDecoder(BartPretrainedModel):
    """
    Transformer decoder consisting of *config.decoder_layers* layers. Each layer is a [`BartDecoderLayer`]

    Args:
        config: BartConfig
        embed_tokens (nn.Embedding): output embedding
    """

    def __init__(self, config: BartConfig, embed_tokens: Optional[nn.Embedding] = None):
        super().__init__(config)
        self.dropout = config.dropout
        self.layerdrop = config.decoder_layerdrop
        self.padding_idx = config.pad_token_id
        self.max_target_positions = config.max_position_embeddings
        self.embed_scale = math.sqrt(config.d_model) if config.scale_embedding else 1.0

        if embed_tokens is not None:
            self.embed_tokens = embed_tokens
        else:
            self.embed_tokens = nn.Embedding(config.vocab_size, config.d_model, self.padding_idx)

        self.embed_positions = BartLearnedPositionalEmbedding(
            config.max_position_embeddings,
            config.d_model,
        )
        self.layers = nn.ModuleList([BartDecoderLayer(config) for _ in range(config.decoder_layers)])
        self.layernorm_embedding = nn.LayerNorm(config.d_model)

        self.gradient_checkpointing = False
        # Initialize weights and apply final processing
        self.post_init()

    def get_input_embeddings(self):
        return self.embed_tokens

    def set_input_embeddings(self, value):
        self.embed_tokens = value

    def _prepare_decoder_attention_mask(self, attention_mask, input_shape, inputs_embeds, past_key_values_length):
        # create causal mask
        # [bsz, seq_len] -> [bsz, 1, tgt_seq_len, src_seq_len]
        combined_attention_mask = None
        if input_shape[-1] > 1:
            combined_attention_mask = _make_causal_mask(
                input_shape, inputs_embeds.dtype, past_key_values_length=past_key_values_length
            ).to(inputs_embeds.device)

        if attention_mask is not None:
            # [bsz, seq_len] -> [bsz, 1, tgt_seq_len, src_seq_len]
            expanded_attn_mask = _expand_mask(attention_mask, inputs_embeds.dtype, tgt_len=input_shape[-1])
            combined_attention_mask = (
                expanded_attn_mask if combined_attention_mask is None else expanded_attn_mask + combined_attention_mask
            )

        return combined_attention_mask

    def forward(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        encoder_hidden_states: Optional[torch.FloatTensor] = None,
        encoder_attention_mask: Optional[torch.LongTensor] = None,
        head_mask: Optional[torch.Tensor] = None,
        cross_attn_head_mask: Optional[torch.Tensor] = None,
        past_key_values: Optional[List[torch.FloatTensor]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple, BaseModelOutputWithPastAndCrossAttentions]:
        r"""
        Args:
            input_ids (`torch.LongTensor` of shape `(batch_size, sequence_length)`):
                Indices of input sequence tokens in the vocabulary. Padding will be ignored by default should you
                provide it.

                Indices can be obtained using [`BartTokenizer`]. See [`PreTrainedTokenizer.encode`] and
                [`PreTrainedTokenizer.__call__`] for details.

                [What are input IDs?](../glossary#input-ids)
            attention_mask (`torch.Tensor` of shape `(batch_size, sequence_length)`, *optional*):
                Mask to avoid performing attention on padding token indices. Mask values selected in `[0, 1]`:

                - 1 for tokens that are **not masked**,
                - 0 for tokens that are **masked**.

                [What are attention masks?](../glossary#attention-mask)
            encoder_hidden_states (`torch.FloatTensor` of shape `(batch_size, encoder_sequence_length, hidden_size)`, *optional*):
                Sequence of hidden-states at the output of the last layer of the encoder. Used in the cross-attention
                of the decoder.
            encoder_attention_mask (`torch.LongTensor` of shape `(batch_size, encoder_sequence_length)`, *optional*):
                Mask to avoid performing cross-attention on padding tokens indices of encoder input_ids. Mask values
                selected in `[0, 1]`:

                - 1 for tokens that are **not masked**,
                - 0 for tokens that are **masked**.

                [What are attention masks?](../glossary#attention-mask)
            head_mask (`torch.Tensor` of shape `(decoder_layers, decoder_attention_heads)`, *optional*):
                Mask to nullify selected heads of the attention modules. Mask values selected in `[0, 1]`:

                - 1 indicates the head is **not masked**,
                - 0 indicates the head is **masked**.

            cross_attn_head_mask (`torch.Tensor` of shape `(decoder_layers, decoder_attention_heads)`, *optional*):
                Mask to nullify selected heads of the cross-attention modules in the decoder to avoid performing
                cross-attention on hidden heads. Mask values selected in `[0, 1]`:

                - 1 indicates the head is **not masked**,
                - 0 indicates the head is **masked**.

            past_key_values (`tuple(tuple(torch.FloatTensor))`, *optional*, returned when `use_cache=True` is passed or when `config.use_cache=True`):
                Tuple of `tuple(torch.FloatTensor)` of length `config.n_layers`, with each tuple having 2 tensors of
                shape `(batch_size, num_heads, sequence_length, embed_size_per_head)`) and 2 additional tensors of
                shape `(batch_size, num_heads, encoder_sequence_length, embed_size_per_head)`.

                Contains pre-computed hidden-states (key and values in the self-attention blocks and in the
                cross-attention blocks) that can be used (see `past_key_values` input) to speed up sequential decoding.

                If `past_key_values` are used, the user can optionally input only the last `decoder_input_ids` (those
                that don't have their past key value states given to this model) of shape `(batch_size, 1)` instead of
                all `decoder_input_ids` of shape `(batch_size, sequence_length)`. inputs_embeds (`torch.FloatTensor` of
                shape `(batch_size, sequence_length, hidden_size)`, *optional*): Optionally, instead of passing
                `input_ids` you can choose to directly pass an embedded representation. This is useful if you want more
                control over how to convert `input_ids` indices into associated vectors than the model's internal
                embedding lookup matrix.
            output_attentions (`bool`, *optional*):
                Whether or not to return the attentions tensors of all attention layers. See `attentions` under
                returned tensors for more detail.
            output_hidden_states (`bool`, *optional*):
                Whether or not to return the hidden states of all layers. See `hidden_states` under returned tensors
                for more detail.
            return_dict (`bool`, *optional*):
                Whether or not to return a [`~utils.ModelOutput`] instead of a plain tuple.
        """
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        use_cache = use_cache if use_cache is not None else self.config.use_cache
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        # retrieve input_ids and inputs_embeds
        if input_ids is not None and inputs_embeds is not None:
            raise ValueError("You cannot specify both decoder_input_ids and decoder_inputs_embeds at the same time")
        elif input_ids is not None:
            input_shape = input_ids.size()
            input_ids = input_ids.view(-1, input_shape[-1])
        elif inputs_embeds is not None:
            input_shape = inputs_embeds.size()[:-1]
        else:
            raise ValueError("You have to specify either decoder_input_ids or decoder_inputs_embeds")

        # past_key_values_length
        past_key_values_length = past_key_values[0][0].shape[2] if past_key_values is not None else 0

        if inputs_embeds is None:
            inputs_embeds = self.embed_tokens(input_ids) * self.embed_scale

        attention_mask = self._prepare_decoder_attention_mask(
            attention_mask, input_shape, inputs_embeds, past_key_values_length
        )

        # expand encoder attention mask
        if encoder_hidden_states is not None and encoder_attention_mask is not None:
            # [bsz, seq_len] -> [bsz, 1, tgt_seq_len, src_seq_len]
            encoder_attention_mask = _expand_mask(encoder_attention_mask, inputs_embeds.dtype, tgt_len=input_shape[-1])

        # embed positions
        positions = self.embed_positions(input_shape, past_key_values_length)

        hidden_states = inputs_embeds + positions
        hidden_states = self.layernorm_embedding(hidden_states)

        hidden_states = nn.functional.dropout(hidden_states, p=self.dropout, training=self.training)

        # decoder layers
        all_hidden_states = () if output_hidden_states else None
        all_self_attns = () if output_attentions else None
        all_cross_attentions = () if (output_attentions and encoder_hidden_states is not None) else None
        next_decoder_cache = () if use_cache else None

        # check if head_mask/cross_attn_head_mask has a correct number of layers specified if desired
        for attn_mask, mask_name in zip([head_mask, cross_attn_head_mask], ["head_mask", "cross_attn_head_mask"]):
            if attn_mask is not None:
                if attn_mask.size()[0] != (len(self.layers)):
                    raise ValueError(
                        f"The `{mask_name}` should be specified for {len(self.layers)} layers, but it is for"
                        f" {head_mask.size()[0]}."
                    )

        for idx, decoder_layer in enumerate(self.layers):
            # add LayerDrop (see https://arxiv.org/abs/1909.11556 for description)
            if output_hidden_states:
                all_hidden_states += (hidden_states,)
            dropout_probability = random.uniform(0, 1)
            if self.training and (dropout_probability < self.layerdrop):
                continue

            past_key_value = past_key_values[idx] if past_key_values is not None else None

            if self.gradient_checkpointing and self.training:

                if use_cache:
                    logger.warning(
                        "`use_cache=True` is incompatible with gradient checkpointing. Setting `use_cache=False`..."
                    )
                    use_cache = False

                def create_custom_forward(module):
                    def custom_forward(*inputs):
                        # None for past_key_value
                        return module(*inputs, output_attentions, use_cache)

                    return custom_forward

                layer_outputs = torch.utils.checkpoint.checkpoint(
                    create_custom_forward(decoder_layer),
                    hidden_states,
                    attention_mask,
                    encoder_hidden_states,
                    encoder_attention_mask,
                    head_mask[idx] if head_mask is not None else None,
                    cross_attn_head_mask[idx] if cross_attn_head_mask is not None else None,
                    None,
                )
            else:

                layer_outputs = decoder_layer(
                    hidden_states,
                    attention_mask=attention_mask,
                    encoder_hidden_states=encoder_hidden_states,
                    encoder_attention_mask=encoder_attention_mask,
                    layer_head_mask=(head_mask[idx] if head_mask is not None else None),
                    cross_attn_layer_head_mask=(
                        cross_attn_head_mask[idx] if cross_attn_head_mask is not None else None
                    ),
                    past_key_value=past_key_value,
                    output_attentions=output_attentions,
                    use_cache=use_cache,
                )
            hidden_states = layer_outputs[0]

            if use_cache:
                next_decoder_cache += (layer_outputs[3 if output_attentions else 1],)

            if output_attentions:
                all_self_attns += (layer_outputs[1],)

                if encoder_hidden_states is not None:
                    all_cross_attentions += (layer_outputs[2],)

        # add hidden states from the last decoder layer
        if output_hidden_states:
            all_hidden_states += (hidden_states,)

        next_cache = next_decoder_cache if use_cache else None
        if not return_dict:
            return tuple(
                v
                for v in [hidden_states, next_cache, all_hidden_states, all_self_attns, all_cross_attentions]
                if v is not None
            )
        return BaseModelOutputWithPastAndCrossAttentions(
            last_hidden_state=hidden_states,
            past_key_values=next_cache,
            hidden_states=all_hidden_states,
            attentions=all_self_attns,
            cross_attentions=all_cross_attentions,
        )


@add_start_docstrings(
    "The bare BART Model outputting raw hidden-states without any specific head on top.",
    BART_START_DOCSTRING,
)
class BartModel(BartPretrainedModel):
    def __init__(self, config: BartConfig):
        super().__init__(config)

        padding_idx, vocab_size = config.pad_token_id, config.vocab_size
        self.shared = nn.Embedding(vocab_size, config.d_model, padding_idx)

        self.encoder = BartEncoder(config, self.shared)
        self.decoder = BartDecoder(config, self.shared)

        # Initialize weights and apply final processing
        self.post_init()

    def get_input_embeddings(self):
        return self.shared

    def set_input_embeddings(self, value):
        self.shared = value
        self.encoder.embed_tokens = self.shared
        self.decoder.embed_tokens = self.shared

    def get_encoder(self):
        return self.encoder

    def get_decoder(self):
        return self.decoder

    @add_start_docstrings_to_model_forward(BART_INPUTS_DOCSTRING)
    @add_code_sample_docstrings(
        processor_class=_TOKENIZER_FOR_DOC,
        checkpoint=_CHECKPOINT_FOR_DOC,
        output_type=Seq2SeqModelOutput,
        config_class=_CONFIG_FOR_DOC,
        expected_output=_EXPECTED_OUTPUT_SHAPE,
    )
    def forward(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        decoder_input_ids: Optional[torch.LongTensor] = None,
        decoder_attention_mask: Optional[torch.LongTensor] = None,
        head_mask: Optional[torch.Tensor] = None,
        decoder_head_mask: Optional[torch.Tensor] = None,
        cross_attn_head_mask: Optional[torch.Tensor] = None,
        encoder_outputs: Optional[List[torch.FloatTensor]] = None,
        past_key_values: Optional[List[torch.FloatTensor]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        decoder_inputs_embeds: Optional[torch.FloatTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple, Seq2SeqModelOutput]:

        # different to other models, Bart automatically creates decoder_input_ids from
        # input_ids if no decoder_input_ids are provided
        if decoder_input_ids is None and decoder_inputs_embeds is None:
            if input_ids is None:
                raise ValueError(
                    "If no `decoder_input_ids` or `decoder_inputs_embeds` are "
                    "passed, `input_ids` cannot be `None`. Please pass either "
                    "`input_ids` or `decoder_input_ids` or `decoder_inputs_embeds`."
                )

            decoder_input_ids = shift_tokens_right(
                input_ids, self.config.pad_token_id, self.config.decoder_start_token_id
            )

        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        use_cache = use_cache if use_cache is not None else self.config.use_cache
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        if encoder_outputs is None:
            encoder_outputs = self.encoder(
                input_ids=input_ids,
                attention_mask=attention_mask,
                head_mask=head_mask,
                inputs_embeds=inputs_embeds,
                output_attentions=output_attentions,
                output_hidden_states=output_hidden_states,
                return_dict=return_dict,
            )
        # If the user passed a tuple for encoder_outputs, we wrap it in a BaseModelOutput when return_dict=True
        elif return_dict and not isinstance(encoder_outputs, BaseModelOutput):
            encoder_outputs = BaseModelOutput(
                last_hidden_state=encoder_outputs[0],
                hidden_states=encoder_outputs[1] if len(encoder_outputs) > 1 else None,
                attentions=encoder_outputs[2] if len(encoder_outputs) > 2 else None,
            )

        # decoder outputs consists of (dec_features, past_key_value, dec_hidden, dec_attn)
        decoder_outputs = self.decoder(
            input_ids=decoder_input_ids,
            attention_mask=decoder_attention_mask,
            encoder_hidden_states=encoder_outputs[0],
            encoder_attention_mask=attention_mask,
            head_mask=decoder_head_mask,
            cross_attn_head_mask=cross_attn_head_mask,
            past_key_values=past_key_values,
            inputs_embeds=decoder_inputs_embeds,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        if not return_dict:
            return decoder_outputs + encoder_outputs

        return Seq2SeqModelOutput(
            last_hidden_state=decoder_outputs.last_hidden_state,
            past_key_values=decoder_outputs.past_key_values,
            decoder_hidden_states=decoder_outputs.hidden_states,
            decoder_attentions=decoder_outputs.attentions,
            cross_attentions=decoder_outputs.cross_attentions,
            encoder_last_hidden_state=encoder_outputs.last_hidden_state,
            encoder_hidden_states=encoder_outputs.hidden_states,
            encoder_attentions=encoder_outputs.attentions,
        )



"""
PyTorch MultiTaskBartForConditionalGeneration Class for DSTC-11 SIMMC 2.1
Changed Based on BartForConditionalGeneration
Updated by Yirong Chen 
Used for [SIMMC 2.1](https://github.com/facebookresearch/simmc2)
Mail: [yrchen5@iflytek.com](mailto:yrchen5@iflytek.com) or [eeyirongchen@mail.scut.edu.cn](mailto:eeyirongchen@mail.scut.edu.cn)
Date: 2022/08/04
"""
class BoxEmbedding(nn.Module):
    """ Copy from DSTC-10 SIMMC2.0 Team 4
        Updated by Yirong Chen on 2022/08/04
        Reference: 
            https://github.com/KAIST-AILab/DSTC10-SIMMC/blob/dstc10-simmc2-v1/scripts/run_bart_multi_task.py
            https://aclanthology.org/2022.findings-naacl.61/
        Used for Scene Box Embedding
        对各个object的图像方块的box信息建模作为表征。输入为6维
        [x1/w-0.5, y1/h-0.5, x2/w-0.5, y2/h-0.5, (x2-x1)(y2-y1)/(h*w), z_value/largest_z_value]
        See line 175 of https://github.com/KAIST-AILab/DSTC10-SIMMC/tree/dstc10-simmc2-v1/scripts/convert.py
        Which is different from the description of the paper.
    """
    def __init__(self, hidden_dim, dropout_rate=0.0):
        super(BoxEmbedding, self).__init__()
        self.box_linear = nn.Linear(6, hidden_dim)  
        self.box_layer_norm = nn.LayerNorm(hidden_dim)
        #self.dropout = nn.Dropout(dropout_rate) # 新增
    def forward(self, box_feat):
        """
            box_feat: [x1/w-0.5, y1/h-0.5, x2/w-0.5, y2/h-0.5, (x2-x1)(y2-y1)/(h*w), z_value/largest_z_value]
        """
        transformed_box = self.box_layer_norm(self.box_linear(box_feat))
        #transformed_box = self.dropout(transformed_box) # 新增
        return transformed_box

class NoCorefHead(nn.Module):
    """ Copy from DSTC-10 SIMMC2.0 Team 4
        Updated by Yirong Chen on 2022/08/04
        Reference: 
            https://github.com/KAIST-AILab/DSTC10-SIMMC/blob/dstc10-simmc2-v1/scripts/run_bart_multi_task.py
            https://aclanthology.org/2022.findings-naacl.61/
        Used for NoCorefHead
    """
    def __init__(self, hidden_dim, dropout_rate=0.0):
        super(NoCorefHead, self).__init__()
        self.no_coref_linear = nn.Linear(hidden_dim, 2)  
        #self.dropout = nn.Dropout(dropout_rate) # 新增
    def forward(self, no_coref_vector):
        coref_cls = self.no_coref_linear(no_coref_vector)
        #coref_cls = self.dropout(coref_cls) # 新增
        return coref_cls

class DisambiguationHead(nn.Module):
    """ Copy from DSTC-10 SIMMC2.0 Team 4
        Updated by Yirong Chen on 2022/08/04
        Reference: 
            https://github.com/KAIST-AILab/DSTC10-SIMMC/blob/dstc10-simmc2-v1/scripts/run_bart_multi_task.py
            https://aclanthology.org/2022.findings-naacl.61/
        Used for DisambiguationHead
        DSTC-10的subtask-1 二分类任务
    """
    def __init__(self, hidden_dim, dropout_rate=0.0):
        super(DisambiguationHead, self).__init__()
        self.disamb_linear = nn.Linear(hidden_dim, 2)  
        #self.dropout = nn.Dropout(dropout_rate) # 新增
    def forward(self, x):
        #return self.dropout(self.disamb_linear(x))
        return self.disamb_linear(x)

class FashionEncoderHead(nn.Module):
    """ Copy from DSTC-10 SIMMC2.0 Team 4
        Updated by Yirong Chen on 2022/08/04
        Reference: 
            https://github.com/KAIST-AILab/DSTC10-SIMMC/blob/dstc10-simmc2-v1/scripts/run_bart_multi_task.py
            https://aclanthology.org/2022.findings-naacl.61/
        Used for FashionEncoderHead
    """
    def __init__(self, hidden_dim, dropout_rate=0.0):
        super(FashionEncoderHead, self).__init__()
        self.aggregator = nn.Linear(2*hidden_dim, 2*hidden_dim)
        self.disamb_linear = nn.Linear(2*hidden_dim, 2) # 用于判断每个object是否为歧义候选disambiguation candidate, Updated by Yirong Chen
        self.coref_linear = nn.Linear(2*hidden_dim, 2) # 用于判断每个object是否为MM-Coref object
        self.size_linear = nn.Linear(2*hidden_dim, 6)
        self.available_sizes_linear = nn.Linear(2*hidden_dim, 6)  # sigmoid is applied later by 
        self.brand_linear = nn.Linear(2*hidden_dim, 26)
        self.color_linear = nn.Linear(2*hidden_dim, 71)
        self.pattern_linear = nn.Linear(2*hidden_dim, 36)
        self.sleeve_length_linear = nn.Linear(2*hidden_dim, 6)
        self.asset_type_linear = nn.Linear(2*hidden_dim, 12)
        self.type_linear = nn.Linear(2*hidden_dim, 18)
        self.price_linear = nn.Linear(2*hidden_dim, 45)
        self.customer_review_linear = nn.Linear(2*hidden_dim, 26)
        self.dropout = nn.Dropout(dropout_rate) # 新增
    def forward(self, concat_vector):
        ''' concat_vector: concat of obj_index_vector and st_vector '''
        aggregated = self.aggregator(concat_vector)
        aggregated = self.dropout(aggregated)
        disamb = self.disamb_linear(aggregated) # 用于判断每个object是否为歧义候选disambiguation candidate, Updated by Yirong Chen
        #disamb = self.dropout(disamb)
        coref = self.coref_linear(aggregated) # 用于判断每个object是否为MM-Coref object
        #coref = self.dropout(coref)
        size = self.size_linear(aggregated)
        available_sizes = self.available_sizes_linear(aggregated)
        brand = self.brand_linear(aggregated)
        color = self.color_linear(aggregated)
        pattern = self.pattern_linear(aggregated)
        sleeve_length = self.sleeve_length_linear(aggregated)
        asset_type = self.asset_type_linear(aggregated)
        type_ = self.type_linear(aggregated)
        price = self.price_linear(aggregated)
        customer_review = self.customer_review_linear(aggregated)
        return disamb, coref, size, available_sizes, brand, color, pattern, sleeve_length, asset_type, type_, \
               price, customer_review

class FurnitureEncoderHead(nn.Module):
    """ Copy from DSTC-10 SIMMC2.0 Team 4
        Updated by Yirong Chen on 2022/08/04
        Reference: 
            https://github.com/KAIST-AILab/DSTC10-SIMMC/blob/dstc10-simmc2-v1/scripts/run_bart_multi_task.py
            https://aclanthology.org/2022.findings-naacl.61/
        Used for FurnitureEncoderHead
    """
    def __init__(self, hidden_dim, dropout_rate=0.0):
        super(FurnitureEncoderHead, self).__init__()
        self.aggregator = nn.Linear(2*hidden_dim, 2*hidden_dim)
        self.disamb_linear = nn.Linear(2*hidden_dim, 2) # 用于判断每个object是否为歧义候选disambiguation candidate, Updated by Yirong Chen
        self.coref_linear = nn.Linear(2*hidden_dim, 2) # 用于判断每个object是否为MM-Coref object
        self.brand_linear = nn.Linear(2*hidden_dim, 12)
        self.color_linear = nn.Linear(2*hidden_dim, 9)
        self.materials_linear = nn.Linear(2*hidden_dim, 7)
        self.type_linear = nn.Linear(2*hidden_dim, 10)
        self.price_linear = nn.Linear(2*hidden_dim, 10)
        self.customer_review_linear = nn.Linear(2*hidden_dim, 19)
        self.dropout = nn.Dropout(dropout_rate) # 新增
    def forward(self, concat_vector):
        ''' concat_vector: concat of obj_index_vector and st_vector '''
        aggregated = self.aggregator(concat_vector)
        aggregated = self.dropout(aggregated)
        disamb = self.disamb_linear(aggregated) # 用于判断每个object是否为歧义候选disambiguation candidate, Updated by Yirong Chen
        #disamb = self.dropout(disamb)
        coref = self.coref_linear(aggregated)
        #coref = self.dropout(coref)
        brand = self.brand_linear(aggregated)
        color = self.color_linear(aggregated)
        materials = self.materials_linear(aggregated)
        type_ = self.type_linear(aggregated)
        price = self.price_linear(aggregated)
        customer_review = self.customer_review_linear(aggregated)
        return disamb, coref, brand, color, materials, type_, price, customer_review


"""
PyTorch MultiTaskBartForConditionalGenerationWithDisamb Class for DSTC-11 SIMMC 2.1
Changed Based on BartForConditionalGeneration
Updated by Yirong Chen 
Used for [SIMMC 2.1](https://github.com/facebookresearch/simmc2)
Mail: [yrchen5@iflytek.com](mailto:yrchen5@iflytek.com) or [eeyirongchen@mail.scut.edu.cn](mailto:eeyirongchen@mail.scut.edu.cn)
Date: 2022/08/04
"""

@add_start_docstrings(
    "The BART Model with a language modeling head and several heads for other tasks. Can be used for Multi task.", BART_START_DOCSTRING
)
class MultiTaskBartForConditionalGenerationWithDisambUseAttr(BartPretrainedModel):
    base_model_prefix = "model"
    _keys_to_ignore_on_load_missing = [r"final_logits_bias", r"lm_head.weight"]

    def __init__(self, config: BartConfig):
        super().__init__(config)
        self.model = BartModel(config)
        self.register_buffer("final_logits_bias", torch.zeros((1, self.model.shared.num_embeddings)))
        self.lm_head = nn.Linear(config.d_model, self.model.shared.num_embeddings, bias=False)
        # Add module here
        # Updated by Yirong Chen
        self.box_embedding = BoxEmbedding(config.d_model, dropout_rate=config.dropout)
        self.nocoref_head = NoCorefHead(config.d_model, dropout_rate=config.dropout)
        self.fashion_enc_head = FashionEncoderHead(config.d_model, dropout_rate=config.dropout)
        self.furniture_enc_head = FurnitureEncoderHead(config.d_model, dropout_rate=config.dropout)
        self.disambiguation_head = DisambiguationHead(config.d_model, dropout_rate=config.dropout)

        # Initialize weights and apply final processing
        self.post_init()

    def get_encoder(self):
        return self.model.get_encoder()

    def get_decoder(self):
        return self.model.get_decoder()

    def resize_token_embeddings(self, new_num_tokens: int) -> nn.Embedding:
        new_embeddings = super().resize_token_embeddings(new_num_tokens)
        self._resize_final_logits_bias(new_num_tokens)
        return new_embeddings

    def _resize_final_logits_bias(self, new_num_tokens: int) -> None:
        old_num_tokens = self.final_logits_bias.shape[-1]
        if new_num_tokens <= old_num_tokens:
            new_bias = self.final_logits_bias[:, :new_num_tokens]
        else:
            extra_bias = torch.zeros((1, new_num_tokens - old_num_tokens), device=self.final_logits_bias.device)
            new_bias = torch.cat([self.final_logits_bias, extra_bias], dim=1)
        self.register_buffer("final_logits_bias", new_bias)

    def get_output_embeddings(self):
        return self.lm_head

    def set_output_embeddings(self, new_embeddings):
        self.lm_head = new_embeddings

    @add_start_docstrings_to_model_forward(BART_INPUTS_DOCSTRING)
    @replace_return_docstrings(output_type=Seq2SeqLMOutput, config_class=_CONFIG_FOR_DOC)
    @add_end_docstrings(BART_GENERATION_EXAMPLE)
    def forward(
        self,
        input_ids: torch.LongTensor = None, # enc_input
        attention_mask: Optional[torch.Tensor] = None, # enc_attention_mask
        decoder_input_ids: Optional[torch.LongTensor] = None, # =decoder_input[:, :-1]
        decoder_attention_mask: Optional[torch.LongTensor] = None, # =decoder_attention_mask[:, :-1]
        head_mask: Optional[torch.Tensor] = None,
        decoder_head_mask: Optional[torch.Tensor] = None,
        cross_attn_head_mask: Optional[torch.Tensor] = None,
        encoder_outputs: Optional[List[torch.FloatTensor]] = None,
        past_key_values: Optional[List[torch.FloatTensor]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None, # 输入的为经过inputs_embeds[b_idx][pos] += box_embedded[obj_idx]后的
        decoder_inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None, # =decoder_input[:, 1:].contiguous()
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        boxes: Optional[List[List[List[float]]]] = None, # 新增 考虑batch_size, # batch, num_obj_per_line, 6
        misc: Optional[List[List[Dict]]] = None, # 新增 [ [ {pos, coref_label, misc_labels(dict), is_fashion}, ... ], ...], 考虑batch_size, # batch, num_obj_per_line, dict
        nocoref: Optional[List] = None, # 新增 [(position, label), (position, label), (position, label), ...] or [position, position, position, ...]
        response: Optional[torch.LongTensor] = None, # 新增
        response_attention_mask: Optional[torch.LongTensor] = None, # 新增
        disambiguation_labels: Optional[torch.LongTensor] = None, # 新增
        do_retrieval: Optional[bool] = False, # 新增，指定是否有检索任务
        alpha_masked_lm_loss: Optional[float] = 1.0, # 新增，生成回复的损失占比
        alpha_nocoref_loss: Optional[float] = 0.1, # 新增，二分类任务判断是否存在共指消解的损失占比
        alpha_misc_loss: Optional[float] = 0.1, # 新增，属性识别损失占比
        alpha_disam_loss: Optional[float] = 0.1, # 新增，歧义句子分类损失占比
        alpha_retrieval_loss: Optional[float] = 0.4, # 新增，检索回复损失占比
        alpha_disamb_candi_loss: Optional[float] = 0.8, # 新增，识别对象是否为歧义候选的损失占比
        alpha_coref_loss: Optional[float] = 0.8, # 新增，识别对象是否为多模态共指消解的损失占比
        use_focal_loss: Optional[bool] = False,
        focal_loss_gamma: Optional[float] = 2.0, # 0.2, 0.5, 1.0, 2.0, 5.0
        focal_loss_alpha: Optional[float] = 0.25, # 大多数样本的标签对应的权重，取值范围为：0~1
        object_attr_input_ids: Optional[List] = None, # List[List[List[Tensor]]] # (batch_size, obj_num, attr_num, 1)
        use_non_visual_attrs: Optional[bool] = False, #
    ) -> Union[Tuple, Seq2SeqLMOutputForSIMMC]:
        r"""
        labels (`torch.LongTensor` of shape `(batch_size, sequence_length)`, *optional*):
            Labels for computing the masked language modeling loss. Indices should either be in `[0, ...,
            config.vocab_size]` or -100 (see `input_ids` docstring). Tokens with indices set to `-100` are ignored
            (masked), the loss is only computed for the tokens with labels in `[0, ..., config.vocab_size]`.

        Returns:
        """
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        # 损失函数统一放置在这里
        # Updated by Yirong Chen
        ce_loss_fct = CrossEntropyLoss()
        bce_loss_fct = BCEWithLogitsLoss()
        if use_focal_loss:
            # 参考：https://zhuanlan.zhihu.com/p/122542747
            # 参考：https://github.com/alexmeredith8299/focal_loss_pytorch/blob/main/focal_loss_pytorch/focal_loss.py
            # alpha: 大多数样本的权重, 1-alpha: 少数样本的权重
            # 例如在二分类当中，标签0的样本有：10000，标签1的样本有：200，则
            # 标签0对应alpha，标签1对应：1-alpha
            alpha1 = torch.tensor([[focal_loss_alpha], [1-focal_loss_alpha]]) # disambiguation_labels，标签为0的样本占大多数
            alpha2 = torch.tensor([[1-focal_loss_alpha], [focal_loss_alpha]]) # no_coref，标签为1的样本占大多数
            alpha3 = torch.tensor([[focal_loss_alpha], [1-focal_loss_alpha]]) # object是否为歧义候选，标签0的样本占大多数
            alpha4 = torch.tensor([[focal_loss_alpha], [1-focal_loss_alpha]]) # object是否为多模态共指，标签0的样本占大多数
            focal_loss_fct1 = FocalLoss(class_num=2, alpha=alpha1, gamma=focal_loss_gamma, size_average=True) # disambiguation_labels，标签为0的样本占大多数
            focal_loss_fct2 = FocalLoss(class_num=2, alpha=alpha2, gamma=focal_loss_gamma, size_average=True) 
            focal_loss_fct3 = FocalLoss(class_num=2, alpha=alpha3, gamma=focal_loss_gamma, size_average=True)
            focal_loss_fct4 = FocalLoss(class_num=2, alpha=alpha4, gamma=focal_loss_gamma, size_average=True)

        if labels is not None:
            if use_cache:
                logger.warning("The `use_cache` argument is changed to `False` since `labels` is provided.")
            use_cache = False
            if decoder_input_ids is None and decoder_inputs_embeds is None:
                decoder_input_ids = shift_tokens_right(
                    labels, self.config.pad_token_id, self.config.decoder_start_token_id
                )
        if misc is not None:
            batch_size = len(misc)
        elif input_ids is not None:
            batch_size = len(input_ids)
        elif boxes is not None:
            batch_size = len(boxes)
        elif encoder_outputs is not None and not isinstance(encoder_outputs, BaseModelOutput):
            batch_size = len(encoder_outputs[0])
        elif encoder_outputs is not None:
            batch_size = len(encoder_outputs.last_hidden_state)
        elif inputs_embeds is not None:
            batch_size = len(inputs_embeds)
        elif decoder_input_ids is not None:
            batch_size = len(decoder_input_ids)
        elif decoder_inputs_embeds is not None:
            batch_size = len(decoder_inputs_embeds)

        # 增加box_embedding到这个位置

        # follow `class BartEncoder`. shape of (batch, seqlen, d_model)
        if boxes is not None: # 考虑box_embedding
            inputs_embeds = self.model.encoder.embed_tokens(input_ids) * self.model.encoder.embed_scale

            for b_idx in range(batch_size):  # in a batch
                # Add Box embeddings
                box_embedded = self.box_embedding(torch.tensor(boxes[b_idx]).to(input_ids.device))  # (num_obj_per_line, d_model)
                for obj_idx in range(len(misc[b_idx])):
                    pos = misc[b_idx][obj_idx]['pos']
                    inputs_embeds[b_idx][pos] += box_embedded[obj_idx]

                # Add Attribute embeddings
                if object_attr_input_ids is not None and use_non_visual_attrs:

                    #for line_object_attr_input_ids in object_attr_input_ids:
                    line_object_embeddings = []
                    for object_attr_input_id in object_attr_input_ids[b_idx]:  # (obj_num, attr_num, 1)
                        # object_attr_input_id: [tensor([50847]), tensor([50863]), tensor([50910]), tensor([448])]
                        object_embeddings = [torch.sum(self.model.encoder.embed_tokens(obj_tok.to(inputs_embeds.device)), dim=0) # summing over columns handling multiple integer tokens
                                                for obj_tok in object_attr_input_id]
                        # object_embeddings: [torch.Tensor([, ...]), ...] size: (attr_num, 768)
                        line_object_embeddings.append(object_embeddings) # size: (obj_num, attr_num, 768)

                    # 将line_object_embeddings叠加到inputs_embeds当中
                    for idx, abs_id_embs in enumerate(line_object_embeddings):
                        pos = misc[b_idx][idx]['pos']
                        for embs in abs_id_embs:
                            inputs_embeds[b_idx][pos] += torch.reshape(embs, (-1,))

            outputs = self.model(
                input_ids=None,
                attention_mask=attention_mask,
                decoder_input_ids=decoder_input_ids,
                encoder_outputs=encoder_outputs,
                decoder_attention_mask=decoder_attention_mask,
                head_mask=head_mask,
                decoder_head_mask=decoder_head_mask,
                cross_attn_head_mask=cross_attn_head_mask,
                past_key_values=past_key_values,
                inputs_embeds=inputs_embeds, # 输入的为叠加了box_embedded的表征
                decoder_inputs_embeds=decoder_inputs_embeds,
                use_cache=use_cache,
                output_attentions=output_attentions,
                output_hidden_states=output_hidden_states,
                return_dict=True,
            )

        else: # 不考虑box_embedding
            outputs = self.model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                decoder_input_ids=decoder_input_ids,
                encoder_outputs=encoder_outputs,
                decoder_attention_mask=decoder_attention_mask,
                head_mask=head_mask,
                decoder_head_mask=decoder_head_mask,
                cross_attn_head_mask=cross_attn_head_mask,
                past_key_values=past_key_values,
                inputs_embeds=inputs_embeds,
                decoder_inputs_embeds=decoder_inputs_embeds,
                use_cache=use_cache,
                output_attentions=output_attentions,
                output_hidden_states=output_hidden_states,
                return_dict=True,
            )
        #print("outputs[0]=",outputs[0])
        #decoder_outputs.last_hidden_state

        lm_logits = self.lm_head(outputs[0]) + self.final_logits_bias

        #masked_lm_loss = None # model_loss
        masked_lm_loss = 0 # model_loss
        if labels is not None:
            #ce_loss_fct = CrossEntropyLoss()
            masked_lm_loss = ce_loss_fct(lm_logits.view(-1, self.config.vocab_size), labels.view(-1))

        # 增加处理代码在该位置
        # by Yirong Chen

        enc_last_state = outputs.encoder_last_hidden_state  # (bs, seqlen, d_model)


        if do_retrieval: # 训练阶段
            # For Biencoder
            response_vec = self.model.encoder(input_ids=response, attention_mask=response_attention_mask)[0][:, 0, :] # bs, dim
            context_vec = enc_last_state[:, 0, :] # bs, dim
            dot_product = torch.matmul(context_vec, response_vec.t())  # bs, bs
            retrieval_loss = ce_loss_fct(dot_product, torch.arange(batch_size).to(context_vec.device))
        else: # 验证或测试阶段或该任务不参与训练
            retrieval_loss = 0

        # For Disambiguation
        disambiguation_logits = self.disambiguation_head(enc_last_state[:, 1, :]) # bs, d_model --> bs, 2
        #disambiguation_label = torch.argmax(disambiguation_logits, dim=-1).squeeze()
        if disambiguation_labels is not None: # 训练阶段
            if use_focal_loss:
                disam_loss = focal_loss_fct1(disambiguation_logits, disambiguation_labels.view(-1))
            else:
                disam_loss = ce_loss_fct(disambiguation_logits, disambiguation_labels.view(-1))
        else: # 验证或测试阶段或该任务不参与训练
            disam_loss = 0
        
        if (nocoref is not None) and (isinstance(nocoref[0],tuple)): # 训练阶段
            # [(position, label), (position, label), (position, label),...]
            nocoref_logits = torch.stack([self.nocoref_head(enc_last_state[b_idx][nocoref[b_idx][0]]) for b_idx in range(batch_size) ])
            nocoref_labels = torch.tensor([nocoref[b_idx][1] for b_idx in range(batch_size)]).to(input_ids.device)
            if use_focal_loss:
                nocoref_loss = focal_loss_fct2(nocoref_logits, nocoref_labels)
            else:
                nocoref_loss = ce_loss_fct(nocoref_logits, nocoref_labels)
        elif nocoref is not None: # 验证或测试阶段或该任务不参与训练
            # [position, position, position, ...]
            nocoref_logits = torch.stack([self.nocoref_head(enc_last_state[b_idx][nocoref[b_idx][0]]) for b_idx in range(batch_size)])
            #is_nocoref = nocoref_logits.argmax(dim=1).bool()
            nocoref_loss = 0
        else:
            nocoref_logits = None
            nocoref_loss = 0

        misc_loss = 0
        
        if misc is not None and "coref_label" in misc[0][0]:
            # 思考这里的判断条件
            """ train 阶段，计算loss
            """
            # 不能并行

            enc_head_results = []

            for b_idx in range(batch_size):  # in a batch
                is_fashion = misc[b_idx][0]['is_fashion']
                coref_label = [misc[b_idx][obj_idx]['coref_label'] for obj_idx in range(len(misc[b_idx]))]  # (num_obj)  0 or 1
                disamb_label = [misc[b_idx][obj_idx]['disamb_label'] for obj_idx in range(len(misc[b_idx]))]  # (num_obj)  0 or 1
                if is_fashion:
                    fashion_size_label = [misc[b_idx][obj_idx]['misc_labels']['size'] for obj_idx in range(len(misc[b_idx]))]  # (num_obj)
                    fashion_available_sizes_label = [misc[b_idx][obj_idx]['misc_labels']['available_sizes'] for obj_idx in range(len(misc[b_idx]))]  # (num_obj, 6)
                    fashion_brand_label = [misc[b_idx][obj_idx]['misc_labels']['brand'] for obj_idx in range(len(misc[b_idx]))]
                    fashion_color_label = [misc[b_idx][obj_idx]['misc_labels']['color'] for obj_idx in range(len(misc[b_idx]))]
                    fashion_pattern_label = [misc[b_idx][obj_idx]['misc_labels']['pattern'] for obj_idx in range(len(misc[b_idx]))]
                    fashion_sleeve_length_label = [misc[b_idx][obj_idx]['misc_labels']['sleeve_length'] for obj_idx in range(len(misc[b_idx]))]
                    fashion_asset_type_label = [misc[b_idx][obj_idx]['misc_labels']['asset_type'] for obj_idx in range(len(misc[b_idx]))]
                    fashion_type_label = [misc[b_idx][obj_idx]['misc_labels']['type'] for obj_idx in range(len(misc[b_idx]))]
                    fashion_price_label = [misc[b_idx][obj_idx]['misc_labels']['price'] for obj_idx in range(len(misc[b_idx]))]
                    fashion_customer_review_label = [misc[b_idx][obj_idx]['misc_labels']['customer_review'] for obj_idx in range(len(misc[b_idx]))]
                else:
                    furniture_brand_label = [misc[b_idx][obj_idx]['misc_labels']['brand'] for obj_idx in range(len(misc[b_idx]))]  # (num_obj)
                    furniture_color_label = [misc[b_idx][obj_idx]['misc_labels']['color'] for obj_idx in range(len(misc[b_idx]))]
                    furniture_materials_label = [misc[b_idx][obj_idx]['misc_labels']['materials'] for obj_idx in range(len(misc[b_idx]))]
                    furniture_type_label = [misc[b_idx][obj_idx]['misc_labels']['type'] for obj_idx in range(len(misc[b_idx]))]
                    furniture_price_label = [misc[b_idx][obj_idx]['misc_labels']['price'] for obj_idx in range(len(misc[b_idx]))]
                    furniture_customer_review_label = [misc[b_idx][obj_idx]['misc_labels']['customer_review'] for obj_idx in range(len(misc[b_idx]))]
                
                for obj_idx in range(len(misc[b_idx])):
                    pos = misc[b_idx][obj_idx]['pos']
                    # hidden_concat: (num_obj, 2*model)
                    if obj_idx == 0:
                        hidden_concat = torch.reshape(enc_last_state[b_idx][pos:pos+2], (1,-1))
                    else:
                        hidden_concat = torch.cat([hidden_concat, torch.reshape(enc_last_state[b_idx][pos:pos+2], (1,-1))], dim=0)
                    # hidden_concat = torch.reshape(enc_last_state[b_idx][pos:pos+2], (-1,))  # (2*d_model)  -> 
                """
                # 为了解决报错：
                By Yirong Chen on 2022/08/10
                RuntimeError: Expected to have finished reduction in the prior iteration before starting a new one. 
                This error indicates that your module has parameters that were not used in producing loss. Since `find_unused_parameters=True` is enabled, 
                this likely  means that not all `forward` outputs participate in computing loss. You can fix this by making sure all `forward` function 
                outputs participate in calculating loss.
                """
                fashion_disamb, fashion_coref, fashion_size, fashion_available_sizes, fashion_brand, fashion_color, fashion_pattern, fashion_sleeve_length, \
                fashion_asset_type, fashion_type_, fashion_price, fashion_customer_review = self.fashion_enc_head(hidden_concat)  # (num_obj, num_logits)
                furniture_disamb, furniture_coref, furniture_brand, furniture_color, furniture_materials, furniture_type_, furniture_price, furniture_customer_review = self.furniture_enc_head(hidden_concat)  # (num_obj, num_logits)
                
                if is_fashion:
                    enc_head_results.append((fashion_disamb, fashion_coref, fashion_size, fashion_available_sizes, fashion_brand, fashion_color, fashion_pattern, fashion_sleeve_length, \
                    fashion_asset_type, fashion_type_, fashion_price, fashion_customer_review))
                    if use_focal_loss:
                        loss_per_line = alpha_disamb_candi_loss * focal_loss_fct3(fashion_disamb, torch.tensor(disamb_label, dtype=torch.long).to(input_ids.device)) + \
                                        alpha_coref_loss * focal_loss_fct4(fashion_coref, torch.tensor(coref_label, dtype=torch.long).to(input_ids.device)) + \
                                        alpha_misc_loss * ce_loss_fct(fashion_size, torch.tensor(fashion_size_label, dtype=torch.long).to(input_ids.device)) + \
                                        alpha_misc_loss * bce_loss_fct(fashion_available_sizes, torch.tensor(fashion_available_sizes_label, dtype=torch.float32).to(input_ids.device)) + \
                                        alpha_misc_loss * ce_loss_fct(fashion_brand, torch.tensor(fashion_brand_label, dtype=torch.long).to(input_ids.device)) + \
                                        alpha_misc_loss * ce_loss_fct(fashion_color, torch.tensor(fashion_color_label, dtype=torch.long).to(input_ids.device)) + \
                                        alpha_misc_loss * ce_loss_fct(fashion_pattern, torch.tensor(fashion_pattern_label, dtype=torch.long).to(input_ids.device)) + \
                                        alpha_misc_loss * ce_loss_fct(fashion_sleeve_length, torch.tensor(fashion_sleeve_length_label, dtype=torch.long).to(input_ids.device)) + \
                                        alpha_misc_loss * ce_loss_fct(fashion_asset_type, torch.tensor(fashion_asset_type_label, dtype=torch.long).to(input_ids.device)) + \
                                        alpha_misc_loss * ce_loss_fct(fashion_type_, torch.tensor(fashion_type_label, dtype=torch.long).to(input_ids.device)) + \
                                        alpha_misc_loss * ce_loss_fct(fashion_price, torch.tensor(fashion_price_label, dtype=torch.long).to(input_ids.device)) + \
                                        alpha_misc_loss * ce_loss_fct(fashion_customer_review, torch.tensor(fashion_customer_review_label, dtype=torch.long).to(input_ids.device)) + \
                                        0 * ce_loss_fct(furniture_coref, torch.tensor(coref_label, dtype=torch.long).to(input_ids.device))  # 增加该行
                    else:
                        loss_per_line = alpha_disamb_candi_loss * ce_loss_fct(fashion_disamb, torch.tensor(disamb_label, dtype=torch.long).to(input_ids.device)) + \
                                        alpha_coref_loss * ce_loss_fct(fashion_coref, torch.tensor(coref_label, dtype=torch.long).to(input_ids.device)) + \
                                        alpha_misc_loss * ce_loss_fct(fashion_size, torch.tensor(fashion_size_label, dtype=torch.long).to(input_ids.device)) + \
                                        alpha_misc_loss * bce_loss_fct(fashion_available_sizes, torch.tensor(fashion_available_sizes_label, dtype=torch.float32).to(input_ids.device)) + \
                                        alpha_misc_loss * ce_loss_fct(fashion_brand, torch.tensor(fashion_brand_label, dtype=torch.long).to(input_ids.device)) + \
                                        alpha_misc_loss * ce_loss_fct(fashion_color, torch.tensor(fashion_color_label, dtype=torch.long).to(input_ids.device)) + \
                                        alpha_misc_loss * ce_loss_fct(fashion_pattern, torch.tensor(fashion_pattern_label, dtype=torch.long).to(input_ids.device)) + \
                                        alpha_misc_loss * ce_loss_fct(fashion_sleeve_length, torch.tensor(fashion_sleeve_length_label, dtype=torch.long).to(input_ids.device)) + \
                                        alpha_misc_loss * ce_loss_fct(fashion_asset_type, torch.tensor(fashion_asset_type_label, dtype=torch.long).to(input_ids.device)) + \
                                        alpha_misc_loss * ce_loss_fct(fashion_type_, torch.tensor(fashion_type_label, dtype=torch.long).to(input_ids.device)) + \
                                        alpha_misc_loss * ce_loss_fct(fashion_price, torch.tensor(fashion_price_label, dtype=torch.long).to(input_ids.device)) + \
                                        alpha_misc_loss * ce_loss_fct(fashion_customer_review, torch.tensor(fashion_customer_review_label, dtype=torch.long).to(input_ids.device)) + \
                                        0 * ce_loss_fct(furniture_coref, torch.tensor(coref_label, dtype=torch.long).to(input_ids.device))  # 增加该行
                else: 
                    enc_head_results.append((furniture_disamb, furniture_coref, furniture_brand, furniture_color, furniture_materials, furniture_type_, furniture_price, furniture_customer_review))
                    if use_focal_loss:
                        loss_per_line = alpha_disamb_candi_loss * focal_loss_fct3(furniture_disamb, torch.tensor(disamb_label, dtype=torch.long).to(input_ids.device)) + \
                                        alpha_coref_loss * focal_loss_fct4(furniture_coref, torch.tensor(coref_label, dtype=torch.long).to(input_ids.device)) + \
                                        alpha_misc_loss * ce_loss_fct(furniture_brand, torch.tensor(furniture_brand_label, dtype=torch.long).to(input_ids.device)) + \
                                        alpha_misc_loss * ce_loss_fct(furniture_color, torch.tensor(furniture_color_label, dtype=torch.long).to(input_ids.device)) + \
                                        alpha_misc_loss * ce_loss_fct(furniture_materials, torch.tensor(furniture_materials_label, dtype=torch.long).to(input_ids.device)) + \
                                        alpha_misc_loss * ce_loss_fct(furniture_type_, torch.tensor(furniture_type_label, dtype=torch.long).to(input_ids.device)) + \
                                        alpha_misc_loss * ce_loss_fct(furniture_price, torch.tensor(furniture_price_label, dtype=torch.long).to(input_ids.device)) + \
                                        alpha_misc_loss * ce_loss_fct(furniture_customer_review, torch.tensor(furniture_customer_review_label, dtype=torch.long).to(input_ids.device)) + \
                                        0 * ce_loss_fct(fashion_coref, torch.tensor(coref_label, dtype=torch.long).to(input_ids.device))  # 增加该行
                    else:
                        loss_per_line = alpha_disamb_candi_loss * ce_loss_fct(furniture_disamb, torch.tensor(disamb_label, dtype=torch.long).to(input_ids.device)) + \
                                        alpha_coref_loss * ce_loss_fct(furniture_coref, torch.tensor(coref_label, dtype=torch.long).to(input_ids.device)) + \
                                        alpha_misc_loss * ce_loss_fct(furniture_brand, torch.tensor(furniture_brand_label, dtype=torch.long).to(input_ids.device)) + \
                                        alpha_misc_loss * ce_loss_fct(furniture_color, torch.tensor(furniture_color_label, dtype=torch.long).to(input_ids.device)) + \
                                        alpha_misc_loss * ce_loss_fct(furniture_materials, torch.tensor(furniture_materials_label, dtype=torch.long).to(input_ids.device)) + \
                                        alpha_misc_loss * ce_loss_fct(furniture_type_, torch.tensor(furniture_type_label, dtype=torch.long).to(input_ids.device)) + \
                                        alpha_misc_loss * ce_loss_fct(furniture_price, torch.tensor(furniture_price_label, dtype=torch.long).to(input_ids.device)) + \
                                        alpha_misc_loss * ce_loss_fct(furniture_customer_review, torch.tensor(furniture_customer_review_label, dtype=torch.long).to(input_ids.device)) + \
                                        0 * ce_loss_fct(fashion_coref, torch.tensor(coref_label, dtype=torch.long).to(input_ids.device))  # 增加该行
                
                misc_loss += loss_per_line
            misc_loss /= batch_size

        elif misc is not None:
            """ eval and test
                不计算loss
            """
            enc_head_results = []

            for b_idx in range(batch_size):
                #coref_obj_each_batch = []
                for obj_idx in range(len(misc[b_idx])):
                    pos = misc[b_idx][obj_idx]['pos']
                    # hidden_concat: (num_obj, 2*model)
                    if obj_idx == 0:
                        hidden_concat = torch.reshape(enc_last_state[b_idx][pos:pos+2], (1,-1))
                    else:
                        hidden_concat = torch.cat([hidden_concat, torch.reshape(enc_last_state[b_idx][pos:pos+2], (1,-1))], dim=0)
                
                objs_pos = [misc[b_idx][obj_idx]['pos'] for obj_idx in range(len(misc[b_idx]))]
                #obj_indices = [tokenizer_id2token[enc_input[b_idx][pos].item()] for pos in objs_pos]  # ex) [<11>, <41>, ...]

                is_fashion = misc[b_idx][0]['is_fashion']
                if is_fashion:
                    enc_head_results_tuple = self.fashion_enc_head(hidden_concat)
                    #disamb, coref, size, available_sizes, brand, color, pattern, sleeve_length, \
                    #asset_type, type_, price, customer_review = fashion_enc_head(hidden_concat)
                else:
                    enc_head_results_tuple = self.furniture_enc_head(hidden_concat)
                    #disamb, coref, brand, color, materials, type_, price, customer_review = furniture_enc_head(hidden_concat)

                #coref_predict = coref.argmax(dim=1).tolist()  # (num_objs)
                #for i, coref_signal in enumerate(coref_predict):
                #    if coref_signal:
                #        coref_obj_each_batch.append(obj_indices[i])
                #coref_obj_list.append(coref_obj_each_batch)
                #coref_check.append(True if len(coref_obj_each_batch) > 0 else False)

                enc_head_results.append(enc_head_results_tuple)
        else:
            enc_head_results = None

        loss = alpha_masked_lm_loss*masked_lm_loss + alpha_nocoref_loss*nocoref_loss + misc_loss + alpha_disam_loss*disam_loss + alpha_retrieval_loss*retrieval_loss

        if not return_dict:
            return [loss,masked_lm_loss,nocoref_loss,misc_loss,disam_loss,retrieval_loss]
            #output = (lm_logits,) + outputs[1:]
            #return ((loss, masked_lm_loss,nocoref_loss,misc_loss,disam_loss,retrieval_loss) + output) if masked_lm_loss is not None else output

        return Seq2SeqLMOutputForSIMMC(
            loss=loss,
            masked_lm_loss=masked_lm_loss,
            nocoref_loss=nocoref_loss,
            misc_loss=misc_loss,
            disam_loss=disam_loss,
            retrieval_loss=retrieval_loss,
            logits=lm_logits,
            disambiguation_logits=disambiguation_logits,
            enc_head_results=enc_head_results,
            nocoref_logits=nocoref_logits,
            past_key_values=outputs.past_key_values,
            decoder_hidden_states=outputs.decoder_hidden_states,
            decoder_attentions=outputs.decoder_attentions,
            cross_attentions=outputs.cross_attentions,
            encoder_last_hidden_state=outputs.encoder_last_hidden_state,
            encoder_hidden_states=outputs.encoder_hidden_states,
            encoder_attentions=outputs.encoder_attentions,
        )

    def prepare_inputs_for_generation(
        self,
        decoder_input_ids,
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
            decoder_input_ids = decoder_input_ids[:, -1:]

        return {
            "input_ids": None,  # encoder_outputs is defined. input_ids not needed
            "encoder_outputs": encoder_outputs,
            "past_key_values": past,
            "decoder_input_ids": decoder_input_ids,
            "attention_mask": attention_mask,
            "head_mask": head_mask,
            "decoder_head_mask": decoder_head_mask,
            "cross_attn_head_mask": cross_attn_head_mask,
            "use_cache": use_cache,  # change this to avoid caching (presumably for debugging)
        }

    def prepare_decoder_input_ids_from_labels(self, labels: torch.Tensor):
        return shift_tokens_right(labels, self.config.pad_token_id, self.config.decoder_start_token_id)

    @staticmethod
    def _reorder_cache(past, beam_idx):
        reordered_past = ()
        for layer_past in past:
            # cached cross_attention states don't have to be reordered -> they are always the same
            reordered_past += (
                tuple(past_state.index_select(0, beam_idx) for past_state in layer_past[:2]) + layer_past[2:],
            )
        return reordered_past







