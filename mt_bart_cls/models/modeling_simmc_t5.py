# coding=utf-8
# Copyright 2018 Mesh TensorFlow authors, T5 Authors and HuggingFace Inc. team.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
import copy
import math
import os
import warnings
from typing import List, Optional, Tuple, Union, Dict

import torch
from torch import nn
from torch.nn import BCEWithLogitsLoss, CrossEntropyLoss, MSELoss
from torch.utils.checkpoint import checkpoint

from transformers.activations import ACT2FN
from transformers.modeling_outputs import (
    BaseModelOutput,
    BaseModelOutputWithPastAndCrossAttentions,
    Seq2SeqLMOutput,
    Seq2SeqModelOutput,
)
from transformers.modeling_utils import PreTrainedModel
from transformers.pytorch_utils import find_pruneable_heads_and_indices, prune_linear_layer
from transformers.utils import (
    DUMMY_INPUTS,
    DUMMY_MASK,
    add_start_docstrings,
    add_start_docstrings_to_model_forward,
    is_torch_fx_proxy,
    logging,
    replace_return_docstrings,
)
from transformers.utils.model_parallel_utils import assert_device_map, get_device_map
from transformers.models.t5 import T5PreTrainedModel, T5Config
from transformers.models.t5.modeling_t5 import *
from transformers.models.t5.modeling_t5 import __HEAD_MASK_WARNING_MSG

from .simmc_utils import *


logger = logging.get_logger(__name__)

_CONFIG_FOR_DOC = "T5Config"
_TOKENIZER_FOR_DOC = "T5Tokenizer"
_CHECKPOINT_FOR_DOC = "t5-small"


class MultiTaskT5ForConditionalGeneration(T5PreTrainedModel):
    _keys_to_ignore_on_load_missing = [
        r"encoder.embed_tokens.weight",
        r"decoder.embed_tokens.weight",
        r"lm_head.weight",
    ]
    _keys_to_ignore_on_load_unexpected = [
        r"decoder.block.0.layer.1.EncDecAttention.relative_attention_bias.weight",
    ]

    def __init__(self, config: T5Config):
        super().__init__(config)
        self.model_dim = config.d_model
        self.embed_scale = math.sqrt(self.model_dim)

        self.shared = nn.Embedding(config.vocab_size, config.d_model)

        encoder_config = copy.deepcopy(config)
        encoder_config.is_decoder = False
        encoder_config.use_cache = False
        encoder_config.is_encoder_decoder = False
        self.encoder = T5Stack(encoder_config, self.shared)

        decoder_config = copy.deepcopy(config)
        decoder_config.is_decoder = True
        decoder_config.is_encoder_decoder = False
        decoder_config.num_layers = config.num_decoder_layers
        self.decoder = T5Stack(decoder_config, self.shared)

        self.lm_head = nn.Linear(config.d_model, config.vocab_size, bias=False)
        # Add module here
        # Updated by Yirong Chen
        self.box_embedding = BoxEmbedding(config.d_model)
        self.nocoref_head = NoCorefHead(config.d_model)
        self.fashion_enc_head = FashionEncoderHeadV2(config.d_model)
        self.furniture_enc_head = FurnitureEncoderHeadV2(config.d_model)
        self.disambiguation_head = DisambiguationHead(config.d_model)

        # Initialize weights and apply final processing
        self.post_init()

        # Model parallel
        self.model_parallel = False
        self.device_map = None

    @add_start_docstrings(PARALLELIZE_DOCSTRING)
    def parallelize(self, device_map=None):
        self.device_map = (
            get_device_map(len(self.encoder.block), range(torch.cuda.device_count()))
            if device_map is None
            else device_map
        )
        assert_device_map(self.device_map, len(self.encoder.block))
        self.encoder.parallelize(self.device_map)
        self.decoder.parallelize(self.device_map)
        self.box_embedding = self.box_embedding.to(self.encoder.first_device) # 加到并行函数当中
        self.lm_head = self.lm_head.to(self.decoder.first_device)
        # 并行
        self.nocoref_head = self.nocoref_head.to(self.decoder.first_device)
        self.fashion_enc_head = self.fashion_enc_head.to(self.encoder.first_device)
        self.furniture_enc_head = self.furniture_enc_head.to(self.encoder.first_device)
        self.disambiguation_head = self.disambiguation_head.to(self.encoder.first_device)

        self.model_parallel = True

    @add_start_docstrings(DEPARALLELIZE_DOCSTRING)
    def deparallelize(self):
        self.encoder.deparallelize()
        self.decoder.deparallelize()
        self.encoder = self.encoder.to("cpu")
        self.decoder = self.decoder.to("cpu")
        self.lm_head = self.lm_head.to("cpu")
        self.box_embedding = self.box_embedding.to("cpu")
        self.nocoref_head = self.nocoref_head.to("cpu")
        self.fashion_enc_head = self.fashion_enc_head.to("cpu")
        self.furniture_enc_head = self.furniture_enc_head.to("cpu")
        self.disambiguation_head = self.disambiguation_head.to("cpu")
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

    @add_start_docstrings_to_model_forward(T5_INPUTS_DOCSTRING)
    @replace_return_docstrings(output_type=Seq2SeqLMOutput, config_class=_CONFIG_FOR_DOC)
    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None, # enc_input
        attention_mask: Optional[torch.FloatTensor] = None, # enc_attention_mask
        decoder_input_ids: Optional[torch.LongTensor] = None, # =decoder_input[:, :-1]
        decoder_attention_mask: Optional[torch.BoolTensor] = None, # =decoder_attention_mask[:, :-1]
        head_mask: Optional[torch.FloatTensor] = None,
        decoder_head_mask: Optional[torch.FloatTensor] = None,
        cross_attn_head_mask: Optional[torch.Tensor] = None,
        encoder_outputs: Optional[Tuple[Tuple[torch.Tensor]]] = None,
        past_key_values: Optional[Tuple[Tuple[torch.Tensor]]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        decoder_inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
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
    ) -> Union[Tuple[torch.FloatTensor], Seq2SeqLMOutputForSIMMC]:
        r"""
        labels (`torch.LongTensor` of shape `(batch_size,)`, *optional*):
            Labels for computing the sequence classification/regression loss. Indices should be in `[-100, 0, ...,
            config.vocab_size - 1]`. All labels set to `-100` are ignored (masked), the loss is only computed for
            labels in `[0, ..., config.vocab_size]`

        Returns:

        Examples:

        ```python
        >>> from transformers import T5Tokenizer, MultiTaskT5ForConditionalGeneration

        >>> tokenizer = T5Tokenizer.from_pretrained("t5-small")
        >>> model = MultiTaskT5ForConditionalGeneration.from_pretrained("t5-small")

        >>> # training
        >>> input_ids = tokenizer("The <extra_id_0> walks in <extra_id_1> park", return_tensors="pt").input_ids
        >>> labels = tokenizer("<extra_id_0> cute dog <extra_id_1> the <extra_id_2>", return_tensors="pt").input_ids
        >>> outputs = model(input_ids=input_ids, labels=labels)
        >>> loss = outputs.loss
        >>> logits = outputs.logits

        >>> # inference
        >>> input_ids = tokenizer(
        ...     "summarize: studies have shown that owning a dog is good for you", return_tensors="pt"
        ... ).input_ids  # Batch size 1
        >>> outputs = model.generate(input_ids)
        >>> print(tokenizer.decode(outputs[0], skip_special_tokens=True))
        >>> # studies have shown that owning a dog is good for you.
        ```"""
        use_cache = use_cache if use_cache is not None else self.config.use_cache
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        # 损失函数统一放置在这里
        # Updated by Yirong Chen
        ce_loss_fct = CrossEntropyLoss()
        bce_loss_fct = BCEWithLogitsLoss()

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


        # FutureWarning: head_mask was separated into two input args - head_mask, decoder_head_mask
        if head_mask is not None and decoder_head_mask is None:
            if self.config.num_layers == self.config.num_decoder_layers:
                warnings.warn(__HEAD_MASK_WARNING_MSG, FutureWarning)
                decoder_head_mask = head_mask

        if boxes is not None: # 考虑box_embedding
            inputs_embeds = self.encoder.embed_tokens(input_ids) * self.embed_scale

            for b_idx in range(batch_size):  # in a batch
                box_embedded = self.box_embedding(torch.tensor(boxes[b_idx]).to(input_ids.device))  # (num_obj_per_line, d_model)
                for obj_idx in range(len(misc[b_idx])):
                    pos = misc[b_idx][obj_idx]['pos']
                    inputs_embeds[b_idx][pos] += box_embedded[obj_idx]

            # Encode if needed (training, first prediction pass)
            if encoder_outputs is None:
                # Convert encoder inputs in embeddings if needed
                encoder_outputs = self.encoder(
                    input_ids=None,
                    attention_mask=attention_mask,
                    inputs_embeds=inputs_embeds,
                    head_mask=head_mask,
                    output_attentions=output_attentions,
                    output_hidden_states=output_hidden_states,
                    return_dict=True,
                )
            elif return_dict and not isinstance(encoder_outputs, BaseModelOutput):
                encoder_outputs = BaseModelOutput(
                    last_hidden_state=encoder_outputs[0],
                    hidden_states=encoder_outputs[1] if len(encoder_outputs) > 1 else None,
                    attentions=encoder_outputs[2] if len(encoder_outputs) > 2 else None,
                )

        else: # 不考虑box_embedding
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
                    return_dict=True,
                )
            elif return_dict and not isinstance(encoder_outputs, BaseModelOutput):
                encoder_outputs = BaseModelOutput(
                    last_hidden_state=encoder_outputs[0],
                    hidden_states=encoder_outputs[1] if len(encoder_outputs) > 1 else None,
                    attentions=encoder_outputs[2] if len(encoder_outputs) > 2 else None,
                )



        hidden_states = encoder_outputs[0]
        enc_last_state = encoder_outputs.last_hidden_state  # (bs, seqlen, d_model)

        if do_retrieval: # 训练阶段
            # For Biencoder
            response_vec = self.encoder(input_ids=response, attention_mask=response_attention_mask)[0][:, 0, :] # bs, dim
            context_vec = enc_last_state[:, 0, :] # bs, dim
            if self.model_parallel:
                response_vec = response_vec.to(self.encoder.first_device)
                context_vec = context_vec.to(self.encoder.first_device)
            dot_product = torch.matmul(context_vec, response_vec.t())  # bs, bs
            retrieval_loss = ce_loss_fct(dot_product, torch.arange(batch_size).to(context_vec.device))
        else: # 验证或测试阶段或该任务不参与训练
            retrieval_loss = 0


        if self.model_parallel:
            torch.cuda.set_device(self.decoder.first_device)

        if labels is not None and decoder_input_ids is None and decoder_inputs_embeds is None:
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
        )

        sequence_output = decoder_outputs[0]

        # Set device for model parallelism
        if self.model_parallel:
            torch.cuda.set_device(self.encoder.first_device)
            self.lm_head = self.lm_head.to(self.encoder.first_device)
            enc_last_state = enc_last_state.to(self.encoder.first_device) # 所有利用到该部分的在decoder.first_device进行计算
            sequence_output = sequence_output.to(self.lm_head.weight.device)

        if self.config.tie_word_embeddings:
            # Rescale output before projecting on vocab
            # See https://github.com/tensorflow/mesh/blob/fa19d69eafc9a482aff0b59ddd96b025c0cb207d/mesh_tensorflow/transformer/transformer.py#L586
            sequence_output = sequence_output * (self.model_dim**-0.5)

        lm_logits = self.lm_head(sequence_output)

        #loss = None
        masked_lm_loss = 0 # model_loss
        if labels is not None:
            loss_fct = CrossEntropyLoss(ignore_index=-100)
            masked_lm_loss = loss_fct(lm_logits.view(-1, lm_logits.size(-1)), labels.view(-1))
            # TODO(thom): Add z_loss https://github.com/tensorflow/mesh/blob/fa19d69eafc9a482aff0b59ddd96b025c0cb207d/mesh_tensorflow/layers.py#L666

        # For Disambiguation
        # 2022.08.17 20:00 注意<DISAM>的位置在第一个位置，T5没有在输入前面加开始符号
        # disambiguation_logits = self.disambiguation_head(enc_last_state[:, 0, :]) # bs, d_model --> bs, 2
        # 2022.08.19 20:00 更新了本模型的输入格式，见tokenization_t5.py，所以在输入前面增加了<s>
        # <s> <DISAM> ...
        #  0     1    ...
        disambiguation_logits = self.disambiguation_head(enc_last_state[:, 0, :])
        # disambiguation_logits = self.disambiguation_head(enc_last_state[:, 1, :]) # bs, d_model --> bs, 2
        #disambiguation_label = torch.argmax(disambiguation_logits, dim=-1).squeeze()
        if disambiguation_labels is not None: # 训练阶段
            disam_loss = ce_loss_fct(disambiguation_logits, disambiguation_labels.view(-1))
        else: # 验证或测试阶段或该任务不参与训练
            disam_loss = 0
        
        if (nocoref is not None) and (isinstance(nocoref[0],tuple)): # 训练阶段
            # [(position, label), (position, label), (position, label),...]
            nocoref_logits = torch.stack([self.nocoref_head(enc_last_state[b_idx][nocoref[b_idx][0]]) for b_idx in range(batch_size) ])
            nocoref_labels = torch.tensor([nocoref[b_idx][1] for b_idx in range(batch_size)]).to(input_ids.device)
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
                    loss_per_line = 9 * ce_loss_fct(fashion_disamb, torch.tensor(disamb_label, dtype=torch.long).to(input_ids.device)) + \
                                    9 * ce_loss_fct(fashion_coref, torch.tensor(coref_label, dtype=torch.long).to(input_ids.device)) + \
                                    ce_loss_fct(fashion_size, torch.tensor(fashion_size_label, dtype=torch.long).to(input_ids.device)) + \
                                    bce_loss_fct(fashion_available_sizes, torch.tensor(fashion_available_sizes_label, dtype=torch.float32).to(input_ids.device)) + \
                                    ce_loss_fct(fashion_brand, torch.tensor(fashion_brand_label, dtype=torch.long).to(input_ids.device)) + \
                                    ce_loss_fct(fashion_color, torch.tensor(fashion_color_label, dtype=torch.long).to(input_ids.device)) + \
                                    ce_loss_fct(fashion_pattern, torch.tensor(fashion_pattern_label, dtype=torch.long).to(input_ids.device)) + \
                                    ce_loss_fct(fashion_sleeve_length, torch.tensor(fashion_sleeve_length_label, dtype=torch.long).to(input_ids.device)) + \
                                    ce_loss_fct(fashion_asset_type, torch.tensor(fashion_asset_type_label, dtype=torch.long).to(input_ids.device)) + \
                                    ce_loss_fct(fashion_type_, torch.tensor(fashion_type_label, dtype=torch.long).to(input_ids.device)) + \
                                    ce_loss_fct(fashion_price, torch.tensor(fashion_price_label, dtype=torch.long).to(input_ids.device)) + \
                                    ce_loss_fct(fashion_customer_review, torch.tensor(fashion_customer_review_label, dtype=torch.long).to(input_ids.device)) + \
                                    0 * ce_loss_fct(furniture_coref, torch.tensor(coref_label, dtype=torch.long).to(input_ids.device))  # 增加该行
                else: 
                    loss_per_line = 9 * ce_loss_fct(furniture_disamb, torch.tensor(disamb_label, dtype=torch.long).to(input_ids.device)) + \
                                    9 * ce_loss_fct(furniture_coref, torch.tensor(coref_label, dtype=torch.long).to(input_ids.device)) + \
                                    ce_loss_fct(furniture_brand, torch.tensor(furniture_brand_label, dtype=torch.long).to(input_ids.device)) + \
                                    ce_loss_fct(furniture_color, torch.tensor(furniture_color_label, dtype=torch.long).to(input_ids.device)) + \
                                    ce_loss_fct(furniture_materials, torch.tensor(furniture_materials_label, dtype=torch.long).to(input_ids.device)) + \
                                    ce_loss_fct(furniture_type_, torch.tensor(furniture_type_label, dtype=torch.long).to(input_ids.device)) + \
                                    ce_loss_fct(furniture_price, torch.tensor(furniture_price_label, dtype=torch.long).to(input_ids.device)) + \
                                    ce_loss_fct(furniture_customer_review, torch.tensor(furniture_customer_review_label, dtype=torch.long).to(input_ids.device)) + \
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
                    #coref, size, available_sizes, brand, color, pattern, sleeve_length, \
                    #asset_type, type_, price, customer_review = fashion_enc_head(hidden_concat)
                else:
                    enc_head_results_tuple = self.furniture_enc_head(hidden_concat)
                    #coref, brand, color, materials, type_, price, customer_review = furniture_enc_head(hidden_concat)

                #coref_predict = coref.argmax(dim=1).tolist()  # (num_objs)
                #for i, coref_signal in enumerate(coref_predict):
                #    if coref_signal:
                #        coref_obj_each_batch.append(obj_indices[i])
                #coref_obj_list.append(coref_obj_each_batch)
                #coref_check.append(True if len(coref_obj_each_batch) > 0 else False)

                enc_head_results.append(enc_head_results_tuple)
        else:
            enc_head_results = None
        
        if self.model_parallel:
            # 保证多任务所有loss加在一起时，是在同一块GPU
            torch.cuda.set_device(self.encoder.first_device)
            if isinstance(masked_lm_loss, torch.Tensor):
                masked_lm_loss = masked_lm_loss.to(self.encoder.first_device)
            if isinstance(nocoref_loss, torch.Tensor):
                nocoref_loss = nocoref_loss.to(self.encoder.first_device)
            if isinstance(misc_loss, torch.Tensor):
                misc_loss = misc_loss.to(self.encoder.first_device)
            if isinstance(disam_loss, torch.Tensor):
                disam_loss = disam_loss.to(self.encoder.first_device)
            if isinstance(retrieval_loss, torch.Tensor):
                retrieval_loss = retrieval_loss.to(self.encoder.first_device)

        #loss = masked_lm_loss + 0.1*nocoref_loss + 0.1*misc_loss + 0.1*disam_loss + 0.4*retrieval_loss
        loss = masked_lm_loss + 0.2*nocoref_loss + 0.1*misc_loss + 0.2*disam_loss + 0.4*retrieval_loss


        if not return_dict:
            return [loss,masked_lm_loss,nocoref_loss,misc_loss,disam_loss,retrieval_loss]
            #output = (lm_logits,) + decoder_outputs[1:] + encoder_outputs
            #return ((loss,) + output) if loss is not None else output

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

