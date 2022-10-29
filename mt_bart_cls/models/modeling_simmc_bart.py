# coding=utf-8
# Copyright 2021 The Fairseq Authors and The HuggingFace Inc. team. All rights reserved.
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
from typing import List, Optional, Tuple, Union, Dict

import torch
import torch.utils.checkpoint
from torch import nn
from torch.nn import BCEWithLogitsLoss, CrossEntropyLoss

from transformers.models.bart import *
from transformers.modeling_outputs import (
    BaseModelOutput,
)
from transformers.utils import (
    logging,
)
from .simmc_utils import *
logger = logging.get_logger(__name__)


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



class MultiTaskBartForConditionalGeneration(BartPretrainedModel):
    base_model_prefix = "model"
    _keys_to_ignore_on_load_missing = [r"final_logits_bias", r"lm_head.weight"]

    def __init__(self, config: BartConfig):
        super().__init__(config)
        self.model = BartModel(config)
        self.register_buffer("final_logits_bias", torch.zeros((1, self.model.shared.num_embeddings)))
        self.lm_head = nn.Linear(config.d_model, self.model.shared.num_embeddings, bias=False)
        # Add module here
        # Updated by Yirong Chen
        self.box_embedding = BoxEmbedding(config.d_model)
        self.nocoref_head = NoCorefHead(config.d_model)
        self.fashion_enc_head = FashionEncoderHead(config.d_model)
        self.furniture_enc_head = FurnitureEncoderHead(config.d_model)
        self.disambiguation_head = DisambiguationHead(config.d_model)

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
        boxes: Optional[List[List[List[float]]]] = None,
        misc: Optional[List[List[Dict]]] = None,
        nocoref: Optional[List] = None,
        response: Optional[torch.LongTensor] = None, # 新增
        response_attention_mask: Optional[torch.LongTensor] = None, # 新增
        disambiguation_labels: Optional[torch.LongTensor] = None, # 新增
        do_retrieval: Optional[bool] = False,   # 新增，指定是否有检索任务
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
                box_embedded = self.box_embedding(torch.tensor(boxes[b_idx]).to(input_ids.device))  # (num_obj_per_line, d_model)
                for obj_idx in range(len(misc[b_idx])):
                    pos = misc[b_idx][obj_idx]['pos']
                    inputs_embeds[b_idx][pos] += box_embedded[obj_idx]

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
        # disambiguation_label = torch.argmax(disambiguation_logits, dim=-1).squeeze()
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
            # is_nocoref = nocoref_logits.argmax(dim=1).bool()
            nocoref_loss = 0
        else:
            nocoref_logits = None
            nocoref_loss = 0

        misc_loss = 0
        
        if misc is not None and "coref_label" in misc[0][0]:
            """ train 阶段，计算loss
            """
            enc_head_results = []

            for b_idx in range(batch_size):  # in a batch
                is_fashion = misc[b_idx][0]['is_fashion']
                coref_label = [misc[b_idx][obj_idx]['coref_label'] for obj_idx in range(len(misc[b_idx]))]  # (num_obj)  0 or 1
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

                fashion_coref, fashion_size, fashion_available_sizes, fashion_brand, fashion_color, fashion_pattern, fashion_sleeve_length, \
                fashion_asset_type, fashion_type_, fashion_price, fashion_customer_review = self.fashion_enc_head(hidden_concat)  # (num_obj, num_logits)
                furniture_coref, furniture_brand, furniture_color, furniture_materials, furniture_type_, furniture_price, furniture_customer_review = self.furniture_enc_head(hidden_concat)  # (num_obj, num_logits)
                
                if is_fashion:
                    loss_per_line = 8 * ce_loss_fct(fashion_coref, torch.tensor(coref_label, dtype=torch.long).to(input_ids.device)) + \
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
                    loss_per_line = 8 * ce_loss_fct(furniture_coref, torch.tensor(coref_label, dtype=torch.long).to(input_ids.device)) + \
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

        loss = masked_lm_loss + 0.1*nocoref_loss + 0.1*misc_loss + 0.1*disam_loss + 0.4*retrieval_loss

        if not return_dict:
            return [loss]
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


