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
from transformers.models.roberta import *
from transformers.modeling_outputs import (
    BaseModelOutput,
)
from transformers.utils import (
    logging,
)
from .simmc_utils import *
import pdb

logger = logging.get_logger(__name__)


class RobertaForSIMMC(RobertaPreTrainedModel):
    # base_model_prefix = "roberta"
    _keys_to_ignore_on_load_missing = [r"position_ids"]

    def __init__(self, config):
        super().__init__(config)
        self.roberta = RobertaModel(config)

        self.box_embedding = BoxEmbedding(config.hidden_size)
        self.nocoref_head = NoCorefHead(config.hidden_size)
        self.fashion_enc_head = FashionEncoderHeadV2(config.hidden_size)
        self.furniture_enc_head = FurnitureEncoderHeadV2(config.hidden_size)
        self.disambiguation_head = DisambiguationHead(config.hidden_size)
        self.dst_head = DSTHeadV2(config.hidden_size)

        # Initialize weights and apply final processing
        self.post_init()


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
        dst: Optional[List[Dict]] = None,
        id2word: Dict = None
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
        # follow `class BartEncoder`. shape of (batch, seqlen, hidden_size)
        if boxes is not None: # 考虑box_embedding
            inputs_embeds = self.roberta.embeddings(input_ids)
            for b_idx in range(batch_size):  # in a batch
                box_embedded = self.box_embedding(torch.tensor(boxes[b_idx]).to(input_ids.device))  # (num_obj_per_line, hidden_size)
                for obj_idx in range(len(misc[b_idx])):
                    pos = misc[b_idx][obj_idx]['pos']
                    inputs_embeds[b_idx][pos] += box_embedded[obj_idx]

            outputs = self.roberta(
                input_ids=None,
                attention_mask=attention_mask,
                head_mask=head_mask,
                inputs_embeds=inputs_embeds,
                output_attentions=output_attentions,
                output_hidden_states=output_hidden_states,
                return_dict=True,
                )
        else: # 不考虑box_embedding
            outputs = self.roberta(
                input_ids=input_ids,
                attention_mask=attention_mask,
                head_mask=head_mask,
                inputs_embeds=None,
                output_attentions=output_attentions,
                output_hidden_states=output_hidden_states,
                return_dict=True,
            )

        # pdb.set_trace()
        enc_last_state = outputs.last_hidden_state  # (bs, seqlen, hidden_size)

        if do_retrieval: # 训练阶段
            # For Biencoder
            response_vec = self.roberta.encoder(input_ids=response, attention_mask=response_attention_mask)[0][:, 0, :] # bs, dim
            context_vec = enc_last_state[:, 0, :] # bs, dim
            dot_product = torch.matmul(context_vec, response_vec.t())  # bs, bs
            retrieval_loss = ce_loss_fct(dot_product, torch.arange(batch_size).to(context_vec.device))
        else: # 验证或测试阶段或该任务不参与训练
            retrieval_loss = 0

        # For Disambiguation
        disambiguation_logits = self.disambiguation_head(enc_last_state[:, 1, :]) # bs, hidden_size --> bs, 2
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

        ## DST 任务
        dst_loss = 0
        if dst is not None:
            # pdb.set_trace()
            for bidx in range(batch_size):
                dst_pos = dst[bidx]["pos"]
                dst_tokens = [id2word[int(input_ids[bidx][dst_pos+i])] for i in range(13)]
                logger.info("====DST TOKENS====\t%s" % " ".join(dst_tokens))
                print("====DST TOKENS====\t%s" % " ".join(dst_tokens))
            ## 每个 special token 上面做预测，context后面拼了13个special token
            dst_id_state = torch.stack([enc_last_state[b_idx, dst[b_idx]["pos"]:dst[b_idx]["pos"]+13, :] for b_idx in range(batch_size)])
            dst_act, dst_request_slots, dst_type, dst_price, dst_customer_review, dst_brand, dst_size, \
                dst_pattern, dst_color, dst_sleeve_length, dst_available_sizes, dst_materials, \
                    dst_customer_rating = self.dst_head(dst_id_state)
            dst_act_label = [dst[b_idx]["dst_label"]["act"] for b_idx in range(batch_size)]
            dst_request_slots_label = [dst[b_idx]["dst_label"]["request_slots"] for b_idx in range(batch_size)]
            dst_type_label = [dst[b_idx]["dst_label"]["type"] for b_idx in range(batch_size)]
            dst_price_label = [dst[b_idx]["dst_label"]["price"] for b_idx in range(batch_size)]
            dst_customer_review_label = [dst[b_idx]["dst_label"]["customerReview"] for b_idx in range(batch_size)]
            dst_brand_label = [dst[b_idx]["dst_label"]["brand"] for b_idx in range(batch_size)]
            dst_size_label = [dst[b_idx]["dst_label"]["size"] for b_idx in range(batch_size)]
            dst_pattern_label = [dst[b_idx]["dst_label"]["pattern"] for b_idx in range(batch_size)]
            dst_color_label = [dst[b_idx]["dst_label"]["color"] for b_idx in range(batch_size)]
            dst_sleeve_length_label = [dst[b_idx]["dst_label"]["sleeveLength"] for b_idx in range(batch_size)]
            dst_available_sizes_label = [dst[b_idx]["dst_label"]["availableSizes"] for b_idx in range(batch_size)]
            dst_materials_label = [dst[b_idx]["dst_label"]["materials"] for b_idx in range(batch_size)]
            dst_customer_rating_label = [dst[b_idx]["dst_label"]["customerRating"] for b_idx in range(batch_size)]
            dst_loss = ce_loss_fct(dst_act, torch.tensor(dst_act_label, dtype=torch.long).to(input_ids.device)) + \
                            bce_loss_fct(dst_request_slots, torch.tensor(dst_request_slots_label, dtype=torch.float32).to(input_ids.device)) + \
                            ce_loss_fct(dst_type, torch.tensor(dst_type_label, dtype=torch.long).to(input_ids.device)) + \
                            ce_loss_fct(dst_price, torch.tensor(dst_price_label, dtype=torch.long).to(input_ids.device)) + \
                            ce_loss_fct(dst_customer_review, torch.tensor(dst_customer_review_label, dtype=torch.long).to(input_ids.device)) + \
                            ce_loss_fct(dst_brand, torch.tensor(dst_brand_label, dtype=torch.long).to(input_ids.device)) + \
                            ce_loss_fct(dst_size, torch.tensor(dst_size_label, dtype=torch.long).to(input_ids.device)) + \
                            ce_loss_fct(dst_pattern, torch.tensor(dst_pattern_label, dtype=torch.long).to(input_ids.device)) + \
                            ce_loss_fct(dst_color, torch.tensor(dst_color_label, dtype=torch.long).to(input_ids.device)) + \
                            ce_loss_fct(dst_sleeve_length, torch.tensor(dst_sleeve_length_label, dtype=torch.long).to(input_ids.device)) + \
                            bce_loss_fct(dst_available_sizes, torch.tensor(dst_available_sizes_label, dtype=torch.float32).to(input_ids.device)) + \
                            ce_loss_fct(dst_materials, torch.tensor(dst_materials_label, dtype=torch.long).to(input_ids.device)) + \
                            ce_loss_fct(dst_customer_rating, torch.tensor(dst_customer_rating_label, dtype=torch.long).to(input_ids.device))

        misc_loss = 0
        # pdb.set_trace()
        if misc is not None and "coref_label" in misc[0][0]:
            """ train 阶段，计算loss
            """
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
                    # hidden_concat = torch.reshape(enc_last_state[b_idx][pos:pos+2], (-1,))  # (2*hidden_size)  -> 

                fashion_disamb, fashion_coref, fashion_size, fashion_available_sizes, fashion_brand, fashion_color, fashion_pattern, fashion_sleeve_length, \
                fashion_asset_type, fashion_type_, fashion_price, fashion_customer_review = self.fashion_enc_head(hidden_concat)  # (num_obj, num_logits)
                furniture_disamb, furniture_coref, furniture_brand, furniture_color, furniture_materials, furniture_type_, furniture_price, furniture_customer_review = self.furniture_enc_head(hidden_concat)  # (num_obj, num_logits)
                
                # pdb.set_trace()
                if is_fashion:
                    loss_per_line = 8 * ce_loss_fct(fashion_disamb, torch.tensor(disamb_label, dtype=torch.long).to(input_ids.device)) + \
                                    8 * ce_loss_fct(fashion_coref, torch.tensor(coref_label, dtype=torch.long).to(input_ids.device)) + \
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
                    loss_per_line = 8 * ce_loss_fct(furniture_disamb, torch.tensor(disamb_label, dtype=torch.long).to(input_ids.device)) + \
                                    8 * ce_loss_fct(furniture_coref, torch.tensor(coref_label, dtype=torch.long).to(input_ids.device)) + \
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
                
                is_fashion = misc[b_idx][0]['is_fashion']
                if is_fashion:
                    enc_head_results_tuple = self.fashion_enc_head(hidden_concat)
                else:
                    enc_head_results_tuple = self.furniture_enc_head(hidden_concat)

                enc_head_results.append(enc_head_results_tuple)
        else:
            enc_head_results = None

        # loss = 0.1 * nocoref_loss + 0.1 * misc_loss + 0.1 * disam_loss + 0.4 * retrieval_loss
        loss = nocoref_loss + misc_loss + disam_loss

        if not return_dict:
            return [loss ,nocoref_loss, misc_loss, disam_loss, retrieval_loss]

        return SequenceClassifierOutputSIMMC(
            loss=loss,
            nocoref_loss=nocoref_loss,
            misc_loss=misc_loss,
            disam_loss=disam_loss,
            retrieval_loss=retrieval_loss,
            dst_loss=dst_loss,
            disambiguation_logits=disambiguation_logits,
            nocoref_logits=nocoref_logits,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )


