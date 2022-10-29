from torch import nn
from transformers.utils import ModelOutput
from typing import List, Optional, Tuple, Union, Dict
import torch
from dataclasses import dataclass


class BoxEmbedding(nn.Module):
    def __init__(self, hidden_dim):
        super(BoxEmbedding, self).__init__()
        self.box_linear = nn.Linear(6, hidden_dim)  
        self.box_layer_norm = nn.LayerNorm(hidden_dim)
    def forward(self, box_feat):
        """
            box_feat: [x1/w-0.5, y1/h-0.5, x2/w-0.5, y2/h-0.5, (x2-x1)(y2-y1)/(h*w), z_value/largest_z_value]
        """
        transformed_box = self.box_layer_norm(self.box_linear(box_feat))
        return transformed_box

class NoCorefHead(nn.Module):
    def __init__(self, hidden_dim):
        super(NoCorefHead, self).__init__()
        self.no_coref_linear = nn.Linear(hidden_dim, 2)  
    def forward(self, no_coref_vector):
        coref_cls = self.no_coref_linear(no_coref_vector)
        return coref_cls

class DisambiguationHead(nn.Module):
    def __init__(self, hidden_dim):
        super(DisambiguationHead, self).__init__()
        self.disamb_linear = nn.Linear(hidden_dim, 2)  
    def forward(self, x):
        return self.disamb_linear(x)

class FashionEncoderHead(nn.Module):
    def __init__(self, hidden_dim):
        super(FashionEncoderHead, self).__init__()
        self.aggregator = nn.Linear(2*hidden_dim, 2*hidden_dim)
        self.coref_linear = nn.Linear(2*hidden_dim, 2)
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
    def forward(self, concat_vector):
        ''' concat_vector: concat of obj_index_vector and st_vector '''
        aggregated = self.aggregator(concat_vector)
        coref = self.coref_linear(aggregated)
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
        return coref, size, available_sizes, brand, color, pattern, sleeve_length, asset_type, type_, \
               price, customer_review

class FurnitureEncoderHead(nn.Module):
    def __init__(self, hidden_dim):
        super(FurnitureEncoderHead, self).__init__()
        self.aggregator = nn.Linear(2*hidden_dim, 2*hidden_dim)
        self.coref_linear = nn.Linear(2*hidden_dim, 2)
        self.brand_linear = nn.Linear(2*hidden_dim, 12)
        self.color_linear = nn.Linear(2*hidden_dim, 9)
        self.materials_linear = nn.Linear(2*hidden_dim, 7)
        self.type_linear = nn.Linear(2*hidden_dim, 10)
        self.price_linear = nn.Linear(2*hidden_dim, 10)
        self.customer_review_linear = nn.Linear(2*hidden_dim, 19)
    def forward(self, concat_vector):
        ''' concat_vector: concat of obj_index_vector and st_vector '''
        aggregated = self.aggregator(concat_vector)
        coref = self.coref_linear(aggregated)
        brand = self.brand_linear(aggregated)
        color = self.color_linear(aggregated)
        materials = self.materials_linear(aggregated)
        type_ = self.type_linear(aggregated)
        price = self.price_linear(aggregated)
        customer_review = self.customer_review_linear(aggregated)
        return coref, brand, color, materials, type_, price, customer_review

class FashionEncoderHeadV2(nn.Module):
    def __init__(self, hidden_dim):
        super(FashionEncoderHeadV2, self).__init__()
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
    def forward(self, concat_vector):
        ''' concat_vector: concat of obj_index_vector and st_vector '''
        aggregated = self.aggregator(concat_vector)
        disamb = self.disamb_linear(aggregated) # 用于判断每个object是否为歧义候选disambiguation candidate, Updated by Yirong Chen
        coref = self.coref_linear(aggregated) # 用于判断每个object是否为MM-Coref object
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

class FurnitureEncoderHeadV2(nn.Module):
    def __init__(self, hidden_dim):
        super(FurnitureEncoderHeadV2, self).__init__()
        self.aggregator = nn.Linear(2*hidden_dim, 2*hidden_dim)
        self.disamb_linear = nn.Linear(2*hidden_dim, 2) # 用于判断每个object是否为歧义候选disambiguation candidate, Updated by Yirong Chen
        self.coref_linear = nn.Linear(2*hidden_dim, 2) # 用于判断每个object是否为MM-Coref object
        self.brand_linear = nn.Linear(2*hidden_dim, 12)
        self.color_linear = nn.Linear(2*hidden_dim, 9)
        self.materials_linear = nn.Linear(2*hidden_dim, 7)
        self.type_linear = nn.Linear(2*hidden_dim, 10)
        self.price_linear = nn.Linear(2*hidden_dim, 10)
        self.customer_review_linear = nn.Linear(2*hidden_dim, 19)
    def forward(self, concat_vector):
        ''' concat_vector: concat of obj_index_vector and st_vector '''
        aggregated = self.aggregator(concat_vector)
        disamb = self.disamb_linear(aggregated) # 用于判断每个object是否为歧义候选disambiguation candidate, Updated by Yirong Chen
        coref = self.coref_linear(aggregated)
        brand = self.brand_linear(aggregated)
        color = self.color_linear(aggregated)
        materials = self.materials_linear(aggregated)
        type_ = self.type_linear(aggregated)
        price = self.price_linear(aggregated)
        customer_review = self.customer_review_linear(aggregated)
        return disamb, coref, brand, color, materials, type_, price, customer_review

class DSTHead(nn.Module):
    def __init__(self, hidden_dim):
        super(DSTHead, self).__init__()
        self.aggregator = nn.Linear(hidden_dim, hidden_dim)
        self.dst_act = nn.Linear(hidden_dim, 7)
        self.dst_request_slots = nn.Linear(hidden_dim, 10)
        self.dst_type = nn.Linear(hidden_dim, 28)
        self.dst_price = nn.Linear(hidden_dim, 58)
        self.dst_customer_review = nn.Linear(hidden_dim, 28)
        self.dst_brand = nn.Linear(hidden_dim, 27)
        self.dst_size = nn.Linear(hidden_dim, 7)
        self.dst_pattern = nn.Linear(hidden_dim, 34)
        self.dst_color = nn.Linear(hidden_dim, 70)
        self.dst_sleeve_length = nn.Linear(hidden_dim, 7)
        self.dst_available_sizes = nn.Linear(hidden_dim, 6)
        self.dst_materials = nn.Linear(hidden_dim, 7)
        self.dst_customer_rating = nn.Linear(hidden_dim, 20)
    def forward(self, concat_vector):
        aggregated = self.aggregator(concat_vector)
        dst_act = self.dst_act(aggregated)
        dst_request_slots = self.dst_request_slots(aggregated)
        dst_type = self.dst_type(aggregated)
        dst_price = self.dst_price(aggregated)
        dst_customer_review = self.dst_customer_review(aggregated)
        dst_brand = self.dst_brand(aggregated)
        dst_size = self.dst_size(aggregated)
        dst_pattern = self.dst_pattern(aggregated)
        dst_color = self.dst_color(aggregated)
        dst_sleeve_length = self.dst_sleeve_length(aggregated)
        dst_available_sizes = self.dst_available_sizes(aggregated)
        dst_materials = self.dst_materials(aggregated)
        dst_customer_rating = self.dst_customer_rating(aggregated)
        return dst_act, dst_request_slots, dst_type, dst_price, dst_customer_review, dst_brand, dst_size, \
            dst_pattern, dst_color, dst_sleeve_length, dst_available_sizes, dst_materials, dst_customer_rating

class DSTHeadV2(nn.Module):
    def __init__(self, hidden_dim):
        super(DSTHeadV2, self).__init__()
        self.aggregator = nn.Linear(hidden_dim, hidden_dim)
        self.dst_act = nn.Linear(hidden_dim, 7)
        self.dst_request_slots = nn.Linear(hidden_dim, 10)
        self.dst_type = nn.Linear(hidden_dim, 28)
        self.dst_price = nn.Linear(hidden_dim, 58)
        self.dst_customer_review = nn.Linear(hidden_dim, 28)
        self.dst_brand = nn.Linear(hidden_dim, 27)
        self.dst_size = nn.Linear(hidden_dim, 7)
        self.dst_pattern = nn.Linear(hidden_dim, 34)
        self.dst_color = nn.Linear(hidden_dim, 70)
        self.dst_sleeve_length = nn.Linear(hidden_dim, 7)
        self.dst_available_sizes = nn.Linear(hidden_dim, 6)
        self.dst_materials = nn.Linear(hidden_dim, 7)
        self.dst_customer_rating = nn.Linear(hidden_dim, 20)

    def forward(self, concat_vector):
        aggregated = self.aggregator(concat_vector)
        dst_act = self.dst_act(aggregated[:, 0, :])
        dst_request_slots = self.dst_request_slots(aggregated[:, 1, :])
        dst_type = self.dst_type(aggregated[:, 2, :])
        dst_price = self.dst_price(aggregated[:, 3, :])
        dst_customer_review = self.dst_customer_review(aggregated[:, 4, :])
        dst_brand = self.dst_brand(aggregated[:, 5, :])
        dst_size = self.dst_size(aggregated[:, 6, :])
        dst_pattern = self.dst_pattern(aggregated[:, 7, :])
        dst_color = self.dst_color(aggregated[:, 8, :])
        dst_sleeve_length = self.dst_sleeve_length(aggregated[:, 9, :])
        dst_available_sizes = self.dst_available_sizes(aggregated[:, 10, :])
        dst_materials = self.dst_materials(aggregated[:, 11, :])
        dst_customer_rating = self.dst_customer_rating(aggregated[:, 12, :])
        return dst_act, dst_request_slots, dst_type, dst_price, dst_customer_review, dst_brand, dst_size, \
            dst_pattern, dst_color, dst_sleeve_length, dst_available_sizes, dst_materials, dst_customer_rating


@dataclass
class Seq2SeqLMOutputForSIMMC(ModelOutput):
    loss: Optional[torch.FloatTensor] = None
    masked_lm_loss: Optional[torch.FloatTensor] = None
    nocoref_loss: Optional[torch.FloatTensor] = None
    misc_loss: Optional[torch.FloatTensor] = None
    disam_loss: Optional[torch.FloatTensor] = None
    retrieval_loss: Optional[torch.FloatTensor] = None
    dst_loss: Optional[torch.FloatTensor] = None
    logits: torch.FloatTensor = None
    disambiguation_logits: Optional[torch.FloatTensor] = None
    nocoref_logits: Optional[torch.FloatTensor] = None
    enc_head_results: Optional[List] = None
    past_key_values: Optional[Tuple[Tuple[torch.FloatTensor]]] = None
    decoder_hidden_states: Optional[Tuple[torch.FloatTensor]] = None
    decoder_attentions: Optional[Tuple[torch.FloatTensor]] = None
    cross_attentions: Optional[Tuple[torch.FloatTensor]] = None
    encoder_last_hidden_state: Optional[torch.FloatTensor] = None
    encoder_hidden_states: Optional[Tuple[torch.FloatTensor]] = None
    encoder_attentions: Optional[Tuple[torch.FloatTensor]] = None

@dataclass
class SequenceClassifierOutputSIMMC(ModelOutput):
    loss: Optional[torch.FloatTensor] = None
    nocoref_loss: Optional[torch.FloatTensor] = None
    misc_loss: Optional[torch.FloatTensor] = None
    disam_loss: Optional[torch.FloatTensor] = None
    retrieval_loss: Optional[torch.FloatTensor] = None
    dst_loss: Optional[torch.FloatTensor] = None
    disambiguation_logits: Optional[torch.FloatTensor] = None
    nocoref_logits: Optional[torch.FloatTensor] = None
    hidden_states: Optional[Tuple[torch.FloatTensor]] = None
    attentions: Optional[Tuple[torch.FloatTensor]] = None