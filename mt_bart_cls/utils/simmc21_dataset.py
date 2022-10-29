# coding=utf-8
import re
import json
import logging
import torch
import pdb
from torch.utils.data import Dataset
from transformers import PreTrainedTokenizer, BlenderbotTokenizer

from .metadata import (FASHION_SIZES, FASHION_AVAILABLE_SIZES, FASHION_BRAND, FASHION_COLOR, 
FASHION_PATTERN, FASHION_SLEEVE_LENGTH, FASHION_ASSET_TYPE, FASHION_TYPE, 
FASHION_PRICE, FASHION_CUSTOMER_REVIEW, FURNITURE_BRAND, FURNITURE_COLOR, 
FURNITURE_MATERIALS, FURNITURE_TYPE, FURNITURE_PRICE, FURNITURE_CUSTOMER_RATING)
from .dst_label import INIT_DST_DICT, DST_DICT
import copy

logger = logging.getLogger(__name__)

fashion_meta_attrs = {
    'size': FASHION_SIZES,
    'available_sizes': FASHION_AVAILABLE_SIZES,
    'brand': FASHION_BRAND,
    'color': FASHION_COLOR,
    'pattern': FASHION_PATTERN,
    'sleeve_length': FASHION_SLEEVE_LENGTH,
    'asset_type': FASHION_ASSET_TYPE,
    'type': FASHION_TYPE,
    'price': FASHION_PRICE,
    'customer_review': FASHION_CUSTOMER_REVIEW,
    }
furniture_meta_attrs = {
    'brand': FURNITURE_BRAND,
    'color': FURNITURE_COLOR,
    'materials': FURNITURE_MATERIALS,
    'type': FURNITURE_TYPE,
    'price': FURNITURE_PRICE,
    'customer_review': FURNITURE_CUSTOMER_RATING  # key is "review"!!
}
available_sizes2st = {
    'XS': '<A>',
    'S': '<B>',
    'M': '<C>',
    'L': '<D>',
    'XL': '<E>',
    'XXL': '<F>' 
}


NUM_FASHION_ITEMS = 288
NUM_FURNITURE_ITEMS = 57
FASHION_SPECIAL_TOKENS = [f"<@1{i:03}>" for i in range(NUM_FASHION_ITEMS)]
FURNITURE_SPECIAL_TOKENS = [f"<@2{i:03}>" for i in range(NUM_FURNITURE_ITEMS)]

MAX_NUM_OBJ_IN_SCENE = 200
OBJECT_INDICES = [f"<{i}>" for i in range(MAX_NUM_OBJ_IN_SCENE)]

START_OF_MULTIMODAL_CONTEXTS = "<SOM>"
END_OF_MULTIMODAL_CONTEXTS = "<EOM>"
START_OF_OBJ_TOKEN = "<SOO>"
END_OF_OBJ_TOKEN = "<EOO>"
NO_COREF = "<NOCOREF>"
BELIEF_STATE = "<DST>"


def get_input_id(tokenizer, tokens):
    '''返回给定字符的id
       特别地，BlenderbotTokenizerFast与BartTokenizerFast并不相同
       BlenderbotTokenizerFast返回的如下所示：{'input_ids': [8638, 2], 'attention_mask': [1, 1]}
       BartTokenizerFast返回的如下所示：{'input_ids': [0, 50893, 2], 'attention_mask': [1, 1, 1]}
    '''
    token_ids = tokenizer(tokens)
    if isinstance(tokenizer, BlenderbotTokenizer) or isinstance(tokenizer, BlenderbotTokenizer):
        #return tokenizer(tokens).input_ids[-2:-1]
        return tokenizer(tokens).input_ids[0:-1]
    elif token_ids.input_ids[0] == tokenizer.bos_token_id:
        # 第一个为表示bos的id
        return token_ids.input_ids[1:-1]
    else:
        return tokenizer(tokens).input_ids[0:-1]


def id_converter(tokenizer):
    id2index = {get_input_id(tokenizer, index)[0]: index for index in OBJECT_INDICES}
    id2fashion_st = {get_input_id(tokenizer, st)[0]: st for st in FASHION_SPECIAL_TOKENS}
    id2furniture_st = {get_input_id(tokenizer, st)[0]: st for st in FURNITURE_SPECIAL_TOKENS}
    return id2index, id2fashion_st, id2furniture_st


class LineByLineDataset(Dataset):
    def __init__(self, input_file, tokenizer: PreTrainedTokenizer, all_objects_meta):
        print(f"Data file : {input_file}")

        self.contexts = []
        self.generation = []
        self.response = []
        self.corefs = []
        self.disambs = []
        self.disambiguation_labels = []
        self.nocorefs = []
        self.boxes = []
        self.dst_dicts = []

        vocab2id = tokenizer.get_vocab()
        id2vocab = {v: k for k, v in vocab2id.items()}
        SOM_id = vocab2id[START_OF_MULTIMODAL_CONTEXTS]
        EOM_id = vocab2id[END_OF_MULTIMODAL_CONTEXTS]

        # pdb.set_trace()
        with open(input_file, "r") as fr:
            for line in fr.readlines():
                line = line.strip()
                context, dst_str, target, obj_boxes, disam_cands, coref_objs, is_disam = line.split("\t")
                context = "<DISAM> " + context
                response = re.sub("<SOB>.*<EOB> ", "", target)
                self.contexts.append(context)
                self.generation.append(target)
                self.response.append(response)
                self.disambiguation_labels.append(int(is_disam))
                self.corefs.append(json.loads(coref_objs))
                self.disambs.append(json.loads(disam_cands))
                self.boxes.append(json.loads(obj_boxes))
                self.dst_dicts.append(json.loads(dst_str))

        ## 输入转成 id
        encode_contexts = tokenizer(self.contexts, add_special_tokens=True)
        self.examples = encode_contexts.input_ids
        self.examples_attention_mask = encode_contexts.attention_mask

        ## 获取判别 <NOCOREF> id
        nocoref_id = get_input_id(tokenizer, NO_COREF)[0]
        self.nocoref = []  # [(position, label), (position, label), (position, label)], 
        self.misc = []  # [ [ {pos, coref_label, misc_labels(dict), is_fashion}, ... ], ...]
        self.dsts = []
        id2index, id2fashion_st, id2furniture_st = id_converter(tokenizer)
        for idx, tokenized_line in enumerate(self.examples):
            tl = tokenized_line

            EOM_indices = [i for i, tokenized_id in enumerate(tl) if tokenized_id == EOM_id]
            if EOM_indices:
                EOM_last_idx = EOM_indices[-1]
            else:
                EOM_last_idx = -1

            self.nocoref.append((tl.index(nocoref_id), 1 if not self.corefs[idx] else 0))

            ## 处理状态跟踪
            dst_id = get_input_id(tokenizer, BELIEF_STATE)[0]
            dst_dict = self.dst_dicts[idx]
            dst_label = copy.deepcopy(INIT_DST_DICT)
            # pdb.set_trace()
            if len(dst_dict) != 0:
                act = dst_dict["act"]
                dst_label["act"] = DST_DICT["act"].index(act)
                for rs in dst_dict["act_attributes"]["request_slots"]:
                    dst_label["request_slots"][DST_DICT["request_slots"].index(rs)] = 1
                for key, value in dst_dict["act_attributes"]["slot_values"].items():
                    if key == "availableSizes":
                        for v in value:
                            dst_label["availableSizes"][DST_DICT["availableSizes"].index(v)] = 1
                    else:
                        ## TODO 后面修改更全面的属性取值范围
                        if value in DST_DICT[key]:
                            dst_label[key] = DST_DICT[key].index(value)
                        else:
                            dst_label[key] = DST_DICT[key].index("None")
            for dst_pos, token_id in enumerate(tl):
                if token_id == dst_id:
                    break
            # pdb.set_trace()
            self.dsts.append({"pos": dst_pos, "dst_label": dst_label})


            ## 处理输入场景中每个对象的属性，以及是否是歧义候选对象和指代消解对象
            is_fashion = True
            for token_id in tl:
                if token_id in id2fashion_st:
                    break
                if token_id in id2furniture_st:
                    is_fashion = False
                    break

            line_labels = []
            if is_fashion:
                for i, token_id in enumerate(tl):
                    if token_id in id2index and i > EOM_last_idx:  # this token is for item index
                        temp = dict()
                        pos = i; item_index = id2index[token_id]; fashion_st = id2fashion_st[tl[i+1]]
                        temp['is_fashion'] = True
                        temp['pos'] = pos
                        temp['disamb_label'] = 1 if item_index in self.disambs[idx] else 0 # 歧义候选识别
                        temp['coref_label'] = 1 if item_index in self.corefs[idx] else 0
                        temp['misc_labels'] = dict()
                        for attr_name, attr_value in all_objects_meta[fashion_st].items():
                            if attr_name != 'available_sizes':
                                temp['misc_labels'][attr_name] = fashion_meta_attrs[attr_name].index(attr_value)
                            else:
                                temp['misc_labels'][attr_name] = [1 if x in attr_value else 0 for x in fashion_meta_attrs[attr_name]]
                        line_labels.append(temp)
            else:
                for i, token_id in enumerate(tl):
                    if token_id in id2index and i > EOM_last_idx:  # this token is for item index
                        temp = dict()
                        pos = i; item_index = id2index[token_id]; furniture_st = id2furniture_st[tl[i+1]]
                        temp['is_fashion'] = False
                        temp['pos'] = pos
                        temp['disamb_label'] = 1 if item_index in self.disambs[idx] else 0 # 歧义候选识别
                        temp['coref_label'] = 1 if item_index in self.corefs[idx] else 0
                        temp['misc_labels'] = dict()
                        for attr_name, attr_value in all_objects_meta[furniture_st].items():
                            temp['misc_labels'][attr_name] = furniture_meta_attrs[attr_name].index(attr_value)
                        line_labels.append(temp)

            if len(line_labels) != len(self.boxes[idx]):
                pdb.set_trace()
            self.misc.append(line_labels)
        print("Done Load Main File....")
    
    def __len__(self):
        return len(self.examples)

    def __getitem__(self, i):
        return torch.tensor(self.examples[i], dtype=torch.long), torch.tensor(self.examples_attention_mask[i], dtype=torch.long), \
                self.generation[i], self.boxes[i], self.misc[i], self.nocoref[i], torch.tensor(self.disambiguation_labels[i], dtype=torch.long), \
                self.response[i], self.dsts[i]