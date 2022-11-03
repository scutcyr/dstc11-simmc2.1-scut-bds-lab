# coding=utf-8
# Copyright 2022 Research Center of Body Data Science from South China University of Technology. All rights reserved.

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

from .image import read_image
from PIL import Image
from torchvision import transforms

# 用于OFA模型的图像处理
mean, std = [0.5, 0.5, 0.5], [0.5, 0.5, 0.5]
resolution = 480 # 统一转换为480*480
patch_resize_transform = transforms.Compose([
        lambda image: image.convert("RGB"),
        transforms.Resize((resolution, resolution), interpolation=Image.BICUBIC),
        transforms.ToTensor(), 
        transforms.Normalize(mean=mean, std=std)
    ])

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
DISAM = "<DISAM>"


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


class LineByLineDatasetFromSingleFileForOFA(Dataset):
    def __init__(self, input_file, tokenizer: PreTrainedTokenizer, all_objects_meta, image_path_file=None, image_dir=None, evaluation=False):
        print(f"Data file : {input_file}")
        self.evaluation = evaluation

        # 从image_path_file文件当中读取image_name_list，用于后续读取图片
        self.do_extract_image_feature = False
        if image_path_file is not None:
            self.image_names = []
            with open(image_path_file, encoding="utf-8") as f:
                for line in f.read().splitlines():
                    if (len(line) > 0 and not line.isspace()):
                        self.image_names.append(line) # ['cloth_store_1_5_0.png', ... ]
            print("Done Load image_path_file File....")
            if image_dir is not None:
                self.image_dir = image_dir
                self.do_extract_image_feature = True

        self.contexts = []
        self.generation = []
        self.response = []
        self.corefs = []
        self.disambs = []
        self.disambiguation_labels = []
        self.nocorefs = []
        self.boxes = []
        self.miscs = []

        vocab2id = tokenizer.get_vocab()
        id2vocab = {v: k for k, v in vocab2id.items()}
        SOM_id = vocab2id[START_OF_MULTIMODAL_CONTEXTS]
        EOM_id = vocab2id[END_OF_MULTIMODAL_CONTEXTS]
        DISAM_id = get_input_id(tokenizer, DISAM)[0]
        SOO_id = get_input_id(tokenizer, START_OF_OBJ_TOKEN)[0]
        EOO_id = get_input_id(tokenizer, END_OF_OBJ_TOKEN)[0]

        with open(input_file, "r") as fr:
            for line in fr.readlines():
                line = line.strip()
                context, target, obj_boxes, disam_cands, coref_objs, is_disam = line.split("\t")
                context = "<DISAM> " + context
                response = re.sub("<SOB>.*<EOB> ", "", target)
                self.contexts.append(context)
                self.generation.append(target)
                self.response.append(response)
                self.disambiguation_labels.append(int(is_disam))
                self.corefs.append(json.loads(coref_objs))
                self.disambs.append(json.loads(disam_cands))
                self.boxes.append(json.loads(obj_boxes))

        encode_contexts = tokenizer(self.contexts, add_special_tokens=True)
        self.examples = encode_contexts.input_ids
        self.examples_attention_mask = encode_contexts.attention_mask

        nocoref_id = get_input_id(tokenizer, NO_COREF)[0]
        disam_id = get_input_id(tokenizer, DISAM)[0]
        self.nocoref = []  # [(position, label), (position, label), (position, label)], 
        self.disam = []  # [(position, label), (position, label), (position, label)], 
        self.misc = []  # [ [ {pos, coref_label, misc_labels(dict), is_fashion}, ... ], ...]
        id2index, id2fashion_st, id2furniture_st = id_converter(tokenizer)
        for idx, tokenized_line in enumerate(self.examples):
            tl = tokenized_line

            SOO_id_index = tl.index(SOO_id)
            EOO_id_index = tl.index(EOO_id)

            EOM_indices = [i for i, tokenized_id in enumerate(tl) if tokenized_id == EOM_id]
            if EOM_indices:
                EOM_last_idx = EOM_indices[-1]
            else:
                EOM_last_idx = -1

            self.nocoref.append((tl.index(nocoref_id), 1 if not self.corefs[idx] else 0))
            self.disam.append((tl.index(disam_id), 1 if self.disambs[idx] else 0))

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
                    # if token_id in id2index and i > EOM_last_idx:  # this token is for item index
                    if token_id in id2index and i > SOO_id_index and i < EOO_id_index:
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
                    # if token_id in id2index and i > EOM_last_idx:  # this token is for item index
                    if token_id in id2index and i > SOO_id_index and i < EOO_id_index:
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
        if self.do_extract_image_feature:
            # 读取图像特征
            image = read_image(self.image_dir, self.image_names[i])
            if image == False:
                # 图像不存在，则返回全0的Tensor
                image_feature = torch.zeros([3, resolution, resolution], dtype=torch.float32)
            else:
                image_feature = patch_resize_transform(image)

            if not self.evaluation:
                return torch.tensor(self.examples[i], dtype=torch.long), torch.tensor(self.examples_attention_mask[i], dtype=torch.long), \
                    self.generation[i], self.boxes[i], self.misc[i], self.nocoref[i], torch.tensor(self.disambiguation_labels[i], dtype=torch.long), \
                    self.response[i], self.disam[i], image_feature
            else:
                return torch.tensor(self.examples[i], dtype=torch.long), torch.tensor(self.examples_attention_mask[i], dtype=torch.long), \
                    self.generation[i], self.boxes[i], self.misc[i], self.nocoref[i], self.disam[i], image_feature
        if not self.evaluation:
            return torch.tensor(self.examples[i], dtype=torch.long), torch.tensor(self.examples_attention_mask[i], dtype=torch.long), \
                    self.generation[i], self.boxes[i], self.misc[i], self.nocoref[i], torch.tensor(self.disambiguation_labels[i], dtype=torch.long), \
                    self.response[i], self.disam[i]
        else:
            return torch.tensor(self.examples[i], dtype=torch.long), torch.tensor(self.examples_attention_mask[i], dtype=torch.long), \
                    self.generation[i], self.boxes[i], self.misc[i], self.nocoref[i], self.disam[i]

