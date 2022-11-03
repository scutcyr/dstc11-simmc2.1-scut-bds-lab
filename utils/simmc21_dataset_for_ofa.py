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


""" Dataset Class for DSTC-11 SIMMC 2.1

Updated by Yirong Chen 
Used for [SIMMC 2.1](https://github.com/facebookresearch/simmc2)
Mail: [eeyirongchen@mail.scut.edu.cn](mailto:eeyirongchen@mail.scut.edu.cn)
Date: 2022/10/10

# 关键包版本说明：
# python==3.8
# pytorch==1.12.0
# pytorch-ignite==0.4.8
# transformers==4.20.0

# SIMMC2.1 数据集说明
# 数据集存储形式如下：
# data/
# 场景图片文件夹
#      simmc2_scene_images_dstc10_public/
#      simmc2_scene_images_dstc10_teststd/
# 场景图片的json标注信息文件夹
#      simmc2_scene_jsons_dstc10_public/
#      simmc2_scene_jsons_dstc10_teststd/
# 物品的metadata信息
#      fashion_prefab_metadata_all.json
#      furniture_prefab_metadata_all.json
# 对话及其标注
#      simmc2.1_dials_dstc11_dev.json
#      simmc2.1_dials_dstc11_devtest.json
#      simmc2.1_dials_dstc11_mini.json
#      simmc2.1_dials_dstc11_train.json

Usage: 


"""

import re
import os
import ast
import copy
import json
import random
import logging
import torch

from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader, Dataset, RandomSampler, SequentialSampler
from transformers import (PreTrainedTokenizer, AutoTokenizer, BlenderbotTokenizer, 
    BlenderbotTokenizerFast, BartTokenizerFast
)

# 本文件夹package
#import api#, util
from .metadata import (FASHION_SIZES, FASHION_AVAILABLE_SIZES, FASHION_BRAND, FASHION_COLOR, 
FASHION_PATTERN, FASHION_SLEEVE_LENGTH, FASHION_ASSET_TYPE, FASHION_TYPE, 
FASHION_PRICE, FASHION_CUSTOMER_REVIEW, FURNITURE_BRAND, FURNITURE_COLOR, 
FURNITURE_MATERIALS, FURNITURE_TYPE, FURNITURE_PRICE, FURNITURE_CUSTOMER_RATING)

from .image import read_image
from PIL import Image
from torchvision import transforms
# 
#from ..models.simmc21_ul2 import UL2Tokenizer

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

'''
train_api = api.PromptAPI(dial_split="train", 
                          data_dir="~/dstc11_simmc2.1_scut-bds-lab/data", 
                          dialogue_name_prefix="simmc2.1_dials_dstc11_",
                          jsons_dir_name="simmc2_scene_jsons_dstc10_public",
                          images_dir_name="simmc2_scene_images_dstc10_public")
'''

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
    elif token_ids.input_ids[0] == tokenizer.bos_token_id or token_ids.input_ids[0] == tokenizer.sep_token_id or token_ids.input_ids[0] == tokenizer.cls_token_id:
        # 第一个为表示bos的id
        return token_ids.input_ids[1:-1]
    else:
        return tokenizer(tokens).input_ids[0:-1]

    return token_ids.input_ids[1:-1]

def id_converter(tokenizer):
    id2index = {get_input_id(tokenizer, index)[0]: index for index in OBJECT_INDICES}
    id2fashion_st = {get_input_id(tokenizer, st)[0]: st for st in FASHION_SPECIAL_TOKENS}
    id2furniture_st = {get_input_id(tokenizer, st)[0]: st for st in FURNITURE_SPECIAL_TOKENS}
    return id2index, id2fashion_st, id2furniture_st

class LineByLineDatasetForOFA(Dataset):
    def __init__(self, 
                 input_file, 
                 target_file, 
                 disambiguation_file, 
                 response_file, 
                 tokenizer: PreTrainedTokenizer, 
                 all_objects_meta, 
                 evaluation=False, 
                 model_type=None, 
                 use_OBJ=True,
                 image_path_file=None,
                 image_dir=None
                 ):
        '''
        image_path_file: a line-by-line .txt file, a line is name of the scene image
        image_dir: the dir of the scene image
        image_path = os.path.join(image_dir,image_name)

        '''

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

        if not evaluation:
            # Disambiguation File
            self.disambiguation_labels = []
            with open(disambiguation_file, encoding="utf-8") as f:
                for line in f.read().splitlines():
                    self.disambiguation_labels.append(int(line))
            print("Done Load Disambiguation File....")
            # Response File
            response_list = []
            response =  open(response_file, encoding="utf-8")
            for line in response.read().splitlines():
                if (len(line) > 0 and not line.isspace()):
                    #if isinstance(ult, UL2Tokenizer):
                    if model_type=="mt-ul2":
                        response_list.append("<pad>"+line) # <pad作为开头>
                    else:
                        response_list.append(line)
            self.response = response_list
            print("Done Load Response File....")
        
        # Other tasks
        lines = []
        self.boxes = []
        vocab2id = tokenizer.get_vocab()
        id2vocab = {v: k for k, v in vocab2id.items()}
        SOM_id = vocab2id[START_OF_MULTIMODAL_CONTEXTS]
        EOM_id = vocab2id[END_OF_MULTIMODAL_CONTEXTS]
        DISAM_id = get_input_id(tokenizer, DISAM)[0]
        SOO_id = get_input_id(tokenizer, START_OF_OBJ_TOKEN)[0]
        EOO_id = get_input_id(tokenizer, END_OF_OBJ_TOKEN)[0]

        # extract input sequence to BART, and bbox info to be embedded
        with open(input_file, encoding="utf-8") as f:
            for line in f.read().splitlines():
                if (len(line) > 0 and not line.isspace()):
                    # [[0.2, 0.3, 0.1, 0.2, 0.43, 3.4], [0.2, 0.1, 0.1, 0.2, 0.43, 3.7], ...]
                    line_boxes = [ast.literal_eval(position.replace('(', '').replace(')', '')) for position in re.findall(r"\[\([^\)]+\)\]", line)]
                    self.boxes.append(line_boxes)
                    line = re.sub(r"\[\([^\)]*\)\]", "", line)

                    if not use_OBJ:
                        # 去掉<OBJ>标志
                        line = re.sub(r"<OBJ>", "", line)
                    
                    #if isinstance(ult, UL2Tokenizer):
                    if model_type=="mt-ul2":
                        # 参考：https://huggingface.co/google/ul2
                        # 
                        lines.append("[S2S] <DISAM> "+line)
                        # ['[', 'S', '2', 'S', ']', '<DISAM>', ..., '</s>']

                    else:
                        lines.append("<DISAM> "+line)

        encode_text = tokenizer(lines, add_special_tokens=True)
        self.examples = encode_text.input_ids
        self.examples_attention_mask = encode_text.attention_mask
        # extract generation target
        targets = []
        with open(target_file, encoding="utf-8") as f:
            target_lines = [
                line
                for line in f.read().splitlines()
                if (len(line) > 0 and not line.isspace())
            ]
        
        assert len(target_lines) == len(self.examples)

        
        corefs = []  # [ [corefobj1(index), corefobj2], [corefobj1], [...], ...]
        disambs = [] #
        for line in target_lines:
            dst_start = line.index('Belief State : ')
            dst_end = line.index('<EOB>')
            dst = line[dst_start:dst_end]
            if (r"|") in dst: # SIMMC2.1
                # dst="Belief State : REQUEST:COMPARE [  ] () < <59>, <82> > | <1>, <2> |"
                coref_of_dst = re.search(r"< .*? >",dst).group() # < <59>, <82> >
                coref_referred = [obj_index for obj_index in re.findall(r"<[^<^>^ ]+>", coref_of_dst)] # ['<59>', '<82>']
                corefs.append(coref_referred)

                disamb_of_dst = re.search(r"\| .*? \|",dst).group() # | <1>, <2> |
                disamb_referred = [obj_index for obj_index in re.findall(r"<[^<^>^ ]+>", disamb_of_dst)] # ['<1>', '<2>']
                disambs.append(disamb_referred)



            else: # SIMMC2.0
                # dst="Belief State : REQUEST:COMPARE [  ] () < <59>, <82> >"
                coref_referred = [obj_index for obj_index in re.findall(r"<[^<^>^ ]+>", dst)]
                corefs.append(coref_referred)
                disamb_referred = []
                disambs.append(disamb_referred)


            # if 'availableSizes =' in dst:                
            #     available_sizes = [ast.literal_eval(availSize.split('=')[1].strip()) for availSize in re.findall(r"availableSizes = \[.*\]", dst)][0]
            #     available_sizes = [available_sizes2st[size] for size in available_sizes]
            line_split = line.split('Belief State : ')
            after_belief_state = line_split[1]
            after_belief_state = re.sub(r"<((<[0-9]+>)|,| )*>", "", after_belief_state)
            after_belief_state = re.sub(r"\|((<[0-9]+>)|,| )*\|", "", after_belief_state) # REQUEST:COMPARE [  ] ()  |  | <EOB> Yes, the brown one costs $199.99 and the black one costs $44.99. <EOS>
            # if 'availableSizes =' in after_belief_state:
            #     after_belief_state = re.sub(r"availableSizes = \[.*\]", str(available_sizes), after_belief_state)
            # targets.append('=====' + after_belief_state)
            targets.append(after_belief_state)
        self.generation = targets

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

            EOM_indices = [i for i, tokenized_id in enumerate(tl) if tokenized_id ==EOM_id]
            if EOM_indices:
                EOM_last_idx = EOM_indices[-1]
            else:
                EOM_last_idx = -1

            self.nocoref.append((tl.index(nocoref_id), 1 if not corefs[idx] else 0))
            self.disam.append((tl.index(disam_id), 1 if disambs[idx] else 0))

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
                    if token_id in id2index and i > SOO_id_index and i < EOO_id_index:  # this token is for item index
                        temp = dict()
                        pos = i; item_index = id2index[token_id]; fashion_st = id2fashion_st[tl[i+1]]
                        temp['is_fashion'] = True
                        temp['pos'] = pos
                        temp['disamb_label'] = 1 if item_index in disambs[idx] else 0 # 歧义候选识别
                        temp['coref_label'] = 1 if item_index in corefs[idx] else 0
                        temp['misc_labels'] = dict()
                        for attr_name, attr_value in all_objects_meta[fashion_st].items():
                            if attr_name != 'available_sizes':
                                temp['misc_labels'][attr_name] = fashion_meta_attrs[attr_name].index(attr_value)
                            else:
                                temp['misc_labels'][attr_name] = [1 if x in attr_value else 0
                                                                  for x in fashion_meta_attrs[attr_name]]
                        line_labels.append(temp)
            else:
                for i, token_id in enumerate(tl):
                    if token_id in id2index and i > SOO_id_index and i < EOO_id_index:  # this token is for item index
                        temp = dict()
                        pos = i; item_index = id2index[token_id]; furniture_st = id2furniture_st[tl[i+1]]
                        temp['is_fashion'] = False
                        temp['pos'] = pos
                        temp['disamb_label'] = 1 if item_index in disambs[idx] else 0 # 歧义候选识别
                        temp['coref_label'] = 1 if item_index in corefs[idx] else 0
                        temp['misc_labels'] = dict()
                        for attr_name, attr_value in all_objects_meta[furniture_st].items():
                            temp['misc_labels'][attr_name] = furniture_meta_attrs[attr_name].index(attr_value)
                        line_labels.append(temp)
            self.misc.append(line_labels)
        print("Done Load Main File....")
        if not evaluation:        
            assert len(self.examples) == len(self.disambiguation_labels) == len(self.response)
    
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

            return torch.tensor(self.examples[i], dtype=torch.long), torch.tensor(self.examples_attention_mask[i], dtype=torch.long), \
                self.generation[i], self.boxes[i], self.misc[i], self.nocoref[i], self.disam[i], image_feature

        else:
            if not self.evaluation:
                return torch.tensor(self.examples[i], dtype=torch.long), torch.tensor(self.examples_attention_mask[i], dtype=torch.long), \
                        self.generation[i], self.boxes[i], self.misc[i], self.nocoref[i], torch.tensor(self.disambiguation_labels[i], dtype=torch.long), \
                        self.response[i], self.disam[i]

            return torch.tensor(self.examples[i], dtype=torch.long), torch.tensor(self.examples_attention_mask[i], dtype=torch.long), \
                self.generation[i], self.boxes[i], self.misc[i], self.nocoref[i], self.disam[i]



def get_dataset_for_ofa(args, tokenizer, all_objects_meta, train=True):
    if train:
        dataset = LineByLineDatasetForOFA(input_file=args.train_input_file, 
                                          target_file=args.train_target_file, 
                                          disambiguation_file=args.disambiguation_file, 
                                          response_file=args.response_file, 
                                          tokenizer=tokenizer, 
                                          all_objects_meta=all_objects_meta, 
                                          evaluation=False,
                                          model_type=args.model_type, 
                                          use_OBJ=not args.not_use_OBJ,
                                          image_path_file=args.train_image_path_file,
                                          image_dir=args.train_image_dir)
    else:
        dataset = LineByLineDatasetForOFA(input_file=args.eval_input_file, 
                                          target_file=args.eval_target_file, 
                                          disambiguation_file=None, 
                                          response_file=None, 
                                          tokenizer=tokenizer, 
                                          all_objects_meta=all_objects_meta, 
                                          evaluation=True, 
                                          model_type=args.model_type, 
                                          use_OBJ=not args.not_use_OBJ,
                                          image_path_file=args.eval_image_path_file,
                                          image_dir=args.eval_image_dir)
    # Unknown issues have been reported around not being able to handle incomplete batches (e.g. w/ older CUDA 9.2)
    # Below is a workaround in case you encounter this issue.
    # Alternatively, --nocuda could avoid this issue too.
    # Comment out the following if you do not encounuter this issue or if you are not using any GPU.
    n = len(dataset) % args.train_batch_size if train else len(dataset) % args.eval_batch_size
    if n != 0:
        print(f"Truncating from {len(dataset.examples)} examples to {len(dataset.examples[:-n])}")
        dataset.examples = dataset.examples[:-n]
        dataset.generation = dataset.generation[:-n]
        dataset.boxes = dataset.boxes[:-n]
        dataset.misc = dataset.misc[:-n]
        dataset.nocoref = dataset.nocoref[:-n]
        if dataset.do_extract_image_feature: 
            dataset.image_names = dataset.image_names[:-n]
        if train:
            dataset.disambiguation_labels = dataset.disambiguation_labels[:-n]
            dataset.response = dataset.response[:-n]
    return dataset



class GenerationDatasetForOFA(Dataset):
    '''用于DSTC11的4个任务的验证、测试阶段
       特别地，在本项目当中主要用于eval_model_all_task.py任务

    '''
    def __init__(self, 
                 prompts_from_file, 
                 tokenizer, 
                 model_type=None, 
                 use_OBJ=True,
                 image_path_file=None,
                 image_dir=None):
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

        lines = []
        self.original_lines = []
        self.boxes = []

        vocab2id = tokenizer.get_vocab()
        id2vocab = {v: k for k, v in vocab2id.items()}
        SOM_id = vocab2id[START_OF_MULTIMODAL_CONTEXTS] # "<SOM>"的ID
        EOM_id = vocab2id[END_OF_MULTIMODAL_CONTEXTS] # "<EOM>"的ID
        DISAM_id = get_input_id(tokenizer, DISAM)[0]
        SOO_id = get_input_id(tokenizer, START_OF_OBJ_TOKEN)[0]
        EOO_id = get_input_id(tokenizer, END_OF_OBJ_TOKEN)[0]

        # extract input sequence to BART, and bbox info to be embedded
        with open(prompts_from_file, encoding="utf-8") as f:
            for line in f.read().splitlines():
                if (len(line) > 0 and not line.isspace()):
                    # [[0.2, 0.3, 0.1, 0.2, 0.4, 0.8], [0.2, 0.1, 0.1, 0.2, 0.3, 0.1], ...]
                    # --re.findall(r"\[\([^\)]+\)\]", line) ----> "[(-0.1038,-0.4987,-0.0311,-0.0557,0.0322,0.8933)]" 
                    # --position.replace('(', '').replace(')', '')--> "[-0.1038,-0.4987,-0.0311,-0.0557,0.0322,0.8933]"
                    # --ast.literal_eval()--> [-0.1038,-0.4987,-0.0311,-0.0557,0.0322,0.8933]
                    # --[]-->[[-0.1038,-0.4987,-0.0311,-0.0557,0.0322,0.8933]]
                    line_boxes = [ast.literal_eval(position.replace('(', '').replace(')', '')) for position in re.findall(r"\[\([^\)]+\)\]", line)]
                    self.boxes.append(line_boxes)

                    startPos = line.find("<EOO>")
                    endPos = line.find("=> Belief State :")
                    # ...<EOO> INFORM:GET [ type = shirt, customerReview = good ] (  )  => Belief State :  
                    # 转换为：
                    # ...<EOO> => Belief State :  
                    original_line = line[:startPos+5]+" "+line[endPos:]
                    original_line = re.sub(r"\[\([^\)]*\)\]", "", original_line) # 将原始line中的表示位置的如"[(-0.1038,-0.4987,-0.0311,-0.0557,0.0322,0.8933)]"部分去掉

                    line = re.sub(r"\[\([^\)]*\)\]", "", line) # 将原始line中的表示位置的如"[(-0.1038,-0.4987,-0.0311,-0.0557,0.0322,0.8933)]"部分去掉

                    if not use_OBJ:
                        # 去掉<OBJ>标志
                        line = re.sub(r"<OBJ>", "", line)


                    #original_line = copy.deepcopy(line)
                    # 将<SOO><NOCOREF><OBJ><55>[(-0.4427,-0.1846,-0.3335,0.1398,0.0354,0.3617)]<@1240><OBJ><56>[(-0.0461,-0.1687,0.0371,0.1139,0.0235,0.2396)]<@1269><OBJ><85>[(0.0785,-0.1657,0.1029,0.1189,0.0069,0.3242)]<@1007><OBJ><57>[(0.1220,-0.1547,0.1654,0.1070,0.0114,0.2804)]<@1214><OBJ><59>[(0.2137,-0.1547,0.2561,0.1308,0.0121,1.0000)]<@1237><OBJ><58>[(0.2185,-0.1687,0.2752,0.1169,0.0162,0.3440)]<@1228><OBJ><86>[(0.2694,-0.1736,0.3123,0.1189,0.0126,0.3768)]<@1006><OBJ><62>[(0.0875,0.1866,0.1543,0.4980,0.0208,0.3827)]<@1241><OBJ><63>[(0.2110,0.1657,0.2784,0.4970,0.0223,0.3781)]<@1015><OBJ><87>[(0.3293,0.1507,0.3765,0.4502,0.0141,0.5338)]<@1184><OBJ><65>[(-0.4300,0.2383,-0.3303,0.4980,0.0259,0.4574)]<@1241><OBJ><61>[(-0.0376,0.1995,0.0292,0.4980,0.0199,0.3585)]<@1013><EOO>
                    # 部分去掉，得到类似：
                    # "User : Do you have any size XS jeans?  => Belief State : "的形式
                    original_line = re.sub(r" <SOO.*EOO>", "", original_line)
                    #lines.append(line)
                    #if isinstance(ult, UL2Tokenizer):
                    if model_type=="mt-ul2":
                        # 参考：https://huggingface.co/google/ul2
                        # 
                        lines.append("[S2S] <DISAM> "+line)
                    else:
                        lines.append("<DISAM>" +line)
                    self.original_lines.append(original_line)
        encode_text = tokenizer(lines, add_special_tokens=True)
        self.examples = encode_text.input_ids
        self.examples_attention_mask = encode_text.attention_mask
        
        nocoref_id = get_input_id(tokenizer, NO_COREF)[0]
        disam_id = get_input_id(tokenizer, DISAM)[0]
        self.nocoref = []  # [position, position, position, ...]
        self.disam = []  # [position, position, position, ...]
        self.misc = []  # [ [ {pos, is_fashion}, ... ], ...]
        id2index, id2fashion_st, id2furniture_st = id_converter(tokenizer)
        for idx, tokenized_line in enumerate(self.examples):
            tl = tokenized_line

            SOO_id_index = tl.index(SOO_id)
            EOO_id_index = tl.index(EOO_id)

            EOM_indices = [i for i, tokenized_id in enumerate(tl) if tokenized_id ==EOM_id]
            if EOM_indices:
                EOM_last_idx = EOM_indices[-1]
            else:
                EOM_last_idx = -1

            self.nocoref.append(tl.index(nocoref_id))
            self.disam.append(tl.index(disam_id))

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
                    if token_id in id2index and i > SOO_id_index and i < EOO_id_index:  # this token is for item index
                        temp = dict()
                        pos = i
                        temp['is_fashion'] = True
                        temp['pos'] = pos
                        
                        line_labels.append(temp)
            else:
                for i, token_id in enumerate(tl):
                    if token_id in id2index and i > SOO_id_index and i < EOO_id_index:  # this token is for item index
                        temp = dict()
                        pos = i
                        temp['is_fashion'] = False
                        temp['pos'] = pos
                        line_labels.append(temp)
            self.misc.append(line_labels)

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

            return torch.tensor(self.examples[i], dtype=torch.long), torch.tensor(self.examples_attention_mask[i], dtype=torch.long), \
                self.original_lines[i], self.boxes[i], self.misc[i], self.nocoref[i], self.disam[i], image_feature

        return torch.tensor(self.examples[i], dtype=torch.long), torch.tensor(self.examples_attention_mask[i], dtype=torch.long), \
            self.original_lines[i], self.boxes[i], self.misc[i], self.nocoref[i], self.disam[i]