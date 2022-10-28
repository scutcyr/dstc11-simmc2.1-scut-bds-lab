# coding=utf-8
# Copyright 2022 iFLYTEK, The State Key Laboratory of Cognitive Intelligence. All rights reserved.

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


# File: eval_model.py
# Description: The evaluating model code for DSTC-11 SIMMC 2.1
# Repository: https://github.com/scutcyr/dstc11-simmc2.1-iflytek
# Mail: [yrchen5@iflytek.com](mailto:yrchen5@iflytek.com) or [eeyirongchen@mail.scut.edu.cn](mailto:eeyirongchen@mail.scut.edu.cn)
# Date: 2022/10/18
# Usage:


"""
## 使用示例
WORK_DIR=/yrfs1/intern/yrchen5/dstc11_simmc2.1_iflytek/dstc11-simmc2.1-iflytek
INIT_DATA_DIR=/yrfs1/intern/yrchen5/dstc11_simmc2.1_iflytek/data
PREPROCESS_DATA_DIR=/yrfs1/intern/yrchen5/dstc11_simmc2.1_iflytek/dstc11-simmc2.1-iflytek/data_convert
MODEL_SAVE_DIR=/yrfs1/intern/yrchen5/dstc11_simmc2.1_iflytek/dstc11-simmc2.1-iflytek/runs
CONTEXT_LENGTH=4 # 2,4,6,8
# cd working path
cd $WORK_DIR

python eval_model.py \
   --model_type=mt-bart-large-disamb \
   --model_dir=$MODEL_SAVE_DIR/simmc21_bart_ctxlen4_20221017_1040 \
   --path_output=$MODEL_SAVE_DIR/simmc21_bart_ctxlen4_20221017_1040/devtest_predict_results \
   --output_path_csv_report=$MODEL_SAVE_DIR/simmc21_bart_ctxlen4_20221017_1040/devtest_predict_results/report_summary_simmc21_bart_ctxlen4_20221017_1040.csv \
   --prompts_from_file=$PREPROCESS_DATA_DIR/simmc2.1_dials_dstc11_dev_predict_with_sys_state_ctxlen${CONTEXT_LENGTH}.txt \
   --item2id=$INIT_DATA_DIR/item2id.json \
   --data_json_path=$INIT_DATA_DIR/simmc2.1_dials_dstc11_devtest.json \
   --input_path_target=$PREPROCESS_DATA_DIR/simmc2.1_dials_dstc11_devtest_target_ctxlen${CONTEXT_LENGTH}.txt \
   --dialog_meta_data=$PREPROCESS_DATA_DIR/simmc2.1_dials_dstc11_devtest_inference_disambiguation.json \
   --batch_size=1 \
   --length=150 \
   --do_sample \
   --num_beams=4 \
   --temperature=0.9 \
   --repetition_penalty=1.0 \
   --k=0 \
   --p=0.9 \
   --num_return_sequences=1 \
   --eval_all_checkpoints \
   --single_round_evaluation \
   --do_calculate_score


## BART预训练模型版本差异分析
    原作者提供的模型训练版本： "transformers_version": "4.11.1"
    我们复现时使用的版本： "transformers_version": "4.20.0"
    调用模型时出错，分析发现是两个版本的模型的```config.json```出现了差别。
    我们下载的BART模型的```config.json```多了一个设置：
    ```bash
    "forced_bos_token_id": 0,
    ```
    这会导致模型有可能不会生成<EOB>字符就结束了？
    所以我们把该部分删除就可以了！

"""
import os
import re
import ast
import copy
import json
import nltk
import glob
import logging
#import ipdb 

import torch
import numpy as np
import pandas as pd
from tqdm import tqdm
from torch.utils.data import DataLoader, Dataset, SequentialSampler
from torch.nn.utils.rnn import pad_sequence
from transformers import WEIGHTS_NAME, AutoTokenizer, BartForConditionalGeneration, BartTokenizerFast, BlenderbotTokenizer, BlenderbotTokenizerFast, T5Tokenizer


from evaluation_tools.convert import parse_flattened_results_from_file
from evaluation_tools.evaluate_dst import evaluate_from_flat_list, evaluate_from_flat_list_mentioned_object
from evaluation_tools.response_evaluation import evaluate_response_generation

# 导入模型类
from models.simmc21_bart import MultiTaskBartForConditionalGeneration
from models.simmc21_bart import MultiTaskBartForConditionalGenerationWithDisamb
from models.simmc21_bart import MultiTaskBartForConditionalGenerationJointDisambCoref
from models.simmc21_bart import MultiTaskBartForConditionalGenerationWithDisambAndIntent
from models.simmc21_bart import MultiTaskBartForConditionalGenerationWithDisambUseAttr
from models.simmc21_blenderbot import MultiTaskBlenderbotForConditionalGeneration, SIMMC21BlenderbotTokenizer
from models.simmc21_t5 import MultiTaskT5ForConditionalGeneration, SIMMC21T5Tokenizer
from models.simmc21_ul2 import MultiTaskUL2ForConditionalGeneration, UL2Tokenizer
from models.simmc21_flava import MultiTaskFlavaModel
from models.simmc21_ofa import OFAModelForSIMMCGeneration, MultiTaskOFAModelForConditionalGeneration, OFATokenizer


# 导入数据处理类
from utils.simmc21_dataset import (get_input_id, id_converter, GenerationDataset, get_dataset, 
fashion_meta_attrs, furniture_meta_attrs, available_sizes2st, NUM_FASHION_ITEMS, NUM_FURNITURE_ITEMS,
FASHION_SPECIAL_TOKENS, FURNITURE_SPECIAL_TOKENS, MAX_NUM_OBJ_IN_SCENE, OBJECT_INDICES,
START_OF_MULTIMODAL_CONTEXTS, END_OF_MULTIMODAL_CONTEXTS, START_OF_OBJ_TOKEN, END_OF_OBJ_TOKEN, 
NO_COREF)

from utils.simmc21_dataset_with_image import (LineByLineDatasetWithImage, get_dataset_with_image)
from utils.simmc21_dataset_joint_disam_coref import (LineByLineDatasetJointDisamAndCoref, get_dataset_jointdisamandcoref)
from utils.simmc21_dataset_for_ofa import (LineByLineDatasetForOFA, GenerationDatasetForOFA, get_dataset_for_ofa)
from utils.simmc21_dataset_from_single_file import LineByLineDatasetFromSingleFile # 适合涛哥预处理的数据集文件读取
from utils.simmc21_dataset_from_single_file_for_ofa import LineByLineDatasetFromSingleFileForOFA
from utils.simmc21_dataset_add_attr_embedding import GenerationDatasetAddAttr

from eval_model_args import parser # 导入模型所需参数

# 日志模块定义
logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s",
    datefmt="%m/%d/%Y %H:%M:%S",
    level=logging.INFO,
)
logger = logging.getLogger(__name__)

MAX_LENGTH = int(10000)  # Hardcoded max length to avoid infinite loop


def normalize_sentence(sentence):
    """Normalize the sentences and tokenize."""
    return nltk.tokenize.word_tokenize(sentence.lower())


def parse_response_from_file(input_path):
    """Parses the response from a flattened file.

    Args:
        input_path: Path to read the responses from.
    """
    lines = []
    with open(input_path, "r") as file_id:
        for ii in file_id.readlines():
            split_line = ii.split("<EOB>", 1)
            # print(split_line)
            # 增加该判断，因为部分样例没有生成回复，没有"<EOB>"标记
            # Updated by Yirong Chen
            if len(split_line)==1:
                lines.append(
                    (split_line[0].strip("\n"), "Sorry, I don't understand what you mean.")
                )
            else:
                lines.append(
                    (split_line[0].strip("\n"), split_line[1].strip("\n").strip("<EOS>"))
                )
    return lines

def set_seed(args):
    """增加注释 by Yirong Chen on 2022/08/03
    用途：设置全局随机种子

    """
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if args.n_gpu > 0:
        torch.cuda.manual_seed_all(args.seed)


def correct_action(text, correction_dict):
    for k, v in correction_dict.items():
        text = text.replace(k, v)
    return text

def correct_available_sizes(text):
    """增加注释 by Yirong Chen on 2022/08/03

    """
    SIZES =['<A>', '<B>', '<C>', '<D>', '<E>', '<F>']
    try:
        if 'availableSizes =' in text:
            available_sizes_str_list = [(m.start(0), m.end(0)) for m in re.finditer(r"availableSizes =", text)]
            if not available_sizes_str_list:  # empty available_sizes_str_list: in case of (availableSizes)
                return text
            availableSizes_idx = available_sizes_str_list[0][1]
            start_bracket_idx = -1
            end_bracket_idx = -1
            for i in range(70):
                if text[availableSizes_idx+i] == '[':
                    start_bracket_idx = availableSizes_idx+i
                if text[availableSizes_idx+i] == ']':
                    end_bracket_idx = availableSizes_idx+i
                if start_bracket_idx != -1 and end_bracket_idx != -1:
                    break
            assert start_bracket_idx != -1 and end_bracket_idx != -1, f"ERROR AT def correct_available_sizes!!\n{text}"
            list_str = text[start_bracket_idx:end_bracket_idx].replace("'", "")
            new_list = []
            for size in SIZES:
                if size in list_str:
                    new_list.append(size)
            new = ", ".join(new_list)
            return text[:start_bracket_idx] + '['+new + text[end_bracket_idx:]
        else:
            return text
    except:
        print('text:', text)

def remove_bos_eos_startequal(text):
    text = text.split("</s>")[0].replace('<s>', '')
    return text

def replace_special_chars(text):
    def rep(match_re_obj):
        return match_re_obj.group(0).replace('<','').replace('>','')
    available_sizes_st_list = [('<A>', "'XS'"), ('<B>', "'S'"), ('<C>', "'M'"), ('<D>', "'L'"), ('<E>', "'XL'"), ('<F>', "'XXL'")]
    for size_tuple in available_sizes_st_list:
        text = text.replace(size_tuple[0], size_tuple[1])
    text = re.sub("<[0-9]+>", rep, text)
    return text

def insert_coref(text, coref_chars: list):
    """ coref_chars: [<11>, <44>, ...] 
    Updated by Yirong Chen on 2022/08/11
    增加判断条件，因为有时候会生成不带<EOB>的情形，直接生成<EOS>结束了，
    例如：<s><s>INFORM:GET, S, M, L, XL, and XXL. <EOS>

    <s><s>INFORM:GET, S, M, L, XL, and XXL. <EOS>
    ------->
    <s><s>INFORM:GET, S, M, L, XL, and XXL. <EOB> I'm sorry, I don't see anything else like that. <EOS>

    
    """
    try:
        #print("Before re.finditer: ",text)
        if "<EOB>" in text:
            # 历史版本模型错误，没有将数据|  |去掉 
            text = re.sub(r"\|(([0-9]+)|,| )*\|", "", text)
            text = re.sub(r"\|", "", text)
            text = re.sub(r" .<EOB>", " <EOB>", text)
            #text = re.sub(r"\)   <EOB>", "\) <EOB>", text)
            if r"\)" in text:
                coref_pos_start, coref_pos_end = [(m.start(0), m.end(0)) for m in re.finditer(r"\) *<EOB>", text)][0]
            elif ">   <EOB>" in text or ">   <EOB>" in text or "> <EOB>" in text:
                text = re.sub(r"\> *<EOB>", ")  <EOB>", text)
                coref_pos_start, coref_pos_end = [(m.start(0), m.end(0)) for m in re.finditer(r"\) *<EOB>", text)][0]
            else:
                coref_pos_start, coref_pos_end = [(m.start(0), m.end(0)) for m in re.finditer(r" *<EOB>", text)][0]

        elif "|" in text: # 有 |  |而没有 "<EOB>"
            # 历史版本模型错误，没有将数据|  |去掉
            # 替换
            nPos = text.find("|") # 查找位置
            text = text[:nPos+1] +" <EOB> "+text[nPos+1:]
            text = re.sub(r"\|(([0-9]+)|,| )*\|", "", text)
            text = re.sub(r"\|", "", text)
            coref_pos_start, coref_pos_end = [(m.start(0), m.end(0)) for m in re.finditer(r"\) *<EOB>", text)][0]

        elif ")" in text: # 有 ")"
            print(text)
            # User : Do you have any jackets by Cats Are Great? => Belief State : <pad> [  ] ()  brand  =  Cats Are Great . I do that now.  <EOS>
            nPos = text.find(")") # 查找位置
            text = text[:nPos+1] +" <EOB> "+text[nPos+1:]

            coref_pos_start, coref_pos_end = [(m.start(0), m.end(0)) for m in re.finditer(r"\) *<EOB>", text)][0]

        elif "(" in text: # 有 "("
            print(text)
            # User : Do you have any jackets by Cats Are Great? => Belief State : <pad> [  ] ()  brand  =  Cats Are Great . I do that now.  <EOS>
            nPos = text.find("(") # 查找位置
            text = text[:nPos+1] +") <EOB> "+text[nPos+1:]

            coref_pos_start, coref_pos_end = [(m.start(0), m.end(0)) for m in re.finditer(r"\) *<EOB>", text)][0]

        elif "]" in text: # 有 ")"
            print(text)
            # User : Do you have any jackets by Cats Are Great? => Belief State : <pad> [  ] ()  brand  =  Cats Are Great . I do that now.  <EOS>
            nPos = text.find("]") # 查找位置
            text = text[:nPos+1] +" () <EOB> "+text[nPos+1:]

            coref_pos_start, coref_pos_end = [(m.start(0), m.end(0)) for m in re.finditer(r"\) *<EOB>", text)][0]

        elif "<EOS>" in text:
            print(text)
            #nPos = text.rfind("<EOS>") # 查找位置
            #text = text[:nPos] + "<EOB> I'm sorry, I don't see anything else like that. "+text[nPos:]
            #coref_pos_start, coref_pos_end = [(m.start(0), m.end(0)) for m in re.finditer(r"\) *<EOB>", text)][0]

            act_pos_start, act_pos_end = [(m.start(0), m.end(0)) for m in re.finditer(r"REQUEST:GET|REQUEST:ADD_TO_CART|REQUEST:COMPARE|INFORM:GET|INFORM:REFINE|INFORM:DISAMBIGUATE|ASK:GET", text)][0]
            text = text[:act_pos_end] + " [  ] () " + "<EOB> I'm sorry, I don't see anything else like that. <EOS>"
            coref_pos_start, coref_pos_end = [(m.start(0), m.end(0)) for m in re.finditer(r"\) *<EOB>", text)][0]

        else:
            print(text)
            act_pos_start, act_pos_end = [(m.start(0), m.end(0)) for m in re.finditer(r"REQUEST:GET|REQUEST:ADD_TO_CART|REQUEST:COMPARE|INFORM:GET|INFORM:REFINE|INFORM:DISAMBIGUATE|ASK:GET", text)][0]
            text = text[:act_pos_end] + " [  ] () " + "<EOB> I'm sorry, I don't see anything else like that. <EOS>"
            coref_pos_start, coref_pos_end = [(m.start(0), m.end(0)) for m in re.finditer(r"\) *<EOB>", text)][0]


    except:
        ipdb.set_trace()
        
    coref_list = [int(coref.replace('<', '').replace('>', '')) for coref in coref_chars]
    coref_str = str(coref_list).replace('[', '< ').replace(']',' >') if coref_list else '<  >'
    return text[:coref_pos_start+1] + ' ' + coref_str + ' <EOB>' + text[coref_pos_end:]

def insert_disamb(text, disamb_chars: list):
    """ coref_chars: [<11>, <44>, ...] 
    Updated by Yirong Chen on 2022/08/11
    增加判断条件，因为有时候会生成不带<EOB>的情形，直接生成<EOS>结束了，
    例如：<s><s>INFORM:GET, S, M, L, XL, and XXL. <EOS>

    <s><s>INFORM:GET, S, M, L, XL, and XXL. <EOS>
    ------->
    <s><s>INFORM:GET, S, M, L, XL, and XXL. <EOB> I'm sorry, I don't see anything else like that. <EOS>
    """
    try:
        #print("Before re.finditer: ",text)
        if "<EOB>" in text:
            disamb_pos_start, disamb_pos_end = [(m.start(0), m.end(0)) for m in re.finditer(r"> *<EOB>", text)][0]
        elif "<EOS>" in text:
            nPos = text.rfind("<EOS>") # 查找位置
            text = text[:nPos] + "<EOB> I'm sorry, I don't see anything else like that. "+text[nPos:]
            disamb_pos_start, disamb_pos_end = [(m.start(0), m.end(0)) for m in re.finditer(r"> *<EOB>", text)][0]
    except:
        ipdb.set_trace()
    disamb_list = [int(disamb.replace('<', '').replace('>', '')) for disamb in disamb_chars]
    disamb_str = str(disamb_list).replace('[', '| ').replace(']',' |') if disamb_list else '|  |'
    return text[:disamb_pos_start+1] + ' ' + disamb_str + ' <EOB>' + text[disamb_pos_end:]

def adjust_length_to_model(length, max_sequence_length):
    """增加注释 by Yirong Chen on 2022/08/03
    用途：将通过args传入的length设置与max_sequence_length比较并产生正确的length

    """
    if length < 0 and max_sequence_length > 0:
        length = max_sequence_length
    elif 0 < max_sequence_length < length:
        length = max_sequence_length  # No generation bigger than model size
    elif length < 0:
        length = MAX_LENGTH  # avoid infinite loop
    return length


def evaluate(args, model, tokenizer, prefix="",bad_words_ids=None, test_dataset=None):
    '''evaluate: 给定模型，返回测试结果

    '''
    if args.model_type != 'gen-ofa' and args.model_type != 'mt-ofa': # 排除只做任务4的模型
        box_embedding = model.box_embedding
        disambiguation_head = model.disambiguation_head
        nocoref_head= model.nocoref_head
        fashion_enc_head = model.fashion_enc_head
        furniture_enc_head = model.furniture_enc_head

    if args.model_type == 'mt-bart-attr':
        def collate_bart(examples):
            enc_input = list(map(lambda x: x[0], examples))
            enc_attention_mask = list(map(lambda x: x[1], examples))
            original_lines = list(map(lambda x: x[2], examples))
            boxes = list(map(lambda x: x[3], examples))
            misc = list(map(lambda x: x[4], examples))
            nocoref = list(map(lambda x: x[5], examples))
            disam = list(map(lambda x: x[6], examples))
            obj_ids_per_line = list(map(lambda x: x[7], examples))
            object_attr_input_ids_per_line = list(map(lambda x: x[8], examples))

            if tokenizer._pad_token is None:
                enc_input_pad = pad_sequence(enc_input, batch_first=True)
            else:
                enc_input_pad = pad_sequence(enc_input, batch_first=True, padding_value=tokenizer.pad_token_id)
            enc_attention_pad = pad_sequence(enc_attention_mask, batch_first=True, padding_value=0) # 0表示mask
            return enc_input_pad, enc_attention_pad, original_lines, boxes, misc, nocoref, disam, obj_ids_per_line, object_attr_input_ids_per_line

    else:

        def collate_bart(examples):
            enc_input = list(map(lambda x: x[0], examples))
            enc_attention_mask = list(map(lambda x: x[1], examples))
            original_lines = list(map(lambda x: x[2], examples))
            boxes = list(map(lambda x: x[3], examples))
            misc = list(map(lambda x: x[4], examples))
            nocoref = list(map(lambda x: x[5], examples))
            disam = list(map(lambda x: x[6], examples))
            if len(examples[0])>7:
                image_feature = list(map(lambda x: x[7], examples))
                if (len(image_feature[0].size())<=1):
                    image_feature_pad = torch.vstack(image_feature)
                else:
                    image_feature_pad = pad_sequence(image_feature, batch_first=True, padding_value=0) # torch.Size([batch_size, 3, 224, 224])
            else:
                image_feature_pad = None

            if tokenizer._pad_token is None:
                enc_input_pad = pad_sequence(enc_input, batch_first=True)
            else:
                enc_input_pad = pad_sequence(enc_input, batch_first=True, padding_value=tokenizer.pad_token_id)
            enc_attention_pad = pad_sequence(enc_attention_mask, batch_first=True, padding_value=0) # 0表示mask
            return enc_input_pad, enc_attention_pad, original_lines, boxes, misc, nocoref, disam, image_feature_pad
    
    with open(args.item2id, 'r') as f:
        item2id = json.load(f)
    
    
    decode_sampler = SequentialSampler(test_dataset)
    decode_dataloader = DataLoader(test_dataset, sampler=decode_sampler, batch_size=args.batch_size, collate_fn=collate_bart)

    tokenizer_id2token = {v: k for k, v in tokenizer.get_vocab().items()}
    results = []
    results_coref_and_disamb_replaced = []
    n_prompts = len(test_dataset)

    for index, batch in enumerate(tqdm(decode_dataloader, desc='Decoding')):  # should be 1-batchsized batch

        # 输入数据规范化
        if args.model_parallel:
            enc_input = batch[0].to(model.encoder.first_device)
            enc_input_attention = batch[1].to(model.encoder.first_device)
            original_lines = batch[2]
            boxes = batch[3] # batch, num_obj_per_line, 6
            misc = batch[4]  # batch, num_obj_per_line, dict
            nocoref = batch[5]
            disam = batch[6]
            if args.model_type == 'gen-ofa' or args.model_type == 'mt-ofa':
                image_feature = batch[7].to(model.encoder.first_device)

            if args.model_type == 'mt-bart-attr':
                obj_ids_per_line = batch[7]
                object_attr_input_ids = batch[8]


            batch_size = len(misc)
            decoder_input_ids = torch.full([batch_size,1], model.config.decoder_start_token_id).to(model.encoder.first_device)

        else:
            enc_input = batch[0].to(args.device)
            enc_input_attention = batch[1].to(args.device)
            original_lines = batch[2]
            boxes = batch[3] # batch, num_obj_per_line, 6
            misc = batch[4]  # batch, num_obj_per_line, dict
            nocoref = batch[5]
            disam = batch[6]
            if args.model_type == 'gen-ofa' or args.model_type == 'mt-ofa':
                image_feature = batch[7].to(args.device)
            
            if args.model_type == 'mt-bart-attr':
                obj_ids_per_line = batch[7]
                object_attr_input_ids = batch[8]
            
            batch_size = len(misc)
            
            decoder_input_ids = torch.full([batch_size,1], model.config.decoder_start_token_id).to(args.device)

        with torch.no_grad():
            if args.model_type == 'gen-ofa':
                output_sequences = model.generate(input_ids=enc_input,
                                                  patch_images=image_feature,
                                                  max_length=args.length + enc_input.size()[1],
                                                  min_length=args.min_length,
                                                  temperature=args.temperature,
                                                  top_k=args.k,
                                                  top_p=args.p,
                                                  bad_words_ids=bad_words_ids,
                                                  length_penalty=args.length_penalty,
                                                  repetition_penalty=args.repetition_penalty,
                                                  remove_invalid_values=True, # Whether to remove possible nan and inf outputs of the model to prevent the generation method to crash. Note that using remove_invalid_values can slow down generation.
                                                  do_sample=args.do_sample,
                                                  num_beams=args.num_beams)
            elif args.model_type == 'mt-ofa':
                model_outputs = model(
                            input_ids=enc_input,
                            decoder_input_ids=decoder_input_ids,
                            patch_images=image_feature,
                            boxes=boxes,
                            misc=misc,
                            nocoref=nocoref,
                            return_dict=True)
                disambiguation_logits=model_outputs.disambiguation_logits
                is_disamb = disambiguation_logits.argmax(dim=-1).squeeze().bool() # 1表示有歧义
                is_nodisamb = ~ is_disamb # 取反
                nocoref_logits = model_outputs.nocoref_logits
                is_nocoref = nocoref_logits.argmax(dim=1).bool() # 1表示无多模态共指消解

                disamb_obj_list = []
                disamb_check = []

                coref_obj_list = []
                coref_check = []
                batch_size = len(misc)
                enc_head_results = model_outputs.enc_head_results
                for b_idx in range(batch_size):
                    disamb_obj_each_batch = []
                    coref_obj_each_batch = []

                    objs_pos = [misc[b_idx][obj_idx]['pos'] for obj_idx in range(len(misc[b_idx]))]
                    obj_indices = [tokenizer_id2token[enc_input[b_idx][pos].item()] for pos in objs_pos]  # ex) [<11>, <41>, ...]

                    is_fashion = misc[b_idx][0]['is_fashion']
                    if is_fashion:
                        disamb, coref, size, available_sizes, brand, color, pattern, sleeve_length, \
                        asset_type, type_, price, customer_review = enc_head_results[b_idx]
                    else:
                        disamb, coref, brand, color, materials, type_, price, customer_review = enc_head_results[b_idx]

                    disamb_predict = disamb.argmax(dim=1).tolist()  # (num_objs)
                    for i, disamb_signal in enumerate(disamb_predict):
                        if disamb_signal:
                            disamb_obj_each_batch.append(obj_indices[i])

                    disamb_obj_list.append(disamb_obj_each_batch)
                    disamb_check.append(True if len(disamb_obj_each_batch) > 0 else False)

                    coref_predict = coref.argmax(dim=1).tolist()  # (num_objs)
                    for i, coref_signal in enumerate(coref_predict):
                        if coref_signal:
                            coref_obj_each_batch.append(obj_indices[i])

                    coref_obj_list.append(coref_obj_each_batch)
                    coref_check.append(True if len(coref_obj_each_batch) > 0 else False)
                
                disamb_check_result = torch.logical_and(is_nodisamb.cpu(), torch.tensor(disamb_check, dtype=torch.bool))
                # 检查歧义句子识别结果与歧义候选识别结果是否有冲突
                if args.check_disamb_candi and disamb_check_result.any():
                    print("is_nodisamb and object is both on!!! THIS SHOULD NOT HAPPEN!")
                    idx = (disamb_check_result == True).nonzero(as_tuple=True)[0].tolist()
                    for i in idx:
                        disamb_obj_list[i] = []

                # 检查歧义识别结果是1时，多模态共指消解结果是否存在
                disamb_and_coref_check_result = torch.logical_and(is_disamb.cpu(), torch.tensor(coref_check, dtype=torch.bool))
                if args.check_disamb_and_coref and disamb_and_coref_check_result.any():
                    print("is_disamb and object is both on!!! THIS SHOULD NOT HAPPEN!")
                    idx = (disamb_and_coref_check_result == True).nonzero(as_tuple=True)[0].tolist()
                    for i in idx:
                        coref_obj_list[i] = []

                coref_check_result = torch.logical_and(is_nocoref.cpu(), torch.tensor(coref_check, dtype=torch.bool))
                # 检查多模态共指消解结果与识别结果是否有冲突
                if args.check_isnocoref_and_coref and coref_check_result.any():
                    print("is_nocoref and object is both on!!! THIS SHOULD NOT HAPPEN!")
                    idx = (coref_check_result == True).nonzero(as_tuple=True)[0].tolist()
                    for i in idx:
                        coref_obj_list[i] = []
                
                output_sequences = model.generate(input_ids=enc_input,
                                                  patch_images=image_feature,
                                                  max_length=args.length + enc_input.size()[1],
                                                  min_length=args.min_length,
                                                  temperature=args.temperature,
                                                  top_k=args.k,
                                                  top_p=args.p,
                                                  bad_words_ids=bad_words_ids,
                                                  length_penalty=args.length_penalty,
                                                  repetition_penalty=args.repetition_penalty,
                                                  remove_invalid_values=True, # Whether to remove possible nan and inf outputs of the model to prevent the generation method to crash. Note that using remove_invalid_values can slow down generation.
                                                  do_sample=args.do_sample,
                                                  num_beams=args.num_beams)
            
            
            else:

                if args.model_type == 'mt-t5':
                    # T5的模型结构当中encoder、decoder直接写在MultiTaskT5ForConditionalGeneration类当中
                    inputs_embeds = model.encoder.embed_tokens(enc_input) * model.embed_scale
                    for b_idx in range(batch_size):  # in a batch
                        box_embedded = box_embedding(torch.tensor(boxes[b_idx]).to(args.device))  # (num_obj_per_line, d_model)
                        for obj_idx in range(len(misc[b_idx])):
                            pos = misc[b_idx][obj_idx]['pos']
                            inputs_embeds[b_idx][pos] += box_embedded[obj_idx]

                    encoder_outputs = model.encoder(inputs_embeds=inputs_embeds, attention_mask=enc_input_attention, return_dict=True)  # check this line

                else:
                    inputs_embeds = model.model.encoder.embed_tokens(enc_input) * model.model.encoder.embed_scale
                    for b_idx in range(batch_size):  # in a batch
                        box_embedded = box_embedding(torch.tensor(boxes[b_idx]).to(args.device))  # (num_obj_per_line, d_model)
                        for obj_idx in range(len(misc[b_idx])):
                            pos = misc[b_idx][obj_idx]['pos']
                            inputs_embeds[b_idx][pos] += box_embedded[obj_idx]
                    
                        if args.model_type == 'mt-bart-attr':
                            #for line_object_attr_input_ids in object_attr_input_ids:
                            line_object_embeddings = []
                            for object_attr_input_id in object_attr_input_ids[b_idx]:  # (obj_num, attr_num, 1)
                                # object_attr_input_id: [tensor([50847]), tensor([50863]), tensor([50910]), tensor([448])]
                                object_embeddings = [torch.sum(model.model.encoder.embed_tokens(obj_tok.to(inputs_embeds.device)), dim=0) # summing over columns handling multiple integer tokens
                                                        for obj_tok in object_attr_input_id]
                                # object_embeddings: [torch.Tensor([, ...]), ...] size: (attr_num, 768)
                                line_object_embeddings.append(object_embeddings) # size: (obj_num, attr_num, 768)

                            # 将line_object_embeddings叠加到inputs_embeds当中
                            for idx, abs_id_embs in enumerate(line_object_embeddings):
                                pos = misc[b_idx][idx]['pos']
                                for embs in abs_id_embs:
                                    inputs_embeds[b_idx][pos] += torch.reshape(embs, (-1,))
                    
                    encoder_outputs = model.model.encoder(inputs_embeds=inputs_embeds, attention_mask=enc_input_attention, return_dict=True)  # check this line
            
                enc_last_hidden_state = encoder_outputs.last_hidden_state

                if args.model_parallel:
                    enc_last_hidden_state = enc_last_hidden_state.to(model.encoder.first_device)


                if args.model_type=="mt-ul2":
                    # ['[', 'S', '2', 'S', ']', '<DISAM>', ..., '</s>']
                    disambiguation_logits = disambiguation_head(enc_last_hidden_state[:, 5, :]) # bs, d_model --> bs, 2

                else:
                    # ['<s>', '<DISAM>', ..., '</s>']
                    disambiguation_logits = disambiguation_head(enc_last_hidden_state[:, 1, :]) # bs, d_model --> bs, 2

                is_disamb = disambiguation_logits.argmax(dim=-1).squeeze().bool() # 1表示有歧义
                is_nodisamb = ~ is_disamb # 取反

                nocoref_logits = torch.stack([nocoref_head(enc_last_hidden_state[b_idx][nocoref[b_idx]]) for b_idx in range(batch_size)])
                is_nocoref = nocoref_logits.argmax(dim=1).bool() # 1表示无多模态共指消解

                disamb_obj_list = []
                disamb_check = []

                coref_obj_list = []
                coref_check = []
                for b_idx in range(batch_size):
                    disamb_obj_each_batch = []
                    coref_obj_each_batch = []
                    for obj_idx in range(len(misc[b_idx])):
                        pos = misc[b_idx][obj_idx]['pos']
                        # hidden_concat: (num_obj, 2*model)
                        if obj_idx == 0:
                            hidden_concat = torch.reshape(enc_last_hidden_state[b_idx][pos:pos+2], (1,-1))
                        else:
                            hidden_concat = torch.cat([hidden_concat, torch.reshape(enc_last_hidden_state[b_idx][pos:pos+2], (1,-1))], dim=0)
                    
                    objs_pos = [misc[b_idx][obj_idx]['pos'] for obj_idx in range(len(misc[b_idx]))]
                    obj_indices = [tokenizer_id2token[enc_input[b_idx][pos].item()] for pos in objs_pos]  # ex) [<11>, <41>, ...]

                    is_fashion = misc[b_idx][0]['is_fashion']
                    if is_fashion:
                        disamb, coref, size, available_sizes, brand, color, pattern, sleeve_length, \
                        asset_type, type_, price, customer_review = fashion_enc_head(hidden_concat)
                    else:
                        disamb, coref, brand, color, materials, type_, price, customer_review = furniture_enc_head(hidden_concat)

                    disamb_predict = disamb.argmax(dim=1).tolist()  # (num_objs)
                    for i, disamb_signal in enumerate(disamb_predict):
                        if disamb_signal:
                            disamb_obj_each_batch.append(obj_indices[i])

                    disamb_obj_list.append(disamb_obj_each_batch)
                    disamb_check.append(True if len(disamb_obj_each_batch) > 0 else False)

                    coref_predict = coref.argmax(dim=1).tolist()  # (num_objs)
                    for i, coref_signal in enumerate(coref_predict):
                        if coref_signal:
                            coref_obj_each_batch.append(obj_indices[i])

                    coref_obj_list.append(coref_obj_each_batch)
                    coref_check.append(True if len(coref_obj_each_batch) > 0 else False)
                
                disamb_check_result = torch.logical_and(is_nodisamb.cpu(), torch.tensor(disamb_check, dtype=torch.bool))
                # 检查歧义句子识别结果与歧义候选识别结果是否有冲突
                if args.check_disamb_candi and disamb_check_result.any():
                    print("is_nodisamb and object is both on!!! THIS SHOULD NOT HAPPEN!")
                    idx = (disamb_check_result == True).nonzero(as_tuple=True)[0].tolist()
                    for i in idx:
                        disamb_obj_list[i] = []

                # 检查歧义识别结果是1时，多模态共指消解结果是否存在
                disamb_and_coref_check_result = torch.logical_and(is_disamb.cpu(), torch.tensor(coref_check, dtype=torch.bool))
                if args.check_disamb_and_coref and disamb_and_coref_check_result.any():
                    print("is_disamb and object is both on!!! THIS SHOULD NOT HAPPEN!")
                    idx = (disamb_and_coref_check_result == True).nonzero(as_tuple=True)[0].tolist()
                    for i in idx:
                        coref_obj_list[i] = []

                coref_check_result = torch.logical_and(is_nocoref.cpu(), torch.tensor(coref_check, dtype=torch.bool))
                # 检查多模态共指消解结果与识别结果是否有冲突
                if args.check_isnocoref_and_coref and coref_check_result.any():
                    print("is_nocoref and object is both on!!! THIS SHOULD NOT HAPPEN!")
                    idx = (coref_check_result == True).nonzero(as_tuple=True)[0].tolist()
                    for i in idx:
                        coref_obj_list[i] = []

                output_sequences = model.generate(
                    max_length=args.length + inputs_embeds.size()[1],
                    min_length=args.min_length,
                    temperature=args.temperature,
                    top_k=args.k,
                    top_p=args.p,
                    bad_words_ids=bad_words_ids,
                    length_penalty=args.length_penalty,
                    repetition_penalty=args.repetition_penalty,
                    remove_invalid_values=True, # Whether to remove possible nan and inf outputs of the model to prevent the generation method to crash. Note that using remove_invalid_values can slow down generation.
                    do_sample=args.do_sample,
                    num_beams=args.num_beams,
                    encoder_outputs=encoder_outputs)


        # Remove the batch dimension when returning multiple sequences
        if len(output_sequences.shape) > 2:
            output_sequences.squeeze_()
        
        generated_sequences = []
        generated_sequences_coref_and_disamb_replaced = []
        
        predicts = tokenizer.batch_decode(output_sequences, include_special_token=True)
        for sequence_idx, text in enumerate(predicts):
            '''
            #if sequence_idx == 0:
            print(
                "=== GENERATED SEQUENCE {sequence_idx}, {promt_idx}/{n_prompts} ===".format(
                    sequence_idx=sequence_idx + 1,
                    promt_idx=index + 1,
                    n_prompts=n_prompts,
                )
            )
            print("Before replace : " + text.split("</s>")[0].strip())
            '''
            
            text = remove_bos_eos_startequal(text)
            text = correct_available_sizes(text)
            text_coref_replaced = copy.deepcopy(text)
            if text_coref_replaced is None:
                text_coref_replaced = "INFORM:DISAMBIGUATE [  ] () <EOB> I don't understand what you say, can you explain in detail? <EOS>"
            total_sequence = replace_special_chars(original_lines[sequence_idx] + text_coref_replaced)
            #print("Before insert_coref : ", total_sequence)
            #print("coref_obj_list=", coref_obj_list)

            if args.model_type == 'gen-ofa':
                # 只做任务4类模型
                #total_sequence_coref_replaced = insert_coref(total_sequence, []) # 插入共指消解的结果 < <int>, ... >
                #total_sequence_disamb_replaced = insert_disamb(total_sequence_coref_replaced, []) # 插入歧义候选识别结果 | <int>, ... |
                generated_sequences_coref_and_disamb_replaced.append(total_sequence)

            else:
                total_sequence_coref_replaced = insert_coref(total_sequence, coref_obj_list[sequence_idx]) # 插入共指消解的结果 < <int>, ... >
                total_sequence_disamb_replaced = insert_disamb(total_sequence_coref_replaced, disamb_obj_list[sequence_idx]) # 插入歧义候选识别结果 | <int>, ... |
                generated_sequences_coref_and_disamb_replaced.append(total_sequence_disamb_replaced)

            #if sequence_idx == 0:
            #    print('total_sequence_coref_replaced_and_disamb_replaced:', total_sequence_disamb_replaced, '\n')

        results_coref_and_disamb_replaced.extend(generated_sequences_coref_and_disamb_replaced)
        

    # 保存结果文件到path_output，或者path_output指定的目录下面
    if args.eval_all_checkpoints:
        # "pred-results-of-"+model_name+"_"+checkpoint，例如： pred-results-of-mt-bart-large_context_before_objects_20220914_gas_1_checkpoint-13600.txt 或者 
        # pred-results-of-mt-bart-large_context_before_objects_20220914_gas_1_.txt
        now_file_name = "pred-results-of-" + prefix + ".txt" # 一个字符串
        now_save_path = os.path.join(args.path_output, now_file_name)
        with open(now_save_path, "w") as f_out:
            f_out.write("\n".join(results_coref_and_disamb_replaced))
        return now_save_path # 返回保存的结果文件路径
    else:
        now_file_name = "pred-results-of-" + prefix + ".txt" # 一个字符串
        now_save_path = os.path.join(args.path_output, now_file_name)
        with open(now_save_path, "w") as f_out:
            f_out.write("\n".join(results_coref_and_disamb_replaced))
        return now_save_path # 返回保存的结果文件路径

        #with open(args.path_output, "w") as f_out:
        #    f_out.write("\n".join(results_coref_and_disamb_replaced))

        #return args.path_output # 返回保存的结果文件路径


def main():
    # Get the arguments for training model
    args = parser.parse_args()

    if not args.not_generate_predict_file_again or not os.path.exists(args.path_output):
        # 不指定args.not_generate_predict_file_again或者不存在args.path_output，则运行本部分代码

        args.device = torch.device(
            "cuda" if torch.cuda.is_available() and not args.no_cuda else "cpu"
        )
        args.n_gpu = 0 if args.no_cuda else torch.cuda.device_count()
        args.eval_input_file = args.prompts_from_file
        set_seed(args)

        if args.prompts_from_file and not os.path.exists(args.prompts_from_file):
            raise Exception(f"prompt file '{args.prompts_from_file}' not found")

        # 创建保存模型预测结果的路径
        os.makedirs(args.path_output, exist_ok=True)
        logger.info("Saving model evaluating results to %s", args.path_output)

        # 根据模型类型确定模型类以及tokenizer
        if args.model_type == 'mt-bart-large':
            model_class, tokenizer_class = MultiTaskBartForConditionalGeneration, BartTokenizerFast
        elif args.model_type == 'mt-bart-large-disamb':
            model_class, tokenizer_class = MultiTaskBartForConditionalGenerationWithDisamb, BartTokenizerFast
        elif args.model_type == 'mt-bart-attr':
            model_class, tokenizer_class = MultiTaskBartForConditionalGenerationWithDisambUseAttr, BartTokenizerFast
        elif args.model_type == 'mt-bart_joint_disam_coref':
            model_class, tokenizer_class = MultiTaskBartForConditionalGenerationJointDisambCoref, BartTokenizerFast
        elif args.model_type == 'mt-bart_add_intent':
            model_class, tokenizer_class = MultiTaskBartForConditionalGenerationWithDisambAndIntent, BartTokenizerFast
        elif args.model_type == 'mt-blenderbot':
            model_class, tokenizer_class = MultiTaskBlenderbotForConditionalGeneration, SIMMC21BlenderbotTokenizer
        elif args.model_type == 'mt-t5':
            model_class, tokenizer_class = MultiTaskT5ForConditionalGeneration, SIMMC21T5Tokenizer
        elif args.model_type == 'mt-ul2':
            model_class, tokenizer_class = MultiTaskUL2ForConditionalGeneration, UL2Tokenizer
        elif args.model_type == 'mt-flava':
            model_class, tokenizer_class = MultiTaskFlavaModel, BertTokenizerFast
        elif args.model_type == 'gen-ofa':
            model_class, tokenizer_class = OFAModelForSIMMCGeneration, OFATokenizer
        elif args.model_type == 'mt-ofa':
            model_class, tokenizer_class = MultiTaskOFAModelForConditionalGeneration, OFATokenizer

        # Load tokenizer from model_dir
        if args.model_type == 'mt-t5':
            tokenizer = tokenizer_class.from_pretrained(args.model_dir, model_max_length=512)
        else:
            tokenizer = tokenizer_class.from_pretrained(args.model_dir)

        # 2022/09/16 by Yirong Chen 此处用于初始化图像相关的特征提取器
        if args.model_type == 'mt-flava':
            feature_extractor = FlavaFeatureExtractor.from_pretrained(args.model_name_or_path)
            processor = FlavaProcessor(feature_extractor=feature_extractor, tokenizer=tokenizer)
            #processor = FlavaProcessor.from_pretrained(args.model_name_or_path)
        else:
            # 避免先前的版本参数受到影响
            feature_extractor = None
            processor = None

        # 解码方式设置
        if args.sampling_method == 'greedy-decoding':
            args.num_beams = 1 # 强制为1
            args.do_sample = False
            #num_beam_groups = 1
        elif args.sampling_method == 'multinomial-sampling':
            args.num_beams = 1 # 强制为1
            args.do_sample = True
            #num_beam_groups = 1
        elif args.sampling_method == 'beam-search-decoding':
            args.do_sample = False
            #num_beam_groups = 1
        elif args.sampling_method == 'beam-search-multinomial-sampling':
            args.do_sample = True
        
        # 打印args参数
        logger.info(args)

        # 读取待评测的数据集文件
        # Load the train and eval dataset
        # 2022/10/12 兼容涛哥预处理的数据集，使用LineByLineDatasetFromSingleFile
        #            只需要指定args.train_input_file和args.eval_input_file, 其他为None
        logger.info("Loading Test Dataset!!!")
        if args.model_type == 'gen-ofa' or args.model_type == 'mt-ofa':
            test_dataset = GenerationDatasetForOFA(prompts_from_file=args.prompts_from_file, 
                                                tokenizer=tokenizer, 
                                                model_type=args.model_type, 
                                                use_OBJ=True, 
                                                image_path_file=args.image_path_file, 
                                                image_dir=args.image_dir)
        elif args.model_type == 'mt-bart-attr':
            test_dataset = GenerationDatasetAddAttr(prompts_from_file=args.prompts_from_file, 
                                                    tokenizer=tokenizer, 
                                                    model_type=args.model_type,
                                                    fashion_meta_file=os.path.join(args.data_dir, "fashion_prefab_metadata_all.json"),
                                                    furniture_meta_file=os.path.join(args.data_dir, "furniture_prefab_metadata_all.json"))

        else:
            test_dataset = GenerationDataset(prompts_from_file=args.prompts_from_file, 
                                            tokenizer=tokenizer, 
                                            model_type=args.model_type)

        # 增加bad_words_ids: List of token ids that are not allowed to be generated.
        if args.add_bad_words:
            with open(args.add_bad_words, 'r') as f:
                add_bad_words = json.load(f)
            bad_words = add_bad_words["bad_words_for_generate"]
            bad_words_ids = tokenizer(bad_words, add_prefix_space=True, add_special_tokens=False).input_ids
            print("bad_words_ids=", bad_words_ids)
        else:
            bad_words_ids = None

        # 对模型的所有checkpoint做测试
        checkpoints = [args.model_dir]
        result_file_paths = {} # {}
        if args.eval_all_checkpoints:
            checkpoints = [os.path.dirname(c) for c in sorted(glob.glob(args.model_dir + "/**/" + WEIGHTS_NAME, recursive=True))]
            logging.getLogger("transformers.modeling_utils").setLevel(logging.WARN)  # Reduce logging
        
        logger.info("Evaluate the following checkpoints: %s", checkpoints)
        for checkpoint in checkpoints:
            logger.info("Evaluating the checkpoint: %s", checkpoint)
            global_step = checkpoint.split("-")[-1] if len(checkpoints) > 1 else ""
            prefix = (args.model_dir.split("/")[-1]+"_"+checkpoint.split("/")[-1] if checkpoint.find("checkpoint") != -1 else "") # model_name+"_"+checkpoint-数字
            
            ## 初始化模型
            if args.model_type == 'mt-ul2':
                model = model_class.from_pretrained(checkpoint, low_cpu_mem_usage=True, torch_dtype=torch.bfloat16)
            elif args.model_type == 'gen-ofa':
                model = model_class.from_pretrained(checkpoint, use_cache=False)
            else:
                model = model_class.from_pretrained(checkpoint)

            if args.model_parallel and not args.no_cuda and torch.cuda.is_available(): # T5-11B, UL-2
                logger.info("Execute model.parallelize()")
                model.parallelize()
            else:
                logger.info("put model to %s", args.device)
                model.to(args.device)

            ## 对当前模型执行预测
            result_file_path = evaluate(args=args, 
                                        model=model, 
                                        tokenizer=tokenizer, 
                                        prefix=prefix, 
                                        bad_words_ids=bad_words_ids,
                                        test_dataset=test_dataset)

            result_file_paths[prefix] = result_file_path # 完整的路径

    else:
        # 指定args.not_generate_predict_file_again且存在args.path_output，则运行本部分代码
        glob_path = glob.glob(os.path.join(args.path_output, "*.txt"))
        result_file_paths = {}
        for file_path in glob_path:
            prefix=file_path.split("/")[-1][16:-4]
            result_file_paths[prefix] = file_path # 完整的路径


    # 生成用于最终提交的任务4的json文件
    if args.output_json_response_path is not None:

        os.makedirs(args.output_json_response_path, exist_ok=True)

        for prefix, input_path_predicted in result_file_paths.items():
            # Convert the data from the GPT-2 friendly format to JSON
            list_predicted = parse_flattened_results_from_file(input_path_predicted)
            # Subtask 4
            with open(args.data_json_path, "r") as file_id:
                gt_responses = json.load(file_id)

            #print(gt_responses)
            #print(gt_responses[0])

            dialog_meta_data = json.load(open(args.dialog_meta_data)) # List[Dict]

            predicted_response = [] 

            with open(input_path_predicted, 'r') as f:
                lines = f.readlines()
                assert len(lines) == len(dialog_meta_data)
                for line, meta in zip(lines, dialog_meta_data):
                    response = line.split("<EOB>")[1].split("<EOS>")[0].strip()
                    predicted_response.append({
                        "dialog_id" : meta["dialog_id"],
                        "predictions" : [{
                            "turn_id" : meta["turn_id"],
                            "response" : response
                        }]
                    })

            if prefix:
                json_file_name = prefix+".json"
            else: # prefix == ""
                json_file_name = "final_checkpoint"+".json"

            json.dump(predicted_response, open(os.path.join(args.output_json_response_path, json_file_name), "w"), indent=4)




    # 计算测试结果
    if args.do_calculate_score:
        report_of_all_models = {}
        for prefix, input_path_predicted in result_file_paths.items():
            # Convert the data from the GPT-2 friendly format to JSON
            list_target = parse_flattened_results_from_file(args.input_path_target)
            list_predicted = parse_flattened_results_from_file(input_path_predicted)
            # Evaluate Subtask 1 ~ Subtask 3
            if args.cal_diff_f1_based_on_previously_mentioned and args.multimodal_context_json_file is not None:
                with open(args.multimodal_context_json_file, "r") as file_id:
                    mentioned_objects = json.load(file_id)

                report = evaluate_from_flat_list_mentioned_object(list_target, list_predicted, mentioned_objects)
            else:

                report = evaluate_from_flat_list(list_target, list_predicted)
            #print(report)

            # Evaluate Subtask 4
            if args.single_round_evaluation:

                with open(args.data_json_path, "r") as file_id:
                    gt_responses = json.load(file_id)

                #print(gt_responses)
                #print(gt_responses[0])

                dialog_meta_data = json.load(open(args.dialog_meta_data)) # List[Dict]

                predicted_response = [] 

                with open(input_path_predicted, 'r') as f:
                    lines = f.readlines()
                    assert len(lines) == len(dialog_meta_data)
                    for line, meta in zip(lines, dialog_meta_data):
                        response = line.split("<EOB>")[1].split("<EOS>")[0].strip()
                        predicted_response.append({
                            "dialog_id" : meta["dialog_id"],
                            "predictions" : [{
                                "turn_id" : meta["turn_id"],
                                "response" : response
                            }]
                        })
                if args.output_json_response_path is not None:

                    if prefix:
                        json_file_name = prefix+".json"
                    else: # prefix == ""
                        json_file_name = "final_checkpoint"+".json"

                    json.dump(predicted_response, open(os.path.join(args.output_json_response_path, json_file_name), "w"), indent=4)


                bleu_score, bleu_std_err = evaluate_response_generation(
                    gt_responses, predicted_response, args.single_round_evaluation
                )
                print(f"BLEU Score: {bleu_score} +- {bleu_std_err}")

                report["bleu"] = bleu_score
                report["bleu_stderr"] = bleu_std_err
            else:
                # Convert the data from the model friendly format to JSON
                list_target = parse_response_from_file(args.input_path_target)
                list_predicted = parse_response_from_file(input_path_predicted)
                # Compute BLEU scores.
                bleu_scores = []
                # Smoothing function.
                chencherry = nltk.translate.bleu_score.SmoothingFunction()

                for response, gt_response in zip(list_predicted, list_target):
                    #print("预测回复：", response[0])
                    #print("真实回复：", gt_response[0])
                    #assert response[0] == gt_response[0], "Input contexts do not match!"
                    bleu_score = nltk.translate.bleu_score.sentence_bleu(
                        [normalize_sentence(gt_response[1])],
                        normalize_sentence(response[1]),
                        smoothing_function=chencherry.method7,
                    )
                    bleu_scores.append(bleu_score)
                mean_bleu_scores = np.mean(bleu_scores)
                mean_bleu_scores_std = np.std(bleu_scores) / np.sqrt(len(bleu_scores))

                report["bleu"] = mean_bleu_scores
                report["bleu_stderr"] = mean_bleu_scores_std

                print(
                    "BLEU score: {} +- {}".format(
                        mean_bleu_scores, mean_bleu_scores_std
                    )
                )

            report_of_all_models[prefix] = report # {"": {}, ...}
        
        report_list = []
        for prefix, report in report_of_all_models.items():
            if args.cal_diff_f1_based_on_previously_mentioned and args.multimodal_context_json_file is not None:
                temp_list = [prefix, report["disamb_candidate_prec"], report["disamb_candidate_rec"], report["disamb_candidate_f1"], 
                    report["disamb_candidate_prec_mentioned_object"], report["disamb_candidate_rec_mentioned_object"], report["disamb_candidate_f1_mentioned_object"],
                    report["disamb_candidate_prec_not_mentioned_object"], report["disamb_candidate_rec_not_mentioned_object"], report["disamb_candidate_f1_not_mentioned_object"],
                    report["object_prec"], report["object_rec"], report["object_f1"], 
                    report["object_prec_mentioned_object"], report["object_rec_mentioned_object"], report["object_f1_mentioned_object"], 
                    report["object_prec_not_mentioned_object"], report["object_rec_not_mentioned_object"], report["object_f1_not_mentioned_object"], 
                    report["slot_f1"], report["act_f1"], report["bleu"]]

            else:
                temp_list = [prefix, report["disamb_candidate_prec"], report["disamb_candidate_rec"], report["disamb_candidate_f1"], report["object_prec"],
                                report["object_rec"], report["object_f1"], report["slot_f1"], report["act_f1"], report["bleu"]]

            report_list.append(temp_list)
        
        # 将结果转为.csv文件并且保存：
        # "Model_Checkpoint" "Subtask-1-Amb.-Candi.-F1" "Subtask-2-MM-Coref-F1" "Subtask-3-MM-DST-Slot-F1" "Subtask-3-MM-DST-Intent-F1" "Subtask-4-Response-Gen.-BLEU-4"
        if args.cal_diff_f1_based_on_previously_mentioned and args.multimodal_context_json_file is not None:
            df = pd.DataFrame(report_list, columns=['model_name', "disamb_candidate_prec", "disamb_candidate_rec", "disamb_candidate_f1", 
                    "disamb_candidate_prec_mentioned_object", "disamb_candidate_rec_mentioned_object", "disamb_candidate_f1_mentioned_object", 
                    "disamb_candidate_prec_not_mentioned_object", "disamb_candidate_rec_not_mentioned_object", "disamb_candidate_f1_not_mentioned_object", 
                    "object_prec", "object_rec", "object_f1", 
                    "object_prec_mentioned_object", "object_rec_mentioned_object", "object_f1_mentioned_object",
                    "object_prec_not_mentioned_object", "object_rec_not_mentioned_object", "object_f1_not_mentioned_object",
                    "slot_f1", "act_f1", "bleu"])

        else:
            df = pd.DataFrame(report_list, columns=['model_name', "disamb_candidate_prec", "disamb_candidate_rec", "disamb_candidate_f1", "object_prec",
                                                "object_rec", "object_f1", "slot_f1", "act_f1", "bleu"])
        df.to_csv(args.output_path_csv_report)
    

if __name__ == "__main__":
    main()
