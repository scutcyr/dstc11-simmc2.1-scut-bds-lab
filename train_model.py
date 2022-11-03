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


# File: train_model.py
# Description: The training model code for DSTC-11 SIMMC 2.1
# Repository: https://github.com/scutcyr/dstc11-simmc2.1-scut-bds-lab
# Mail: [eeyirongchen@mail.scut.edu.cn](mailto:eeyirongchen@mail.scut.edu.cn)
# Date: 2022/10/10
# Usage:
'''
# 以4块A100 48GB显卡运行OFA模型为例
WORK_DIR=~/dstc11_simmc2.1_scut-bds-lab/dstc11-simmc2.1-scut-bds-lab
INIT_DATA_DIR=../data
PREPROCESS_DATA_DIR=./data_convert
CONTEXT_LENGTH=4 # 2,4,6,8
# cd working path
cd $WORK_DIR

# 科研平台上：--master_addr $MASTER_ADDR --master_port $MASTER_PORT
# 本地运行：--master_addr 127.0.0.1 --master_port 9129
torchrun --nnodes 1 --nproc_per_node 4 --master_addr $MASTER_ADDR --master_port 9011 train_model.py \
    --model_type=gen-ofa \
    --model_name_or_path=~/pretraining_model/OFA-large \
    --add_special_tokens=$INIT_DATA_DIR/simmc_special_tokens.json \
    --item2id=$INIT_DATA_DIR/item2id.json \
    --train_input_file=$PREPROCESS_DATA_DIR/simmc2.1_dials_dstc11_train_predict_ctxlen${CONTEXT_LENGTH}.txt \
    --train_target_file=$PREPROCESS_DATA_DIR/simmc2.1_dials_dstc11_train_target_ctxlen${CONTEXT_LENGTH}.txt  \
    --disambiguation_file=$PREPROCESS_DATA_DIR/simmc2.1_dials_dstc11_train_disambiguation_label.txt \
    --response_file=$PREPROCESS_DATA_DIR/simmc2.1_dials_dstc11_train_response.txt \
    --eval_input_file=$PREPROCESS_DATA_DIR/simmc2.1_dials_dstc11_devtest_predict_ctxlen${CONTEXT_LENGTH}.txt \
    --eval_target_file=$PREPROCESS_DATA_DIR/simmc2.1_dials_dstc11_devtest_target_ctxlen${CONTEXT_LENGTH}.txt \
    --train_image_path_file=$PREPROCESS_DATA_DIR/simmc2.1_dials_dstc11_train_scene_name.txt \
    --train_image_dir=$INIT_DATA_DIR/simmc2_scene_images_dstc10_public \
    --eval_image_path_file=$PREPROCESS_DATA_DIR/simmc2.1_dials_dstc11_devtest_scene_name.txt \
    --eval_image_dir=$INIT_DATA_DIR/simmc2_scene_images_dstc10_public \
    --output_dir=./runs/simmc21_ofa_ctxlen${CONTEXT_LENGTH}_20221012_1650 \
    --output_eval_file=./runs/simmc21_ofa_ctxlen${CONTEXT_LENGTH}_20221012_1650/eval_report.txt \
    --overwrite_output_dir \
    --num_train_epochs=12 \
    --evaluate_during_training \
    --log_steps=10 \
    --save_total_limit=12 \
    --embedding_train_epochs_start=400 \
    --do_train_embedding_clip_way_during_training \
    --embedding_train_steps=200 \
    --embedding_train_epochs_ongoing=100 \
    --per_gpu_train_batch_size=8 \
    --per_gpu_eval_batch_size=8 \
    --weight_decay=0.1 \
    --adam_epsilon=1e-8 \
    --max_grad_norm=1.0 \
    --seed=2022 \
    --warm_up_ratio=0.1 \
    --learning_rate=5e-5 \
    --scheduler=get_linear_schedule_with_warmup \
    --optimizer=AdamW \
    --gradient_accumulation_steps=1
'''


import re
import os
os.environ["TOKENIZERS_PARALLELISM"] = 'true' # 为了模型并行
import ast
import copy
import json
import glob
import torch
import random
import argparse
import logging
import shutil
import numpy as np
from pprint import pformat
from tqdm import tqdm, trange
#import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from typing import Dict, List, Tuple

from torch import nn
import torch.nn.functional as F
from torch.optim import AdamW as torch_AdamW
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data.distributed import DistributedSampler
from torch.cuda.amp import autocast as autocast # 用于使用自动混合精度，要求torch版本为1.6+
from torch.utils.data import DataLoader, Dataset, RandomSampler, SequentialSampler
from transformers import (MODEL_WITH_LM_HEAD_MAPPING, WEIGHTS_NAME, AutoConfig, AutoModelWithLMHead, AutoTokenizer,
    BertTokenizerFast, BartTokenizerFast, FlavaProcessor, FlavaFeatureExtractor)

from transformers.optimization import (AdamW, Adafactor, get_linear_schedule_with_warmup, get_constant_schedule_with_warmup, 
    get_constant_schedule, get_cosine_schedule_with_warmup, get_adafactor_schedule)

# 导入模型类
from models.simmc21_bart import MultiTaskBartForConditionalGeneration # 没有歧义候选识别任务
from models.simmc21_bart import MultiTaskBartForConditionalGenerationWithDisamb # 有歧义候选识别任务
from models.simmc21_bart import MultiTaskBartForConditionalGenerationJointDisambCoref
from models.simmc21_bart import MultiTaskBartForConditionalGenerationWithDisambAndIntent
from models.simmc21_bart import MultiTaskBartForConditionalGenerationWithDisambUseAttr
from models.simmc21_blenderbot import MultiTaskBlenderbotForConditionalGeneration, SIMMC21BlenderbotTokenizer
from models.simmc21_t5 import MultiTaskT5ForConditionalGeneration, SIMMC21T5Tokenizer
from models.simmc21_ul2 import MultiTaskUL2ForConditionalGeneration, UL2Tokenizer
from models.simmc21_flava import MultiTaskFlavaModel
from models.simmc21_ofa import OFAModelForSIMMCGeneration, MultiTaskOFAModelForConditionalGeneration, OFATokenizer
from models import (compute_kl_loss, count_trainable_parameters, count_total_parameters, show_trainable_parameters)

# 导入数据处理类
from utils import api#, util
from utils.simmc21_dataset import (get_input_id, id_converter, LineByLineDataset, get_dataset, fashion_meta_attrs, furniture_meta_attrs, 
    available_sizes2st, NUM_FASHION_ITEMS, NUM_FURNITURE_ITEMS,FASHION_SPECIAL_TOKENS, FURNITURE_SPECIAL_TOKENS, MAX_NUM_OBJ_IN_SCENE, OBJECT_INDICES,
    START_OF_MULTIMODAL_CONTEXTS, END_OF_MULTIMODAL_CONTEXTS, START_OF_OBJ_TOKEN, END_OF_OBJ_TOKEN, NO_COREF)
from utils.simmc21_dataset_with_image import (LineByLineDatasetWithImage, get_dataset_with_image)
from utils.simmc21_dataset_joint_disam_coref import (LineByLineDatasetJointDisamAndCoref, get_dataset_jointdisamandcoref)
from utils.simmc21_dataset_for_ofa import (LineByLineDatasetForOFA, get_dataset_for_ofa)
from utils.metadata import (FASHION_SIZES, FASHION_AVAILABLE_SIZES, FASHION_BRAND, FASHION_COLOR, FASHION_PATTERN, FASHION_SLEEVE_LENGTH, 
    FASHION_ASSET_TYPE, FASHION_TYPE, FASHION_PRICE, FASHION_CUSTOMER_REVIEW, FURNITURE_BRAND, FURNITURE_COLOR, FURNITURE_MATERIALS, FURNITURE_TYPE, 
    FURNITURE_PRICE, FURNITURE_CUSTOMER_RATING)
from utils.simmc21_dataset_from_single_file import LineByLineDatasetFromSingleFile # 适合涛哥预处理的数据集文件读取
from utils.simmc21_dataset_from_single_file_for_ofa import LineByLineDatasetFromSingleFileForOFA
from utils.simmc21_dataset_add_attr_embedding import LineByLineDatasetWithOBJList, get_dataset_with_obj_attr

from train_model_args import parser # 导入模型所需参数

try:
    from torch.utils.tensorboard import SummaryWriter
except ImportError:
    from tensorboardX import SummaryWriter

logger = logging.getLogger(__file__)

ce_loss_fct = nn.CrossEntropyLoss()
bce_loss_fct = nn.BCEWithLogitsLoss()
KLMeanLoss = nn.KLDivLoss(reduction="mean", log_target=True) # 对于分类任务
KLBatchMeanLoss = nn.KLDivLoss(reduction="batchmean", log_target=True) # 对于生成任务


def setup_seed(seed, n_gpu):
    ''' 设置随机种子 '''
    torch.manual_seed(seed)
    if n_gpu > 0:
        torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True

def average_distributed_scalar(scalar, args):
    ''' Average a scalar over the nodes if we are in distributed training. We use this for distributed evaluation. '''
    if args.local_rank == -1:
        return scalar
    scalar_t = torch.tensor(scalar, dtype=torch.float, device=args.device) / torch.distributed.get_world_size()
    torch.distributed.all_reduce(scalar_t, op=torch.distributed.ReduceOp.SUM)
    return scalar_t.item()

def _sorted_checkpoints(args, checkpoint_prefix="checkpoint", use_mtime=False) -> List[str]:
    ''' 对所有的checkpoints进行排序 '''
    ordering_and_checkpoint_path = []
    glob_checkpoints = glob.glob(os.path.join(args.output_dir, "{}-*".format(checkpoint_prefix)))

    for path in glob_checkpoints:
        if use_mtime:
            ordering_and_checkpoint_path.append((os.path.getmtime(path), path))
        else:
            regex_match = re.match(".*{}-([0-9]+)".format(checkpoint_prefix), path)
            if regex_match and regex_match.groups():
                ordering_and_checkpoint_path.append((int(regex_match.groups()[0]), path))

    checkpoints_sorted = sorted(ordering_and_checkpoint_path)
    checkpoints_sorted = [checkpoint[1] for checkpoint in checkpoints_sorted]
    return checkpoints_sorted


def _rotate_checkpoints(args, checkpoint_prefix="checkpoint", use_mtime=False) -> None:
    ''' 删除多出的checkpoint '''
    if not args.save_total_limit or args.save_total_limit <= 0:
        return
    # Check if we should delete older checkpoint(s)
    checkpoints_sorted = _sorted_checkpoints(args, checkpoint_prefix, use_mtime)
    if len(checkpoints_sorted) <= args.save_total_limit:
        return

    number_of_checkpoints_to_delete = max(0, len(checkpoints_sorted) - args.save_total_limit)
    checkpoints_to_be_deleted = checkpoints_sorted[:number_of_checkpoints_to_delete]
    for checkpoint in checkpoints_to_be_deleted:
        logger.info("Deleting older checkpoint [{}] due to args.save_total_limit".format(checkpoint))
        shutil.rmtree(checkpoint)

def train_embedding_clip_way(args, model, tokenizer, all_objects_meta, num_iter=50, do_tsne=False):
    ''' train special token embedding '''
    # 参考：https://blog.csdn.net/qq_43332629/article/details/125322182
    if hasattr(model, "module"): # 模型DDP时，该部分可以正常运行
        if args.model_type == 'mt-t5' or args.model_type == 'mt-ul2' or args.model_type == 'gen-ofa' or args.model_type == 'mt-ofa':
            emb = model.module.encoder.embed_tokens
        elif args.model_type == 'mt-flava':
            emb = model.module.text_model.embeddings.word_embeddings
        else:
            emb = model.module.model.encoder.embed_tokens
    else:
        if args.model_type == 'mt-t5' or args.model_type == 'mt-ul2' or args.model_type == 'gen-ofa' or args.model_type == 'mt-ofa':
            emb = model.encoder.embed_tokens
        elif args.model_type == 'mt-flava':
            emb = model.text_model.embeddings.word_embeddings
        else:
            emb = model.model.encoder.embed_tokens

    emb.weight.requires_grad = True
    emb_weight_clone = emb.weight.detach().clone()
    emb_opt = torch_AdamW(emb.parameters())

    # fashion_attr_dict: {'sleeve_length': {'sleeveless':[(tok1, ctr1), (tok2, ctr2)], 'long':[(tok3,ctr3)] ...},
    #                'color':{'red':[(tok1, ctr1)], ...}, 
    #                ...}
    fashion_attr_dict = dict()
    fashion_meta_attrs_copied = copy.deepcopy(fashion_meta_attrs)
    for attr_name, attr_values in fashion_meta_attrs_copied.items():
        fashion_attr_dict[attr_name] = dict()
        if '' in attr_values:
            attr_values.remove('')
            attr_values.append('none')
        attr_values = list(attr_values)
        attr_values.sort()
        accum_token_counter = 0
        for attr_value in attr_values:
            fashion_attr_dict[attr_name][attr_value] = [(x, accum_token_counter + i) for i, x in enumerate(get_input_id(tokenizer, attr_value))]
            accum_token_counter += len(get_input_id(tokenizer, attr_value))

    furniture_attr_dict = dict()
    for attr_name, attr_values in furniture_meta_attrs.items():
        furniture_attr_dict[attr_name] = dict()
        attr_values = list(attr_values)
        attr_values.sort()
        accum_token_counter = 0
        for attr_value in attr_values:
            if not attr_value:  # skip empty string
                continue
            furniture_attr_dict[attr_name][attr_value] = [(x, accum_token_counter + i) for i, x in enumerate(get_input_id(tokenizer, attr_value))]
            accum_token_counter += len(get_input_id(tokenizer, attr_value))
    
    # fashion_item_label: {
    #    '<@1001>': {'sleeve_length': [pos1, pos2], 'color': [pos1], ...}
    # }
    fashion_item_label = {fashion_st: dict() for fashion_st in FASHION_SPECIAL_TOKENS}
    for attr_name, token_dict in fashion_attr_dict.items():
        for fashion_st in FASHION_SPECIAL_TOKENS:
            if attr_name == 'available_sizes':
                item_meta = all_objects_meta[fashion_st]
                sizes = item_meta[attr_name]
                sizes.sort()
                #print(token_dict[sizes[0]])
                #print(sizes)
                # sizes = ['^'+size for size in attr_value.split('^') if size]
                fashion_item_label[fashion_st][attr_name] = [token_dict[size][0][1] for size in sizes] 
            else:
                item_meta = all_objects_meta[fashion_st]
                attr_value = item_meta[attr_name] if item_meta[attr_name] != '' else 'none'  # for sleeve_length ''
                fashion_item_label[fashion_st][attr_name] = [idx for tok, idx in token_dict[attr_value]]

    furniture_item_label = {furniture_st: dict() for furniture_st in FURNITURE_SPECIAL_TOKENS}
    for attr_name, token_dict in furniture_attr_dict.items():
        for furniture_st in FURNITURE_SPECIAL_TOKENS:
            item_meta = all_objects_meta[furniture_st]
            attr_value = item_meta[attr_name]
            furniture_item_label[furniture_st][attr_name] = [idx for tok, idx in token_dict[attr_value]]
    
    # fashion_ce_loss_fct_label: {attr_name: [gt1, gt2, gt3, ...], ...} 
    fashion_ce_loss_fct_label = dict()
    for attr_name in fashion_attr_dict.keys():
        gt_list = []
        for item in FASHION_SPECIAL_TOKENS:
            gt_list.extend(fashion_item_label[item][attr_name])
        fashion_ce_loss_fct_label[attr_name] = torch.tensor(gt_list).to(args.device)
    furniture_ce_loss_fct_label = dict()
    for attr_name in furniture_attr_dict.keys():
        gt_list = []
        for item in FURNITURE_SPECIAL_TOKENS:
            gt_list.extend(furniture_item_label[item][attr_name])
        furniture_ce_loss_fct_label[attr_name] = torch.tensor(gt_list).to(args.device)
    
    fashion_attr_embed_matrix = dict()
    for attr_name, tok_dict in fashion_attr_dict.items():
        fashion_attr_embed_matrix[attr_name] = torch.stack([emb_weight_clone[t[0]] for tl in tok_dict.values() for t in tl]).to(args.device)
    furniture_attr_embed_matrix = dict()
    for attr_name, tok_dict in furniture_attr_dict.items():
        furniture_attr_embed_matrix[attr_name] = torch.stack([emb_weight_clone[t[0]] for tl in tok_dict.values() for t in tl]).to(args.device)

    for i in range(num_iter):
        for j, attr_name in enumerate(fashion_attr_dict.keys()):
            st_indices = []
            for fashion_st in FASHION_SPECIAL_TOKENS:
                st_repeat = len(fashion_item_label[fashion_st][attr_name])
                st_indices.extend(get_input_id(tokenizer, fashion_st) * st_repeat)
            # logits: (num_possibly_duplicated_items, num_concatenated_tokens)
            logits = emb(torch.tensor(st_indices).to(args.device)) @ fashion_attr_embed_matrix[attr_name].t()
            if j == 0:
                fashion_emb_loss = ce_loss_fct(logits, fashion_ce_loss_fct_label[attr_name])
            else: 
                fashion_emb_loss += ce_loss_fct(logits, fashion_ce_loss_fct_label[attr_name])
        for j, attr_name in enumerate(furniture_attr_dict.keys()):
            st_indices = []
            for furniture_st in FURNITURE_SPECIAL_TOKENS:
                st_repeat = len(furniture_item_label[furniture_st][attr_name])
                st_indices.extend(get_input_id(tokenizer, furniture_st) * st_repeat)
            # logits: (num_possibly_duplicated_items, num_concatenated_tokens)
            logits = emb(torch.tensor(st_indices).to(args.device)) @ furniture_attr_embed_matrix[attr_name].t()
            if j == 0:
                furniture_emb_loss = ce_loss_fct(logits, furniture_ce_loss_fct_label[attr_name])
            else:
                furniture_emb_loss += ce_loss_fct(logits, furniture_ce_loss_fct_label[attr_name])
        (fashion_emb_loss + furniture_emb_loss).backward()
        emb_opt.step()
        emb.zero_grad()
    
    if do_tsne:
        tsne = TSNE()
        st_indices = []
        st = []
        for fashion_st in FASHION_SPECIAL_TOKENS:
            st_indices.extend(get_input_id(tokenizer, fashion_st))
            st.append(fashion_st)
        for furniture_st in FURNITURE_SPECIAL_TOKENS:
            st_indices.extend(get_input_id(tokenizer, furniture_st))
            st.append(furniture_st)

        tsne_logits = emb(torch.tensor(st_indices).to(args.device)).detach().cpu().numpy()  # (num_items (fashion and furniture), d_model)
        tsne_fitted = tsne.fit_transform(tsne_logits)
        '''
        plt.figure(figsize=(20, 20))
        for i in range(len(tsne_fitted)):
            # print(f"x: {tsne_fitted[i,0]}, y: {tsne_fitted[i,1]}")
            plt.text(tsne_fitted[i,0], tsne_fitted[i,1], str(st[i]), color='black', fontdict={'weight': 'bold', 'size':9})
        plt.savefig('fig1.png', dpi=300)
        '''


def train(args, model, tokenizer, all_objects_meta, train_dataset, eval_dataset=None):
    '''
    Training the model
    '''
    logger.info("***** Preparing for training *****")
    if args.local_rank in [-1, 0]:
        os.makedirs(args.output_dir, exist_ok=True)
        tb_writer = SummaryWriter(log_dir=args.output_dir)


    if args.model_type == 'mt-bart-attr':
        def collate_train(examples):
            '''
            # 兼容2022.09.02之前的版本的collate_bart
            # 可根据examples的size判别为不同的输入
            # examples: [(),(),(), ...]
            '''
            #print("len(examples[0])=", len(examples[0]))
            enc_input = list(map(lambda x: x[0], examples))
            enc_attention_mask = list(map(lambda x: x[1], examples))
            decoder_input = list(map(lambda x: x[2], examples))
            boxes = list(map(lambda x: x[3], examples))  
            misc = list(map(lambda x: x[4], examples))
            nocoref = list(map(lambda x: x[5], examples))
            disambiguation_labels = list(map(lambda x: x[6], examples)) # in get_dataset_jointdisamandcoref, this represent disam_and_coref_labels
            response = list(map(lambda x: x[7], examples))
            if len(examples[0])>8:
                disam = list(map(lambda x: x[8], examples))
            else:
                disam = None
            if len(examples[0])>9:
                obj_ids_per_line = list(map(lambda x: x[9], examples))
            else:
                obj_ids_per_line = None

            if len(examples[0])>10:
                object_attr_input_ids_per_line = list(map(lambda x: x[10], examples))
            else:
                object_attr_input_ids_per_line = None

            if tokenizer._pad_token is None:
                enc_input_pad = pad_sequence(enc_input, batch_first=True)
            else:
                enc_input_pad = pad_sequence(enc_input, batch_first=True, padding_value=tokenizer.pad_token_id)
            
            enc_attention_pad = pad_sequence(enc_attention_mask, batch_first=True, padding_value=0) # 0表示被mask掉
            decoder_input_pad = tokenizer(decoder_input, padding="longest", truncation=True, return_tensors="pt")
            response_pad = tokenizer(response, padding="longest", truncation=True, return_tensors="pt")

            if len(examples[0])>10: # 增加obj_ids_per_line, object_attr_input_ids_per_line
                return enc_input_pad, enc_attention_pad, decoder_input_pad.input_ids, decoder_input_pad.attention_mask, \
                        boxes, misc, nocoref, torch.vstack(disambiguation_labels), response_pad.input_ids, response_pad.attention_mask, \
                        obj_ids_per_line, object_attr_input_ids_per_line

            elif len(examples[0])>9: # obj_ids_per_line
                return enc_input_pad, enc_attention_pad, decoder_input_pad.input_ids, decoder_input_pad.attention_mask, \
                        boxes, misc, nocoref, torch.vstack(disambiguation_labels), response_pad.input_ids, response_pad.attention_mask, obj_ids_per_line
            # 2022.09.02之前的版本输出
            return enc_input_pad, enc_attention_pad, decoder_input_pad.input_ids, decoder_input_pad.attention_mask, \
                    boxes, misc, nocoref, torch.vstack(disambiguation_labels), response_pad.input_ids, response_pad.attention_mask

    else:
        def collate_train(examples):
            '''
            # 兼容2022.09.02之前的版本的collate_bart
            # 可根据examples的size判别为不同的输入
            # examples: [(),(),(), ...]
            '''
            #print("len(examples[0])=", len(examples[0]))
            enc_input = list(map(lambda x: x[0], examples))
            enc_attention_mask = list(map(lambda x: x[1], examples))
            decoder_input = list(map(lambda x: x[2], examples))
            boxes = list(map(lambda x: x[3], examples))  
            misc = list(map(lambda x: x[4], examples))
            nocoref = list(map(lambda x: x[5], examples))
            disambiguation_labels = list(map(lambda x: x[6], examples)) # in get_dataset_jointdisamandcoref, this represent disam_and_coref_labels
            response = list(map(lambda x: x[7], examples))
            if len(examples[0])>8:
                disam = list(map(lambda x: x[8], examples))
            else:
                disam = None
            
            if len(examples[0])>9:
                # in get_dataset_jointdisamandcoref, this represent user_act_labels
                image_feature = list(map(lambda x: x[9], examples))
                #print("image_feature=", image_feature)
                if (len(image_feature[0].size())<=1):
                    image_feature_pad = torch.vstack(image_feature)
                else:
                    image_feature_pad = pad_sequence(image_feature, batch_first=True, padding_value=0) # torch.Size([batch_size, 3, 224, 224]) or torch.Size([batch_size, 3, 480, 480])
            else:
                image_feature_pad = None
            
            if len(examples[0])>10:
                # in get_dataset_jointdisamandcoref, this represent system_act_labels
                enc_token_type_ids = list(map(lambda x: x[10], examples))
                if (len(enc_token_type_ids[0].size())<=1):
                    enc_token_type_ids_pad = torch.vstack(enc_token_type_ids)
                else:
                    enc_token_type_ids_pad = pad_sequence(enc_token_type_ids, batch_first=True, padding_value=0)
            else:
                enc_token_type_ids_pad = None

            if tokenizer._pad_token is None:
                enc_input_pad = pad_sequence(enc_input, batch_first=True)
            else:
                enc_input_pad = pad_sequence(enc_input, batch_first=True, padding_value=tokenizer.pad_token_id)
            
            enc_attention_pad = pad_sequence(enc_attention_mask, batch_first=True, padding_value=0) # 0表示被mask掉
            decoder_input_pad = tokenizer(decoder_input, padding="longest", truncation=True, return_tensors="pt")
            response_pad = tokenizer(response, padding="longest", truncation=True, return_tensors="pt")

            if len(examples[0])>10: # 增加图像特征，增加token_type_ids
                return enc_input_pad, enc_attention_pad, decoder_input_pad.input_ids, decoder_input_pad.attention_mask, \
                        boxes, misc, nocoref, torch.vstack(disambiguation_labels), response_pad.input_ids, response_pad.attention_mask, \
                        image_feature_pad, enc_token_type_ids_pad

            elif len(examples[0])>9: # 增加图像特征
                return enc_input_pad, enc_attention_pad, decoder_input_pad.input_ids, decoder_input_pad.attention_mask, \
                        boxes, misc, nocoref, torch.vstack(disambiguation_labels), response_pad.input_ids, response_pad.attention_mask, image_feature_pad
            # 2022.09.02之前的版本输出
            return enc_input_pad, enc_attention_pad, decoder_input_pad.input_ids, decoder_input_pad.attention_mask, \
                    boxes, misc, nocoref, torch.vstack(disambiguation_labels), response_pad.input_ids, response_pad.attention_mask

    train_sampler = RandomSampler(train_dataset) if args.local_rank == -1 or args.model_parallel else DistributedSampler(train_dataset)
    train_dataloader = DataLoader(train_dataset, sampler=train_sampler, batch_size=args.train_batch_size, collate_fn=collate_train)
    if args.max_steps > 0:
        t_total = args.max_steps
        args.num_train_epochs = args.max_steps // (len(train_dataloader) // args.gradient_accumulation_steps) + 1
    else:
        t_total = len(train_dataloader) // args.gradient_accumulation_steps * args.num_train_epochs

    # 设置需要进行L2正则化的参数
    no_decay = ["bias", "LayerNorm.weight"]
    optimizer_grouped_parameters = [
        {"params": [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)], "weight_decay": args.weight_decay},
        {"params": [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)], "weight_decay": 0.0,}
        ]

    # 训练时的优化器选择
    if args.optimizer == "AdamW":
        optimizer = AdamW(optimizer_grouped_parameters, lr=args.learning_rate, eps=args.adam_epsilon)
    elif args.optimizer == "Adafactor":
        optimizer = Adafactor(optimizer_grouped_parameters, lr=None, eps=(1e-30, 1e-3), clip_threshold=1.0, scale_parameter=True, relative_step=True, warmup_init=True)
    elif args.optimizer == "Adafactor-srwf":
        # 此种情形无需scheduler
        optimizer = Adafactor(optimizer_grouped_parameters, lr=args.learning_rate, eps=(1e-30, 1e-3), clip_threshold=1.0, scale_parameter=False, relative_step=False, warmup_init=False)

    # 学习率模式选择，2022/08/30 新增 by Yirong Chen
    if args.scheduler == "get_linear_schedule_with_warmup":
        scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=int(args.warm_up_ratio*t_total), num_training_steps=t_total)
    elif args.scheduler == "get_constant_schedule_with_warmup":
        scheduler = get_constant_schedule_with_warmup(optimizer, num_warmup_steps=int(args.warm_up_ratio*t_total))
    elif args.scheduler == "get_constant_schedule":
        scheduler = get_constant_schedule(optimizer)
    elif args.scheduler == "get_cosine_schedule_with_warmup":
        scheduler = get_cosine_schedule_with_warmup(optimizer, num_warmup_steps=int(args.warm_up_ratio*t_total), num_training_steps=t_total, num_cycles=0.5)
    elif args.scheduler == "get_adafactor_schedule":
        # 配合Adafactor进行使用
        # min_step = 1e-6 * param_state["step"] if param_group["warmup_init"] else 1e-2
        # KeyError: 'step'
        scheduler = get_adafactor_schedule(optimizer, initial_lr=args.learning_rate)
    elif args.scheduler == "no_schedule":
        logger.info("***** Not use any scheduler!!! *****")

    # Check if saved optimizer or scheduler states exist
    if (args.model_name_or_path and os.path.isfile(os.path.join(args.model_name_or_path, "optimizer.pt")) and os.path.isfile(os.path.join(args.model_name_or_path, "scheduler.pt"))):
        optimizer.load_state_dict(torch.load(os.path.join(args.model_name_or_path, "optimizer.pt")))
        if args.scheduler != "no_schedule":
            scheduler.load_state_dict(torch.load(os.path.join(args.model_name_or_path, "scheduler.pt")))

    if args.fp16:
        try:
            from apex import amp
        except ImportError:
            raise ImportError("Please install apex from https://www.github.com/nvidia/apex to use fp16 training.")
        model, optimizer = amp.initialize(model, optimizer, opt_level=args.fp16_opt_level)
    elif args.autocast:
        # 混合精度训练
        # 参考: https://pytorch.org/docs/1.9.0/amp.html?highlight=torch%20cuda%20amp%20gradscaler
        #       https://pytorch.org/docs/1.9.0/notes/amp_examples.html#amp-examples
        scaler = torch.cuda.amp.GradScaler()  # pytorch版本要求：1.6+

    # multi-gpu training (should be after apex fp16 initialization)
    if args.n_gpu > 1 and args.local_rank == -1 and not args.model_parallel:
        model = torch.nn.DataParallel(model)
    # Distributed training (should be after apex fp16 initialization)
    if args.local_rank != -1 and not args.model_parallel:
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.local_rank], output_device=args.local_rank,
                                                          find_unused_parameters=not args.not_find_unused_parameters)
    # Train!
    logger.info("***** Running training *****")
    logger.info("  Num examples = %d", len(train_dataset))
    logger.info("  Num Epochs = %d", args.num_train_epochs)
    logger.info("  Train batch size = %d", args.train_batch_size)
    logger.info("  Instantaneous batch size per GPU = %d", args.per_gpu_train_batch_size)
    logger.info("  Total train batch size (w. parallel, distributed & accumulation) = %d",
                   args.train_batch_size * args.gradient_accumulation_steps * (torch.distributed.get_world_size() if args.local_rank != -1 else 1))
    logger.info("  Gradient Accumulation steps = %d", args.gradient_accumulation_steps)
    logger.info("  Total optimization steps = %d", t_total)

    global_step = 0
    epochs_trained = 0
    steps_trained_in_current_epoch = 0

    # Check if continuing training from a checkpoint
    if args.model_name_or_path and os.path.exists(args.model_name_or_path):
        try:
            # set global_step to gobal_step of last saved checkpoint from model path
            checkpoint_suffix = args.model_name_or_path.split("-")[-1].split("/")[0]
            global_step = int(checkpoint_suffix)
            epochs_trained = global_step // (len(train_dataloader) // args.gradient_accumulation_steps)
            steps_trained_in_current_epoch = global_step % (len(train_dataloader) // args.gradient_accumulation_steps)
            logger.info("  Continuing training from checkpoint, will skip to saved global_step")
            logger.info("  Continuing training from epoch %d", epochs_trained)
            logger.info("  Continuing training from global step %d", global_step)
            logger.info("  Will skip the first %d steps in the first epoch", steps_trained_in_current_epoch)
        except ValueError:
            logger.info("  Starting fine-tuning.")

    tr_loss, logging_loss = 0.0, 0.0
    tr_masked_lm_loss, logging_masked_lm_loss = 0.0, 0.0
    tr_nocoref_loss, logging_nocoref_loss = 0.0, 0.0
    tr_misc_loss, logging_misc_loss = 0.0, 0.0
    tr_disam_loss, logging_disam_loss = 0.0, 0.0
    tr_retrieval_loss, logging_retrieval_loss = 0.0, 0.0
    tr_kl_loss, logging_kl_loss = 0.0, 0.0
    if args.model_type == 'mt-bart_joint_disam_coref' or args.model_type == 'mt-bart_add_intent':
        tr_user_act_loss, logging_user_act_loss = 0.0, 0.0
        tr_system_act_loss, logging_system_act_loss = 0.0, 0.0
        tr_disamb_and_coref_loss, logging_disamb_and_coref_loss = 0.0, 0.0
        tr_disam_and_coref_candi_loss, logging_disam_and_coref_candi_loss = 0.0, 0.0

    model.zero_grad()
    train_iterator = trange(epochs_trained, int(args.num_train_epochs), desc="Epoch", disable=args.local_rank not in [-1, 0])
    setup_seed(args.seed, args.n_gpu)
    
    # do R-Drop, so we need the logits for calculate!
    if args.do_rdrop:
        logger.info("Run training the model with R-Drop!")
        return_dict=True
    else:
        return_dict=False

    for epoch_i in train_iterator:
        epoch_iterator = tqdm(train_dataloader, desc="Iteration", disable=args.local_rank not in [-1, 0])
        for step, batch in enumerate(epoch_iterator):
            # Skip past any already trained steps if resuming training
            if steps_trained_in_current_epoch > 0:
                steps_trained_in_current_epoch -= 1
                continue
            
            # 输入数据规范化
            if args.model_parallel:
                enc_input = batch[0].to(model.encoder.first_device)
                enc_attention_mask = batch[1].to(model.encoder.first_device)
                decoder_input = batch[2].to(model.encoder.first_device)
                decoder_attention_mask = batch[3].to(model.encoder.first_device)
                boxes = batch[4] # batch, num_obj_per_line, 6
                misc = batch[5]  # batch, num_obj_per_line, dict
                nocoref = batch[6]
                disambiguation_labels = batch[7].to(model.encoder.first_device)
                response = batch[8].to(model.encoder.first_device)
                response_attention_mask = batch[9].to(model.encoder.first_device)
                if args.model_type == 'mt-flava':
                    image_feature = batch[10].to(model.encoder.first_device)
                    enc_token_type_ids = batch[11].to(model.encoder.first_device)
                elif args.model_type == 'mt-bart_joint_disam_coref' or args.model_type == 'mt-bart_add_intent':
                    user_act_labels = batch[10].to(model.encoder.first_device)
                    system_act_labels = batch[11].to(model.encoder.first_device)
                elif args.model_type == 'gen-ofa' or args.model_type == 'mt-ofa':
                    image_feature = batch[10].to(model.encoder.first_device)
                elif args.model_type == 'mt-bart-attr':
                    obj_ids_per_line = batch[10]
                    object_attr_input_ids = batch[11]

            else:
                enc_input = batch[0].to(args.device)
                enc_attention_mask = batch[1].to(args.device)
                decoder_input = batch[2].to(args.device)
                decoder_attention_mask = batch[3].to(args.device)
                boxes = batch[4] # batch, num_obj_per_line, 6
                misc = batch[5]  # batch, num_obj_per_line, dict
                nocoref = batch[6]
                disambiguation_labels = batch[7].to(args.device)
                response = batch[8].to(args.device)
                response_attention_mask = batch[9].to(args.device)
                if args.model_type == 'mt-flava':
                    image_feature = batch[10].to(args.device)
                    enc_token_type_ids = batch[11].to(args.device)
                elif args.model_type == 'mt-bart_joint_disam_coref' or args.model_type == 'mt-bart_add_intent':
                    user_act_labels = batch[10].to(args.device)
                    system_act_labels = batch[11].to(args.device)
                elif args.model_type == 'gen-ofa' or args.model_type == 'mt-ofa':
                    image_feature = batch[10].to(args.device)
                elif args.model_type == 'mt-bart-attr':
                    obj_ids_per_line = batch[10]
                    object_attr_input_ids = batch[11]
            
            # 对labels的特殊处理
            # 修复原先的bug: labels=decoder_input[:, 1:].contiguous(), 此时pad填充的是pad_token_id，并非-100，计算loss时会把pad部分也计算进去
            lm_labels=decoder_input[:, 1:].contiguous() # 哪怕输入是 <pad> .... <seq> 也没有影响，因为第一个token不作为labels
            lm_labels[lm_labels==tokenizer.pad_token_id]=-100 # 将labels的所有pad_token_id替换为-100，因为CrossEntropyLoss默认的ignore_index=-100
            lm_labels.contiguous()

            model.train()
            # 根据模型类型的不同采用不同的输入，进行模型前向计算
            if args.model_type == 'mt-flava':
                # 所有需要场景图的模型
                model_outputs = model(
                            input_ids=enc_input,
                            pixel_values=image_feature, # (batch_size, 3, 224, 224)
                            attention_mask=enc_attention_mask,
                            token_type_ids=enc_token_type_ids,
                            boxes=boxes,
                            misc=misc,
                            nocoref=nocoref,
                            response=response,
                            response_attention_mask=response_attention_mask,
                            disambiguation_labels=disambiguation_labels,
                            do_retrieval=args.do_retrieval,
                            return_dict=return_dict,
                            alpha_masked_lm_loss= args.alpha_masked_lm_loss, # 新增，生成回复的损失占比
                            alpha_nocoref_loss= args.alpha_nocoref_loss, # 新增，二分类任务判断是否存在共指消解的损失占比
                            alpha_misc_loss= args.alpha_misc_loss, # 新增，属性识别损失占比
                            alpha_disam_loss= args.alpha_disam_loss, # 新增，歧义句子分类损失占比
                            alpha_retrieval_loss= args.alpha_retrieval_loss, # 新增，检索回复损失占比
                            alpha_disamb_candi_loss= args.alpha_disamb_candi_loss, # 新增，识别对象是否为歧义候选的损失占比
                            alpha_coref_loss= args.alpha_coref_loss,
                            use_focal_loss=args.use_focal_loss,
                            focal_loss_gamma=args.focal_loss_gamma,
                            focal_loss_alpha=args.focal_loss_alpha)
                if args.do_rdrop:
                    model_outputs2 = model(
                            input_ids=enc_input,
                            pixel_values=image_feature, # (batch_size, 3, 224, 224)
                            attention_mask=enc_attention_mask,
                            token_type_ids=enc_token_type_ids,
                            boxes=boxes,
                            misc=misc,
                            nocoref=nocoref,
                            response=response,
                            response_attention_mask=response_attention_mask,
                            disambiguation_labels=disambiguation_labels,
                            do_retrieval=args.do_retrieval,
                            return_dict=return_dict,
                            alpha_masked_lm_loss= args.alpha_masked_lm_loss, # 新增，生成回复的损失占比
                            alpha_nocoref_loss= args.alpha_nocoref_loss, # 新增，二分类任务判断是否存在共指消解的损失占比
                            alpha_misc_loss= args.alpha_misc_loss, # 新增，属性识别损失占比
                            alpha_disam_loss= args.alpha_disam_loss, # 新增，歧义句子分类损失占比
                            alpha_retrieval_loss= args.alpha_retrieval_loss, # 新增，检索回复损失占比
                            alpha_disamb_candi_loss= args.alpha_disamb_candi_loss, # 新增，识别对象是否为歧义候选的损失占比
                            alpha_coref_loss= args.alpha_coref_loss,
                            use_focal_loss=args.use_focal_loss,
                            focal_loss_gamma=args.focal_loss_gamma,
                            focal_loss_alpha=args.focal_loss_alpha)

            elif args.model_type == 'mt-bart_joint_disam_coref':
                model_outputs = model(
                            input_ids=enc_input,
                            attention_mask=enc_attention_mask,
                            decoder_input_ids=decoder_input[:, :-1],
                            decoder_attention_mask=decoder_attention_mask[:, :-1],
                            labels=lm_labels,
                            boxes=boxes,
                            misc=misc,
                            nocoref=nocoref,
                            response=response,
                            response_attention_mask=response_attention_mask,
                            disam_and_coref_labels=disambiguation_labels,
                            user_act_labels=user_act_labels,
                            system_act_labels=system_act_labels,
                            do_retrieval=args.do_retrieval,
                            return_dict=return_dict,
                            alpha_masked_lm_loss= args.alpha_masked_lm_loss, # 新增，生成回复的损失占比
                            alpha_misc_loss= args.alpha_misc_loss, # 新增，属性识别损失占比
                            alpha_retrieval_loss= args.alpha_retrieval_loss, # 新增，检索回复损失占比
                            alpha_user_act_loss= args.alpha_user_act_loss, # 新增，识别用户意图的损失占比
                            alpha_system_act_loss= args.alpha_system_act_loss, # 新增，预测系统意图的损失占比
                            alpha_disamb_and_coref_loss=args.alpha_disamb_and_coref_loss, # 新增，预测当前对话是否存在歧义候选或者多模态共指现象的损失占比
                            alpha_disam_and_coref_candi_loss= args.alpha_disam_and_coref_candi_loss, # 新增，判断object是否属于歧义候选或者多模态共指的损失占比
                            use_focal_loss=args.use_focal_loss,
                            focal_loss_gamma=args.focal_loss_gamma)
                if args.do_rdrop:
                    model_outputs2 = model(
                            input_ids=enc_input,
                            attention_mask=enc_attention_mask,
                            decoder_input_ids=decoder_input[:, :-1],
                            decoder_attention_mask=decoder_attention_mask[:, :-1],
                            labels=lm_labels,
                            boxes=boxes,
                            misc=misc,
                            nocoref=nocoref,
                            response=response,
                            response_attention_mask=response_attention_mask,
                            disam_and_coref_labels=disambiguation_labels,
                            user_act_labels=user_act_labels,
                            system_act_labels=system_act_labels,
                            do_retrieval=args.do_retrieval,
                            return_dict=return_dict,
                            alpha_masked_lm_loss= args.alpha_masked_lm_loss, # 新增，生成回复的损失占比
                            alpha_misc_loss= args.alpha_misc_loss, # 新增，属性识别损失占比
                            alpha_retrieval_loss= args.alpha_retrieval_loss, # 新增，检索回复损失占比
                            alpha_user_act_loss= args.alpha_user_act_loss, # 新增，识别用户意图的损失占比
                            alpha_system_act_loss= args.alpha_system_act_loss, # 新增，预测系统意图的损失占比
                            alpha_disamb_and_coref_loss=args.alpha_disamb_and_coref_loss, # 新增，预测当前对话是否存在歧义候选或者多模态共指现象的损失占比
                            alpha_disam_and_coref_candi_loss= args.alpha_disam_and_coref_candi_loss, # 新增，判断object是否属于歧义候选或者多模态共指的损失占比
                            use_focal_loss=args.use_focal_loss,
                            focal_loss_gamma=args.focal_loss_gamma)

            elif args.model_type == 'mt-bart_add_intent':
                model_outputs = model(
                            input_ids=enc_input,
                            attention_mask=enc_attention_mask,
                            decoder_input_ids=decoder_input[:, :-1],
                            decoder_attention_mask=decoder_attention_mask[:, :-1],
                            labels=lm_labels,
                            boxes=boxes,
                            misc=misc,
                            nocoref=nocoref,
                            response=response,
                            response_attention_mask=response_attention_mask,
                            disambiguation_labels=disambiguation_labels,
                            user_act_labels=user_act_labels,
                            system_act_labels=system_act_labels,
                            do_retrieval=args.do_retrieval,
                            return_dict=return_dict,
                            alpha_masked_lm_loss= args.alpha_masked_lm_loss, # 新增，生成回复的损失占比
                            alpha_nocoref_loss= args.alpha_nocoref_loss, # 新增，二分类任务判断是否存在共指消解的损失占比
                            alpha_misc_loss= args.alpha_misc_loss, # 新增，属性识别损失占比
                            alpha_disam_loss= args.alpha_disam_loss, # 新增，歧义句子分类损失占比
                            alpha_retrieval_loss= args.alpha_retrieval_loss, # 新增，检索回复损失占比
                            alpha_disamb_candi_loss= args.alpha_disamb_candi_loss, # 新增，识别对象是否为歧义候选的损失占比
                            alpha_coref_loss= args.alpha_coref_loss,
                            alpha_user_act_loss= args.alpha_user_act_loss, # 新增，识别用户意图的损失占比
                            alpha_system_act_loss= args.alpha_system_act_loss, # 新增，预测系统意图的损失占比
                            use_focal_loss=args.use_focal_loss,
                            focal_loss_gamma=args.focal_loss_gamma)
                if args.do_rdrop:
                    model_outputs2 = model(
                            input_ids=enc_input,
                            attention_mask=enc_attention_mask,
                            decoder_input_ids=decoder_input[:, :-1],
                            decoder_attention_mask=decoder_attention_mask[:, :-1],
                            labels=lm_labels,
                            boxes=boxes,
                            misc=misc,
                            nocoref=nocoref,
                            response=response,
                            response_attention_mask=response_attention_mask,
                            disambiguation_labels=disambiguation_labels,
                            user_act_labels=user_act_labels,
                            system_act_labels=system_act_labels,
                            do_retrieval=args.do_retrieval,
                            return_dict=return_dict,
                            alpha_masked_lm_loss= args.alpha_masked_lm_loss, # 新增，生成回复的损失占比
                            alpha_nocoref_loss= args.alpha_nocoref_loss, # 新增，二分类任务判断是否存在共指消解的损失占比
                            alpha_misc_loss= args.alpha_misc_loss, # 新增，属性识别损失占比
                            alpha_disam_loss= args.alpha_disam_loss, # 新增，歧义句子分类损失占比
                            alpha_retrieval_loss= args.alpha_retrieval_loss, # 新增，检索回复损失占比
                            alpha_disamb_candi_loss= args.alpha_disamb_candi_loss, # 新增，识别对象是否为歧义候选的损失占比
                            alpha_coref_loss= args.alpha_coref_loss,
                            alpha_user_act_loss= args.alpha_user_act_loss, # 新增，识别用户意图的损失占比
                            alpha_system_act_loss= args.alpha_system_act_loss, # 新增，预测系统意图的损失占比
                            use_focal_loss=args.use_focal_loss,
                            focal_loss_gamma=args.focal_loss_gamma)

            elif args.model_type == 'gen-ofa':
                model_outputs = model(
                            input_ids=enc_input,
                            patch_images=image_feature,
                            decoder_input_ids=decoder_input[:, :-1],
                            attention_mask=decoder_attention_mask[:, :-1],  # attention mask for decoding.
                            labels=lm_labels,
                            boxes=boxes,
                            misc=misc,
                            sample_patch_num=args.sample_patch_num)
                if args.do_rdrop:
                    model_outputs2 = model(
                            input_ids=enc_input,
                            patch_images=image_feature,
                            decoder_input_ids=decoder_input[:, :-1],
                            attention_mask=decoder_attention_mask[:, :-1],  # attention mask for decoding.
                            labels=lm_labels,
                            boxes=boxes,
                            misc=misc,
                            sample_patch_num=args.sample_patch_num)
            elif args.model_type == 'mt-ofa':
                model_outputs = model(
                            input_ids=enc_input,
                            patch_images=image_feature,
                            decoder_input_ids=decoder_input[:, :-1],
                            attention_mask=decoder_attention_mask[:, :-1],  # attention mask for decoding.
                            labels=lm_labels,
                            boxes=boxes,
                            misc=misc,
                            nocoref=nocoref,
                            disambiguation_labels=disambiguation_labels,
                            return_dict=return_dict,
                            alpha_masked_lm_loss= args.alpha_masked_lm_loss, # 新增，生成回复的损失占比
                            alpha_nocoref_loss= args.alpha_nocoref_loss, # 新增，二分类任务判断是否存在共指消解的损失占比
                            alpha_misc_loss= args.alpha_misc_loss, # 新增，属性识别损失占比
                            alpha_disam_loss= args.alpha_disam_loss, # 新增，歧义句子分类损失占比
                            alpha_retrieval_loss= args.alpha_retrieval_loss, # 新增，检索回复损失占比
                            alpha_disamb_candi_loss= args.alpha_disamb_candi_loss, # 新增，识别对象是否为歧义候选的损失占比
                            alpha_coref_loss= args.alpha_coref_loss,
                            use_focal_loss=args.use_focal_loss,
                            focal_loss_gamma=args.focal_loss_gamma,
                            focal_loss_alpha=args.focal_loss_alpha,
                            sample_patch_num=args.sample_patch_num
                            )
                if args.do_rdrop:
                    model_outputs2 = model(
                            input_ids=enc_input,
                            patch_images=image_feature,
                            decoder_input_ids=decoder_input[:, :-1],
                            attention_mask=decoder_attention_mask[:, :-1],  # attention mask for decoding.
                            labels=lm_labels,
                            boxes=boxes,
                            misc=misc,
                            nocoref=nocoref,
                            disambiguation_labels=disambiguation_labels,
                            return_dict=return_dict,
                            alpha_masked_lm_loss= args.alpha_masked_lm_loss, # 新增，生成回复的损失占比
                            alpha_nocoref_loss= args.alpha_nocoref_loss, # 新增，二分类任务判断是否存在共指消解的损失占比
                            alpha_misc_loss= args.alpha_misc_loss, # 新增，属性识别损失占比
                            alpha_disam_loss= args.alpha_disam_loss, # 新增，歧义句子分类损失占比
                            alpha_retrieval_loss= args.alpha_retrieval_loss, # 新增，检索回复损失占比
                            alpha_disamb_candi_loss= args.alpha_disamb_candi_loss, # 新增，识别对象是否为歧义候选的损失占比
                            alpha_coref_loss= args.alpha_coref_loss,
                            use_focal_loss=args.use_focal_loss,
                            focal_loss_gamma=args.focal_loss_gamma,
                            focal_loss_alpha=args.focal_loss_alpha,
                            sample_patch_num=args.sample_patch_num
                            )

            elif args.model_type == 'mt-bart-attr':
                model_outputs = model(
                            input_ids=enc_input,
                            attention_mask=enc_attention_mask,
                            decoder_input_ids=decoder_input[:, :-1],
                            decoder_attention_mask=decoder_attention_mask[:, :-1],
                            labels=lm_labels,
                            boxes=boxes,
                            misc=misc,
                            nocoref=nocoref,
                            response=response,
                            response_attention_mask=response_attention_mask,
                            disambiguation_labels=disambiguation_labels,
                            do_retrieval=args.do_retrieval,
                            return_dict=return_dict,
                            alpha_masked_lm_loss= args.alpha_masked_lm_loss, # 新增，生成回复的损失占比
                            alpha_nocoref_loss= args.alpha_nocoref_loss, # 新增，二分类任务判断是否存在共指消解的损失占比
                            alpha_misc_loss= args.alpha_misc_loss, # 新增，属性识别损失占比
                            alpha_disam_loss= args.alpha_disam_loss, # 新增，歧义句子分类损失占比
                            alpha_retrieval_loss= args.alpha_retrieval_loss, # 新增，检索回复损失占比
                            alpha_disamb_candi_loss= args.alpha_disamb_candi_loss, # 新增，识别对象是否为歧义候选的损失占比
                            alpha_coref_loss= args.alpha_coref_loss,
                            use_focal_loss=args.use_focal_loss,
                            focal_loss_gamma=args.focal_loss_gamma,
                            focal_loss_alpha=args.focal_loss_alpha,
                            object_attr_input_ids=object_attr_input_ids,
                            use_non_visual_attrs=True)    
                if args.do_rdrop:
                    model_outputs2 = model(
                            input_ids=enc_input,
                            attention_mask=enc_attention_mask,
                            decoder_input_ids=decoder_input[:, :-1],
                            decoder_attention_mask=decoder_attention_mask[:, :-1],
                            labels=lm_labels,
                            boxes=boxes,
                            misc=misc,
                            nocoref=nocoref,
                            response=response,
                            response_attention_mask=response_attention_mask,
                            disambiguation_labels=disambiguation_labels,
                            do_retrieval=args.do_retrieval,
                            return_dict=return_dict,
                            alpha_masked_lm_loss= args.alpha_masked_lm_loss, # 新增，生成回复的损失占比
                            alpha_nocoref_loss= args.alpha_nocoref_loss, # 新增，二分类任务判断是否存在共指消解的损失占比
                            alpha_misc_loss= args.alpha_misc_loss, # 新增，属性识别损失占比
                            alpha_disam_loss= args.alpha_disam_loss, # 新增，歧义句子分类损失占比
                            alpha_retrieval_loss= args.alpha_retrieval_loss, # 新增，检索回复损失占比
                            alpha_disamb_candi_loss= args.alpha_disamb_candi_loss, # 新增，识别对象是否为歧义候选的损失占比
                            alpha_coref_loss= args.alpha_coref_loss,
                            use_focal_loss=args.use_focal_loss,
                            focal_loss_gamma=args.focal_loss_gamma,
                            focal_loss_alpha=args.focal_loss_alpha,
                            object_attr_input_ids=object_attr_input_ids,
                            use_non_visual_attrs=True)  

            else:
                model_outputs = model(
                            input_ids=enc_input,
                            attention_mask=enc_attention_mask,
                            decoder_input_ids=decoder_input[:, :-1],
                            decoder_attention_mask=decoder_attention_mask[:, :-1],
                            labels=lm_labels,
                            boxes=boxes,
                            misc=misc,
                            nocoref=nocoref,
                            response=response,
                            response_attention_mask=response_attention_mask,
                            disambiguation_labels=disambiguation_labels,
                            do_retrieval=args.do_retrieval,
                            return_dict=return_dict,
                            alpha_masked_lm_loss= args.alpha_masked_lm_loss, # 新增，生成回复的损失占比
                            alpha_nocoref_loss= args.alpha_nocoref_loss, # 新增，二分类任务判断是否存在共指消解的损失占比
                            alpha_misc_loss= args.alpha_misc_loss, # 新增，属性识别损失占比
                            alpha_disam_loss= args.alpha_disam_loss, # 新增，歧义句子分类损失占比
                            alpha_retrieval_loss= args.alpha_retrieval_loss, # 新增，检索回复损失占比
                            alpha_disamb_candi_loss= args.alpha_disamb_candi_loss, # 新增，识别对象是否为歧义候选的损失占比
                            alpha_coref_loss= args.alpha_coref_loss,
                            use_focal_loss=args.use_focal_loss,
                            focal_loss_gamma=args.focal_loss_gamma,
                            focal_loss_alpha=args.focal_loss_alpha)    
                if args.do_rdrop:
                    model_outputs2 = model(
                            input_ids=enc_input,
                            attention_mask=enc_attention_mask,
                            decoder_input_ids=decoder_input[:, :-1],
                            decoder_attention_mask=decoder_attention_mask[:, :-1],
                            labels=lm_labels,
                            boxes=boxes,
                            misc=misc,
                            nocoref=nocoref,
                            response=response,
                            response_attention_mask=response_attention_mask,
                            disambiguation_labels=disambiguation_labels,
                            do_retrieval=args.do_retrieval,
                            return_dict=return_dict,
                            alpha_masked_lm_loss= args.alpha_masked_lm_loss, # 新增，生成回复的损失占比
                            alpha_nocoref_loss= args.alpha_nocoref_loss, # 新增，二分类任务判断是否存在共指消解的损失占比
                            alpha_misc_loss= args.alpha_misc_loss, # 新增，属性识别损失占比
                            alpha_disam_loss= args.alpha_disam_loss, # 新增，歧义句子分类损失占比
                            alpha_retrieval_loss= args.alpha_retrieval_loss, # 新增，检索回复损失占比
                            alpha_disamb_candi_loss= args.alpha_disamb_candi_loss, # 新增，识别对象是否为歧义候选的损失占比
                            alpha_coref_loss= args.alpha_coref_loss,
                            use_focal_loss=args.use_focal_loss,
                            focal_loss_gamma=args.focal_loss_gamma,
                            focal_loss_alpha=args.focal_loss_alpha)

            # 计算Loss
            if args.do_rdrop:
                if args.model_type == 'mt-bart_joint_disam_coref':
                    loss = 0.5*(model_outputs.loss + model_outputs2.loss)
                    masked_lm_loss = 0.5*(model_outputs.masked_lm_loss + model_outputs2.masked_lm_loss)
                    retrieval_loss = 0.5*(model_outputs.retrieval_loss + model_outputs2.retrieval_loss)
                    user_act_loss = 0.5*(model_outputs.user_act_loss + model_outputs2.user_act_loss)
                    system_act_loss = 0.5*(model_outputs.system_act_loss + model_outputs2.system_act_loss)
                    disamb_and_coref_loss = 0.5*(model_outputs.disamb_and_coref_loss + model_outputs2.disamb_and_coref_loss)
                    disam_and_coref_candi_loss = 0.5*(model_outputs.disam_and_coref_candi_loss + model_outputs2.disam_and_coref_candi_loss)
                    misc_loss = 0.5*(model_outputs.misc_loss + model_outputs2.misc_loss)
                    # 计算KL Loss
                    if args.alpha_masked_lm_loss != 0:
                        kl_loss_lm_logits = compute_kl_loss(model_outputs.logits, model_outputs2.logits, attention_mask=decoder_attention_mask[:, 1:], reduction=args.rdrop_reduction)
                    else:
                        kl_loss_lm_logits = torch.tensor(0.0).to(enc_input.device)
                    
                    if args.alpha_user_act_loss != 0:    
                        kl_loss_user_act = compute_kl_loss(model_outputs.user_act_logits, model_outputs2.user_act_logits, reduction=args.rdrop_reduction)
                    else:
                        kl_loss_user_act = torch.tensor(0.0).to(enc_input.device)
                    
                    if args.alpha_system_act_loss != 0:    
                        kl_loss_system_act = compute_kl_loss(model_outputs.system_act_logits, model_outputs2.system_act_logits, reduction=args.rdrop_reduction)
                    else:
                        kl_loss_system_act = torch.tensor(0.0).to(enc_input.device)
                    
                    if args.alpha_disamb_and_coref_loss != 0:    
                        kl_loss_disamb_and_coref = compute_kl_loss(model_outputs.disamb_and_coref_logits, model_outputs2.disamb_and_coref_logits, reduction=args.rdrop_reduction)
                    else:
                        kl_loss_disamb_and_coref= torch.tensor(0.0).to(enc_input.device)

                    # 对于歧义候选和多模态共指消解联合任务
                    kl_loss_disam_and_coref_candi = torch.tensor(0.0).to(enc_input.device)
                    for b_idx in range(len(enc_input)):
                        # 对象数目不固定，取总平均
                        if args.alpha_disam_and_coref_candi_loss != 0:
                            kl_loss_disam_and_coref_candi += compute_kl_loss(model_outputs.enc_head_results[b_idx][0], model_outputs2.enc_head_results[b_idx][0], reduction="mean")
                    if args.rdrop_reduction=="mean" or "batchmean":
                        kl_loss_disam_and_coref_candi /= len(enc_input)

                    kl_loss = kl_loss_lm_logits + kl_loss_user_act + kl_loss_system_act + kl_loss_disamb_and_coref + kl_loss_disam_and_coref_candi
                    loss = loss + args.alpha_rdrop*kl_loss

                elif args.model_type == 'gen-ofa':
                    loss = 0.5*(model_outputs.loss + model_outputs2.loss)
                    kl_loss = compute_kl_loss(model_outputs.logits, model_outputs2.logits, attention_mask=decoder_attention_mask[:, 1:], reduction=args.rdrop_reduction)
                    loss = loss + args.alpha_rdrop*kl_loss

                else:
                    loss = 0.5*(model_outputs.loss + model_outputs2.loss)
                    masked_lm_loss = 0.5*(model_outputs.masked_lm_loss + model_outputs2.masked_lm_loss)
                    nocoref_loss = 0.5*(model_outputs.nocoref_loss + model_outputs2.nocoref_loss)
                    misc_loss = 0.5*(model_outputs.misc_loss + model_outputs2.misc_loss)
                    disam_loss = 0.5*(model_outputs.disam_loss + model_outputs2.disam_loss)
                    retrieval_loss = 0.5*(model_outputs.retrieval_loss + model_outputs2.retrieval_loss)

                    if args.model_type == 'mt-bart_add_intent':
                        user_act_loss = 0.5*(model_outputs.user_act_loss + model_outputs2.user_act_loss)
                        system_act_loss = 0.5*(model_outputs.system_act_loss + model_outputs2.system_act_loss)
                        if args.alpha_user_act_loss != 0:    
                            kl_loss_user_act = compute_kl_loss(model_outputs.user_act_logits, model_outputs2.user_act_logits, reduction=args.rdrop_reduction)
                        else:
                            kl_loss_user_act = torch.tensor(0.0).to(enc_input.device)
                        if args.alpha_system_act_loss != 0:    
                            kl_loss_system_act = compute_kl_loss(model_outputs.system_act_logits, model_outputs2.system_act_logits, reduction=args.rdrop_reduction)
                        else:
                            kl_loss_system_act = torch.tensor(0.0).to(enc_input.device)

                    # 计算多任务的KL Loss
                    # compute_kl_loss(p, q, pad_mask=None, attention_mask=None, reduction="batchmean")
                    if args.alpha_masked_lm_loss != 0:
                        kl_loss_lm_logits = compute_kl_loss(model_outputs.logits, model_outputs2.logits, attention_mask=decoder_attention_mask[:, 1:], reduction=args.rdrop_reduction)
                    else:
                        kl_loss_lm_logits = torch.tensor(0.0).to(enc_input.device)
                    if args.alpha_disam_loss != 0:    
                        kl_loss_disamb = compute_kl_loss(model_outputs.disambiguation_logits, model_outputs2.disambiguation_logits, reduction=args.rdrop_reduction)
                    else:
                        kl_loss_disamb = torch.tensor(0.0).to(enc_input.device)
                    if args.alpha_nocoref_loss != 0:
                        kl_loss_nocoref = compute_kl_loss(model_outputs.nocoref_logits, model_outputs2.nocoref_logits, reduction=args.rdrop_reduction)
                    else:
                        kl_loss_nocoref = torch.tensor(0.0).to(enc_input.device)
                    # 对于歧义候选识别任务和多模态共指消解任务
                    kl_loss_disam_candi = torch.tensor(0.0).to(enc_input.device)
                    kl_loss_coref_candi = torch.tensor(0.0).to(enc_input.device)
                    for b_idx in range(len(enc_input)):
                        # 对象数目不固定，取总平均
                        if args.alpha_disamb_candi_loss != 0:
                            kl_loss_disam_candi += compute_kl_loss(model_outputs.enc_head_results[b_idx][0], model_outputs2.enc_head_results[b_idx][0], reduction="mean")
                        if args.alpha_coref_loss != 0:
                            kl_loss_coref_candi += compute_kl_loss(model_outputs.enc_head_results[b_idx][1], model_outputs2.enc_head_results[b_idx][1], reduction="mean")
                    if args.rdrop_reduction=="mean" or "batchmean":
                        kl_loss_disam_candi /= len(enc_input)
                        kl_loss_coref_candi /= len(enc_input)

                    kl_loss = kl_loss_lm_logits + kl_loss_disamb + kl_loss_nocoref + kl_loss_disam_candi + kl_loss_coref_candi
                    if args.model_type == 'mt-bart_add_intent':
                        kl_loss = kl_loss + kl_loss_user_act + kl_loss_system_act
                    loss = loss + args.alpha_rdrop*kl_loss

            else: # No Rdrop
                if args.model_type == 'mt-bart_joint_disam_coref':
                    # [loss,masked_lm_loss,retrieval_loss,user_act_loss,system_act_loss,disamb_and_coref_loss,disam_and_coref_candi_loss,misc_loss]
                    loss = model_outputs[0]
                    masked_lm_loss = model_outputs[1]
                    retrieval_loss = model_outputs[2]
                    user_act_loss = model_outputs[3]
                    system_act_loss = model_outputs[4]
                    disamb_and_coref_loss = model_outputs[5]
                    disam_and_coref_candi_loss = model_outputs[6]
                    misc_loss = model_outputs[7]
                    kl_loss = torch.tensor(0.0).to(enc_input.device)
                elif args.model_type == 'gen-ofa':
                    loss = model_outputs.loss
                else:
                    loss = model_outputs[0] # [loss,masked_lm_loss,nocoref_loss,misc_loss,disam_loss,retrieval_loss]
                    masked_lm_loss = model_outputs[1]
                    nocoref_loss = model_outputs[2]
                    misc_loss = model_outputs[3]
                    disam_loss = model_outputs[4]
                    retrieval_loss = model_outputs[5]
                    if args.model_type == 'mt-bart_add_intent':
                        user_act_loss = model_outputs[6]
                        system_act_loss = model_outputs[7]
                    kl_loss = torch.tensor(0.0).to(enc_input.device)

            if args.n_gpu > 1:
                if args.model_type == 'mt-bart_joint_disam_coref':
                    loss = loss.mean()  # mean() to average on multi-gpu parallel training
                    masked_lm_loss = masked_lm_loss.mean()
                    user_act_loss = user_act_loss.mean()
                    system_act_loss = system_act_loss.mean()
                    disamb_and_coref_loss = disamb_and_coref_loss.mean()
                    disam_and_coref_candi_loss = disam_and_coref_candi_loss.mean()
                    misc_loss = misc_loss.mean()
                    kl_loss = kl_loss.mean()
                    if isinstance(retrieval_loss,torch.Tensor):
                        retrieval_loss = retrieval_loss.mean()
                elif args.model_type == 'gen-ofa':
                    loss = loss.mean()  # mean() to average on multi-gpu parallel training
                else:
                    loss = loss.mean()  # mean() to average on multi-gpu parallel training
                    masked_lm_loss = masked_lm_loss.mean()
                    nocoref_loss = nocoref_loss.mean()
                    misc_loss = misc_loss.mean()
                    disam_loss = disam_loss.mean()
                    if args.model_type == 'mt-bart_add_intent':
                        user_act_loss = user_act_loss.mean()
                        system_act_loss = system_act_loss.mean()
                    kl_loss = kl_loss.mean()
                    if isinstance(retrieval_loss,torch.Tensor):
                        retrieval_loss = retrieval_loss.mean()

            if args.gradient_accumulation_steps > 1:
                if args.model_type == 'mt-bart_joint_disam_coref':
                    loss = loss / args.gradient_accumulation_steps
                    masked_lm_loss = masked_lm_loss / args.gradient_accumulation_steps
                    user_act_loss = user_act_loss / args.gradient_accumulation_steps
                    system_act_loss = system_act_loss / args.gradient_accumulation_steps
                    disamb_and_coref_loss = disamb_and_coref_loss / args.gradient_accumulation_steps
                    disam_and_coref_candi_loss = disam_and_coref_candi_loss / args.gradient_accumulation_steps
                    misc_loss = misc_loss / args.gradient_accumulation_steps
                    retrieval_loss = retrieval_loss / args.gradient_accumulation_steps
                    kl_loss = kl_loss / args.gradient_accumulation_steps
                elif args.model_type == 'gen-ofa':
                    loss = loss / args.gradient_accumulation_steps
                else:
                    loss = loss / args.gradient_accumulation_steps
                    masked_lm_loss = masked_lm_loss / args.gradient_accumulation_steps
                    nocoref_loss = nocoref_loss / args.gradient_accumulation_steps
                    misc_loss = misc_loss / args.gradient_accumulation_steps
                    disam_loss = disam_loss / args.gradient_accumulation_steps
                    retrieval_loss = retrieval_loss / args.gradient_accumulation_steps
                    kl_loss = kl_loss / args.gradient_accumulation_steps
                    if args.model_type == 'mt-bart_add_intent':
                        user_act_loss = user_act_loss / args.gradient_accumulation_steps
                        system_act_loss = system_act_loss / args.gradient_accumulation_steps

            # 梯度后向计算更新参数
            if args.fp16:
                with amp.scale_loss(loss, optimizer) as scaled_loss:
                    scaled_loss.backward()
            else:
                loss.backward()
            
            # loss进行累加
            if args.model_type == 'gen-ofa':
                tr_loss += loss.item()
            else:
                tr_loss += loss.item()
                tr_masked_lm_loss += masked_lm_loss.item()
                tr_kl_loss += kl_loss.item()
                if isinstance(retrieval_loss, torch.Tensor):
                    tr_retrieval_loss += retrieval_loss.item()
                else:
                    tr_retrieval_loss += retrieval_loss
                if isinstance(misc_loss, torch.Tensor):
                    tr_misc_loss += misc_loss.item()
                else:
                    tr_misc_loss += misc_loss

                if args.model_type == 'mt-bart_joint_disam_coref':
                    tr_user_act_loss += user_act_loss.item()
                    tr_system_act_loss += system_act_loss.item()
                    tr_disamb_and_coref_loss += disamb_and_coref_loss.item()
                    tr_disam_and_coref_candi_loss += disam_and_coref_candi_loss.item()
                else:

                    if args.model_type == 'mt-bart_add_intent':
                        tr_user_act_loss += user_act_loss.item()
                        tr_system_act_loss += system_act_loss.item()
                    if isinstance(nocoref_loss,torch.Tensor):
                        tr_nocoref_loss += nocoref_loss.item()
                    else:
                        tr_nocoref_loss += nocoref_loss
                    if isinstance(disam_loss,torch.Tensor):
                        tr_disam_loss += disam_loss.item()
                    else:
                        tr_disam_loss += disam_loss


            if (step + 1) % args.gradient_accumulation_steps == 0:
                if args.fp16:
                    torch.nn.utils.clip_grad_norm_(amp.master_params(optimizer), args.max_grad_norm)
                    optimizer.step()
                    if args.scheduler != "no_schedule":
                        scheduler.step()  # Update learning rate schedule
                    model.zero_grad()
                    global_step += 1
                else:
                    parameters_to_clip = [p for p in model.parameters() if p.grad is not None]
                    torch.nn.utils.clip_grad_norm_(parameters_to_clip, args.max_grad_norm)
                    optimizer.step()
                    if args.scheduler != "no_schedule":
                        scheduler.step()  # Update learning rate schedule
                    model.zero_grad()
                    global_step += 1

                # 训练embedding
                # if global_step % args.embedding_train_steps == 0:
                if (args.do_train_embedding_clip_way_during_training  # and args.local_rank in [-1, 0]
                    and (global_step % args.embedding_train_steps == 0) 
                ):
                    train_embedding_clip_way(args, model, tokenizer, all_objects_meta, args.embedding_train_epochs_ongoing, do_tsne=False)
                
                # 打印日志到终端以及保存到Tensorboard文件
                if (args.local_rank in [-1, 0] and args.log_steps > 0 and (global_step % args.log_steps == 0) and (global_step > 0 )):
                    if args.scheduler != "no_schedule":
                        tb_writer.add_scalar("lr", scheduler.get_lr()[0], global_step)
                    
                    if args.model_type == 'gen-ofa':
                        tb_writer.add_scalar("train/loss", (tr_loss - logging_loss) / args.log_steps, global_step)
                    else:
                        tb_writer.add_scalar("train/loss", (tr_loss - logging_loss) / args.log_steps, global_step)
                        tb_writer.add_scalar("train/masked_lm_loss", (tr_masked_lm_loss - logging_masked_lm_loss) / args.log_steps, global_step)
                        tb_writer.add_scalar("train/misc_loss",(tr_misc_loss - logging_misc_loss) / args.log_steps, global_step)
                        tb_writer.add_scalar("train/retrieval_loss", (tr_retrieval_loss - logging_retrieval_loss) / args.log_steps, global_step)
                        tb_writer.add_scalar("train/kl_loss", (tr_kl_loss - logging_kl_loss) / args.log_steps, global_step)
                        if args.model_type == 'mt-bart_joint_disam_coref':
                            tb_writer.add_scalar("train/user_act_loss", (tr_user_act_loss - logging_user_act_loss) / args.log_steps, global_step)
                            tb_writer.add_scalar("train/system_act_loss", (tr_system_act_loss - logging_system_act_loss) / args.log_steps, global_step)
                            tb_writer.add_scalar("train/disamb_and_coref_loss", (tr_disamb_and_coref_loss - logging_disamb_and_coref_loss) / args.log_steps, global_step)
                            tb_writer.add_scalar("train/disam_and_coref_candi_loss", (tr_disam_and_coref_candi_loss - logging_disam_and_coref_candi_loss) / args.log_steps, global_step)
                        else:
                            if args.model_type == 'mt-bart_add_intent':
                                tb_writer.add_scalar("train/user_act_loss", (tr_user_act_loss - logging_user_act_loss) / args.log_steps, global_step)
                                tb_writer.add_scalar("train/system_act_loss", (tr_system_act_loss - logging_system_act_loss) / args.log_steps, global_step)
                            
                            tb_writer.add_scalar("train/nocoref_loss", (tr_nocoref_loss - logging_nocoref_loss) / args.log_steps, global_step)
                            tb_writer.add_scalar("train/disam_loss", (tr_disam_loss - logging_disam_loss) / args.log_steps, global_step)
                    
                    if args.model_type == 'mt-bart_joint_disam_coref':
                        epoch_iterator.set_postfix(loss=f'{((tr_loss - logging_loss) / args.log_steps):.4f}', 
                                                loss_masked_lm=f'{((tr_masked_lm_loss - logging_masked_lm_loss) / args.log_steps):.4f}', 
                                                loss_retrieval=f'{((tr_retrieval_loss - logging_retrieval_loss) / args.log_steps):.4f}',
                                                loss_user_act=f'{((tr_user_act_loss - logging_user_act_loss) / args.log_steps):.4f}',
                                                loss_system_act=f'{((tr_system_act_loss - logging_system_act_loss) / args.log_steps):.4f}',
                                                loss_disamb_and_coref=f'{((tr_disamb_and_coref_loss - logging_disamb_and_coref_loss) / args.log_steps):.4f}',
                                                loss_disam_and_coref_candi=f'{((tr_disam_and_coref_candi_loss - logging_disam_and_coref_candi_loss) / args.log_steps):.4f}',
                                                loss_misc=f'{((tr_misc_loss - logging_misc_loss) / args.log_steps):.4f}', 
                                                loss_kl=f'{((tr_kl_loss - logging_kl_loss) / args.log_steps):.4f}')
                        logging_loss = tr_loss
                        logging_masked_lm_loss = tr_masked_lm_loss
                        logging_misc_loss = tr_misc_loss
                        logging_user_act_loss = tr_user_act_loss
                        logging_system_act_loss = tr_system_act_loss
                        logging_disamb_and_coref_loss = tr_disamb_and_coref_loss
                        logging_disam_and_coref_candi_loss = tr_disam_and_coref_candi_loss
                        logging_retrieval_loss = tr_retrieval_loss
                        logging_kl_loss = tr_kl_loss
                    elif args.model_type == 'gen-ofa':
                        epoch_iterator.set_postfix(loss=f'{((tr_loss - logging_loss) / args.log_steps):.4f}')
                        logging_loss = tr_loss
                    elif args.model_type == 'mt-bart_add_intent':
                        epoch_iterator.set_postfix(loss=f'{((tr_loss - logging_loss) / args.log_steps):.4f}', 
                                                loss_masked_lm=f'{((tr_masked_lm_loss - logging_masked_lm_loss) / args.log_steps):.4f}', 
                                                loss_nocoref=f'{((tr_nocoref_loss - logging_nocoref_loss) / args.log_steps):.4f}', 
                                                loss_misc=f'{((tr_misc_loss - logging_misc_loss) / args.log_steps):.4f}', 
                                                loss_disam=f'{((tr_disam_loss - logging_disam_loss) / args.log_steps):.4f}', 
                                                loss_retrieval=f'{((tr_retrieval_loss - logging_retrieval_loss) / args.log_steps):.4f}',
                                                loss_user_act=f'{((tr_user_act_loss - logging_user_act_loss) / args.log_steps):.4f}',
                                                loss_system_act=f'{((tr_system_act_loss - logging_system_act_loss) / args.log_steps):.4f}',
                                                loss_kl=f'{((tr_kl_loss - logging_kl_loss) / args.log_steps):.4f}')
                        logging_loss = tr_loss
                        logging_masked_lm_loss = tr_masked_lm_loss
                        logging_nocoref_loss = tr_nocoref_loss
                        logging_misc_loss = tr_misc_loss
                        logging_disam_loss = tr_disam_loss
                        logging_retrieval_loss = tr_retrieval_loss
                        logging_kl_loss = tr_kl_loss
                        logging_user_act_loss = tr_user_act_loss
                        logging_system_act_loss = tr_system_act_loss
                    else:
                        epoch_iterator.set_postfix(loss=f'{((tr_loss - logging_loss) / args.log_steps):.4f}', 
                                                loss_masked_lm=f'{((tr_masked_lm_loss - logging_masked_lm_loss) / args.log_steps):.4f}', 
                                                loss_nocoref=f'{((tr_nocoref_loss - logging_nocoref_loss) / args.log_steps):.4f}', 
                                                loss_misc=f'{((tr_misc_loss - logging_misc_loss) / args.log_steps):.4f}', 
                                                loss_disam=f'{((tr_disam_loss - logging_disam_loss) / args.log_steps):.4f}', 
                                                loss_retrieval=f'{((tr_retrieval_loss - logging_retrieval_loss) / args.log_steps):.4f}',
                                                loss_kl=f'{((tr_kl_loss - logging_kl_loss) / args.log_steps):.4f}')
                        logging_loss = tr_loss
                        logging_masked_lm_loss = tr_masked_lm_loss
                        logging_nocoref_loss = tr_nocoref_loss
                        logging_misc_loss = tr_misc_loss
                        logging_disam_loss = tr_disam_loss
                        logging_retrieval_loss = tr_retrieval_loss
                        logging_kl_loss = tr_kl_loss

        # 每个epoch结束后进入到该位置
        # 每个 epoch 验证一次模型
        if (args.evaluate_during_training and args.local_rank in [-1, 0]):
            results = evaluate(args, model, tokenizer, all_objects_meta, eval_dataset=eval_dataset)
            for key, value in results.items():
                tb_writer.add_scalar("eval/{}".format(key), value, global_step)

        # 每个 epoch 保存一次模型
        if args.local_rank in [-1, 0]:
            checkpoint_prefix = "checkpoint"
            # Save model checkpoint
            output_dir = os.path.join(args.output_dir, "{}-{}".format(checkpoint_prefix, epoch_i))
            os.makedirs(output_dir, exist_ok=True)
            model_to_save = model.module if hasattr(model, "module") else model # Take care of distributed/parallel training
            model_to_save.save_pretrained(output_dir)
            tokenizer.save_pretrained(output_dir)
            torch.save(args, os.path.join(output_dir, "training_args.bin"))
            logger.info("Saving model checkpoint to %s", output_dir)
            _rotate_checkpoints(args, checkpoint_prefix)
            if args.save_optimizer_and_scheduler:
                # 文件太大了，非必要不保存，需要指定--save_optimizer_and_scheduler才保存
                torch.save(optimizer.state_dict(), os.path.join(output_dir, "optimizer.pt"))
                if args.scheduler != "no_schedule":
                    torch.save(scheduler.state_dict(), os.path.join(output_dir, "scheduler.pt"))
                logger.info("Saving optimizer and scheduler states to %s", output_dir)

            if args.max_steps > 0 and global_step > args.max_steps:
                epoch_iterator.close()
                break      
        
        if args.max_steps > 0 and global_step > args.max_steps:
            train_iterator.close()
            break

    if args.local_rank in [-1, 0]:
        tb_writer.close()

    return global_step, tr_loss/global_step


def add_dicts(d1, d2):
    return {k: d1[k] + d2[k] for k in d1}

def rec_prec_f1(n_correct, n_true, n_pred):
    rec = n_correct / n_true if n_true != 0 else 0
    prec = n_correct / n_pred if n_pred != 0 else 0
    f1 = 2 * prec * rec / (prec + rec) if (prec + rec) != 0 else 0
    return rec, prec, f1

def evaluate(args, model, tokenizer, all_objects_meta, eval_dataset, prefix=""):
    '''
    验证模型性能
    '''

    if args.model_type == 'mt-bart-attr':
        def collate_eval(examples):
            # 兼容2022.09.02之前的版本的collate_eval_bart
            # 可根据examples的size判别为不同的输入
            enc_input = list(map(lambda x: x[0], examples))
            enc_attention_mask = list(map(lambda x: x[1], examples))
            decoder_input = list(map(lambda x: x[2], examples))
            boxes = list(map(lambda x: x[3], examples))  
            misc = list(map(lambda x: x[4], examples))
            nocoref = list(map(lambda x: x[5], examples))
            if len(examples[0])>6:
                disam = list(map(lambda x: x[6], examples))
            else:
                disam = None
            if len(examples[0])>7:
                # or disam_and_coref_labels in get_dataset_jointdisamandcoref
                obj_ids_per_line = list(map(lambda x: x[7], examples))
            else:
                obj_ids_per_line = None
            if len(examples[0])>8:
                # or user_act_labels in get_dataset_jointdisamandcoref
                object_attr_input_ids_per_line = list(map(lambda x: x[8], examples))
            else:
                object_attr_input_ids_per_line = None

            if tokenizer._pad_token is None:
                enc_input_pad = pad_sequence(enc_input, batch_first=True)
            else:
                enc_input_pad = pad_sequence(enc_input, batch_first=True, padding_value=tokenizer.pad_token_id)
            enc_attention_pad = pad_sequence(enc_attention_mask, batch_first=True, padding_value=0)
            decoder_input_pad = tokenizer(decoder_input, padding="longest", truncation=True, return_tensors="pt")

            if len(examples[0])>8:
                # or user_act_labels
                return enc_input_pad, enc_attention_pad, decoder_input_pad.input_ids, decoder_input_pad.attention_mask, \
                        boxes, misc, nocoref, disam, obj_ids_per_line, object_attr_input_ids_per_line
            elif len(examples[0])>7:
                # or disam_and_coref_labels
                return enc_input_pad, enc_attention_pad, decoder_input_pad.input_ids, decoder_input_pad.attention_mask, \
                        boxes, misc, nocoref, disam, obj_ids_per_line

            return enc_input_pad, enc_attention_pad, decoder_input_pad.input_ids, decoder_input_pad.attention_mask, \
                    boxes, misc, nocoref, disam

    else:

        def collate_eval(examples):
            # 兼容2022.09.02之前的版本的collate_eval_bart
            # 可根据examples的size判别为不同的输入
            enc_input = list(map(lambda x: x[0], examples))
            enc_attention_mask = list(map(lambda x: x[1], examples))
            decoder_input = list(map(lambda x: x[2], examples))
            boxes = list(map(lambda x: x[3], examples))  
            misc = list(map(lambda x: x[4], examples))
            nocoref = list(map(lambda x: x[5], examples))
            if len(examples[0])>6:
                disam = list(map(lambda x: x[6], examples))
            else:
                disam = None
            if len(examples[0])>7:
                # or disam_and_coref_labels in get_dataset_jointdisamandcoref
                image_feature = list(map(lambda x: x[7], examples))
                if (len(image_feature[0].size())<=1):
                    image_feature_pad = torch.vstack(image_feature)
                else:
                    image_feature_pad = pad_sequence(image_feature, batch_first=True, padding_value=0) # torch.Size([batch_size, 3, 224, 224])
            else:
                image_feature_pad = None
            if len(examples[0])>8:
                # or user_act_labels in get_dataset_jointdisamandcoref
                enc_token_type_ids = list(map(lambda x: x[8], examples))
                if (len(enc_token_type_ids[0].size())<=1):
                    enc_token_type_ids_pad = torch.vstack(enc_token_type_ids)
                else:
                    enc_token_type_ids_pad = pad_sequence(enc_token_type_ids, batch_first=True, padding_value=0)
            else:
                enc_token_type_ids_pad = None
            if len(examples[0])>9:
                # in get_dataset_jointdisamandcoref
                system_act_labels = list(map(lambda x: x[9], examples))
                if (len(system_act_labels[0].size())<=1):
                    system_act_labels_pad = torch.vstack(system_act_labels)
                else:
                    system_act_labels_pad = pad_sequence(system_act_labels, batch_first=True, padding_value=0)

            if tokenizer._pad_token is None:
                enc_input_pad = pad_sequence(enc_input, batch_first=True)
            else:
                enc_input_pad = pad_sequence(enc_input, batch_first=True, padding_value=tokenizer.pad_token_id)
            enc_attention_pad = pad_sequence(enc_attention_mask, batch_first=True, padding_value=0)
            decoder_input_pad = tokenizer(decoder_input, padding="longest", truncation=True, return_tensors="pt")

            if len(examples[0])>9:
                # system_act_labels
                return enc_input_pad, enc_attention_pad, decoder_input_pad.input_ids, decoder_input_pad.attention_mask, \
                        boxes, misc, nocoref, disam, image_feature_pad, enc_token_type_ids_pad, system_act_labels_pad
            elif len(examples[0])>8:
                # or user_act_labels
                return enc_input_pad, enc_attention_pad, decoder_input_pad.input_ids, decoder_input_pad.attention_mask, \
                        boxes, misc, nocoref, disam, image_feature_pad, enc_token_type_ids_pad
            elif len(examples[0])>7:
                # or disam_and_coref_labels
                return enc_input_pad, enc_attention_pad, decoder_input_pad.input_ids, decoder_input_pad.attention_mask, \
                        boxes, misc, nocoref, disam, image_feature_pad

            return enc_input_pad, enc_attention_pad, decoder_input_pad.input_ids, decoder_input_pad.attention_mask, \
                    boxes, misc, nocoref, disam

    eval_sampler = SequentialSampler(eval_dataset)
    eval_dataloader = DataLoader(eval_dataset, sampler=eval_sampler, batch_size=args.eval_batch_size, collate_fn=collate_eval)
    
    logger.info("***** Running evaluation {} *****".format(prefix))
    logger.info("  Num examples = %d", len(eval_dataset))
    logger.info("  Batch size = %d", args.eval_batch_size)
    model.eval()
    eval_loss = 0.0
    disam_loss, nocoref_loss = 0.0, 0.0
    disamb_and_coref_loss = 0.0
    user_act_loss, system_act_loss = 0.0, 0.0
    disam_candi_loss, coref_candi_loss = 0.0, 0.0
    disam_and_coref_candi_loss = 0.0
    misc_loss = 0.0
    nb_eval_steps = 0

    if args.model_type == 'mt-bart_joint_disam_coref':
        report_template = {'eval_loss': 0, 'disam_and_coref': 0, 'fashion_disam_and_coref': 0, 'fashion_size': 0, 'fashion_available_sizes': 0, 
                        'fashion_brand': 0, 'fashion_color': 0, 'fashion_pattern': 0, 'fashion_sleeve_length': 0, 'fashion_asset_type':0, 
                        'fashion_type': 0, 'fashion_price': 0, 'fashion_customer_review': 0, 'furniture_disam_and_coref': 0,
                        'furniture_brand': 0, 'furniture_color': 0, 'furniture_materials': 0, 'furniture_type': 0,
                        'furniture_price': 0, 'furniture_customer_review': 0, 'fashion_misc_loss': 0, 'furniture_misc_loss': 0, 'user_act_loss': 0,
                        'system_act_loss': 0, 'user_act_acc': 0, 'system_act_acc': 0, 'disamb_and_coref_loss': 0, 'disam_and_coref_candi_loss': 0}
    elif args.model_type == 'mt-bart_add_intent':
        report_template = {'eval_loss': 0, 'disam': 0, 'nocoref': 0, 'fashion_disamb': 0, 'fashion_coref': 0, 'fashion_size': 0, 'fashion_available_sizes': 0, 
                        'fashion_brand': 0, 'fashion_color': 0, 'fashion_pattern': 0, 'fashion_sleeve_length': 0, 'fashion_asset_type':0, 
                        'fashion_type': 0, 'fashion_price': 0, 'fashion_customer_review': 0, 'furniture_disamb': 0,
                        'furniture_coref': 0, 'furniture_brand': 0, 'furniture_color': 0, 'furniture_materials': 0, 'furniture_type': 0,
                        'furniture_price': 0, 'furniture_customer_review': 0, 'disam_loss': 0, 'nocoref_loss': 0, 'disam_candi_loss': 0,
                        'coref_candi_loss': 0, 'fashion_misc_loss': 0, 'furniture_misc_loss': 0, 'user_act_loss': 0,
                        'system_act_loss': 0, 'user_act_acc': 0, 'system_act_acc': 0}
    elif args.model_type == 'gen-ofa':
        report_template = {'eval_loss': 0.0, 'generation_perplexity': 0.0}
    else:
        report_template = {'eval_loss': 0, 'disam': 0, 'nocoref': 0, 'fashion_disamb': 0, 'fashion_coref': 0, 'fashion_size': 0, 'fashion_available_sizes': 0, 
                        'fashion_brand': 0, 'fashion_color': 0, 'fashion_pattern': 0, 'fashion_sleeve_length': 0, 'fashion_asset_type':0, 
                        'fashion_type': 0, 'fashion_price': 0, 'fashion_customer_review': 0, 'furniture_disamb': 0,
                        'furniture_coref': 0, 'furniture_brand': 0, 'furniture_color': 0, 'furniture_materials': 0, 'furniture_type': 0,
                        'furniture_price': 0, 'furniture_customer_review': 0, 'disam_loss': 0, 'nocoref_loss': 0, 'disam_candi_loss': 0,
                        'coref_candi_loss': 0, 'fashion_misc_loss': 0, 'furniture_misc_loss': 0}
        
    total_report = copy.deepcopy(report_template)

    n_pred_disambs, n_true_disambs, n_correct_disambs = 0, 0, 0
    n_pred_objects, n_true_objects, n_correct_objects = 0, 0, 0
    num_fashions, num_furnitures = 0, 0

    for batch in tqdm(eval_dataloader, desc="Evaluating"):
        enc_input = batch[0].to(args.device)
        enc_attention_mask = batch[1].to(args.device)
        decoder_input = batch[2].to(args.device)
        decoder_attention_mask = batch[3].to(args.device)
        boxes = batch[4] # batch, num_obj_per_line, 6
        misc = batch[5]  # batch, num_obj_per_line, dict
        nocoref = batch[6]
        disam = batch[7]
        if args.model_type == 'mt-flava':
            # image_feature_pad, enc_token_type_ids_pad
            image_feature = batch[8].to(args.device)
            enc_token_type_ids = batch[9].to(args.device)
        elif args.model_type == 'gen-ofa' or args.model_type == 'mt-ofa':
            image_feature = batch[8].to(args.device)
        elif args.model_type == 'mt-bart_joint_disam_coref':
            disam_and_coref_labels = batch[8].to(args.device)
            user_act_labels = batch[9].to(args.device)
            system_act_labels = batch[10].to(args.device)
        elif args.model_type == 'mt-bart_add_intent':
            disambiguation_labels = batch[8].to(args.device)
            user_act_labels = batch[9].to(args.device)
            system_act_labels = batch[10].to(args.device)
        elif args.model_type == 'mt-bart-attr':
            obj_ids_per_line = batch[8]
            object_attr_input_ids = batch[9]


        # 对labels的特殊处理
        # 修复原先的bug: labels=decoder_input[:, 1:].contiguous(), 此时pad填充的是pad_token_id，并非-100，计算loss时会把pad部分也计算进去
        lm_labels=decoder_input[:, 1:].contiguous() # 哪怕输入是 <pad> .... <seq> 也没有影响，因为第一个token不作为labels
        lm_labels[lm_labels==tokenizer.pad_token_id]=-100 # 将labels的所有pad_token_id替换为-100，因为CrossEntropyLoss默认的ignore_index=-100
        lm_labels.contiguous()



        batch_size = len(misc)
        with torch.no_grad():
            if args.model_type == "mt-flava":
                model_outputs = model(
                            input_ids=enc_input,
                            pixel_values=image_feature,
                            attention_mask=enc_attention_mask,
                            token_type_ids=enc_token_type_ids,
                            boxes=boxes,
                            misc=misc,
                            nocoref=nocoref,
                            return_dict=True)
            elif args.model_type == 'gen-ofa':
                model_outputs = model(
                            input_ids=enc_input,
                            patch_images=image_feature,
                            decoder_input_ids=decoder_input[:, :-1],
                            attention_mask=decoder_attention_mask[:, :-1],  # attention mask for decoding.
                            labels=lm_labels,
                            boxes=boxes,
                            misc=misc,
                            sample_patch_num=args.sample_patch_num)
            elif args.model_type == 'mt-ofa':
                model_outputs = model(
                            input_ids=enc_input,
                            patch_images=image_feature,
                            decoder_input_ids=decoder_input[:, :-1],
                            attention_mask=decoder_attention_mask[:, :-1],  # attention mask for decoding.
                            labels=lm_labels,
                            boxes=boxes,
                            misc=misc,
                            sample_patch_num=args.sample_patch_num,
                            return_dict=True)
            elif args.model_type == 'mt-bart-attr':
                model_outputs = model(
                            input_ids=enc_input,
                            attention_mask=enc_attention_mask,
                            decoder_input_ids=decoder_input[:, :-1],
                            decoder_attention_mask=decoder_attention_mask[:, :-1],
                            labels=lm_labels,
                            boxes=boxes,
                            misc=misc,
                            nocoref=nocoref,
                            return_dict=True,
                            object_attr_input_ids=object_attr_input_ids,
                            use_non_visual_attrs=True) 
            
            else:
                model_outputs = model(
                            input_ids=enc_input,
                            attention_mask=enc_attention_mask,
                            decoder_input_ids=decoder_input[:, :-1],
                            decoder_attention_mask=decoder_attention_mask[:, :-1],
                            labels=lm_labels,
                            boxes=boxes,
                            misc=misc,
                            nocoref=nocoref,
                            return_dict=True)                

        if args.model_type == 'gen-ofa':
            model_loss = model_outputs.loss
            model_loss.mean().item()
        else:
            model_loss = model_outputs.loss
            model_loss.mean().item()
            enc_head_results = model_outputs.enc_head_results
            batch_report = copy.deepcopy(report_template)

            if args.model_type == 'mt-bart_joint_disam_coref':
                disam_and_coref_report = (disam_and_coref_labels == model_outputs.disamb_and_coref_logits.argmax(dim=1)).float().mean()
                batch_report['disam_and_coref'] += disam_and_coref_report
                disamb_and_coref_loss += ce_loss_fct(model_outputs.disamb_and_coref_logits, disam_and_coref_labels.view(-1)).mean().item()

                user_act_report = (user_act_labels == model_outputs.user_act_logits.argmax(dim=1)).float().mean()
                batch_report['user_act_acc'] += user_act_report
                user_act_loss += ce_loss_fct(model_outputs.user_act_logits, user_act_labels.view(-1)).mean().item()

                system_act_report = (system_act_labels == model_outputs.system_act_logits.argmax(dim=1)).float().mean()
                batch_report['system_act_acc'] += system_act_report
                system_act_loss += ce_loss_fct(model_outputs.system_act_logits, system_act_labels.view(-1)).mean().item()

            else:
                disam_labels = torch.tensor([disam[b_idx][1] for b_idx in range(batch_size)]).to(args.device)  # (bs)
                disam_report = (disam_labels == model_outputs.disambiguation_logits.argmax(dim=1)).float().mean()
                disam_loss += ce_loss_fct(model_outputs.disambiguation_logits, disam_labels.view(-1)).mean().item()
                nocoref_labels = torch.tensor([nocoref[b_idx][1] for b_idx in range(batch_size)]).to(args.device)  # (bs)
                nocoref_report = (nocoref_labels == model_outputs.nocoref_logits.argmax(dim=1)).float().mean()
                nocoref_loss += ce_loss_fct(model_outputs.nocoref_logits, nocoref_labels.view(-1)).mean().item()
                batch_report['disam'] += disam_report
                batch_report['nocoref'] += nocoref_report

                if args.model_type == 'mt-bart_add_intent':
                    user_act_report = (user_act_labels == model_outputs.user_act_logits.argmax(dim=1)).float().mean()
                    batch_report['user_act_acc'] += user_act_report
                    user_act_loss += ce_loss_fct(model_outputs.user_act_logits, user_act_labels.view(-1)).mean().item()

                    system_act_report = (system_act_labels == model_outputs.system_act_logits.argmax(dim=1)).float().mean()
                    batch_report['system_act_acc'] += system_act_report
                    system_act_loss += ce_loss_fct(model_outputs.system_act_logits, system_act_labels.view(-1)).mean().item()

            batch_disam_candi_loss = 0.0
            batch_coref_candi_loss = 0.0
            batch_disam_and_coref_candi_loss = 0.0
            for b_idx in range(batch_size):  # in a batch
                is_fashion = misc[b_idx][0]['is_fashion']
                if args.model_type == 'mt-bart_joint_disam_coref':
                    disam_and_coref_label = torch.tensor([misc[b_idx][obj_idx]['disam_and_coref_label'] for obj_idx in range(len(misc[b_idx]))], dtype=torch.long).to(args.device)  # (num_obj)  0 or 1 or 2
                    # 1：有歧义无共指; 2：无歧义有共指
                    if 1 in disam_and_coref_label:
                        n_true_disambs += (disam_and_coref_label == torch.full_like(disam_and_coref_label, 1)).int().sum().item()
                    else:
                        n_true_objects += (disam_and_coref_label == torch.full_like(disam_and_coref_label, 2)).int().sum().item()
                else:
                    disamb_label = torch.tensor([misc[b_idx][obj_idx]['disamb_label'] for obj_idx in range(len(misc[b_idx]))], dtype=torch.long).to(args.device)  # (num_obj)  0 or 1
                    coref_label = torch.tensor([misc[b_idx][obj_idx]['coref_label'] for obj_idx in range(len(misc[b_idx]))], dtype=torch.long).to(args.device)  # (num_obj)  0 or 1
                    n_true_disambs += disamb_label.sum().item()
                    n_true_objects += coref_label.sum().item()

                if is_fashion:
                    num_fashions += 1
                    size_label = torch.tensor([misc[b_idx][obj_idx]['misc_labels']['size'] for obj_idx in range(len(misc[b_idx]))], dtype=torch.long).to(args.device)  # (num_obj)
                    available_sizes_label = torch.tensor([misc[b_idx][obj_idx]['misc_labels']['available_sizes'] for obj_idx in range(len(misc[b_idx]))], dtype=torch.float32).to(args.device)   # (num_obj, 6)
                    brand_label = torch.tensor([misc[b_idx][obj_idx]['misc_labels']['brand'] for obj_idx in range(len(misc[b_idx]))], dtype=torch.long).to(args.device) 
                    color_label = torch.tensor([misc[b_idx][obj_idx]['misc_labels']['color'] for obj_idx in range(len(misc[b_idx]))], dtype=torch.long).to(args.device) 
                    pattern_label = torch.tensor([misc[b_idx][obj_idx]['misc_labels']['pattern'] for obj_idx in range(len(misc[b_idx]))], dtype=torch.long).to(args.device) 
                    sleeve_length_label = torch.tensor([misc[b_idx][obj_idx]['misc_labels']['sleeve_length'] for obj_idx in range(len(misc[b_idx]))], dtype=torch.long).to(args.device) 
                    asset_type_label = torch.tensor([misc[b_idx][obj_idx]['misc_labels']['asset_type'] for obj_idx in range(len(misc[b_idx]))], dtype=torch.long).to(args.device) 
                    type_label = torch.tensor([misc[b_idx][obj_idx]['misc_labels']['type'] for obj_idx in range(len(misc[b_idx]))], dtype=torch.long).to(args.device) 
                    price_label = torch.tensor([misc[b_idx][obj_idx]['misc_labels']['price'] for obj_idx in range(len(misc[b_idx]))], dtype=torch.long).to(args.device) 
                    customer_review_label = torch.tensor([misc[b_idx][obj_idx]['misc_labels']['customer_review'] for obj_idx in range(len(misc[b_idx]))], dtype=torch.long).to(args.device) 
                    if args.model_type == 'mt-bart_joint_disam_coref':
                        disam_and_coref, size, available_sizes, brand, color, pattern, sleeve_length, asset_type, type_, price, customer_review = enc_head_results[b_idx]
                        batch_disam_and_coref_candi_loss += ce_loss_fct(disam_and_coref, disam_and_coref_label).mean().item()
                    else:
                        disamb, coref, size, available_sizes, brand, color, pattern, sleeve_length, asset_type, type_, price, customer_review = enc_head_results[b_idx]
                        batch_disam_candi_loss += ce_loss_fct(disamb, disamb_label).mean().item()
                        batch_coref_candi_loss += ce_loss_fct(coref, coref_label).mean().item()
                    # 计算属性loss
                    fashion_loss_per_line = ce_loss_fct(size, size_label) + \
                                            bce_loss_fct(available_sizes, available_sizes_label) + \
                                            ce_loss_fct(brand, brand_label) + \
                                            ce_loss_fct(color, color_label) + \
                                            ce_loss_fct(pattern, pattern_label) + \
                                            ce_loss_fct(sleeve_length, sleeve_length_label) + \
                                            ce_loss_fct(asset_type, asset_type_label) + \
                                            ce_loss_fct(type_, type_label) + \
                                            ce_loss_fct(price, price_label) + \
                                            ce_loss_fct(customer_review, customer_review_label) 
                    batch_report['fashion_misc_loss'] += fashion_loss_per_line.item()

                    if args.model_type == 'mt-bart_joint_disam_coref':
                        batch_report['fashion_disam_and_coref'] += torch.all(disam_and_coref.argmax(dim=1) == disam_and_coref_label, dim=0).float()  # 1. or 0.
                        # 统计预测为歧义候选的数量
                        disam_and_coref_preds = disam_and_coref.argmax(dim=1)
                        n_pred_disambs += (disam_and_coref_preds == torch.full_like(disam_and_coref_preds, 1)).int().sum().item() # 预测标签为1的数量
                        n_pred_objects += (disam_and_coref_preds == torch.full_like(disam_and_coref_preds, 2)).int().sum().item() # 预测标签为2的数量
                        n_correct_disambs = torch.logical_and((disam_and_coref_preds == torch.full_like(disam_and_coref_preds, 1)), (disam_and_coref_preds == disam_and_coref_label)).int().sum().item() # 预测标签为1且正确预测的数量
                        n_correct_objects = torch.logical_and((disam_and_coref_preds == torch.full_like(disam_and_coref_preds, 2)), (disam_and_coref_preds == disam_and_coref_label)).int().sum().item() # 预测标签为2且正确预测的数量
                    else:
                        # DISAMB CANDI
                        n_pred_disambs += disamb.argmax(dim=1).sum().item()
                        n_correct_disambs += torch.logical_and(disamb.argmax(dim=1), disamb_label).int().sum().item()
                        # MM COREF
                        n_pred_objects += coref.argmax(dim=1).sum().item()
                        n_correct_objects += torch.logical_and(coref.argmax(dim=1), coref_label).int().sum().item()
                        batch_report['fashion_disamb'] += torch.all(disamb.argmax(dim=1) == disamb_label, dim=0).float()  # 1. or 0.
                        batch_report['fashion_coref'] += torch.all(coref.argmax(dim=1) == coref_label, dim=0).float()  # 1. or 0.

                    batch_report['fashion_size'] += (size.argmax(dim=1) == size_label).float().mean()  # accuracy at a line
                    batch_report['fashion_available_sizes'] += torch.all(((available_sizes > 0.5).int() == available_sizes_label).bool(), dim=1).float().mean() # accuracy at a line
                    batch_report['fashion_brand'] += (brand.argmax(dim=1) == brand_label).float().mean()
                    batch_report['fashion_color'] += (color.argmax(dim=1) == color_label).float().mean()
                    batch_report['fashion_pattern'] += (pattern.argmax(dim=1) == pattern_label).float().mean()
                    batch_report['fashion_sleeve_length'] += (sleeve_length.argmax(dim=1) == sleeve_length_label).float().mean()
                    batch_report['fashion_asset_type'] += (asset_type.argmax(dim=1) == asset_type_label).float().mean()
                    batch_report['fashion_type'] += (type_.argmax(dim=1) == type_label).float().mean()
                    batch_report['fashion_price'] += (price.argmax(dim=1) == price_label).float().mean()
                    batch_report['fashion_customer_review'] += (customer_review.argmax(dim=1) == customer_review_label).float().mean()

                else:
                    num_furnitures += 1
                    brand_label = torch.tensor([misc[b_idx][obj_idx]['misc_labels']['brand'] for obj_idx in range(len(misc[b_idx]))], dtype=torch.long).to(args.device)  # (num_obj)
                    color_label = torch.tensor([misc[b_idx][obj_idx]['misc_labels']['color'] for obj_idx in range(len(misc[b_idx]))], dtype=torch.long).to(args.device)
                    materials_label = torch.tensor([misc[b_idx][obj_idx]['misc_labels']['materials'] for obj_idx in range(len(misc[b_idx]))], dtype=torch.long).to(args.device)
                    type_label = torch.tensor([misc[b_idx][obj_idx]['misc_labels']['type'] for obj_idx in range(len(misc[b_idx]))], dtype=torch.long).to(args.device)
                    price_label = torch.tensor([misc[b_idx][obj_idx]['misc_labels']['price'] for obj_idx in range(len(misc[b_idx]))], dtype=torch.long).to(args.device)
                    customer_review_label = torch.tensor([misc[b_idx][obj_idx]['misc_labels']['customer_review'] for obj_idx in range(len(misc[b_idx]))], dtype=torch.long).to(args.device)
                    
                    if args.model_type == 'mt-bart_joint_disam_coref':
                        disam_and_coref, brand, color, materials, type_, price, customer_review = enc_head_results[b_idx]
                        batch_disam_and_coref_candi_loss += ce_loss_fct(disam_and_coref, disam_and_coref_label).mean().item()

                    else:
                        disamb, coref, brand, color, materials, type_, price, customer_review = enc_head_results[b_idx]
                        batch_disam_candi_loss += ce_loss_fct(disamb, disamb_label).mean().item()
                        batch_coref_candi_loss += ce_loss_fct(coref, coref_label).mean().item()

                    # 计算属性loss
                    furniture_loss_per_line = ce_loss_fct(brand, brand_label) + \
                                            ce_loss_fct(color, color_label) + \
                                            ce_loss_fct(materials, materials_label) + \
                                            ce_loss_fct(type_, type_label) + \
                                            ce_loss_fct(price, price_label) + \
                                            ce_loss_fct(customer_review, customer_review_label) 
                    batch_report['furniture_misc_loss'] += furniture_loss_per_line.item()
                    if args.model_type == 'mt-bart_joint_disam_coref':
                        batch_report['furniture_disam_and_coref'] += torch.all(disam_and_coref.argmax(dim=1) == disam_and_coref_label, dim=0).float()  # 1. or 0.
                        # 统计预测为歧义候选的数量
                        disam_and_coref_preds = disam_and_coref.argmax(dim=1)
                        n_pred_disambs += (disam_and_coref_preds == torch.full_like(disam_and_coref_preds, 1)).int().sum().item() # 预测标签为1的数量
                        n_pred_objects += (disam_and_coref_preds == torch.full_like(disam_and_coref_preds, 2)).int().sum().item() # 预测标签为2的数量
                        n_correct_disambs = torch.logical_and((disam_and_coref_preds == torch.full_like(disam_and_coref_preds, 1)), (disam_and_coref_preds == disam_and_coref_label)).int().sum().item() # 预测标签为1且正确预测的数量
                        n_correct_objects = torch.logical_and((disam_and_coref_preds == torch.full_like(disam_and_coref_preds, 2)), (disam_and_coref_preds == disam_and_coref_label)).int().sum().item() # 预测标签为2且正确预测的数量

                    else:
                        # DISAMB CANDI
                        n_pred_disambs += disamb.argmax(dim=1).sum().item()
                        n_correct_disambs += torch.logical_and(disamb.argmax(dim=1), disamb_label).int().sum().item()
                        # MM COREF
                        n_pred_objects += coref.argmax(dim=1).sum().item()
                        n_correct_objects += torch.logical_and(coref.argmax(dim=1), coref_label).int().sum().item()
                        batch_report['furniture_disamb'] += torch.all(disamb.argmax(dim=1) == disamb_label, dim=0).float()  # 1. or 0.
                        batch_report['furniture_coref'] += torch.all(coref.argmax(dim=1) == coref_label, dim=0).float()  # 1. or 0.

                    batch_report['furniture_brand'] += (brand.argmax(dim=1) == brand_label).float().mean()  # accuracy at a line
                    batch_report['furniture_color'] += (color.argmax(dim=1) == color_label).float().mean()  # accuracy at a line
                    batch_report['furniture_materials'] += (materials.argmax(dim=1) == materials_label).float().mean()  # accuracy at a line
                    batch_report['furniture_type'] += (type_.argmax(dim=1) == type_label).float().mean()  # accuracy at a line
                    batch_report['furniture_price'] += (price.argmax(dim=1) == price_label).float().mean()  # accuracy at a line
                    batch_report['furniture_customer_review'] += (customer_review.argmax(dim=1) == customer_review_label).float().mean()  # accuracy at a line
            
            if args.model_type == 'mt-bart_joint_disam_coref':
                disam_and_coref_candi_loss += batch_disam_and_coref_candi_loss/batch_size # batch内平均
            else:
                disam_candi_loss += batch_disam_candi_loss/batch_size # batch内平均
                coref_candi_loss += batch_coref_candi_loss/batch_size # batch内平均
            total_report = add_dicts(total_report, batch_report)

        eval_loss += model_loss.mean().item()
        nb_eval_steps += 1

    if args.model_type == 'gen-ofa':
        total_report = {'eval_loss': 0.0, 'generation_perplexity': 0.0}
        eval_loss /= nb_eval_steps
        total_report['eval_loss'] = eval_loss
        perplexity = torch.exp(torch.tensor(eval_loss))
        total_report['generation_perplexity'] = perplexity
        if args.output_eval_file:
            os.makedirs(args.output_eval_file.rsplit('/', 1)[0], exist_ok=True)
            with open(args.output_eval_file, 'a') as writer:
                for key in total_report.keys():
                    writer.write("%s = %s\n\n" % (key, str(total_report[key])))

        print('EVALUATION:', total_report)
        return total_report

    elif args.model_type == 'mt-bart_joint_disam_coref':
        total_report['disam_and_coref'] /= nb_eval_steps
        total_report['user_act_acc'] /= nb_eval_steps
        total_report['system_act_acc'] /= nb_eval_steps
        disamb_and_coref_loss /= nb_eval_steps
        total_report['disamb_and_coref_loss'] = disamb_and_coref_loss
        user_act_loss /= nb_eval_steps
        total_report['user_act_loss'] = user_act_loss
        system_act_loss /= nb_eval_steps
        total_report['system_act_loss'] = system_act_loss
        disam_and_coref_candi_loss /= nb_eval_steps
        total_report['disam_and_coref_candi_loss'] = disam_and_coref_candi_loss

    else:
        if args.model_type == 'mt-bart_add_intent':
            total_report['user_act_acc'] /= nb_eval_steps
            total_report['system_act_acc'] /= nb_eval_steps
            user_act_loss /= nb_eval_steps
            total_report['user_act_loss'] = user_act_loss
            system_act_loss /= nb_eval_steps
            total_report['system_act_loss'] = system_act_loss

        total_report['disam'] /= nb_eval_steps
        total_report['nocoref'] /= nb_eval_steps
        disam_loss /= nb_eval_steps
        total_report['disam_loss'] = disam_loss
        nocoref_loss /= nb_eval_steps
        total_report['nocoref_loss'] = nocoref_loss
        disam_candi_loss /= nb_eval_steps
        total_report['disam_candi_loss'] = disam_candi_loss
        coref_candi_loss /= nb_eval_steps
        total_report['coref_candi_loss'] = coref_candi_loss

    for k, v in total_report.items():
        if ('fashion' in k) and num_fashions:
            total_report[k] = v/num_fashions
        if ('furniture' in k) and num_furnitures:
            total_report[k] = v/num_furnitures 
    
    eval_loss /= nb_eval_steps
    total_report['eval_loss'] = eval_loss
    perplexity = torch.exp(torch.tensor(eval_loss))
    print('total disamb result:', n_correct_disambs, n_true_disambs, n_pred_disambs)
    disamb_rec, disamb_prec, disamb_f1 = rec_prec_f1(n_correct_disambs, n_true_disambs, n_pred_disambs)
    print('total coref result:', n_correct_objects, n_true_objects, n_pred_objects)
    coref_rec, coref_prec, coref_f1 = rec_prec_f1(n_correct_objects, n_true_objects, n_pred_objects)
    total_report['generation_perplexity'] = perplexity
    total_report['disamb_rec'] = disamb_rec
    total_report['disamb_prec'] = disamb_prec
    total_report['disamb_f1'] = disamb_f1
    total_report['coref_rec'] = coref_rec
    total_report['coref_prec'] = coref_prec
    total_report['coref_f1'] = coref_f1
    if args.output_eval_file:
        os.makedirs(args.output_eval_file.rsplit('/', 1)[0], exist_ok=True)
        with open(args.output_eval_file, 'a') as writer:
            for key in total_report.keys():
                writer.write("%s = %s\n\n" % (key, str(total_report[key])))

    print('EVALUATION:', total_report)
    return total_report


def main():
    # Get the arguments for training model
    args = parser.parse_args()
    # Setup args.local_rank and args.world_size
    if args.no_cuda or args.model_parallel:
        args.local_rank = -1
        args.world_size = 1
    else:
        args.world_size = 1 # 默认值
        if "LOCAL_RANK" in os.environ: # 需要使用torchrun才有！
            args.local_rank = int(os.environ["LOCAL_RANK"])
        if "WORLD_SIZE" in os.environ:
            args.world_size = int(os.environ["WORLD_SIZE"]) # 节点数*每个节点地方任务数
    
    # Load model from checkpoints for continue training
    if args.should_continue:
        sorted_checkpoints = _sorted_checkpoints(args)
        if len(sorted_checkpoints) == 0:
            raise ValueError("Used --should_continue but no checkpoint was found in --output_dir.")
        else:
            args.model_name_or_path = sorted_checkpoints[-1]
    if (os.path.exists(args.output_dir) and os.listdir(args.output_dir) and not args.overwrite_output_dir and not args.should_continue):
        raise ValueError("Output directory ({}) already exists and is not empty. Use --overwrite_output_dir to overcome.".format(args.output_dir))
    
    # Setup CUDA, GPU & distributed training
    if args.local_rank == -1 or args.no_cuda:
        device = torch.device("cuda" if torch.cuda.is_available() and not args.no_cuda else "cpu")
        args.n_gpu = 0 if args.no_cuda else torch.cuda.device_count()
    else:  # Initializes the distributed backend which will take care of sychronizing nodes/GPUs
        # 数据并行DDP训练方式
        torch.cuda.set_device(args.local_rank)
        device = torch.device("cuda", args.local_rank)
        torch.distributed.init_process_group(backend="nccl")
        args.n_gpu = 1
    args.device = device
    
    # Setup logging
    logging.basicConfig(format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s", 
                        datefmt="%m/%d/%Y %H:%M:%S", 
                        level=logging.INFO if args.local_rank in [-1, 0] else logging.WARN)
    logger.warning("Process rank: %s, device: %s, n_gpu: %s, distributed training: %s, 16-bits training: %s", args.local_rank, device, args.n_gpu, bool(args.local_rank != -1), args.fp16)
    
    # Set seed
    setup_seed(args.seed, args.n_gpu)

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
    '''
    # Add new model class here
    elif args.model_type == "XXX":
        model_class, tokenizer_class = ModelClassName, TokenizerClassName
    '''

    # Load tokenizer from pretrained
    if args.model_type == 'mt-t5':
        tokenizer = tokenizer_class.from_pretrained(args.model_name_or_path, model_max_length=512)
    else:
        tokenizer = tokenizer_class.from_pretrained(args.model_name_or_path)

    if args.add_special_tokens:
        if not os.path.exists(args.add_special_tokens):
            raise ValueError("Additional special tokens file {args.add_special_tokens} not found}")
        with open(args.add_special_tokens, "rb") as handle:
                special_tokens_dict = json.load(handle)
        num_added_toks = tokenizer.add_special_tokens(special_tokens_dict)
        logger.info(f"Added {num_added_toks} tokens")
        logger.info(f"All special tokens: {tokenizer.all_special_tokens}")
    
    # 2022/09/02 by Yirong Chen 此处用于初始化图像相关的特征提取器
    if args.model_type == 'mt-flava':
        feature_extractor = FlavaFeatureExtractor.from_pretrained(args.model_name_or_path)
        processor = FlavaProcessor(feature_extractor=feature_extractor, tokenizer=tokenizer)
    else: # 避免先前的版本参数受到影响
        feature_extractor = None
        processor = None
    
    # Load model from pretrained
    if args.model_type == 'mt-ul2':
        model = model_class.from_pretrained(args.model_name_or_path)
    elif args.model_type == 'gen-ofa' or args.model_type == 'mt-ofa':
        model = model_class.from_pretrained(args.model_name_or_path, use_cache=False)
    else:
        model = model_class.from_pretrained(args.model_name_or_path, ignore_mismatched_sizes=args.ignore_mismatched_sizes)

    # Resize token embeddings and lm head
    if args.add_special_tokens:
        model.resize_token_embeddings(len(tokenizer))
        model.vocab_size = len(tokenizer)
    if args.model_type != 'mt-flava':
        model.config.decoder_start_token_id = tokenizer.bos_token_id

    #if args.local_rank == 0:
    #    torch.distributed.barrier()  # Make sure only the first process in distributed training will download model & vocab


    # Set model parallel training
    # 本部分用于配置模型分配不同的层到不同的GPU卡上进行训练，通常参数达到上十亿的模型才需要
    if args.model_parallel and not args.no_cuda and torch.cuda.is_available():
        logger.info("parallelizing...")
        model.parallelize() # 自动计算模型层数，将模型切分到不同的显卡
    else:
        logger.info("put model to GPU")
        model.to(args.device)

    #if args.local_rank not in [-1, 0]:
    #    torch.distributed.barrier()  # Barrier to make sure only the first process in distributed training process the dataset, and the others will use the cache

    if args.autocast: # 混合精度训练
        # 参考: https://pytorch.org/docs/1.9.0/amp.html?highlight=torch%20cuda%20amp%20gradscaler
        #       https://pytorch.org/docs/1.9.0/notes/amp_examples.html#amp-examples
        scaler = torch.cuda.amp.GradScaler()  # pytorch版本要求：1.6+

    if args.no_cuda:
        # 辅助调试代码
        print(model)
        print(model.config)

    logger.info("Training/evaluation parameters %s", args)

    # Prepare dataset
    with open(args.item2id, 'r') as f:
        item2id = json.load(f)
    train_api = api.PromptAPI(dial_split="train", 
                              data_dir=args.data_dir, 
                              dialogue_name_prefix=args.dialogue_name_prefix,
                              jsons_dir_name=args.jsons_dir_name,
                              images_dir_name=args.images_dir_name)
    fashion_meta, furniture_meta = train_api.fashion_meta, train_api.furniture_meta
    all_objects_meta = dict()
    for meta in fashion_meta:
        object_special_id = item2id[meta.name]
        object_meta = {'asset_type': meta.asset_type, 'customer_review': str(meta.customer_review),
        'available_sizes': [available_sizes2st[size] for size in meta.available_sizes], 
        'color': meta.color, 'pattern': meta.pattern, 'brand': meta.brand, 
        'sleeve_length': meta.sleeve_length, 'type': meta.type, 'price': str(meta.price), 'size': meta.size}
        all_objects_meta[object_special_id] = object_meta
    for meta in furniture_meta:
        object_special_id = item2id[meta.name]
        object_meta = {'brand': meta.brand, 'color': meta.color, 'customer_review': str(meta.customer_review),
        'materials': meta.materials, 'price': meta.price, 'type': meta.type}
        all_objects_meta[object_special_id] = object_meta

    # Set args.train_batch_size and args.eval_batch_size
    if args.model_parallel:
        args.train_batch_size = args.per_gpu_train_batch_size
    else: # 分布式训练
        args.train_batch_size = args.per_gpu_train_batch_size * max(1, args.n_gpu)
    args.eval_batch_size = args.per_gpu_eval_batch_size * max(1, args.n_gpu)

    # Load the train and eval dataset
    # 2022/10/12 兼容涛哥预处理的数据集，使用LineByLineDatasetFromSingleFile
    #            只需要指定args.train_input_file和args.eval_input_file, 其他为None
    logger.info("Loading Train and Eval Dataset!!!")
    if args.train_target_file is None and args.eval_target_file is None:
        if args.model_type == 'gen-ofa' or args.model_type == 'mt-ofa':
            train_dataset = LineByLineDatasetFromSingleFileForOFA(args.train_input_file, tokenizer, all_objects_meta, image_path_file=args.train_image_path_file, image_dir=args.train_image_dir, evaluation=False)
            eval_dataset = LineByLineDatasetFromSingleFileForOFA(args.eval_input_file, tokenizer, all_objects_meta, image_path_file=args.eval_image_path_file, image_dir=args.eval_image_dir, evaluation=True)
        else:
            train_dataset = LineByLineDatasetFromSingleFile(args.train_input_file, tokenizer, all_objects_meta, evaluation=False)
            eval_dataset = LineByLineDatasetFromSingleFile(args.eval_input_file, tokenizer, all_objects_meta, evaluation=True)

    else:
        if args.model_type == 'mt-flava':
            train_dataset = get_dataset_with_image(args, tokenizer, all_objects_meta, train=True, feature_extractor=feature_extractor)
            eval_dataset = get_dataset_with_image(args, tokenizer, all_objects_meta, train=False, feature_extractor=feature_extractor)
        elif args.model_type == 'gen-ofa' or args.model_type == 'mt-ofa':
            train_dataset = get_dataset_for_ofa(args, tokenizer, all_objects_meta, train=True)
            eval_dataset = get_dataset_for_ofa(args, tokenizer, all_objects_meta, train=False)
        elif args.model_type == 'mt-bart_joint_disam_coref' or args.model_type == 'mt-bart_add_intent':
            train_dataset = get_dataset_jointdisamandcoref(args, tokenizer, all_objects_meta, train=True)
            eval_dataset = get_dataset_jointdisamandcoref(args, tokenizer, all_objects_meta, train=False)
        elif args.model_type == 'mt-bart-attr':
            train_dataset = get_dataset_with_obj_attr(args, tokenizer, all_objects_meta, train=True)
            eval_dataset = get_dataset_with_obj_attr(args, tokenizer, all_objects_meta, train=False)
        else:
            train_dataset = get_dataset(args, tokenizer, all_objects_meta, train=True)
            eval_dataset = get_dataset(args, tokenizer, all_objects_meta, train=False)

    #if args.local_rank == 0:
    #    torch.distributed.barrier()

    # Training the model
    if not args.no_train: # 如果不指定--no_train，则进行模型训练
        logger.info("***************Training***************")
        logger.info("Running train_embedding_clip_way!!!")
        train_embedding_clip_way(args, model, tokenizer, all_objects_meta, args.embedding_train_epochs_start, do_tsne=False)
        logger.info("Running train!!!")
        global_step, train_loss = train(args, model, tokenizer, all_objects_meta, train_dataset=train_dataset, eval_dataset=eval_dataset)
        logger.info(" global_step = %s, average loss = %s", global_step, train_loss)
        # Saving best-practices: if you use save_pretrained for the model and tokenizer, you can reload them using from_pretrained()
        if args.local_rank == -1 or torch.distributed.get_rank() == 0:
            if args.local_rank in [-1, 0]:  
                os.makedirs(args.output_dir, exist_ok=True)
            logger.info("Saving model checkpoint to %s", args.output_dir)
            model_to_save = (model.module if hasattr(model, "module") else model)  # Take care of distributed/parallel training
            model_to_save.save_pretrained(args.output_dir)
            tokenizer.save_pretrained(args.output_dir)
            torch.save(args, os.path.join(args.output_dir, "training_args.bin"))
            with open(os.path.join(args.output_dir, "training_args.json"),'w',encoding='utf-8') as json_file:
                json.dump(pformat(args),json_file,ensure_ascii=False)
        logger.info("Finishing Train and save model checkpoint!!!")

    # Evaluating the model
    results = {}
    if args.do_eval and args.local_rank in [-1, 0]:
        logger.info("***************Evaluation***************")
        checkpoints = [args.output_dir]
        if args.eval_all_checkpoints:
            checkpoints = [os.path.dirname(c) for c in sorted( glob.glob(args.output_dir + "/**/" + WEIGHTS_NAME, recursive=True))]
            logging.getLogger("transformers.modeling_utils").setLevel(logging.WARN)  # Reduce logging
        logger.info("Evaluate the following checkpoints: %s", checkpoints)
        for checkpoint in checkpoints:
            global_step = checkpoint.split("-")[-1] if len(checkpoints) > 1 else ""
            prefix = checkpoint.split("/")[-1] if checkpoint.find("checkpoint") != -1 else ""
            model = model_class.from_pretrained(checkpoint)
            model.to(args.device)
            result = evaluate(args, model, tokenizer, all_objects_meta, eval_dataset=eval_dataset, prefix=prefix)
            result = {k + "_{}".format(global_step): v for k, v in result.items()}
            results.update(result)

    return results

if __name__ == '__main__':
    main()
