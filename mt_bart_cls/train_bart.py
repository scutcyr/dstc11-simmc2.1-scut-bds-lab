# coding=utf-8
# Copyright 2022 IFLYTEK COG1 Authors.
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
import os
import copy
import json
import random
import logging
import numpy as np
import matplotlib.pyplot as plt
from pprint import pformat
from tqdm import tqdm, trange
from sklearn.manifold import TSNE
from typing import Dict, List, Tuple
import pdb

import torch
from torch import nn
from torch.optim import AdamW as torch_AdamW
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader, RandomSampler
from transformers import (
    AdamW,
    BartTokenizerFast,
    get_linear_schedule_with_warmup
)

# 导入模型类
from models.modeling_simmc_bart_disamcands import MultiTaskBartForConditionalGenerationWithDisamb

# 导入数据处理类
from utils import api
from utils.simmc21_dataset import (
    get_input_id, LineByLineDataset,
    fashion_meta_attrs, furniture_meta_attrs, available_sizes2st,
    FASHION_SPECIAL_TOKENS, FURNITURE_SPECIAL_TOKENS
)

from finetune_args import parser
args = parser.parse_args()

## 配置日志文件
logger = logging.getLogger("execute")
logger.setLevel(logging.INFO)
fh = logging.FileHandler(args.log_file, mode="w")
fmt = "%(asctime)-15s %(levelname)s %(filename)s %(lineno)d %(process)d %(message)s"
datefmt = "%a %d %b %Y %H:%M:%S"
formatter = logging.Formatter(fmt, datefmt)
fh.setFormatter(formatter)
logger.addHandler(fh)


CELoss = nn.CrossEntropyLoss()
BCELoss = nn.BCEWithLogitsLoss()


def set_seed(args):
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if args.n_gpu > 0:
        torch.cuda.manual_seed_all(args.seed)


def train_embedding_clip_way(args, 
                             model, 
                             tokenizer, 
                             all_objects_meta, 
                             num_iter=50, 
                             do_tsne=False):
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
            fashion_attr_dict[attr_name][attr_value] = [
                (x, accum_token_counter + i) for i, x in enumerate(get_input_id(tokenizer, attr_value))
                ]
            accum_token_counter += len(get_input_id(tokenizer, attr_value))
    # print(fashion_attr_dict)
    # furniture_attr_dict: same as fashion_attr_dict
    furniture_attr_dict = dict()
    for attr_name, attr_values in furniture_meta_attrs.items():
        furniture_attr_dict[attr_name] = dict()
        attr_values = list(attr_values)
        attr_values.sort()
        accum_token_counter = 0
        for attr_value in attr_values:
            if not attr_value:  # skip empty string
                continue
            furniture_attr_dict[attr_name][attr_value] = [
                (x, accum_token_counter + i) for i, x in enumerate(get_input_id(tokenizer, attr_value))
                ]
            accum_token_counter += len(get_input_id(tokenizer, attr_value))
    # print(fashion_attr_dict)
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
    
    # fashion_CELoss_label: {attr_name: [gt1, gt2, gt3, ...], ...} 
    fashion_CELoss_label = dict()
    for attr_name in fashion_attr_dict.keys():
        gt_list = []
        for item in FASHION_SPECIAL_TOKENS:
            gt_list.extend(fashion_item_label[item][attr_name])
        fashion_CELoss_label[attr_name] = torch.tensor(gt_list).to(args.device)
    furniture_CELoss_label = dict()
    for attr_name in furniture_attr_dict.keys():
        gt_list = []
        for item in FURNITURE_SPECIAL_TOKENS:
            gt_list.extend(furniture_item_label[item][attr_name])
        furniture_CELoss_label[attr_name] = torch.tensor(gt_list).to(args.device)
    # print(fashion_CELoss_label)
    
    fashion_attr_embed_matrix = dict()
    for attr_name, tok_dict in fashion_attr_dict.items():
        fashion_attr_embed_matrix[attr_name] = torch.stack([emb_weight_clone[t[0]] for tl in tok_dict.values() for t in tl]).to(args.device)
    furniture_attr_embed_matrix = dict()
    for attr_name, tok_dict in furniture_attr_dict.items():
        furniture_attr_embed_matrix[attr_name] = torch.stack([emb_weight_clone[t[0]] for tl in tok_dict.values() for t in tl]).to(args.device)
    # print(furniture_attr_embed_matrix)

    for i in range(num_iter):
        for j, attr_name in enumerate(fashion_attr_dict.keys()):
            st_indices = []
            for fashion_st in FASHION_SPECIAL_TOKENS:
                st_repeat = len(fashion_item_label[fashion_st][attr_name])
                st_indices.extend(get_input_id(tokenizer, fashion_st) * st_repeat)

            # logits: (num_possibly_duplicated_items, num_concatenated_tokens)
            logits = emb(torch.tensor(st_indices).to(args.device)) @ fashion_attr_embed_matrix[attr_name].t()
            if j == 0:
                fashion_emb_loss = CELoss(logits, fashion_CELoss_label[attr_name])
            else: 
                fashion_emb_loss += CELoss(logits, fashion_CELoss_label[attr_name])
        for j, attr_name in enumerate(furniture_attr_dict.keys()):
            st_indices = []
            for furniture_st in FURNITURE_SPECIAL_TOKENS:
                st_repeat = len(furniture_item_label[furniture_st][attr_name])
                st_indices.extend(get_input_id(tokenizer, furniture_st) * st_repeat)
            # logits: (num_possibly_duplicated_items, num_concatenated_tokens)
            logits = emb(torch.tensor(st_indices).to(args.device)) @ furniture_attr_embed_matrix[attr_name].t()
            if j == 0:
                furniture_emb_loss = CELoss(logits, furniture_CELoss_label[attr_name])
            else:
                furniture_emb_loss += CELoss(logits, furniture_CELoss_label[attr_name])

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


def main():
    args.n_gpu = torch.cuda.device_count() if torch.cuda.is_available() else 0
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    args.device = device
    set_seed(args)

    tokenizer = BartTokenizerFast.from_pretrained(args.model_name_or_path)
    id2word = {v: k for k, v in tokenizer.get_vocab().items()}
    if args.add_special_tokens:
        if not os.path.exists(args.add_special_tokens):
            raise ValueError("Additional special tokens file {args.add_special_tokens} not found}")
        with open(args.add_special_tokens, "rb") as handle:
            special_tokens_dict = json.load(handle)
        num_added_toks = tokenizer.add_special_tokens(special_tokens_dict)
        logger.info(f"Added {num_added_toks} tokens")

    # Define Model
    model = MultiTaskBartForConditionalGenerationWithDisamb.from_pretrained(args.model_name_or_path)
    if args.add_special_tokens:
        model.resize_token_embeddings(len(tokenizer))
        model.vocab_size = len(tokenizer)
    model.config.decoder_start_token_id = 0
    model.to(args.device)

    with open(args.item2id, 'r') as f:
        item2id = json.load(f)

    train_api = api.PromptAPI(dial_split="train", 
                              data_dir=args.data_dir, 
                              dialogue_name_prefix=args.dialogue_name_prefix,
                              jsons_dir_name=args.jsons_dir_name,
                              images_dir_name=args.images_dir_name)

    fashion_meta = train_api.fashion_meta
    furniture_meta = train_api.furniture_meta
    all_objects_meta = dict()
    for meta in fashion_meta:
        object_special_id = item2id[meta.name]
        object_meta = {'asset_type': meta.asset_type, 'customer_review': str(meta.customer_review),
        'available_sizes': [available_sizes2st[size] for size in meta.available_sizes], 
        'color': meta.color, 'pattern': meta.pattern, 'brand': meta.brand, 
        'sleeve_length': meta.sleeve_length, 'type': meta.type, 'price': str(meta.price), 'size': meta.size
        }
        all_objects_meta[object_special_id] = object_meta
    for meta in furniture_meta:
        object_special_id = item2id[meta.name]
        object_meta = {'brand': meta.brand, 'color': meta.color, 'customer_review': str(meta.customer_review),
        'materials': meta.materials, 'price': meta.price, 'type': meta.type}
        all_objects_meta[object_special_id] = object_meta

    ## 对 object id 的 embedding 进行训练
    train_embedding_clip_way(args, model, tokenizer, all_objects_meta, args.embedding_train_epochs_start, do_tsne=False)

    def collate_bart(examples):
        # id2word = {v: k for k, v in tokenizer.get_vocab().items()}
        enc_input = list(map(lambda x: x[0], examples))
        enc_attention_mask = list(map(lambda x: x[1], examples))
        decoder_input = list(map(lambda x: x[2], examples))
        boxes = list(map(lambda x: x[3], examples))  
        misc = list(map(lambda x: x[4], examples))
        nocoref = list(map(lambda x: x[5], examples))
        disambiguation_labels = list(map(lambda x: x[6], examples))
        response = list(map(lambda x: x[7], examples))
        dst = list(map(lambda x: x[8], examples))  
        if tokenizer._pad_token is None:
            enc_input_pad = pad_sequence(enc_input, batch_first=True)
        else:
            enc_input_pad = pad_sequence(enc_input, batch_first=True, padding_value=tokenizer.pad_token_id)
        enc_attention_pad = pad_sequence(enc_attention_mask, batch_first=True, padding_value=0) # 0表示被mask掉
        decoder_input_pad = tokenizer(decoder_input, padding="longest", truncation=True, return_tensors="pt")

        response_pad = tokenizer(response, padding="longest", truncation=True, return_tensors="pt")

        # pdb.set_trace()
        return enc_input_pad, enc_attention_pad, decoder_input_pad.input_ids, decoder_input_pad.attention_mask, \
                boxes, misc, nocoref, torch.vstack(disambiguation_labels), response_pad.input_ids, response_pad.attention_mask, dst
    
    # train_dataset = get_dataset(args, tokenizer, all_objects_meta, train=True)
    train_dataset = LineByLineDataset(args.train_input_file, tokenizer, all_objects_meta)
    # pdb.set_trace()
    train_sampler = RandomSampler(train_dataset)
    train_dataloader = DataLoader(train_dataset, sampler=train_sampler, batch_size=args.train_batch_size, collate_fn=collate_bart)

    t_total = len(train_dataloader) * args.num_train_epochs
    warmup_steps = int(t_total * args.warmup_ratio)

    no_decay = ["bias", "LayerNorm.weight"]
    optimizer_grouped_parameters = [
        {
            "params": [
                p
                for n, p in model.named_parameters()
                if not any(nd in n for nd in no_decay)
            ],
            "weight_decay": args.weight_decay,
        },
        {
            "params": [
                p
                for n, p in model.named_parameters()
                if any(nd in n for nd in no_decay)
            ],
            "weight_decay": 0.0,
        }
    ]
    optimizer = AdamW(
        optimizer_grouped_parameters, lr=args.learning_rate, eps=args.adam_epsilon
    )
    scheduler = get_linear_schedule_with_warmup(
        optimizer, num_warmup_steps=warmup_steps, num_training_steps=t_total
    )

    # Check if saved optimizer or scheduler states exist
    if (args.model_name_or_path and os.path.isfile(os.path.join(args.model_name_or_path, "optimizer.pt")) and os.path.isfile(os.path.join(args.model_name_or_path, "scheduler.pt"))):
        optimizer.load_state_dict(torch.load(os.path.join(args.model_name_or_path, "optimizer.pt")))
        scheduler.load_state_dict(torch.load(os.path.join(args.model_name_or_path, "scheduler.pt")))
    
    # Train!
    logger.info("***** Running training *****")
    logger.info("  Num examples = %d", len(train_dataset))
    logger.info("  Num Epochs = %d", args.num_train_epochs)
    logger.info("  Train batch size = %d", args.train_batch_size)
    logger.info("  Total optimization steps = %d", t_total)

    global_step = 0
    epochs_trained = 0
    steps_trained_in_current_epoch = 0

    # Check if continuing training from a checkpoint
    if args.model_name_or_path and os.path.exists(args.model_name_or_path):
        try:
            checkpoint_suffix = args.model_name_or_path.split("-")[-1].split("/")[0]
            global_step = int(checkpoint_suffix)
            epochs_trained = global_step // len(train_dataloader)
            steps_trained_in_current_epoch = global_step % len(train_dataloader)

            logger.info("  Continuing training from checkpoint, will skip to saved global_step")
            logger.info("  Continuing training from epoch %d", epochs_trained)
            logger.info("  Continuing training from global step %d", global_step)
            logger.info("  Will skip the first %d steps in the first epoch", steps_trained_in_current_epoch)
        except ValueError:
            logger.info("  Starting fine-tuning.")

    tr_loss, logging_loss = 0.0, 0.0

    model.zero_grad()
    train_iterator = trange(epochs_trained, int(args.num_train_epochs), desc="Epoch")
    set_seed(args)  # Added here for reproducibility
    for epoch_i in train_iterator:
        epoch_iterator = tqdm(train_dataloader, desc="Iteration")
        for step, batch in enumerate(epoch_iterator):
            # Skip past any already trained steps if resuming training
            if steps_trained_in_current_epoch > 0:
                steps_trained_in_current_epoch -= 1
                continue
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
            dst = batch[10]

            # pdb.set_trace()
            model.train()
            model_outputs = model(
                        input_ids=enc_input,
                        attention_mask=enc_attention_mask,
                        decoder_input_ids=decoder_input[:, :-1],
                        decoder_attention_mask=decoder_attention_mask[:, :-1],
                        labels=decoder_input[:, 1:].contiguous(),
                        boxes=boxes,
                        misc=misc,
                        nocoref=nocoref,
                        response=response,
                        response_attention_mask=response_attention_mask,
                        disambiguation_labels=disambiguation_labels,
                        do_retrieval=args.do_retrieval,
                        return_dict=True,
                        dst=dst)

            loss = model_outputs[0] # [loss, masked_lm_loss, nocoref_loss, misc_loss, disam_loss, retrieval_loss]
            masked_lm_loss = model_outputs[1]
            nocoref_loss = model_outputs[2]
            misc_loss = model_outputs[3]
            disam_loss = model_outputs[4]
            retrieval_loss = model_outputs[5]
            dst_loss = model_outputs[6]
            epoch_iterator.set_postfix(loss=f'{loss:.4f}',
                                       loss_masked_lm=f'{masked_lm_loss:.4f}',
                                       loss_nocoref=f'{nocoref_loss:.4f}',
                                       loss_misc=f'{misc_loss:.4f}',
                                       loss_disam=f'{disam_loss:.4f}',
                                       loss_retrieval=f'{retrieval_loss:.4f}',
                                       loss_dst=f'{dst_loss:.4f}')

            loss.backward()
            tr_loss += loss.item()
            parameters_to_clip = [p for p in model.parameters() if p.grad is not None]
            torch.nn.utils.clip_grad_norm_(parameters_to_clip, args.max_grad_norm)

            optimizer.step()
            scheduler.step()  # Update learning rate schedule
            model.zero_grad()
            global_step += 1

            # 训练embedding
            if args.do_train_embedding_clip_way_during_training and global_step % args.embedding_train_steps == 0:
                train_embedding_clip_way(args, model, tokenizer, all_objects_meta, args.embedding_train_epochs_ongoing, do_tsne=False)

        # 保存模型
        logger.info('checkpoint saving!!')
        output_dir = os.path.join(args.output_dir, "epoch{}".format(epoch_i))
        os.makedirs(output_dir, exist_ok=True)
        model.save_pretrained(output_dir)
        tokenizer.save_pretrained(output_dir)
        torch.save(args, os.path.join(output_dir, "training_args.bin"))
        logger.info("Saving model checkpoint to %s", output_dir)



if __name__ == '__main__':
    main()