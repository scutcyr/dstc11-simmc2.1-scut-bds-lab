# coding=utf-8
# Copyright 2022 Research Center of Body Data Science from South China University of Technology. All rights reserved.
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
os.environ["TOKENIZERS_PARALLELISM"] = 'true' # 为了模型并行
import copy
import json
import torch
import random
import logging
import pdb
import numpy as np
from pprint import pformat
from tqdm import tqdm, trange
#import matplotlib.pyplot as plt
from sklearn.manifold import TSNE

from torch import nn
from torch.optim import AdamW as torch_AdamW
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data.distributed import DistributedSampler
from torch.utils.data import DataLoader, RandomSampler
from transformers import (
    AdamW,
    RobertaTokenizerFast,
    get_linear_schedule_with_warmup,
)


# 导入模型类
from models.modeling_simmc_roberta import RobertaForSIMMC

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
    if hasattr(model, "module"):
        emb = model.module.roberta.embeddings.word_embeddings
    else:
        emb = model.roberta.embeddings.word_embeddings
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
        logger.info("==== train embedding clip ==== %s" % i)
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

def train(args, train_dataset, model, tokenizer, all_objects_meta):
    # pdb.set_trace()
    logger.info("***** Preparing for training *****")

    args.train_batch_size = args.per_gpu_train_batch_size * max(1, args.n_gpu)

    def collate_bart(raw_examples):
        # pdb.set_trace()
        # id2word = {v: k for k, v in tokenizer.get_vocab().items()}
        vocab = tokenizer.get_vocab()
        examples = copy.deepcopy(raw_examples)
        examples = list(examples)
        for idx in range(len(examples)):
            examples[idx] = list(examples[idx])
        for i, exm in enumerate(examples):
            ## 如果输入长度超过 512，从后面的 obj id 开始截断
            if len(exm[0]) > 512:
                # pdb.set_trace()
                input_list = exm[0].tolist()
                att_list = exm[1].tolist()
                boxes_list = exm[3]
                misc_list = exm[4]
                while len(input_list) > 512:
                    last_obj_idx = len(input_list) - input_list[::-1].index(vocab["<OBJ>"]) - 1
                    del input_list[last_obj_idx:last_obj_idx+3]
                    boxes_list.pop()
                    misc_list.pop()
                att_list = att_list[:len(input_list)]
                examples[i][0] = torch.tensor(input_list, dtype=torch.long)
                examples[i][1] = torch.tensor(att_list, dtype=torch.long)
                examples[i][3] = boxes_list
                examples[i][4] = misc_list
                examples[i][8]["pos"] = input_list.index(vocab["<DST>"])

        # for exm in examples:
        #     if len(exm[0]) > 512:
        #         pdb.set_trace()
           
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

        return enc_input_pad, enc_attention_pad, decoder_input_pad.input_ids, decoder_input_pad.attention_mask, \
                boxes, misc, nocoref, torch.vstack(disambiguation_labels), response_pad.input_ids, response_pad.attention_mask, dst
    
    train_sampler = RandomSampler(train_dataset) if args.local_rank == -1 else DistributedSampler(train_dataset)
    train_dataloader = DataLoader(train_dataset, sampler=train_sampler, batch_size=args.train_batch_size, collate_fn=collate_bart)
    
    if args.max_steps > 0:
        t_total = args.max_steps
        args.num_train_epochs = args.max_steps // (len(train_dataloader) // args.gradient_accumulation_steps) + 1
    else:
        t_total = len(train_dataloader) // args.gradient_accumulation_steps * args.num_train_epochs  

    # Prepare optimizer and schedule (linear warmup and decay)
    no_decay = ['bias', 'LayerNorm.weight']
    optimizer_grouped_parameters = [
        {'params': [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)], 'weight_decay': args.weight_decay},
        {'params': [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
        ]
    optimizer = AdamW(optimizer_grouped_parameters, lr=args.learning_rate, eps=args.adam_epsilon)
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=int(args.warmup_ratio*t_total), num_training_steps=t_total)

    if args.fp16:
        try:
            from apex import amp
        except ImportError:
            raise ImportError(
                "Please install apex from https://www.github.com/nvidia/apex to use fp16 training."
            )
        model, optimizer = amp.initialize(
            model, optimizer, opt_level=args.fp16_opt_level
        )

    # multi-gpu training (should be after apex fp16 initialization)
    if args.n_gpu > 1:
        model = torch.nn.DataParallel(model)

    # Distributed training (should be after apex fp16 initialization)
    if args.local_rank != -1:
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.local_rank],
                                                          output_device=args.local_rank,
                                                          find_unused_parameters=True)

    # Train!
    logger.info("***** Running training *****")
    logger.info("  Num examples = %d", len(train_dataset))
    logger.info("  Num Epochs = %d", args.num_train_epochs)
    logger.info("  Instantaneous batch size per GPU = %d", args.per_gpu_train_batch_size)
    logger.info("  Total train batch size (w. parallel, distributed & accumulation) = %d",
                   args.train_batch_size * args.gradient_accumulation_steps * (torch.distributed.get_world_size() if args.local_rank != -1 else 1))
    logger.info("  Gradient Accumulation steps = %d", args.gradient_accumulation_steps)
    logger.info("  Total optimization steps = %d", t_total)
    

    global_step = 0
    tr_loss, logging_loss = 0.0, 0.0
    model.zero_grad()
    train_iterator = trange(int(args.num_train_epochs), desc="Epoch", disable=args.local_rank not in [-1, 0])
    set_seed(args)
    id2word = {v: k for k, v in tokenizer.get_vocab().items()}
    for epoch_i in train_iterator:
        epoch_iterator = tqdm(train_dataloader, desc="Iteration", disable=args.local_rank not in [-1, 0])
        for step, batch in enumerate(epoch_iterator):
            model.train()

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
                        dst=dst,
                        id2word=id2word)

            # loss = model_outputs[0]
            nocoref_loss = model_outputs[1]
            misc_loss = model_outputs[2]
            disam_loss = model_outputs[3]
            retrieval_loss = model_outputs[4]
            dst_loss = model_outputs[5]
            loss = args.lambda_nocoref_loss * nocoref_loss + \
                   args.lambda_misc_loss * misc_loss + \
                   args.lambda_disam_loss * disam_loss + \
                   args.lambda_dst_loss * dst_loss

            if args.local_rank in [-1, 0]:
                epoch_iterator.set_postfix(loss=f'{loss:.4f}',
                                        loss_nocoref=f'{nocoref_loss:.4f}',
                                        loss_misc=f'{misc_loss:.4f}',
                                        loss_disam=f'{disam_loss:.4f}',
                                        loss_dst=f'{dst_loss:.4f}')
                logger.info("epoch: {}, global_step: {}, learning_rate: {}, loss: {}, \
                            loss_nocoref: {}, loss_misc: {}, loss_disam: {}, loss_dst: {}".format(
                                str(epoch_i), str(global_step), str(scheduler.get_lr()[0]), str(loss.item()),
                                str(nocoref_loss.item()), str(misc_loss.item()), str(disam_loss.item()), str(dst_loss)
                            ))

            if args.n_gpu > 1:
                loss = loss.mean()  # mean() to average on multi-gpu parallel training
            if args.gradient_accumulation_steps > 1:
                loss = loss / args.gradient_accumulation_steps

            if args.fp16:
                with amp.scale_loss(loss, optimizer) as scaled_loss:
                    scaled_loss.backward()
                torch.nn.utils.clip_grad_norm_(amp.master_params(optimizer), args.max_grad_norm)
            else:
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
        
            tr_loss += loss.item()
            if (step + 1) % args.gradient_accumulation_steps == 0:
                optimizer.step()
                scheduler.step()  # Update learning rate schedule
                model.zero_grad()
                global_step += 1

                # 训练 object id embedding
                if args.do_train_embedding_clip_way_during_training and global_step % args.embedding_train_steps == 0:
                    train_embedding_clip_way(args, model, tokenizer, all_objects_meta, args.embedding_train_epochs_ongoing, do_tsne=False)
                

        ## 每个 epoch 保存一次模型
        if args.local_rank in [-1, 0]:
            # Save model checkpoint
            output_dir = os.path.join(args.output_dir, 'epoch{}'.format(epoch_i))
            if not os.path.exists(output_dir):
                os.makedirs(output_dir)
            model_to_save = model.module if hasattr(model, 'module') else model  # Take care of distributed/parallel training
            model_to_save.save_pretrained(output_dir)
            tokenizer.save_pretrained(output_dir)
            torch.save(args, os.path.join(output_dir, 'training_args.bin'))
            logger.info("Saving model checkpoint to %s", output_dir)

            if args.max_steps > 0 and global_step > args.max_steps:
                epoch_iterator.close()
                break
        if args.max_steps > 0 and global_step > args.max_steps:
            train_iterator.close()
            break

    return global_step, tr_loss / global_step



def main():
    # pdb.set_trace()
    # Setup CUDA, GPU & distributed training
    if args.local_rank == -1 or args.no_cuda:
        device = torch.device("cuda" if torch.cuda.is_available() and not args.no_cuda else "cpu")
        args.n_gpu = torch.cuda.device_count()
    else:  # Initializes the distributed backend which will take care of sychronizing nodes/GPUs
        torch.cuda.set_device(args.local_rank)
        device = torch.device("cuda", args.local_rank)
        torch.distributed.init_process_group(backend='nccl')
        # torch.distributed.init_process_group(backend='gloo')
        args.n_gpu = 1
    args.device = device

    logger.warning("Process rank: %s, device: %s, n_gpu: %s, distributed training: %s, 16-bits training: %s",
                    args.local_rank, device, args.n_gpu, bool(args.local_rank != -1), args.fp16)

    # Load pretrained model and tokenizer
    if args.local_rank not in [-1, 0]:
        torch.distributed.barrier()  # Make sure only the first process in distributed training will download model & vocab

    # Set seed
    set_seed(args)

    tokenizer = RobertaTokenizerFast.from_pretrained(args.model_name_or_path)
    if args.add_special_tokens:
        if not os.path.exists(args.add_special_tokens):
            raise ValueError("Additional special tokens file {args.add_special_tokens} not found}")
        with open(args.add_special_tokens, "rb") as handle:
                special_tokens_dict = json.load(handle)
        num_added_toks = tokenizer.add_special_tokens(special_tokens_dict)
        logger.info(f"Added {num_added_toks} tokens")
        logger.info(f"All special tokens: {tokenizer.all_special_tokens}")
    id2word = {v: k for k, v in tokenizer.get_vocab().items()}

    # Define Model
    model = RobertaForSIMMC.from_pretrained(args.model_name_or_path)
    if args.add_special_tokens:
        model.resize_token_embeddings(len(tokenizer))
        model.vocab_size = len(tokenizer)
    # model.config.decoder_start_token_id = tokenizer.bos_token_id
    model.config.decoder_start_token_id = 0

    if args.local_rank == 0:
        torch.distributed.barrier()  # Make sure only the first process in distributed training will download model & vocab

    model.to(args.device)

    logger.info("Training/evaluation parameters %s", args)


    if args.local_rank not in [-1, 0]:
        torch.distributed.barrier()  # Barrier to make sure only the first process in distributed training process the dataset, and the others will use the cache
    # Prepare dataset
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

    logger.info("Loading Train Dataset...")
    # train_dataset = get_dataset(args, tokenizer, all_objects_meta, train=True)
    train_dataset = LineByLineDataset(args.train_input_file, tokenizer, all_objects_meta)
    
    if args.local_rank == 0:
        torch.distributed.barrier()

    logger.info("Running train_embedding_clip_way!!!")
    # train_embedding_clip_way(args, model, tokenizer, all_objects_meta, args.embedding_train_epochs_start, do_tsne=False)

    logger.info("Running train!!!")
    global_step, train_loss = train(args, train_dataset, model, tokenizer, all_objects_meta)
    logger.info("global_step = %s, average loss = %s" % (global_step, train_loss))


if __name__ == '__main__':
    main()
