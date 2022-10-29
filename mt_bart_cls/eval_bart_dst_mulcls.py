import os
import re
import ast
import copy
import json
import argparse
import logging
import argparse
from tqdm import tqdm

import torch
import numpy as np

from torch.utils.data import DataLoader, SequentialSampler
from torch.nn.utils.rnn import pad_sequence
from transformers import BartTokenizerFast

from models.modeling_simmc_bart_disamcands_dst_mulcls import MultiTaskBartForConditionalGenerationWithDisamb

import pdb

# 导入数据处理类
from utils import api
from utils.simmc21_dataset import LineByLineDataset, available_sizes2st

from utils.dst_label import DST_DICT

# 日志模块定义
logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s",
    datefmt="%m/%d/%Y %H:%M:%S",
    level=logging.INFO,
)
logger = logging.getLogger(__name__)

MAX_LENGTH = int(10000)  # Hardcoded max length to avoid infinite loop


def set_seed(args):
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if args.n_gpu > 0:
        torch.cuda.manual_seed_all(args.seed)


def remove_bos_eos_startequal(text):
    text = text.split("</s>")[0].replace('<s>', '')
    return text


def adjust_length_to_model(length, max_sequence_length):
    if length < 0 and max_sequence_length > 0:
        length = max_sequence_length
    elif 0 < max_sequence_length < length:
        length = max_sequence_length  # No generation bigger than model size
    elif length < 0:
        length = MAX_LENGTH  # avoid infinite loop
    return length


def main():
    parser = argparse.ArgumentParser()
    # 测试时需要的参数
    parser.add_argument("--prompts_from_file", type=str, default=None, required=True, help="test input file")
    parser.add_argument("--path_output", type=str, required=True, help="test output file")
    parser.add_argument("--test_model_dir", type=str, required=True, help="test model dir")
    parser.add_argument("--log_file", type=str, default="", help='log file')
    parser.add_argument("--item2id", required=True, type=str, help='item2id filepath')
    parser.add_argument("--add_bad_words", type=str, default="", help='bad words avoid generate')
    parser.add_argument("--test_batch_size", type=int, default=36)
    parser.add_argument("--length", type=int, default=150)
    parser.add_argument("--temperature", type=float, default=1.0, help="temperature of 1.0 has no effect, lower tend toward greedy sampling")
    parser.add_argument("--repetition_penalty", type=float, default=1.0, help="primarily useful for CTRL model; in that case, use 1.2")
    parser.add_argument("--k", type=int, default=0)
    parser.add_argument("--p", type=float, default=0.9)
    parser.add_argument("--correct_act", type=str, default=None, help="correct wrongly generated action with correct_act dictionary")
    parser.add_argument("--check_disamb_candi", action="store_true", help="Check whether the is_nodisamb and the predicted disamb_obj_list is conflict.")
    parser.add_argument("--check_disamb_and_coref", action="store_true", help="Check whether the is_disamb and the predicted coref_obj_list is conflict.")
    parser.add_argument("--check_isnocoref_and_coref", action="store_true", help="Check whether the isnocoref and the predicted coref_obj_list is conflict.")
    parser.add_argument("--seed", default=42, type=int)
    parser.add_argument("--no_cuda", action="store_true", help="Avoid using CUDA when available")
    args = parser.parse_args()

    args.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    args.n_gpu = 0 if args.no_cuda else torch.cuda.device_count()
    set_seed(args)

    if args.prompts_from_file and not os.path.exists(args.prompts_from_file):
        raise Exception(f"prompt file '{args.prompts_from_file}' not found")

    tokenizer = BartTokenizerFast.from_pretrained(args.test_model_dir)
    model = MultiTaskBartForConditionalGenerationWithDisamb.from_pretrained(args.test_model_dir)
    model.to(args.device)
    model.eval()
    # 2022.08.19 20:40 统一将 decoder_start_token_id 设置为 tokenizer.bos_token_id
    model.config.decoder_start_token_id = tokenizer.bos_token_id
    
    box_embedding = model.box_embedding
    disambiguation_head = model.disambiguation_head
    nocoref_head= model.nocoref_head
    fashion_enc_head = model.fashion_enc_head
    furniture_enc_head = model.furniture_enc_head

    args.length = adjust_length_to_model(args.length, max_sequence_length=model.config.max_position_embeddings)
    logger.info(args)

    def collate_bart(examples):
        id2word = {v: k for k, v in tokenizer.get_vocab().items()}
        enc_input = list(map(lambda x: x[0], examples))
        enc_attention_mask = list(map(lambda x: x[1], examples))
        original_lines = list(map(lambda x: x[2], examples))
        boxes = list(map(lambda x: x[3], examples))  
        misc = list(map(lambda x: x[4], examples))
        nocoref = list(map(lambda x: x[5][0], examples)) ## 这里需要取一下前面的index
        disambiguation_labels = list(map(lambda x: x[6], examples))
        response = list(map(lambda x: x[7], examples))
        dst = list(map(lambda x: x[8], examples))  

        # pdb.set_trace()
        if tokenizer._pad_token is None:
            enc_input_pad = pad_sequence(enc_input, batch_first=True)
        else:
            enc_input_pad = pad_sequence(enc_input, batch_first=True, padding_value=tokenizer.pad_token_id)
        enc_attention_pad = pad_sequence(enc_attention_mask, batch_first=True, padding_value=0) # 0表示mask
        return enc_input_pad, enc_attention_pad, original_lines, boxes, misc, nocoref, dst
    
    with open(args.item2id, 'r') as f:
        item2id = json.load(f)
    
    train_api = api.PromptAPI(dial_split="train", 
                              data_dir="/ps2/sli/data/data_taowang49/projects/19_dstc11/simmc2.1_solutions/work/simmc2.1-iflytek/data")

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

    
    decode_dataset = LineByLineDataset(args.prompts_from_file, tokenizer, all_objects_meta)
    decode_sampler = SequentialSampler(decode_dataset)
    decode_dataloader = DataLoader(
        decode_dataset,
        sampler=decode_sampler,
        batch_size=args.test_batch_size,
        collate_fn=collate_bart
    )

    # 增加 bad_words_ids
    # List of token ids that are not allowed to be generated.
    if args.add_bad_words:
        with open(args.add_bad_words, 'r') as f:
            add_bad_words = json.load(f)
        bad_words = add_bad_words["bad_words_for_generate"]
        bad_words_ids = tokenizer(bad_words, add_prefix_space=True, add_special_tokens=False).input_ids
        print("bad_words_ids=", bad_words_ids)
    else:
        bad_words_ids = None

    tokenizer_id2token = {v: k for k, v in tokenizer.get_vocab().items()}
    id2word = {v: k for k, v in tokenizer.get_vocab().items()}
    results = []
    n_prompts = len(decode_dataset)
    for i, batch in enumerate(tqdm(decode_dataloader, desc='Decoding')):
        enc_input = batch[0].to(args.device)
        enc_input_attention = batch[1].to(args.device)
        original_lines = batch[2]
        boxes = batch[3] # batch, num_obj_per_line, 6
        misc = batch[4]  # batch, num_obj_per_line, dict
        nocoref = batch[5]
        dst = batch[6]

        batch_size = len(misc)
        # assert batch_size == 1, "batch_size is not 1 !!"
        # pdb.set_trace()
        with torch.no_grad():
            ## 加上 box embedding
            inputs_embeds = model.model.encoder.embed_tokens(enc_input) * model.model.encoder.embed_scale
            for b_idx in range(batch_size):  # in a batch
                box_embedded = box_embedding(torch.tensor(boxes[b_idx]).to(args.device))  # (num_obj_per_line, d_model)
                for obj_idx in range(len(misc[b_idx])):
                    pos = misc[b_idx][obj_idx]['pos']
                    inputs_embeds[b_idx][pos] += box_embedded[obj_idx]
            encoder_outputs = model.model.encoder(inputs_embeds=inputs_embeds, attention_mask=enc_input_attention, return_dict=True)  # check this line

            enc_last_hidden_state = encoder_outputs.last_hidden_state

            disambiguation_logits = disambiguation_head(enc_last_hidden_state[:, 1, :]) # bs, d_model --> bs, 2
            is_disamb = disambiguation_logits.argmax(dim=-1).squeeze().bool() # 1表示有歧义
            is_nodisamb = ~ is_disamb # 取反

            nocoref_logits = torch.stack([nocoref_head(enc_last_hidden_state[b_idx][nocoref[b_idx]]) for b_idx in range(batch_size)])
            is_nocoref = nocoref_logits.argmax(dim=1).bool() # 1表示无多模态共指消解

            ## 对话状态跟踪
            for bidx in range(batch_size):
                dst_pos = dst[bidx]["pos"]
                dst_tokens = [id2word[int(enc_input[bidx][dst_pos+i])] for i in range(13)]
                logger.info("====DST TOKENS====\t%s" % " ".join(dst_tokens))
                print("====DST TOKENS====\t%s" % " ".join(dst_tokens))
            ## 每个 special token 上面做预测，context后面拼了13个special token
            dst_id_state = torch.stack([enc_last_hidden_state[b_idx, dst[b_idx]["pos"]:dst[b_idx]["pos"]+13, :] for b_idx in range(batch_size)])
            dst_act, dst_request_slots, dst_type, dst_price, dst_customer_review, dst_brand, dst_size, \
                dst_pattern, dst_color, dst_sleeve_length, dst_available_sizes, dst_materials, \
                    dst_customer_rating = model.dst_head(dst_id_state)
            # pdb.set_trace()
            dst_act_pred = dst_act.argmax(dim=-1)
            dst_request_slots_pred = (dst_request_slots > 0.5).int()
            dst_type_pred = dst_type.argmax(dim=-1)  
            dst_price_pred = dst_price.argmax(dim=-1)
            dst_customer_review_pred = dst_customer_review.argmax(dim=-1)
            dst_brand_pred = dst_brand.argmax(dim=-1)
            dst_size_pred = dst_size.argmax(dim=-1)
            dst_pattern_pred = dst_pattern.argmax(dim=-1)
            dst_color_pred = dst_color.argmax(dim=-1)
            dst_sleeve_length_pred = dst_sleeve_length.argmax(dim=-1)
            dst_available_sizes_pred = (dst_available_sizes > 0.5).int()
            dst_materials_pred = dst_materials.argmax(dim=-1)
            dst_customer_rating_pred = dst_customer_rating.argmax(dim=-1)

            dst_act_pred_res = [DST_DICT["act"][x] for x in dst_act_pred]
            dst_type_pred_res = [DST_DICT["type"][x] for x in dst_type_pred]
            dst_price_pred_res = [DST_DICT["price"][x] for x in dst_price_pred]
            dst_customer_review_pred_res = [DST_DICT["customerReview"][x] for x in dst_customer_review_pred]
            dst_brand_pred_res = [DST_DICT["brand"][x] for x in dst_brand_pred]
            dst_size_pred_res = [DST_DICT["size"][x] for x in dst_size_pred]
            dst_pattern_pred_res = [DST_DICT["pattern"][x] for x in dst_pattern_pred]
            dst_color_pred_res = [DST_DICT["color"][x] for x in dst_color_pred]
            dst_sleeve_length_pred_res = [DST_DICT["sleeveLength"][x] for x in dst_sleeve_length_pred]
            dst_materials_pred_res = [DST_DICT["materials"][x] for x in dst_materials_pred]
            dst_customer_rating_pred_res = [DST_DICT["customerRating"][x] for x in dst_customer_rating_pred]
            dst_request_slots_pred_res = []
            for request_slots in dst_request_slots_pred:
                tmp = []
                for idx, req_slot in enumerate(request_slots):
                    if req_slot == 1:
                        tmp.append(DST_DICT["request_slots"][idx])
                dst_request_slots_pred_res.append(tmp)
            dst_available_sizes_pred_res = []
            for available_sizes in dst_available_sizes_pred:
                tmp = []
                for idx, ava_size in enumerate(available_sizes):
                    if ava_size == 1:
                        tmp.append(DST_DICT["availableSizes"][idx])
                dst_available_sizes_pred_res.append(tmp)

            dst_list = []
            for idx in range(len(dst_act_pred_res)):
                dst_res = {"act": dst_act_pred_res[idx], "act_attributes": {"slot_values": {}, "request_slots": dst_request_slots_pred_res[idx]}}
                if dst_type_pred_res[idx] != "None": dst_res["act_attributes"]["slot_values"]["type"] = dst_type_pred_res[idx]
                if dst_price_pred_res[idx] != "None": dst_res["act_attributes"]["slot_values"]["price"] = dst_price_pred_res[idx]
                if dst_customer_review_pred_res[idx] != "None": dst_res["act_attributes"]["slot_values"]["customerReview"] = dst_customer_review_pred_res[idx]
                if dst_brand_pred_res[idx] != "None": dst_res["act_attributes"]["slot_values"]["brand"] = dst_brand_pred_res[idx]
                if dst_size_pred_res[idx] != "None": dst_res["act_attributes"]["slot_values"]["size"] = dst_size_pred_res[idx]
                if dst_pattern_pred_res[idx] != "None": dst_res["act_attributes"]["slot_values"]["pattern"] = dst_pattern_pred_res[idx]
                if dst_color_pred_res[idx] != "None": dst_res["act_attributes"]["slot_values"]["color"] = dst_color_pred_res[idx]
                if dst_sleeve_length_pred_res[idx] != "None": dst_res["act_attributes"]["slot_values"]["sleeveLength"] = dst_sleeve_length_pred_res[idx]
                if dst_materials_pred_res[idx] != "None": dst_res["act_attributes"]["slot_values"]["materials"] = dst_materials_pred_res[idx]
                if dst_customer_rating_pred_res[idx] != "None": dst_res["act_attributes"]["slot_values"]["customerRating"] = dst_customer_rating_pred_res[idx]
                if dst_available_sizes_pred_res[idx] != []: dst_res["act_attributes"]["slot_values"]["availableSizes"] = dst_available_sizes_pred_res[idx]
                dst_list.append(dst_res)

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
            
            # 检查歧义句子识别结果与歧义候选识别结果是否有冲突
            disamb_check_result = torch.logical_and(is_nodisamb.cpu(), torch.tensor(disamb_check, dtype=torch.bool))
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

            # output_sequences = model.generate(
            #     max_length=args.length + inputs_embeds.size()[1],
            #     temperature=args.temperature,
            #     top_k=args.k,
            #     top_p=args.p,
            #     bad_words_ids=bad_words_ids,
            #     repetition_penalty=args.repetition_penalty,
            #     do_sample=True,
            #     encoder_outputs=encoder_outputs)
            output_sequences = torch.Tensor([[10, 11, 12] for i in range(batch_size)]).long()

            # Remove the batch dimension when returning multiple sequences
            if len(output_sequences.shape) > 2:
                output_sequences.squeeze_()     

            # pdb.set_trace()
            predicts = tokenizer.batch_decode(output_sequences, include_special_token=True)
            for sequence_idx, text in enumerate(predicts):
                text = remove_bos_eos_startequal(text)
                text = "ASK:GET [  ] ( customerReview ) <EOB> Which one? <EOS>"
                coref_objs = coref_obj_list[sequence_idx]
                disamb_objs = disamb_obj_list[sequence_idx]
                dst_dict = dst_list[sequence_idx]
                results.append((text, coref_objs, disamb_objs, dst_dict))
                
    with open(args.path_output, "w") as f_out:
        for text, coref_objs, disamb_objs, dst_dict in results:
            f_out.write("%s\t%s\t%s\t%s\n" % (text, json.dumps(coref_objs), json.dumps(disamb_objs), json.dumps(dst_dict)))

if __name__ == "__main__":
    main()
