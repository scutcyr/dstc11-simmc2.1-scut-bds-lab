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


# File: eval_model_args.py
# Description: The arguments for evalating model
# Repository: https://github.com/scutcyr/dstc11-simmc2.1-iflytek
# Mail: [yrchen5@iflytek.com](mailto:yrchen5@iflytek.com) or [eeyirongchen@mail.scut.edu.cn](mailto:eeyirongchen@mail.scut.edu.cn)
# Date: 2022/10/17
# Usage:
#    from eval_model_args import parser

import argparse


parser = argparse.ArgumentParser()

# Updated by Yirong Chen 
# model_parallel: 设置模型是否并行，也就是将一个超大模型放在多张GPU上
parser = argparse.ArgumentParser()
# model_parallel: 设置模型是否并行，也就是将一个超大模型放在多张GPU上
parser.add_argument("--model_parallel", action="store_true", help="Set model_parallel=True(设置模型是否流水线并行，主要用于T5-11B和UL-2)")
parser.add_argument(
    "--model_type",
    default="mt-bart-large",
    type=str,
    choices=['gpt-2', 'mt-bart-large', 'mt-bart-large-disamb', 'mt-bart-attr', 'mt-bart_joint_disam_coref', 'mt-bart_add_intent',
                'mt-blenderbot', 'mt-t5', 'mt-ul2', 'mt-flava', 'gen-ofa', 'mt-ofa'],
    help="The model architecture to be trained or fine-tuned.(设置模型的类型)",
)
parser.add_argument(
    "--model_dir",
    type=str,
    required=True,
    help='model dir which contains model, optimizer, scheduler weight files.(指定模型的路径)'
)
parser.add_argument(
    "--prompts_from_file",
    type=str,
    default=None,
    required=True,
    help='.txt file. One line is User : I need a new pair of jeans. <SOO><NOCOREF><OBJ><56>[(-0.2831,-0.1985,-0.1638,0.1697,0.0439,0.2563)]<@1269><OBJ><85>[(-0.1135,-0.1846,-0.0827,0.1498,0.0103,0.3232)]<@1007><OBJ><57>[(-0.0594,-0.1657,-0.0138,0.1289,0.0134,0.2463)]<@1214><OBJ><58>[(0.0392,-0.1716,0.0954,0.1229,0.0166,0.2809)]<@1228><OBJ><59>[(0.0387,-0.1965,0.0769,0.1418,0.0129,1.0000)]<@1237><OBJ><86>[(0.0875,-0.1736,0.1273,0.1179,0.0116,0.3034)]<@1006><OBJ><87>[(0.1442,0.1388,0.1819,0.4144,0.0104,0.4655)]<@1184><OBJ><63>[(0.0376,0.1697,0.1002,0.4980,0.0205,0.3195)]<@1015><OBJ><61>[(-0.2598,0.2622,-0.1654,0.4980,0.0223,0.3961)]<@1013><OBJ><62>[(-0.0928,0.2164,-0.0186,0.4980,0.0209,0.3786)]<@1241><OBJ><66>[(0.1596,0.4672,0.1861,0.4960,0.0008,0.3962)]<@1070><EOO> => Belief State : '
)
parser.add_argument(
    '--item2id',
    type=str,
    required=True
)
parser.add_argument(
    '--batch_size',
    type=int,
    default=36
)
parser.add_argument(
    "--add_special_tokens",
    default=None,
    type=str,
    help="Optional file containing a JSON dictionary of special tokens that should be added to the tokenizer.",
)
parser.add_argument(
    "--add_bad_words",
    default=None,
    type=str,
    help="Optional file containing a JSON dictionary of bad words that are not allowed to be generated.",
)
parser.add_argument("--length", type=int, default=150)
parser.add_argument("--min_length", type=int, default=10)
parser.add_argument(
    "--sampling_method",
    default="None",
    type=str,
    choices=['greedy-decoding', 'multinomial-sampling', 'beam-search-decoding', 'beam-search-multinomial-sampling', 
                'diverse-beam-search-decoding', 'constrained-beam-search-decoding'],
    help="The model generation decoding method",
)
parser.add_argument(
    "--do_sample", action="store_true", help="do_sample in model.generate() method"
)
parser.add_argument(
    "--num_beams", type=int, default=1, help="num_beams"
)
parser.add_argument(
    "--temperature",
    type=float,
    default=1.0,
    help="temperature of 1.0 has no effect, lower tend toward greedy sampling. (默认是1.0，温度越低（小于1），softmax输出的贫富差距越大；温度越高，softmax差距越小。)",
)
parser.add_argument(
    "--repetition_penalty",
    type=float,
    default=1.0,
    help="primarily useful for CTRL model; in that case, use 1.2. (默认是1.0，重复词惩罚)",
)
parser.add_argument(
    "--length_penalty",
    type=float,
    default=1.0,
    help="Exponential penalty to the length. 1.0 means that the beam score is penalized by the sequence length. 0.0 means no penalty. Set to values < 0.0 in order to encourage the model to generate longer sequences, to a value > 0.0 in order to encourage the model to produce shorter sequences.",
)
# length_penalty
parser.add_argument(
    "--num_return_sequences",
    type=int,
    default=1,
    help="The number of samples to generate.",
)
parser.add_argument("--k", type=int, default=0, help="top-k-filtering 算法保留多少个 最高概率的词 作为候选")
parser.add_argument("--p", type=float, default=0.9, help="已知生成各个词的总概率是1（即默认是1.0）如果top_p小于1，则从高到低累加直到top_p，取这前N个词作为候选")
parser.add_argument(
    "--seed", type=int, default=42, help="random seed for initialization"
)
parser.add_argument(
    "--no_cuda", action="store_true", help="Avoid using CUDA when available"
)
parser.add_argument(
    "--correct_act",
    type=str,
    default=None,
    help="correct wrongly generated action with correct_act dictionary",
)
parser.add_argument(
    "--check_disamb_candi", action="store_true", help="Check whether the is_nodisamb and the predicted disamb_obj_list is conflict."
)
parser.add_argument(
    "--check_disamb_and_coref", action="store_true", help="Check whether the is_disamb and the predicted coref_obj_list is conflict."
)
parser.add_argument(
    "--check_isnocoref_and_coref", action="store_true", help="Check whether the isnocoref and the predicted coref_obj_list is conflict."
)
parser.add_argument(
    "--path_output",
    type=str,
    required=True,
    help="Path to output predictions in a line separated text file or the output dir when eval_all_checkpoints.",
)
parser.add_argument(
    "--eval_all_checkpoints",
    action="store_true",
    help="Evaluate all checkpoints starting with the same prefix as model_name_or_path ending and ending with step number",
)

parser.add_argument(
    "--do_calculate_score",
    action="store_true",
    help="对模型的预测结果进行评分（devtest时设置为True, teststd时设置为False）",
)

parser.add_argument(
    "--input_path_target", help="path for target, line-separated format (.txt)"
)
parser.add_argument(
    "--data_json_path",
    default="../data/simmc2_dials_dstc10_devtest.json",
    help="Data with .json format gold responses",
)
parser.add_argument(
    "--output_json_response_path", default=None, help="Responses generated by the model"
)
parser.add_argument(
    "--output_path_csv_report", help="path for saving evaluation summary (.csv)"
)
parser.add_argument(
    "--dialog_meta_data",
    type=str,
    default='../data_object_special/simmc2_dials_dstc10_devtest_inference_disambiguation.json'
)
parser.add_argument(
    "--image_dir",
    type=str,
    default="simmc2_scene_images_dstc10_public",
    help='images_dir_name'
)
parser.add_argument(
    "--image_path_file",
    default=None,
    type=str,
    help='preprocessed train_image_path_file path, line-by-line format'
)
parser.add_argument(
    "--single_round_evaluation",
    dest="single_round_evaluation",
    action="store_true",
    default=False,
    help="Single round evaluation for hidden split",
)
parser.add_argument(
    "--not_generate_predict_file_again",
    action="store_true",
    default=False,
    help="指定该参数时，如果已经存在args.path_output，则不重新进行推理，直接利用之前已经生成的·结果计算模型评分",
)

# 用于OFA模型
parser.add_argument('--sample_patch_num', type=int, default=256, help="图像部分的patch数量")

# 训练模型用到的数据文件路径
parser.add_argument(
    "--data_dir",
    type=str,
    default="/yrfs1/intern/yrchen5/dstc11_simmc2.1_iflytek/data",
    help='the path of the dataset'
)

parser.add_argument(
    "--cal_diff_f1_based_on_previously_mentioned",
    action="store_true",
    default=False,
    help="指定该参数时，根据object_id是否属于previously_mentioned进行分类，然后计算F1",
)

parser.add_argument(
    "--multimodal_context_json_file",
    type=str,
    default=None,
    help='multimodal_context_json_file'
)