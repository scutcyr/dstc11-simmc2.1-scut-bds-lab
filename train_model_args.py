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


# File: train_model_args.py
# Description: The arguments for training model
# Repository: https://github.com/scutcyr/dstc11-simmc2.1-scut-bds-lab
# Mail: [eeyirongchen@mail.scut.edu.cn](mailto:eeyirongchen@mail.scut.edu.cn)
# Date: 2022/10/12
# Usage:
#    from train_model_args import parser
#    args = parser.parse_args()

import argparse


parser = argparse.ArgumentParser()

# Updated by Yirong Chen 
# model_parallel: 设置模型是否并行，也就是将一个超大模型放在多张GPU上
parser.add_argument(
    "--model_parallel",
    action="store_true",
    help="Set model_parallel=True",
)
# 模型读取与保存
parser.add_argument(
    "--model_name_or_path",  # 替换--model_dir
    default="facebook/bart-large",
    type=str,
    required=True,
    help="The model checkpoint for weights initialization. Leave None if you want to train a model from scratch.",
)
parser.add_argument(
    "--model_type",
    default="mt-bart-large",
    type=str,
    choices=['gpt-2', 'mt-bart-large', 'mt-bart-large-disamb', 'mt-bart-attr', 'mt-bart_joint_disam_coref', 'mt-bart_add_intent',
                'mt-blenderbot', 'mt-t5', 'mt-ul2', 'mt-flava', 'gen-ofa', 'mt-ofa'],
    help="The model architecture to be trained or fine-tuned.",
)
parser.add_argument(
    "--output_dir",
    required=True,
    type=str,
    help="The model checkpoint saving path",
)
parser.add_argument(
    "--should_continue",
    action="store_true",
    help="Whether to continue from latest checkpoint in output_dir",
)
parser.add_argument(
    "--save_optimizer_and_scheduler",
    action="store_true",
    help="save optimizer and scheduler in the checkpoint",
)
parser.add_argument(
    "--overwrite_output_dir",
    action="store_true",
    help="Overwrite the content of the output directory",
)
# 训练模型用到的数据文件路径
parser.add_argument(
    "--data_dir",
    type=str,
    default="~/dstc11_simmc2.1_scut-bds-lab/data",
    help='the path of the dataset'
)
parser.add_argument(
    "--dialogue_name_prefix",
    type=str,
    default="simmc2.1_dials_dstc11_",
    help='dialogue_name_prefix of the json file'
)
parser.add_argument(
    "--jsons_dir_name",
    type=str,
    default="simmc2_scene_jsons_dstc10_public",
    help='jsons_dir_name'
)
parser.add_argument(
    "--images_dir_name",
    type=str,
    default="simmc2_scene_images_dstc10_public",
    help='images_dir_name'
)
parser.add_argument(
    "--train_input_file",
    required=True,
    type=str,
    help='preprocessed input file path'
)
parser.add_argument(
    "--disambiguation_file",
    default=None,
    type=str,
    help='preprocessed input file path'
)
parser.add_argument(
    "--response_file",
    default=None,
    type=str,
    help='preprocessed input file path, line-by-line format'
)
parser.add_argument(
    "--train_target_file",
    default=None,
    type=str,
    help='preprocessed target file path, line-by-line format'
)
parser.add_argument(
    "--eval_input_file",
    required=True,
    type=str,
    help='preprocessed input file path, line-by-line format'
)
parser.add_argument(
    "--eval_target_file",
    default=None,
    type=str,
    help='preprocessed target file path, line-by-line format'
)
# <--------2022.09.09 新增：用于合并任务1和任务2，增加系统意图预测，用户意图改为分类任务
parser.add_argument(
    "--train_user_act_file",
    default=None,
    type=str,
    help='train_user_act_file path, .txt, used in simmc21_dataset_joint_disam_coref.py'
)
parser.add_argument(
    "--train_system_act_file",
    default=None,
    type=str,
    help='train_system_act_file path, .txt, used in simmc21_dataset_joint_disam_coref.py'
)
parser.add_argument(
    "--eval_user_act_file",
    default=None,
    type=str,
    help='train_user_act_file path, .txt, used in simmc21_dataset_joint_disam_coref.py'
)
parser.add_argument(
    "--eval_system_act_file",
    default=None,
    type=str,
    help='eval_system_act_file, .txt, used in simmc21_dataset_joint_disam_coref.py'
)
parser.add_argument(
    "--not_no_coref",
    action="store_true",
    help="If True not use <NOCOREF>, used in simmc21_dataset_joint_disam_coref.py",
)
parser.add_argument(
    "--joint_disam_and_coref",
    action="store_true",
    help="If True do joint_disam_and_coref, used in simmc21_dataset_joint_disam_coref.py",
)
# ---------------------------->

# <--------2022.09.02 新增：图像的名称存储.txt文件
parser.add_argument(
    "--train_image_path_file",
    default=None,
    type=str,
    help='preprocessed train_image_path_file path, line-by-line format'
)
parser.add_argument(
    "--train_image_dir",
    default=None,
    type=str,
    help='train_image_dir'
)
parser.add_argument(
    "--eval_image_path_file",
    default=None,
    type=str,
    help='preprocessed eval_image_path_file path, line-by-line format'
)
parser.add_argument(
    "--eval_image_dir",
    default=None,
    type=str,
    help='eval_image_dir'
)
# ---------------------------->

parser.add_argument(
    "--add_special_tokens",
    default=None,
    required=True,
    type=str,
    help="Optional file containing a JSON dictionary of special tokens that should be added to the tokenizer.",
)
parser.add_argument(
    "--item2id",
    required=True,
    type=str,
    help='item2id filepath'
)
# 模型的train_epochs与batch_size
parser.add_argument(
    '--num_train_epochs',
    default=3,
    type=int,
)
parser.add_argument(
    '--start_save_or_eval_from_epoch',
    default=2,
    type=int,
    help="Start to save or evaluate the model from this epoch",
)
parser.add_argument(
    "--per_gpu_train_batch_size",
    default=1,
    type=int,
    help="Batch size per GPU/CPU for training.",
)
parser.add_argument(
    "--per_gpu_eval_batch_size",
    default=1,
    type=int,
    help="Batch size per GPU/CPU for evaluation.",
)
# 模型的梯度加速
parser.add_argument(
    "--gradient_accumulation_steps",
    type=int,
    default=1,
    help="Number of updates steps to accumulate before performing a backward/update pass.",
)
# 模型训练时的超参数
# Dataloder的num_workers
parser.add_argument(
    "--num_workers",
    default=4,
    type=int,
    help="num_workers for Dataloder",
)
parser.add_argument(
    "--optimizer",
    type=str,
    default="AdamW",
    choices=['AdamW', 'Adafactor', 'Adafactor-srwf'],
    help="For optimizer.",
)
parser.add_argument(
    "--scheduler",
    type=str,
    default="get_linear_schedule_with_warmup",
    choices=['get_linear_schedule_with_warmup', 'get_constant_schedule_with_warmup', 'get_constant_schedule',
                'get_cosine_schedule_with_warmup', 'get_adafactor_schedule', 'no_schedule'],
    help="For scheduler.",
)
parser.add_argument(
    "--weight_decay", default=0.0, type=float, help="Weight decay if we apply some."
)
parser.add_argument(
    "--max_steps",
    default=-1,
    type=int,
    help="If > 0: set total number of training steps to perform. Override num_train_epochs.",
)
parser.add_argument(
    "--learning_rate",
    default=5e-5,
    type=float,
    help="The initial learning rate for Adam.",
)
parser.add_argument(
    "--adam_epsilon", default=1e-8, type=float, help="Epsilon for Adam optimizer."
)
parser.add_argument(
    "--warmup_steps", default=8000, type=int, help="Linear warmup over warmup_steps."
)
parser.add_argument(
    "--warm_up_ratio", default=0.1, type=float, help="Linear warmup over warmup_steps(warm_up_ratio*t_total)."
)
parser.add_argument(
    "--max_grad_norm", default=1.0, type=float, help="Max gradient norm."
)
parser.add_argument(
    "--embedding_train_steps", default=200, type=int
)
parser.add_argument(
    "--embedding_train_epochs_start",
    type=int,
    default=400
)
parser.add_argument(
    "--embedding_train_epochs_ongoing",
    type=int,
    default=100
)
parser.add_argument(
    "--do_train_embedding_clip_way_during_training",
    action="store_true",
    help="Run train_embedding_clip_way during training at each embedding_train_step.",
)
parser.add_argument(
    "--do_retrieval",
    action="store_true",
    help="do_retrieval during training.",
)
parser.add_argument(
    "--seed",
    default=42,
    type=int,
)
parser.add_argument(
    "--fp16",
    action="store_true",
    help="Whether to use 16-bit (mixed) precision (through NVIDIA apex) instead of 32-bit",
)
parser.add_argument(
    "--fp16_opt_level",
    type=str,
    default="O1",
    help="For fp16: Apex AMP optimization level selected in ['O0', 'O1', 'O2', and 'O3']."
    "See details at https://nvidia.github.io/apex/amp.html",
)
parser.add_argument(
    "--local_rank",
    type=int,
    default=-1,
    help="For distributed training: local_rank",
)
parser.add_argument(
    "--not_find_unused_parameters", action="store_true", help="If True set find_unused_parameters=False in DDP constructor"
)
# 保存模型与验证模型的设置
parser.add_argument(
    "--no_train", action="store_true", help="Only evaluate the checkpoint and not train"
)
parser.add_argument(
    "--save_steps",
    type=int,
    default=2000,
    help="Save checkpoint every X updates steps.",
)
parser.add_argument(
    "--eval_steps", default=2000, type=int
)
parser.add_argument(
    "--log_steps", default=10, type=int, help="logging output steps."
)
parser.add_argument(
    "--save_total_limit",
    type=int,
    default=None,
    help="Limit the total amount of checkpoints, delete the older checkpoints in the output_dir, does not delete by default",
)
parser.add_argument(
    "--evaluate_during_training",
    action="store_true",
    help="Run evaluation during training at each logging step.",
)
parser.add_argument(
    "--do_eval", action="store_true", help="Whether to run eval on the dev set."
)
parser.add_argument(
    "--ignore_mismatched_sizes", action="store_true", help="Whether to ignore mismatched sizes of the pretrained model."
)
parser.add_argument(
    "--eval_all_checkpoints",
    action="store_true",
    help="Evaluate all checkpoints starting with the same prefix as model_name_or_path ending and ending with step number",
)
parser.add_argument(
    "--output_eval_file",
    type=str,
    default=".txt file"
)
# 其他设置
parser.add_argument(
    "--server_ip", type=str, default="", help="For distant debugging."
)
parser.add_argument(
    "--server_port", type=str, default="", help="For distant debugging."
)
parser.add_argument("--autocast", action='store_true',
                    help="If true using autocast to automatically mix accuracy to accelerate training(开启自动混合精度加速训练)")
parser.add_argument(
    "--no_cuda", action="store_true", help="Avoid using CUDA when available"
)
parser.add_argument(
    "--not_use_OBJ", action="store_true", help="Avoid using <OBJ>"
)
parser.add_argument(
    "--not_use_BF", action="store_true", help="Avoid using \"=> Belief State : \""
)
# R-Drop: Regularized Dropout for Neural Networks
parser.add_argument(
    "--do_rdrop", action="store_true", help="R-Drop: Regularized Dropout for Neural Networks, refer to https://github.com/dropreg/R-Drop"
)
parser.add_argument(
    "--rdrop_reduction", 
    type=str, 
    default="mean", 
    choices=['mean', 'sum', 'batchmean'],
    help="Reduction For calculating KL Loss"
)
parser.add_argument("--alpha_rdrop", type=float, default=1.0, help="--alpha_rdrop, so that total_loss = other_loss + alpha_rdrop*kl_loss")
# 损失函数的各部分占比
parser.add_argument("--alpha_masked_lm_loss", type=float, default=1.0, help="alpha_masked_lm_loss")
parser.add_argument("--alpha_nocoref_loss", type=float, default=0.1, help="alpha_nocoref_loss")
parser.add_argument("--alpha_misc_loss", type=float, default=0.1, help="alpha_misc_loss")
parser.add_argument("--alpha_disam_loss", type=float, default=0.1, help="alpha_disam_loss")
parser.add_argument("--alpha_retrieval_loss", type=float, default=0.4, help="alpha_retrieval_loss")
parser.add_argument("--alpha_disamb_candi_loss", type=float, default=0.8, help="alpha_disamb_candi_loss")
parser.add_argument("--alpha_coref_loss", type=float, default=0.8, help="alpha_coref_loss")

# MultiTaskBartForConditionalGenerationJointDisambCoref新增
parser.add_argument("--alpha_user_act_loss", type=float, default=1.0, help="alpha_user_act_loss")
parser.add_argument("--alpha_system_act_loss", type=float, default=1.0, help="alpha_system_act_loss")
parser.add_argument("--alpha_disamb_and_coref_loss", type=float, default=1.0, help="alpha_disamb_and_coref_loss")
parser.add_argument("--alpha_disam_and_coref_candi_loss", type=float, default=1.0, help="alpha_disam_and_coref_candi_loss")

# Focal Loss
parser.add_argument(
    "--use_focal_loss", action="store_true", help="use_focal_loss"
)
parser.add_argument("--focal_loss_gamma", type=float, default=2.0, help="focal_loss_gamma, in [0.2, 0.5, 1.0, 2.0, 5.0]")
parser.add_argument("--focal_loss_alpha", type=float, default=0.25, help="focal_loss_alpha, in 0~1 (二分类当中，占样本大多数的标签的loss的权重)")



# 冻结模型的部分层
parser.add_argument('--freeze_model', action='store_true', help="If True freeze some layers of the model")
parser.add_argument('--freeze_start_layer', type=int, default=0, help="冻结指定的层范围，格式为start-end，其中start取值范围为0~11，end取值范围为0~11，start<=end")
parser.add_argument('--freeze_end_layer', type=int, default=11, help="冻结指定的层范围，格式为start-end，其中start取值范围为0~11，end取值范围为0~11，start<=end")

# 用于OFA模型
parser.add_argument('--sample_patch_num', type=int, default=256, help="图像部分的patch数量")

