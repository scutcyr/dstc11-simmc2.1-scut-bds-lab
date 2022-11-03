#!/bin/bash
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

# File: run_train_model_simmc21_bart_20221019_1900.sh
# Description: training model scripts for ofa model
# Repository: https://github.com/scutcyr/dstc11-simmc2.1-scut-bds-lab
# Mail: [eeyirongchen@mail.scut.edu.cn](mailto:eeyirongchen@mail.scut.edu.cn)
# Date: 2022/10/11
# Usage:
# $ ./run_train_model_simmc21_bart_20221019_1900.sh
# 科研平台提交任务
# 任务脚本：~/dstc11_simmc2.1_scut-bds-lab/dstc11-simmc2.1-scut-bds-lab/scripts/run_train_model_simmc21_bart_20221019_1900.sh
# 任务日志：~/dstc11_simmc2.1_scut-bds-lab/dstc11-simmc2.1-scut-bds-lab/runs/run_train_model_simmc21_bart_20221019_1900.log
# 模型生成路径：~/dstc11_simmc2.1_scut-bds-lab/dstc11-simmc2.1-scut-bds-lab/runs
# 本地多卡分布式
# CUDA_VISIBLE_DEVICES=2,3 python -m torch.distributed.launch --nnodes 1 --nproc_per_node 2 --master_addr 127.0.0.1 --master_port 9906 train_model.py \
# 本地单卡
# CUDA_VISIBLE_DEVICES=2 python train_model.py \


# 根据CUDA版本source不同的.bashrc，并且激活不同的conda环境
if [ -n "`nvidia-smi | grep 'CUDA Version: 11'`" ];then
    echo "Prepare: source ~/.bashrc_cuda11"
    source ~/.bashrc_cuda11
    echo "Prepare: conda activate py38cu113"
    conda activate py38cu113
else
    echo "Prepare: source ~/.bashrc"
    source ~/.bashrc
    echo "Prepare: conda activate py38"
    conda activate py38
fi

# 针对：bootstrap.cc:40 NCCL WARN Bootstrap : no socket interface found
# 参考: https://blog.csdn.net/m0_37426155/article/details/108129952
export NCCL_SOCKET_IFNAME=en,eth,em,bond

WORK_DIR=~/dstc11_simmc2.1_scut-bds-lab/dstc11-simmc2.1-scut-bds-lab
INIT_DATA_DIR=~/dstc11_simmc2.1_scut-bds-lab/data
PREPROCESS_DATA_DIR=~/dstc11_simmc2.1_scut-bds-lab/dstc11-simmc2.1-scut-bds-lab/data_convert
CONTEXT_LENGTH=6 # 2,4,6,8
MODEL_COMMENT=20221019_1900
# cd working path
cd $WORK_DIR

# 科研平台上：--master_addr $MASTER_ADDR --master_port $MASTER_PORT
# 本地运行：--master_addr 127.0.0.1 --master_port 9129

torchrun --nnodes 1 --nproc_per_node 2 --master_addr $MASTER_ADDR --master_port 9035 train_model.py \
    --model_type=mt-bart-large-disamb \
    --model_name_or_path=~/pretraining_model/bart-large \
    --add_special_tokens=$INIT_DATA_DIR/simmc_special_tokens.json \
    --item2id=$INIT_DATA_DIR/item2id.json \
    --train_input_file=$PREPROCESS_DATA_DIR/simmc2.1_dials_dstc11_train_ctxlen${CONTEXT_LENGTH}_sysana_for_task4.txt \
    --eval_input_file=$PREPROCESS_DATA_DIR/simmc2.1_dials_dstc11_devtest_ctxlen${CONTEXT_LENGTH}_sysana_for_task4.txt \
    --train_image_path_file=$PREPROCESS_DATA_DIR/simmc2.1_dials_dstc11_train_scene_name.txt \
    --train_image_dir=$INIT_DATA_DIR/simmc2_scene_images_dstc10_public \
    --eval_image_path_file=$PREPROCESS_DATA_DIR/simmc2.1_dials_dstc11_devtest_scene_name.txt \
    --eval_image_dir=$INIT_DATA_DIR/simmc2_scene_images_dstc10_public \
    --output_dir=$WORK_DIR/runs/simmc21_bart_ctxlen${CONTEXT_LENGTH}_${MODEL_COMMENT} \
    --output_eval_file=$WORK_DIR/runs/simmc21_bart_ctxlen${CONTEXT_LENGTH}_${MODEL_COMMENT}/eval_report.txt \
    --overwrite_output_dir \
    --num_train_epochs=12 \
    --evaluate_during_training \
    --log_steps=10 \
    --save_total_limit=12 \
    --embedding_train_epochs_start=200 \
    --do_train_embedding_clip_way_during_training \
    --embedding_train_steps=200 \
    --embedding_train_epochs_ongoing=100 \
    --per_gpu_train_batch_size=10 \
    --per_gpu_eval_batch_size=10 \
    --weight_decay=0.1 \
    --adam_epsilon=1e-8 \
    --max_grad_norm=1.0 \
    --seed=2022 \
    --warm_up_ratio=0.1 \
    --learning_rate=5e-5 \
    --scheduler=get_linear_schedule_with_warmup \
    --optimizer=AdamW \
    --gradient_accumulation_steps=1 \
    --alpha_retrieval_loss=0.2 \
    --alpha_masked_lm_loss=1.0 \
    --alpha_disam_loss=0.5 \
    --alpha_nocoref_loss=0.5 \
    --alpha_misc_loss=0.1 \
    --alpha_disamb_candi_loss=1.0 \
    --alpha_coref_loss=1.0