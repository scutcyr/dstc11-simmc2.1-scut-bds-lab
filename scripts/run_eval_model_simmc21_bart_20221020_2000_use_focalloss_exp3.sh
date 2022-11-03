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

# File: run_eval_model_simmc21_bart_20221020_2000_use_focalloss_exp3.sh
# Description: testing model scripts for ofa model
# Repository: https://github.com/scutcyr/dstc11-simmc2.1-scut-bds-lab
# Mail: [eeyirongchen@mail.scut.edu.cn](mailto:eeyirongchen@mail.scut.edu.cn)
# Date: 2022/10/18
# Usage:
# $ ./run_eval_model_simmc21_bart_20221020_2000_use_focalloss_exp3.sh
# 科研平台提交任务
# 任务脚本：~/dstc11_simmc2.1_scut-bds-lab/dstc11-simmc2.1-scut-bds-lab/scripts/run_eval_model_simmc21_bart_20221020_2000_use_focalloss_exp3.sh
# 任务日志：~/dstc11_simmc2.1_scut-bds-lab/dstc11-simmc2.1-scut-bds-lab/runs/run_eval_model_simmc21_bart_20221020_2000_use_focalloss_exp3.log

# 本地单卡
# CUDA_VISIBLE_DEVICES=2 python eval_model.py \


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

WORK_DIR=~/dstc11_simmc2.1_scut-bds-lab/dstc11-simmc2.1-scut-bds-lab
INIT_DATA_DIR=~/dstc11_simmc2.1_scut-bds-lab/data
PREPROCESS_DATA_DIR=~/dstc11_simmc2.1_scut-bds-lab/dstc11-simmc2.1-scut-bds-lab/data_convert
MODEL_SAVE_DIR=~/dstc11_simmc2.1_scut-bds-lab/dstc11-simmc2.1-scut-bds-lab/runs
CONTEXT_LENGTH=6 # 2,4,6,8
MODEL_COMMENT=20221020_2000_use_focalloss_exp3
DATA_TYPE=devtest
# cd working path
cd $WORK_DIR

# 将--temperature=0.9改为--temperature=0.95，针对错误：RuntimeError: probability tensor contains eithe `inf`, `nan` or element < 0，参考：https://bytemeta.vip/repo/aub-mind/arabert/issues/149
python eval_model.py \
   --model_type=mt-bart-large-disamb \
   --model_dir=$MODEL_SAVE_DIR/simmc21_bart_ctxlen${CONTEXT_LENGTH}_${MODEL_COMMENT} \
   --path_output=$MODEL_SAVE_DIR/simmc21_bart_ctxlen${CONTEXT_LENGTH}_${MODEL_COMMENT}/${DATA_TYPE}_predict_results \
   --output_path_csv_report=$MODEL_SAVE_DIR/simmc21_bart_ctxlen${CONTEXT_LENGTH}_${MODEL_COMMENT}/${DATA_TYPE}_predict_results/report_summary_simmc21_bart_ctxlen${CONTEXT_LENGTH}_${MODEL_COMMENT}.csv \
   --prompts_from_file=$PREPROCESS_DATA_DIR/simmc2.1_dials_dstc11_${DATA_TYPE}_predict_ctxlen${CONTEXT_LENGTH}.txt \
   --item2id=$INIT_DATA_DIR/item2id.json \
   --data_json_path=$INIT_DATA_DIR/simmc2.1_dials_dstc11_${DATA_TYPE}.json \
   --input_path_target=$PREPROCESS_DATA_DIR/simmc2.1_dials_dstc11_${DATA_TYPE}_target_ctxlen${CONTEXT_LENGTH}.txt \
   --dialog_meta_data=$PREPROCESS_DATA_DIR/simmc2.1_dials_dstc11_${DATA_TYPE}_inference_disambiguation.json \
   --batch_size=32 \
   --length=150 \
   --do_sample \
   --num_beams=4 \
   --temperature=0.95 \
   --repetition_penalty=1.0 \
   --k=0 \
   --p=0.9 \
   --num_return_sequences=1 \
   --eval_all_checkpoints \
   --single_round_evaluation \
   --do_calculate_score \
   --not_generate_predict_file_again \
   --cal_diff_f1_based_on_previously_mentioned \
   --multimodal_context_json_file=$PREPROCESS_DATA_DIR/simmc2.1_dials_dstc11_${DATA_TYPE}_multimodal_context.json