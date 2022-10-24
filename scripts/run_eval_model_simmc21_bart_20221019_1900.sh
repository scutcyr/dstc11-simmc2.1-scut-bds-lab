#!/bin/bash
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

# File: run_eval_model_simmc21_bart_20221019_1900.sh
# Description: testing model scripts for ofa model
# Repository: https://github.com/scutcyr/dstc11-simmc2.1-iflytek
# Mail: [yrchen5@iflytek.com](mailto:yrchen5@iflytek.com) or [eeyirongchen@mail.scut.edu.cn](mailto:eeyirongchen@mail.scut.edu.cn)
# Date: 2022/10/18
# Usage:
# $ ./run_eval_model_simmc21_bart_20221019_1900.sh
# 科研平台提交任务
# 任务脚本：/yrfs1/intern/yrchen5/dstc11_simmc2.1_iflytek/dstc11-simmc2.1-iflytek/scripts/run_eval_model_simmc21_bart_20221019_1900.sh
# 任务日志：/yrfs1/intern/yrchen5/dstc11_simmc2.1_iflytek/dstc11-simmc2.1-iflytek/runs/run_eval_model_simmc21_bart_20221019_1900.log

# 本地单卡
# CUDA_VISIBLE_DEVICES=2 python eval_model.py \


# 根据CUDA版本source不同的.bashrc，并且激活不同的conda环境
if [ -n "`nvidia-smi | grep 'CUDA Version: 11'`" ];then
    echo "Prepare: source /home/intern/yrchen5/.bashrc_cuda11"
    source /home/intern/yrchen5/.bashrc_cuda11
    echo "Prepare: conda activate py38cu113"
    conda activate py38cu113
else
    echo "Prepare: source /home/intern/yrchen5/.bashrc"
    source /home/intern/yrchen5/.bashrc
    echo "Prepare: conda activate py38"
    conda activate py38
fi

WORK_DIR=/yrfs1/intern/yrchen5/dstc11_simmc2.1_iflytek/dstc11-simmc2.1-iflytek
INIT_DATA_DIR=/yrfs1/intern/yrchen5/dstc11_simmc2.1_iflytek/data
PREPROCESS_DATA_DIR=/yrfs1/intern/yrchen5/dstc11_simmc2.1_iflytek/dstc11-simmc2.1-iflytek/data_convert
MODEL_SAVE_DIR=/yrfs1/intern/yrchen5/dstc11_simmc2.1_iflytek/dstc11-simmc2.1-iflytek/runs
CONTEXT_LENGTH=6 # 2,4,6,8
MODEL_COMMENT=20221019_1900
# cd working path
cd $WORK_DIR

python eval_model.py \
   --model_type=mt-bart-large-disamb \
   --model_dir=$MODEL_SAVE_DIR/simmc21_bart_ctxlen${CONTEXT_LENGTH}_${MODEL_COMMENT} \
   --path_output=$MODEL_SAVE_DIR/simmc21_bart_ctxlen${CONTEXT_LENGTH}_${MODEL_COMMENT}/devtest_predict_results \
   --output_path_csv_report=$MODEL_SAVE_DIR/simmc21_bart_ctxlen${CONTEXT_LENGTH}_${MODEL_COMMENT}/devtest_predict_results/report_summary_simmc21_bart_ctxlen${CONTEXT_LENGTH}_${MODEL_COMMENT}.csv \
   --prompts_from_file=$PREPROCESS_DATA_DIR/simmc2.1_dials_dstc11_devtest_predict_with_sys_state_ctxlen${CONTEXT_LENGTH}.txt \
   --item2id=$INIT_DATA_DIR/item2id.json \
   --data_json_path=$INIT_DATA_DIR/simmc2.1_dials_dstc11_devtest.json \
   --input_path_target=$PREPROCESS_DATA_DIR/simmc2.1_dials_dstc11_devtest_target_ctxlen${CONTEXT_LENGTH}.txt \
   --dialog_meta_data=$PREPROCESS_DATA_DIR/simmc2.1_dials_dstc11_devtest_inference_disambiguation.json \
   --batch_size=32 \
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