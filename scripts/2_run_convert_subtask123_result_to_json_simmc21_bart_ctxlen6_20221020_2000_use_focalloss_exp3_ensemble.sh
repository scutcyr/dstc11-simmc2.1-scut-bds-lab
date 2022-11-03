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


# File: 1_run_cal_subtask4_score.sh
# Description: Preprocessing scripts for dataset SIMMC2.1
# Repository: https://github.com/scutcyr/dstc11-simmc2.1-scut-bds-lab
# Mail: [eeyirongchen@mail.scut.edu.cn](mailto:eeyirongchen@mail.scut.edu.cn)
# Date: 2022/10/25
# Usage:
# $ ./1_run_cal_subtask4_score.sh

# source .bashrc
source ~/.bashrc
# activate python environment
conda activate py38

# 指定当前的工作目录
WORK_DIR=~/dstc11_simmc2.1_scut-bds-lab/dstc11-simmc2.1-scut-bds-lab
INIT_DATA_DIR=~/dstc11_simmc2.1_scut-bds-lab/data
PREPROCESS_DATA_DIR=~/dstc11_simmc2.1_scut-bds-lab/dstc11-simmc2.1-scut-bds-lab/data_convert
MODEL_PATH=$WORK_DIR/runs/simmc21_bart_ctxlen6_20221020_2000_use_focalloss_exp3
cd $WORK_DIR

DATA_TYPE=devtest

python convert.py \
  --input_path_json=$INIT_DATA_DIR/simmc2.1_dials_dstc11_${DATA_TYPE}.json \
  --input_line_by_line_txt_file=$MODEL_PATH/${DATA_TYPE}_predict_results/pred-results-of-combination_model_use_subtask_best_checkpoint_and_mentioned_objects_and_final_turn_of_another_model.txt \
  --output_parse_flattened_results_json_file=$MODEL_PATH/${DATA_TYPE}_predict_results/pred-results-of-combination_model_use_subtask_best_checkpoint_and_mentioned_objects_and_final_turn_of_another_model.json

wait

python evaluation_tools/evaluate_dst.py \
  --input_path_target=$INIT_DATA_DIR/simmc2.1_dials_dstc11_${DATA_TYPE}.json \
  --input_path_predicted=$MODEL_PATH/${DATA_TYPE}_predict_results/pred-results-of-combination_model_use_subtask_best_checkpoint_and_mentioned_objects_and_final_turn_of_another_model.json \
  --output_path_report=$MODEL_PATH/${DATA_TYPE}_predict_results/pred-results-of-combination_model_use_subtask_best_checkpoint_and_mentioned_objects_and_final_turn_of_another_model-result.json

wait

DATA_TYPE=teststd_public

python convert.py \
  --input_path_json=$INIT_DATA_DIR/simmc2.1_dials_dstc11_${DATA_TYPE}.json \
  --input_line_by_line_txt_file=$MODEL_PATH/${DATA_TYPE}_predict_results/pred-results-of-combination_model_use_subtask_best_checkpoint_and_mentioned_objects_and_final_turn_of_another_model.txt \
  --output_parse_flattened_results_json_file=$MODEL_PATH/${DATA_TYPE}_predict_results/pred-results-of-combination_model_use_subtask_best_checkpoint_and_mentioned_objects_and_final_turn_of_another_model.json

