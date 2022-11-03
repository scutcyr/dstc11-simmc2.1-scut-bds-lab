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


# File: 1_run_cal_all_task_score.sh
# Description: Preprocessing scripts for dataset SIMMC2.1
# Repository: https://github.com/scutcyr/dstc11-simmc2.1-scut-bds-lab
# Mail: [eeyirongchen@mail.scut.edu.cn](mailto:eeyirongchen@mail.scut.edu.cn)
# Date: 2022/10/20
# Usage:
# $ ./1_run_cal_all_task_score.sh

# source .bashrc
source ~/.bashrc
# activate python environment
conda activate py38

# 指定当前的工作目录
WORK_DIR=~/dstc11_simmc2.1_scut-bds-lab/dstc11-simmc2.1-scut-bds-lab
cd $WORK_DIR

python ./evaluation_tools/evaluate_all_task.py \
  --data_json_path=../data/simmc2.1_dials_dstc11_devtest.json \
  --input_path_target=../data_object_special_with_disambiguation_candidates/simmc2.1_dials_dstc11_devtest_target_with_disambiguation_candidates.txt \
  --input_path_predicted=./results/devtest_results/dstc11-simmc2.1-devtest-pred-subtask-1-to-4-of-mt-bart-large_context_before_objects_20220907_1620_check_isnocoref_and_coref.txt \
  --dialog_meta_data=../data_object_special/simmc2.1_dials_dstc11_devtest_inference_disambiguation.json \
  --output_path_report=./results/devtest_results/dstc11-simmc2.1-devtest-pred-subtask-1-to-4-of-mt-bart-large_context_before_objects_20220907_1620_check_isnocoref_and_check_disamb_candi_report_of_subtask1_to_4.json \
  --single_round_evaluation \