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


# File: 1_combination_model_result.sh
# Description: Preprocessing scripts for dataset SIMMC2.1
# Repository: https://github.com/scutcyr/dstc11-simmc2.1-iflytek
# Mail: [yrchen5@iflytek.com](mailto:yrchen5@iflytek.com) or [eeyirongchen@mail.scut.edu.cn](mailto:eeyirongchen@mail.scut.edu.cn)
# Date: 2022/10/20
# Usage:
# $ ./1_combination_model_result.sh


# source .bashrc
source /home/intern/yrchen5/.bashrc
# activate python environment
conda activate py38

# 指定当前的工作目录
WORK_DIR=/yrfs1/intern/yrchen5/dstc11_simmc2.1_iflytek/dstc11-simmc2.1-iflytek
cd $WORK_DIR

python combination_model_result.py \
    --input_line_by_line_file_for_task1=$WORK_DIR/runs/simmc21_bart_ctxlen6_20221019_2100_use_focalloss/devtest_predict_results/pred-results-of-.txt \
    --input_line_by_line_file_for_task2=$WORK_DIR/runs/simmc21_bart_ctxlen6_20221019_2100_use_focalloss/devtest_predict_results/pred-results-of-.txt \
    --input_line_by_line_file_for_task3=$WORK_DIR/runs/simmc21_bart_ctxlen6_20221019_2100_use_focalloss/devtest_predict_results/pred-results-of-.txt \
    --input_line_by_line_file_for_task4=$WORK_DIR/runs/simmc21_bart_ctxlen6_20221019_1900/devtest_predict_results/pred-results-of-.txt \
    --output_line_by_line_combination_results=$WORK_DIR/results/line_by_line_combination_model_result_version_1.txt