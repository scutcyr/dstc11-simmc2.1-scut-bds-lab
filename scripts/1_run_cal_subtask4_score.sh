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


# File: 1_run_cal_subtask4_score.sh
# Description: Preprocessing scripts for dataset SIMMC2.1
# Repository: https://github.com/scutcyr/dstc11-simmc2.1-iflytek
# Mail: [yrchen5@iflytek.com](mailto:yrchen5@iflytek.com) or [eeyirongchen@mail.scut.edu.cn](mailto:eeyirongchen@mail.scut.edu.cn)
# Date: 2022/10/25
# Usage:
# $ ./1_run_cal_subtask4_score.sh

# source .bashrc
source /home/intern/yrchen5/.bashrc
# activate python environment
conda activate py38

# 指定当前的工作目录
WORK_DIR=/yrfs1/intern/yrchen5/dstc11_simmc2.1_iflytek/dstc11-simmc2.1-iflytek
INIT_DATA_DIR=/yrfs1/intern/yrchen5/dstc11_simmc2.1_iflytek/data
PREPROCESS_DATA_DIR=/yrfs1/intern/yrchen5/dstc11_simmc2.1_iflytek/dstc11-simmc2.1-iflytek/data_convert
cd $WORK_DIR

python ./evaluation_tools/response_evaluation.py \
  --data_json_path=$INIT_DATA_DIR/simmc2.1_dials_dstc11_devtest.json \
  --model_response_path=/yrfs1/intern/yrchen5/dstc11_simmc2.1_other_team_results/SIMMC2.1/sub4_results/dstc11-simmc-devtest-pred-subtask-4-generation.json \
  --single_round_evaluation \