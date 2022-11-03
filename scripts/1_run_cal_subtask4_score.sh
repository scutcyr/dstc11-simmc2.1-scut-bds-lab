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
cd $WORK_DIR

python ./evaluation_tools/response_evaluation.py \
  --data_json_path=$INIT_DATA_DIR/simmc2.1_dials_dstc11_devtest.json \
  --model_response_path=~/dstc11_simmc2.1_other_team_results/SIMMC2.1/sub4_results/dstc11-simmc-devtest-pred-subtask-4-generation.json \
  --single_round_evaluation \