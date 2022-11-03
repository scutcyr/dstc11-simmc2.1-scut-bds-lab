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


# File: 0_dataset_teststd_preprocessing_predict_with_sys_state.sh
# Description: Preprocessing scripts for dataset SIMMC2.1
# Repository: https://github.com/scutcyr/dstc11-simmc2.1-scut-bds-lab
# Mail: [eeyirongchen@mail.scut.edu.cn](mailto:eeyirongchen@mail.scut.edu.cn)
# Date: 2022/10/10
# Usage:
# Specify the path of the .bashrc file or remove `source ~/.bashrc_cuda11` and `source ~/.bashrc`;
# Specify the conda python environment, e.g. `conda activate py38cu113` or `conda activate py38`;
# Change the INPUT_DIR and WORK_DIR to the actual path you specified, and then run
# $ ./0_dataset_teststd_preprocessing_predict_with_sys_state.sh


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

# 指定原始数据集的绝对路径
INPUT_DIR=~/dstc11_simmc2.1_scut-bds-lab/data
# 指定当前的工作目录，需要保证convert_simmc21_sysana_for_task4.py存放在工作目录下面
WORK_DIR=~/dstc11_simmc2.1_scut-bds-lab/dstc11-simmc2.1-scut-bds-lab
# 指定预处理后的数据的存储路径
OUTPUT_DIR=$WORK_DIR/data_convert

mkdir -p $OUTPUT_DIR

function make_data() {
    context_length=$1
    for file in teststd_public; do
        python convert.py \
            --input_path_json=$INPUT_DIR/simmc2.1_dials_dstc11_$file.json \
            --output_path_predict=$OUTPUT_DIR/simmc2.1_dials_dstc11_${file}_predict_with_sys_state_ctxlen${context_length}.txt \
            --output_path_target=$OUTPUT_DIR/simmc2.1_dials_dstc11_${file}_target_with_sys_state_ctxlen${context_length}.txt \
            --object_special_token_item2id=$INPUT_DIR/item2id.json \
            --scene_json_folder=$INPUT_DIR/simmc2_scene_jsons_dstc10_teststd \
            --image_folder=$INPUT_DIR/simmc2_scene_images_dstc10_teststd \
            --len_context=$context_length \
            --with_target=0 \
            --use_system_transcript_annotated \
            --context_before_objects
    done
}

cd $WORK_DIR
make_data 2 &
make_data 4 &
make_data 6 &
make_data 8 &
cd -

wait
cd $WORK_DIR
python convert.py \
  --input_path_json=$INPUT_DIR/simmc2.1_dials_dstc11_teststd_public.json \
  --input_path_all_turn_predict_lines=$OUTPUT_DIR/simmc2.1_dials_dstc11_teststd_public_predict_with_sys_state_ctxlen2.txt \
  --output_path_final_turn_predict_lines=$OUTPUT_DIR/simmc2.1_dials_dstc11_teststd_public_predict_with_sys_state_ctxlen2_final_turn.txt

python convert.py \
  --input_path_json=$INPUT_DIR/simmc2.1_dials_dstc11_teststd_public.json \
  --input_path_all_turn_predict_lines=$OUTPUT_DIR/simmc2.1_dials_dstc11_teststd_public_predict_with_sys_state_ctxlen4.txt \
  --output_path_final_turn_predict_lines=$OUTPUT_DIR/simmc2.1_dials_dstc11_teststd_public_predict_with_sys_state_ctxlen4_final_turn.txt

python convert.py \
  --input_path_json=$INPUT_DIR/simmc2.1_dials_dstc11_teststd_public.json \
  --input_path_all_turn_predict_lines=$OUTPUT_DIR/simmc2.1_dials_dstc11_teststd_public_predict_with_sys_state_ctxlen6.txt \
  --output_path_final_turn_predict_lines=$OUTPUT_DIR/simmc2.1_dials_dstc11_teststd_public_predict_with_sys_state_ctxlen6_final_turn.txt

python convert.py \
  --input_path_json=$INPUT_DIR/simmc2.1_dials_dstc11_teststd_public.json \
  --input_path_all_turn_predict_lines=$OUTPUT_DIR/simmc2.1_dials_dstc11_teststd_public_predict_with_sys_state_ctxlen8.txt \
  --output_path_final_turn_predict_lines=$OUTPUT_DIR/simmc2.1_dials_dstc11_teststd_public_predict_with_sys_state_ctxlen8_final_turn.txt


echo "end"