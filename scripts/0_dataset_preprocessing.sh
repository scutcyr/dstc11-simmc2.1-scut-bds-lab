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


# File: 0_dataset_preprocessing.sh
# Description: Preprocessing scripts for dataset SIMMC2.1
# Repository: https://github.com/scutcyr/dstc11-simmc2.1-iflytek
# Mail: [yrchen5@iflytek.com](mailto:yrchen5@iflytek.com) or [eeyirongchen@mail.scut.edu.cn](mailto:eeyirongchen@mail.scut.edu.cn)
# Date: 2022/10/10
# Usage:
# Specify the path of the .bashrc file or remove `source /home/intern/yrchen5/.bashrc_cuda11` and `source /home/intern/yrchen5/.bashrc`;
# Specify the conda python environment, e.g. `conda activate py38cu113` or `conda activate py38`;
# Change the INPUT_DIR and WORK_DIR to the actual path you specified, and then run
# $ ./0_dataset_preprocessing.sh


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

# 指定原始数据集的绝对路径
INPUT_DIR=/yrfs1/intern/yrchen5/dstc11_simmc2.1_iflytek/data
# 指定预处理后的数据的存储路径
OUTPUT_DIR=$PWD/data_convert
# 指定当前的工作目录，需要保证convert_simmc21_sysana_for_task4.py存放在工作目录下面
WORK_DIR=/yrfs1/intern/yrchen5/dstc11_simmc2.1_iflytek/dstc11-simmc2.1-iflytek

mkdir -p $OUTPUT_DIR

function make_data() {
    context_length=$1
    for file in train dev devtest; do
        python convert.py \
            --input_path_json=$INPUT_DIR/simmc2.1_dials_dstc11_$file.json \
            --output_path_predict=$OUTPUT_DIR/simmc2.1_dials_dstc11_${file}_predict_ctxlen${context_length}.txt \
            --output_path_target=$OUTPUT_DIR/simmc2.1_dials_dstc11_${file}_target_ctxlen${context_length}.txt \
            --object_special_token_item2id=$INPUT_DIR/item2id.json \
            --scene_json_folder=$INPUT_DIR/simmc2_scene_jsons_dstc10_public \
            --image_folder=$INPUT_DIR/simmc2_scene_images_dstc10_public \
            --len_context=$context_length \
            --context_before_objects
    done
}

cd $WORK_DIR
make_data 2 &
make_data 4 &
make_data 6 &
make_data 8 &

# create simmc2.1_dials_dstc11_{train|dev|devtest}_disambiguation_label.txt
for file in train dev devtest; do
    python convert.py \
        --input_path_json=$INPUT_DIR/simmc2.1_dials_dstc11_$file.json \
        --output_disambiguation_label=$OUTPUT_DIR/simmc2.1_dials_dstc11_${file}_disambiguation_label.txt
done

# create simmc2.1_dials_dstc11_{train|dev|devtest}_response.txt
for file in train dev devtest; do
    python convert.py \
        --input_path_json=$INPUT_DIR/simmc2.1_dials_dstc11_$file.json \
        --output_path_response=$OUTPUT_DIR/simmc2.1_dials_dstc11_${file}_response.txt
done

# create simmc2.1_dials_dstc11_{train|dev|devtest}_scene_name.txt
for file in train dev devtest; do
    python convert.py \
        --input_path_json=$INPUT_DIR/simmc2.1_dials_dstc11_$file.json \
        --output_path_scene_name=$OUTPUT_DIR/simmc2.1_dials_dstc11_${file}_scene_name.txt
done

# create simmc2.1_dials_dstc11_{train|dev|devtest}_system_act.txt
for file in train dev devtest; do
    python convert.py \
        --input_path_json=$INPUT_DIR/simmc2.1_dials_dstc11_$file.json \
        --output_path_system_act=$OUTPUT_DIR/simmc2.1_dials_dstc11_${file}_system_act.txt
done

# create simmc2.1_dials_dstc11_{train|dev|devtest}_user_act.txt
for file in train dev devtest; do
    python convert.py \
        --input_path_json=$INPUT_DIR/simmc2.1_dials_dstc11_$file.json \
        --output_path_user_act=$OUTPUT_DIR/simmc2.1_dials_dstc11_${file}_user_act.txt
done

# create simmc2.1_dials_dstc11_devtest_inference_disambiguation.json
python convert.py \
  --input_path_json=$INPUT_DIR/simmc2.1_dials_dstc11_devtest.json \
  --output_inference_disambiguation=$OUTPUT_DIR/simmc2.1_dials_dstc11_devtest_inference_disambiguation.json \

cd -

wait
echo "end"