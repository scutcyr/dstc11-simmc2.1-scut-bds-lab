#!/bin/bash
if [ -n "`nvidia-smi | grep 'CUDA Version: 11'`" ];then
    source ~/.bashrc_cuda11
    conda activate py3.7_transformers_new_cuda11
else
    source ~/.bashrc
    conda activate py3.7_transformers_new
fi


TEST_MODEL_DIR=
TEST_INPUT_FILE=
TEST_OUTPUT_FILE=
TEST_LOG_FILE=
TEST_BATCH_SIZE=
BASE_DATA_DIR=

if [ -f $TEST_OUTPUT_FILE ]; then
    exit
fi

WORK_DIR=/ps2/sli/data/data_taowang49/projects/19_dstc11/simmc2.1_solutions/work/simmc2.1-iflytek-dst-cls
cd $WORK_DIR
python eval_roberta_dst_mulcls.py \
    --test_model_dir $TEST_MODEL_DIR \
    --prompts_from_file $TEST_INPUT_FILE \
    --path_output $TEST_OUTPUT_FILE \
    --log_file $TEST_LOG_FILE \
    --test_batch_size $TEST_BATCH_SIZE \
    --check_isnocoref_and_coref \
    --item2id $BASE_DATA_DIR/item2id.json \
    --length 150

cd -
