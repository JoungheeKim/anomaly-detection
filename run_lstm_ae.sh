#!/bin/bash

## Train 파라미터
TRAIN_BATCH_SIZE=256
EVAL_BATCH_SIZE=1024
MAX_STEPS=15000
LOGGING_STEP=1000
LEARNING_RATE=0.001

##모델 파라미터
LATENT_DIM=20
WINDOW_SIZE=4
## ae ave 선택
MODEL_TYPE="lstmae"

## PATH 파라미터
ROOT_DATA_PATH="multivariate_lsh_revised"

OUTPUT_DIR="result_lstmae"
PICTURE_DIR="요약"
EXPERIMENT_FILE="experiments/lstm_ae_result.csv"
NUM_LAYERS=2
NORMALIZE_FILE="DC101_selected_vars_info.csv"


for TARGET_RESOURCE in "20180125_20180125" "20171106_20171108" "20181110_20181110" "20180302_20180303" "20190929_20191001" "20180208_20180209" "20190512_20190515" "20181117_20181118" "20181020_20181102" "20191206_20191206" "20190409_20190410"; do

      TRAIN_DATA_FILE=${ROOT_DATA_PATH}"/"${TARGET_RESOURCE}"/train.p"
      VALID_DATA_FILE=${ROOT_DATA_PATH}"/"${TARGET_RESOURCE}"/val.p"
      TEST_DATA_FILE=${ROOT_DATA_PATH}"/"${TARGET_RESOURCE}"/test.p"
      ALL_DATA_FILE=${ROOT_DATA_PATH}"/"${TARGET_RESOURCE}"/all.p"

      mkdir -p ${PICTURE_DIR}"/"${MODEL_TYPE}"/"${TARGET_RESOURCE}

      python train.py\
            --train_file=${TRAIN_DATA_FILE}\
            --valid_file=${VALID_DATA_FILE}\
            --test_file=${TEST_DATA_FILE}\
            --output_dir=${OUTPUT_DIR}\
            --model_type=${MODEL_TYPE}\
            --do_train\
            --do_eval\
            --evaluate_during_training\
            --per_gpu_train_batch_size=${TRAIN_BATCH_SIZE}\
            --per_gpu_eval_batch_size=${EVAL_BATCH_SIZE}\
            --learning_rate=${LEARNING_RATE}\
            --max_steps=${MAX_STEPS}\
            --logging_steps=${LOGGING_STEP}\
            --overwrite_output_dir\
            --experiments_dir=${EXPERIMENT_FILE}\
            --window_size=${WINDOW_SIZE}\
            --latent_dim=${LATENT_DIM}\
            --norm_file=${NORMALIZE_FILE}\
            --num_layers=${NUM_LAYERS}

      python make_picture.py\
            --pre_trained_dir=${OUTPUT_DIR}\
            --picture_dir=${PICTURE_DIR}"/"${MODEL_TYPE}"/"${TARGET_RESOURCE}\
            --test_file=${ALL_DATA_FILE}
done

