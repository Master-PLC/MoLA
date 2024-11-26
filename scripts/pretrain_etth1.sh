#!/bin/bash

GPU=1
DATA_ROOT=./dataset
EXP_NAME=pretrain

seed=2023
des='Exp'
lambda=1.0
rl=$lambda
ax=$(echo "1 - $lambda" | bc)
train_epochs=5
patience=2



model_name=iTransformer
pl_list=(32)
dst="ETTh1"
lr_list=(0.0005)

for lr in ${lr_list[@]}; do
for pl in ${pl_list[@]}; do
    JOB_NAME="${model_name}_${dst}_${pl}_${lr}"
    OUTPUT_DIR="./results/${EXP_NAME}/${JOB_NAME}"

    CHECKPOINTS=$OUTPUT_DIR/checkpoints/
    RESULTS=$OUTPUT_DIR/results/
    TEST_RESULTS=$OUTPUT_DIR/test_results/
    LOG_PATH=$OUTPUT_DIR/result_long_term_forecast.txt

    mkdir -p "${OUTPUT_DIR}/"


    echo "Running command for $JOB_NAME"
    > "${OUTPUT_DIR}/stdout.txt"
    {
        # Set CUDA_VISIBLE_DEVICES for this script and run it in the background
        CUDA_VISIBLE_DEVICES=$GPU python -u run.py \
            --task_name long_term_forecast \
            --is_training 1 \
            --root_path $DATA_ROOT/ETT-small/ \
            --data_path ETTh1.csv \
            --model_id "ETTh1_96_${pl}" \
            --model ${model_name} \
            --data ETTh1 \
            --features M \
            --seq_len 96 \
            --label_len 48 \
            --pred_len ${pl} \
            --label_pred_len ${pl} \
            --e_layers 2 \
            --d_layers 1 \
            --factor 3 \
            --enc_in 7 \
            --dec_in 7 \
            --c_out 7 \
            --des ${des} \
            --d_model 128 \
            --d_ff 128 \
            --learning_rate ${lr} \
            --itr 1 \
            --auxi_lambda ${ax} \
            --rec_lambda ${rl} \
            --fix_seed ${seed} \
            --checkpoints $CHECKPOINTS \
            --results $RESULTS \
            --test_results $TEST_RESULTS \
            --log_path $LOG_PATH \
            --train_epochs ${train_epochs} \
            --patience ${patience}
    } 2>&1 | tee -a "${OUTPUT_DIR}/stdout.txt" &
done
done

wait
