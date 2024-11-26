#!/bin/bash

GPU=0
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
pl_list=(8 24 32 64 84 96 112 180 240 360)
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


model_name=iTransformer
pl_list=(8 24 32 64 84 96 112 180 240 360)
dst="ETTh2"
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
            --data_path ETTh2.csv \
            --model_id "ETTh2_96_${pl}" \
            --model ${model_name} \
            --data ETTh2 \
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



model_name=iTransformer
pl_list=(8 24 32 64 84 96 112 180 240 360)
dst="ETTm1"
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
            --data_path ETTm1.csv \
            --model_id "ETTm1_96_${pl}" \
            --model ${model_name} \
            --data ETTm1 \
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


model_name=iTransformer
pl_list=(8 24 32 64 84 96 112 180 240 360)
dst="ETTm2"
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
            --data_path ETTm2.csv \
            --model_id "ETTm2_96_${pl}" \
            --model ${model_name} \
            --data ETTm2 \
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



model_name=iTransformer
pl_list=(8 24 32 64 84 96 112 180 240 360)
dst="ECL"
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
            --root_path $DATA_ROOT/electricity/ \
            --data_path electricity.csv \
            --model_id "ECL_96_${pl}" \
            --model ${model_name} \
            --data custom \
            --features M \
            --seq_len 96 \
            --label_len 48 \
            --pred_len ${pl} \
            --label_pred_len ${pl} \
            --e_layers 3 \
            --d_layers 1 \
            --factor 3 \
            --enc_in 321 \
            --dec_in 321 \
            --c_out 321 \
            --des ${des} \
            --d_model 512 \
            --d_ff 512 \
            --batch_size 16 \
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


model_name=iTransformer
pl_list=(8 24 32 64 84 96 112 180 240 360)
dst="Traffic"
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
            --root_path $DATA_ROOT/traffic/ \
            --data_path traffic.csv \
            --model_id "Traffic_96_${pl}" \
            --model ${model_name} \
            --data custom \
            --features M \
            --seq_len 96 \
            --label_len 48 \
            --pred_len ${pl} \
            --label_pred_len ${pl} \
            --e_layers 4 \
            --d_layers 1 \
            --factor 3 \
            --enc_in 862 \
            --dec_in 862 \
            --c_out 862 \
            --des ${des} \
            --d_model 512 \
            --d_ff 512 \
            --batch_size 16 \
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


model_name=iTransformer
pl_list=(8 24 32 64 84 96 112 180 240 360)
dst="Weather"
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
            --root_path $DATA_ROOT/weather/ \
            --data_path weather.csv \
            --model_id "Weather_96_${pl}" \
            --model ${model_name} \
            --data custom \
            --features M \
            --seq_len 96 \
            --label_len 48 \
            --pred_len ${pl} \
            --label_pred_len ${pl} \
            --e_layers 3 \
            --d_layers 1 \
            --factor 3 \
            --enc_in 21 \
            --dec_in 21 \
            --c_out 21 \
            --des ${des} \
            --d_model 512 \
            --d_ff 512 \
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


model_name=iTransformer
pl_list=(2 3 4 6 8 12 16 24 48)
dst="PEMS03"
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
            --root_path $DATA_ROOT/PEMS/ \
            --data_path PEMS03.npz \
            --model_id "PEMS03_96_${pl}" \
            --model ${model_name} \
            --data PEMS \
            --freq m \
            --features M \
            --seq_len 96 \
            --label_len 48 \
            --pred_len ${pl} \
            --label_pred_len ${pl} \
            --e_layers 4 \
            --d_layers 1 \
            --factor 3 \
            --enc_in 358 \
            --dec_in 358 \
            --c_out 358 \
            --des ${des} \
            --d_model 512 \
            --d_ff 512 \
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



model_name=iTransformer
pl_list=(2 3 4 6 8 12 16 24 48)
dst="PEMS08"
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
            --root_path $DATA_ROOT/PEMS/ \
            --data_path PEMS08.npz \
            --model_id "PEMS08_96_${pl}" \
            --model ${model_name} \
            --data PEMS \
            --freq m \
            --features M \
            --seq_len 96 \
            --label_len 48 \
            --pred_len ${pl} \
            --label_pred_len ${pl} \
            --e_layers 4 \
            --d_layers 1 \
            --factor 3 \
            --enc_in 170 \
            --dec_in 170 \
            --c_out 170 \
            --des ${des} \
            --d_model 512 \
            --d_ff 512 \
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