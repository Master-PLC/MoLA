#!/bin/bash

GPU=0
DATA_ROOT=./dataset
EXP_NAME=finetune

seed=2023
des='Exp'
tuner_type=lora
target_modules="['encoder.*.ffn']"
train_epochs=10
patience=3
rl=1.0
ax=0.0



model_name=iTransformerMoLora
dst="ETTh1"


lpl_list=(96 192 336 720)
lr_list=(0.001)
r_list=(8)
n_exp_list=(6)


for lr in ${lr_list[@]}; do
for r in ${r_list[@]}; do
for lpl in ${lpl_list[@]}; do
if [ $lpl -eq 96 ]; then
    pl_list=(24 32)
elif [ $lpl -eq 192 ]; then
    pl_list=(48 64)
elif [ $lpl -eq 336 ]; then
    pl_list=(42 84)
elif [ $lpl -eq 720 ]; then
    pl_list=(90 180)
fi
for pl in ${pl_list[@]}; do
for n_exp in ${n_exp_list[@]}; do

    pretrain_model_path=./results/pretrain/iTransformer_ETTh1_${pl}_0.0005/checkpoints/long_term_forecast_ETTh1_96_${pl}_iTransformer_ETTh1_ftM_sl96_ll48_pl${pl}_lpl${pl}_dm128_nh8_el2_dl1_df128_fc3_ebtimeF_dtTrue_ax0.0_rl1.0_axlMAE_mf1_base_0/checkpoint.pth

    num_tuner_online=$((lpl / pl))
    JOB_NAME="${model_name}_${dst}_${lpl}_${pl}_${lr}_${r}_${n_exp}"
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
            --task_name long_term_forecast_finetune_group \
            --is_training 1 \
            --root_path $DATA_ROOT/ETT-small/ \
            --data_path ETTh1.csv \
            --model_id "ETTh1_96_${lpl}" \
            --model ${model_name} \
            --data ETTh1 \
            --features M \
            --seq_len 96 \
            --label_len 48 \
            --pred_len ${pl} \
            --label_pred_len ${lpl} \
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
            --r ${r} \
            --pretrain_model_path ${pretrain_model_path} \
            --tuner_type ${tuner_type} \
            --train_epochs ${train_epochs} \
            --patience ${patience} \
            --target_modules ${target_modules} \
            --num_tuner_online ${num_tuner_online} \
            --n_exp ${n_exp} \
            --init_type normal \
            --std_scale 0.01

        sleep 10
    } 2>&1 | tee -a "${OUTPUT_DIR}/stdout.txt" &
done
done
done
done
done


wait




model_name=iTransformerMoLora
dst="ETTh2"


lpl_list=(96 192 336 720)
lr_list=(0.001)
r_list=(8)
n_exp_list=(6)


for lr in ${lr_list[@]}; do
for r in ${r_list[@]}; do
for lpl in ${lpl_list[@]}; do
if [ $lpl -eq 96 ]; then
    pl_list=(24 32)
elif [ $lpl -eq 192 ]; then
    pl_list=(48 64)
elif [ $lpl -eq 336 ]; then
    pl_list=(42 84)
elif [ $lpl -eq 720 ]; then
    pl_list=(90 180)
fi
for pl in ${pl_list[@]}; do
for n_exp in ${n_exp_list[@]}; do

    pretrain_model_path=./results/pretrain/iTransformer_ETTh2_${pl}_0.0005/checkpoints/long_term_forecast_ETTh2_96_${pl}_iTransformer_ETTh2_ftM_sl96_ll48_pl${pl}_lpl${pl}_dm128_nh8_el2_dl1_df128_fc3_ebtimeF_dtTrue_ax0.0_rl1.0_axlMAE_mf1_base_0/checkpoint.pth

    num_tuner_online=$((lpl / pl))
    JOB_NAME="${model_name}_${dst}_${lpl}_${pl}_${lr}_${r}_${n_exp}"
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
            --task_name long_term_forecast_finetune_group \
            --is_training 1 \
            --root_path $DATA_ROOT/ETT-small/ \
            --data_path ETTh2.csv \
            --model_id "ETTh2_96_${lpl}" \
            --model ${model_name} \
            --data ETTh2 \
            --features M \
            --seq_len 96 \
            --label_len 48 \
            --pred_len ${pl} \
            --label_pred_len ${lpl} \
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
            --r ${r} \
            --pretrain_model_path ${pretrain_model_path} \
            --tuner_type ${tuner_type} \
            --train_epochs ${train_epochs} \
            --patience ${patience} \
            --target_modules ${target_modules} \
            --num_tuner_online ${num_tuner_online} \
            --n_exp ${n_exp} \
            --init_type normal \
            --std_scale 0.01
        
        sleep 10
    } 2>&1 | tee -a "${OUTPUT_DIR}/stdout.txt" &
done
done
done
done
done


wait





model_name=iTransformerMoLora
dst="ETTm1"


lpl_list=(96 192 336 720)
lr_list=(0.001)
r_list=(8)
n_exp_list=(6)



for lr in ${lr_list[@]}; do
for r in ${r_list[@]}; do
for lpl in ${lpl_list[@]}; do
if [ $lpl -eq 96 ]; then
    pl_list=(24 32)
elif [ $lpl -eq 192 ]; then
    pl_list=(48 64)
elif [ $lpl -eq 336 ]; then
    pl_list=(42 84)
elif [ $lpl -eq 720 ]; then
    pl_list=(90 180)
fi
for pl in ${pl_list[@]}; do
for n_exp in ${n_exp_list[@]}; do

    pretrain_model_path=./results/pretrain/iTransformer_ETTm1_${pl}_0.0005/checkpoints/long_term_forecast_ETTm1_96_${pl}_iTransformer_ETTm1_ftM_sl96_ll48_pl${pl}_lpl${pl}_dm128_nh8_el2_dl1_df128_fc3_ebtimeF_dtTrue_ax0.0_rl1.0_axlMAE_mf1_base_0/checkpoint.pth

    num_tuner_online=$((lpl / pl))
    JOB_NAME="${model_name}_${dst}_${lpl}_${pl}_${lr}_${r}_${n_exp}"
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
            --task_name long_term_forecast_finetune_group \
            --is_training 1 \
            --root_path $DATA_ROOT/ETT-small/ \
            --data_path ETTm1.csv \
            --model_id "ETTm1_96_${lpl}" \
            --model ${model_name} \
            --data ETTm1 \
            --features M \
            --seq_len 96 \
            --label_len 48 \
            --pred_len ${pl} \
            --label_pred_len ${lpl} \
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
            --r ${r} \
            --pretrain_model_path ${pretrain_model_path} \
            --tuner_type ${tuner_type} \
            --train_epochs ${train_epochs} \
            --patience ${patience} \
            --target_modules ${target_modules} \
            --num_tuner_online ${num_tuner_online} \
            --n_exp ${n_exp} \
            --init_type normal \
            --std_scale 0.01
        
        sleep 10
    } 2>&1 | tee -a "${OUTPUT_DIR}/stdout.txt" &
done
done
done
done
done


wait





model_name=iTransformerMoLora
dst="ETTm2"


lpl_list=(96 192 336 720)
lr_list=(0.001)
r_list=(8)
n_exp_list=(6)



for lr in ${lr_list[@]}; do
for r in ${r_list[@]}; do
for lpl in ${lpl_list[@]}; do
if [ $lpl -eq 96 ]; then
    pl_list=(24 32)
elif [ $lpl -eq 192 ]; then
    pl_list=(48 64)
elif [ $lpl -eq 336 ]; then
    pl_list=(42 84)
elif [ $lpl -eq 720 ]; then
    pl_list=(90 180)
fi
for pl in ${pl_list[@]}; do
for n_exp in ${n_exp_list[@]}; do

    pretrain_model_path=./results/pretrain/iTransformer_ETTm2_${pl}_0.0005/checkpoints/long_term_forecast_ETTm2_96_${pl}_iTransformer_ETTm2_ftM_sl96_ll48_pl${pl}_lpl${pl}_dm128_nh8_el2_dl1_df128_fc3_ebtimeF_dtTrue_ax0.0_rl1.0_axlMAE_mf1_base_0/checkpoint.pth

    num_tuner_online=$((lpl / pl))
    JOB_NAME="${model_name}_${dst}_${lpl}_${pl}_${lr}_${r}_${n_exp}"
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
            --task_name long_term_forecast_finetune_group \
            --is_training 1 \
            --root_path $DATA_ROOT/ETT-small/ \
            --data_path ETTm2.csv \
            --model_id "ETTm2_96_${lpl}" \
            --model ${model_name} \
            --data ETTm2 \
            --features M \
            --seq_len 96 \
            --label_len 48 \
            --pred_len ${pl} \
            --label_pred_len ${lpl} \
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
            --r ${r} \
            --pretrain_model_path ${pretrain_model_path} \
            --tuner_type ${tuner_type} \
            --train_epochs ${train_epochs} \
            --patience ${patience} \
            --target_modules ${target_modules} \
            --num_tuner_online ${num_tuner_online} \
            --n_exp ${n_exp} \
            --init_type normal \
            --std_scale 0.01
        
        sleep 10
    } 2>&1 | tee -a "${OUTPUT_DIR}/stdout.txt" &
done
done
done
done
done


wait






model_name=iTransformerMoLora
dst="Weather"


lpl_list=(96 192 336 720)
lr_list=(0.001)
r_list=(8)
n_exp_list=(6)



for lr in ${lr_list[@]}; do
for r in ${r_list[@]}; do
for lpl in ${lpl_list[@]}; do
if [ $lpl -eq 96 ]; then
    pl_list=(24 32)
elif [ $lpl -eq 192 ]; then
    pl_list=(48 64)
elif [ $lpl -eq 336 ]; then
    pl_list=(42 84)
elif [ $lpl -eq 720 ]; then
    pl_list=(90 180)
fi
for pl in ${pl_list[@]}; do
for n_exp in ${n_exp_list[@]}; do

    pretrain_model_path=./results/pretrain/iTransformer_Weather_${pl}_0.0005/checkpoints/long_term_forecast_Weather_96_${pl}_iTransformer_custom_ftM_sl96_ll48_pl${pl}_lpl${pl}_dm512_nh8_el3_dl1_df512_fc3_ebtimeF_dtTrue_ax0.0_rl1.0_axlMAE_mf1_base_0/checkpoint.pth

    num_tuner_online=$((lpl / pl))
    JOB_NAME="${model_name}_${dst}_${lpl}_${pl}_${lr}_${r}_${n_exp}"
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
            --task_name long_term_forecast_finetune_group \
            --is_training 1 \
            --root_path $DATA_ROOT/weather/ \
            --data_path weather.csv \
            --model_id "Weather_96_${lpl}" \
            --model ${model_name} \
            --data custom \
            --features M \
            --seq_len 96 \
            --label_len 48 \
            --pred_len ${pl} \
            --label_pred_len ${lpl} \
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
            --r ${r} \
            --pretrain_model_path ${pretrain_model_path} \
            --tuner_type ${tuner_type} \
            --train_epochs ${train_epochs} \
            --patience ${patience} \
            --target_modules ${target_modules} \
            --num_tuner_online ${num_tuner_online} \
            --n_exp ${n_exp} \
            --init_type normal \
            --std_scale 0.01

        sleep 10
    } 2>&1 | tee -a "${OUTPUT_DIR}/stdout.txt" &
done
done
done
done
done


wait





model_name=iTransformerMoLora
dst="PEMS03"


lpl_list=(12 24 36 48)
lr_list=(0.001)
r_list=(8)
n_exp_list=(6)



for lr in ${lr_list[@]}; do
for r in ${r_list[@]}; do
for lpl in ${lpl_list[@]}; do
if [ $lpl -eq 12 ]; then
    pl_list=(3 4)
elif [ $lpl -eq 24 ]; then
    pl_list=(4 6)
elif [ $lpl -eq 36 ]; then
    pl_list=(4 9)
elif [ $lpl -eq 48 ]; then
    pl_list=(8 12)
fi
for pl in ${pl_list[@]}; do
for n_exp in ${n_exp_list[@]}; do

    pretrain_model_path=./results/pretrain/iTransformer_PEMS03_${pl}_0.0005/checkpoints/long_term_forecast_PEMS03_96_${pl}_iTransformer_PEMS_ftM_sl96_ll48_pl${pl}_lpl${pl}_dm512_nh8_el4_dl1_df512_fc3_ebtimeF_dtTrue_ax0.0_rl1.0_axlMAE_mf1_base_0/checkpoint.pth

    num_tuner_online=$((lpl / pl))
    JOB_NAME="${model_name}_${dst}_${lpl}_${pl}_${lr}_${r}_${n_exp}"
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
            --task_name long_term_forecast_finetune_group \
            --is_training 1 \
            --root_path $DATA_ROOT/PEMS/ \
            --data_path PEMS03.npz \
            --model_id "PEMS03_96_${lpl}" \
            --model ${model_name} \
            --data PEMS \
            --freq m \
            --features M \
            --seq_len 96 \
            --label_len 48 \
            --pred_len ${pl} \
            --label_pred_len ${lpl} \
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
            --r ${r} \
            --pretrain_model_path ${pretrain_model_path} \
            --tuner_type ${tuner_type} \
            --train_epochs ${train_epochs} \
            --patience ${patience} \
            --target_modules ${target_modules} \
            --num_tuner_online ${num_tuner_online} \
            --n_exp ${n_exp} \
            --init_type normal \
            --std_scale 0.01
        
        sleep 10
    } 2>&1 | tee -a "${OUTPUT_DIR}/stdout.txt" &
done
done
done
done
done


wait




model_name=iTransformerMoLora
dst="PEMS08"


lpl_list=(12 24 36 48)
lr_list=(0.001)
r_list=(8)
n_exp_list=(6)



for lr in ${lr_list[@]}; do
for r in ${r_list[@]}; do
for lpl in ${lpl_list[@]}; do
if [ $lpl -eq 12 ]; then
    pl_list=(3 4)
elif [ $lpl -eq 24 ]; then
    pl_list=(4 6)
elif [ $lpl -eq 36 ]; then
    pl_list=(4 9)
elif [ $lpl -eq 48 ]; then
    pl_list=(8 12)
fi
for pl in ${pl_list[@]}; do
for n_exp in ${n_exp_list[@]}; do

    pretrain_model_path=./results/pretrain/iTransformer_PEMS08_${pl}_0.0005/checkpoints/long_term_forecast_PEMS08_96_${pl}_iTransformer_PEMS_ftM_sl96_ll48_pl${pl}_lpl${pl}_dm512_nh8_el4_dl1_df512_fc3_ebtimeF_dtTrue_ax0.0_rl1.0_axlMAE_mf1_base_0/checkpoint.pth

    num_tuner_online=$((lpl / pl))
    JOB_NAME="${model_name}_${dst}_${lpl}_${pl}_${lr}_${r}_${n_exp}"
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
            --task_name long_term_forecast_finetune_group \
            --is_training 1 \
            --root_path $DATA_ROOT/PEMS/ \
            --data_path PEMS08.npz \
            --model_id "PEMS08_96_${lpl}" \
            --model ${model_name} \
            --data PEMS \
            --freq m \
            --features M \
            --seq_len 96 \
            --label_len 48 \
            --pred_len ${pl} \
            --label_pred_len ${lpl} \
            --e_layers 3 \
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
            --r ${r} \
            --pretrain_model_path ${pretrain_model_path} \
            --tuner_type ${tuner_type} \
            --train_epochs ${train_epochs} \
            --patience ${patience} \
            --target_modules ${target_modules} \
            --num_tuner_online ${num_tuner_online} \
            --n_exp ${n_exp} \
            --init_type normal \
            --std_scale 0.01
        
        sleep 10
    } 2>&1 | tee -a "${OUTPUT_DIR}/stdout.txt" &
done
done
done
done
done


wait




model_name=iTransformerMoLora
dst="ECL"


lpl_list=(96 192 336 720)
lr_list=(0.001)
r_list=(8)
n_exp_list=(6)



for lr in ${lr_list[@]}; do
for r in ${r_list[@]}; do
for lpl in ${lpl_list[@]}; do
if [ $lpl -eq 96 ]; then
    pl_list=(24 32)
elif [ $lpl -eq 192 ]; then
    pl_list=(48 64)
elif [ $lpl -eq 336 ]; then
    pl_list=(42 84)
elif [ $lpl -eq 720 ]; then
    pl_list=(90 180)
fi
for pl in ${pl_list[@]}; do
for n_exp in ${n_exp_list[@]}; do

    pretrain_model_path=./results/pretrain/iTransformer_ECL_${pl}_0.0005/checkpoints/long_term_forecast_ECL_96_${pl}_iTransformer_custom_ftM_sl96_ll48_pl${pl}_lpl${pl}_dm512_nh8_el3_dl1_df512_fc3_ebtimeF_dtTrue_ax0.0_rl1.0_axlMAE_mf1_base_0/checkpoint.pth

    num_tuner_online=$((lpl / pl))
    JOB_NAME="${model_name}_${dst}_${lpl}_${pl}_${lr}_${r}_${n_exp}"
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
            --task_name long_term_forecast_finetune_group \
            --is_training 1 \
            --root_path $DATA_ROOT/electricity/ \
            --data_path electricity.csv \
            --model_id "ECL_96_${lpl}" \
            --model ${model_name} \
            --data custom \
            --features M \
            --seq_len 96 \
            --label_len 48 \
            --pred_len ${pl} \
            --label_pred_len ${lpl} \
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
            --r ${r} \
            --pretrain_model_path ${pretrain_model_path} \
            --tuner_type ${tuner_type} \
            --train_epochs ${train_epochs} \
            --patience ${patience} \
            --target_modules ${target_modules} \
            --num_tuner_online ${num_tuner_online} \
            --n_exp ${n_exp} \
            --init_type normal \
            --std_scale 0.01

        sleep 10
    } 2>&1 | tee -a "${OUTPUT_DIR}/stdout.txt" &
done
done
done
done
done


wait




model_name=iTransformerMoLora
dst="Traffic"


lpl_list=(96 192 336 720)
lr_list=(0.001)
r_list=(8)
n_exp_list=(6)



for lr in ${lr_list[@]}; do
for r in ${r_list[@]}; do
for lpl in ${lpl_list[@]}; do
if [ $lpl -eq 96 ]; then
    pl_list=(24 32)
elif [ $lpl -eq 192 ]; then
    pl_list=(48 64)
elif [ $lpl -eq 336 ]; then
    pl_list=(42 84)
elif [ $lpl -eq 720 ]; then
    pl_list=(90 180)
fi
for pl in ${pl_list[@]}; do
for n_exp in ${n_exp_list[@]}; do

    pretrain_model_path=./results/pretrain/iTransformer_Traffic_${pl}_0.0005/checkpoints/long_term_forecast_Traffic_96_${pl}_iTransformer_custom_ftM_sl96_ll48_pl${pl}_lpl${pl}_dm512_nh8_el4_dl1_df512_fc3_ebtimeF_dtTrue_ax0.0_rl1.0_axlMAE_mf1_base_0/checkpoint.pth

    num_tuner_online=$((lpl / pl / 2))
    JOB_NAME="${model_name}_${dst}_${lpl}_${pl}_${lr}_${r}_${n_exp}"
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
            --task_name long_term_forecast_finetune_group \
            --is_training 1 \
            --root_path $DATA_ROOT/traffic/ \
            --data_path traffic.csv \
            --model_id "Traffic_96_${lpl}" \
            --model ${model_name} \
            --data custom \
            --features M \
            --seq_len 96 \
            --label_len 48 \
            --pred_len ${pl} \
            --label_pred_len ${lpl} \
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
            --r ${r} \
            --pretrain_model_path ${pretrain_model_path} \
            --tuner_type ${tuner_type} \
            --train_epochs ${train_epochs} \
            --patience ${patience} \
            --target_modules ${target_modules} \
            --num_tuner_online ${num_tuner_online} \
            --n_exp ${n_exp} \
            --init_type normal \
            --std_scale 0.01
        
        sleep 10
    } 2>&1 | tee -a "${OUTPUT_DIR}/stdout.txt" &
done
done
done
done
done


wait
