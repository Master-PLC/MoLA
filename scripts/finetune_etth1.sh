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


lpl_list=(96)
lr_list=(0.001)
r_list=(8)
n_exp_list=(6)


for lr in ${lr_list[@]}; do
for r in ${r_list[@]}; do
for lpl in ${lpl_list[@]}; do
if [ $lpl -eq 96 ]; then
    pl_list=(32)
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
