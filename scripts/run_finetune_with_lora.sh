#!/bin/bash
# Please run this script under ${project_id} in project directory of

deepspeed_args="--master_port=11000"      # Default argument
if [ $# -ge 1 ]; then
  deepspeed_args="$1"
fi

exp_id=finetune_with_lora_chatglm2
project_dir=$(cd "$(dirname $0)"/..; pwd)
output_dir=${project_dir}/output_models/${exp_id}
log_dir=${project_dir}/log/${exp_id}

dataset_path=${project_dir}/data/example_dataset/train

mkdir -p ${output_dir} ${log_dir}
# deepspeed ${deepspeed_args} \
torchrun --nnodes 1 --nproc_per_node 1 \
  examples/finetune.py \
    --model_name_or_path /root/fan.yang/model/chatglm2/ \
    --dataset_path ${dataset_path} \
    --output_dir ${output_dir} --overwrite_output_dir \
    --num_train_epochs 1 \
    --learning_rate 1e-4 \
    --block_size 512 \
    --per_device_train_batch_size 1 \
    --per_device_eval_batch_size 1 \
    --use_lora 1 \
    --lora_r 8 \
    --save_aggregated_lora 0 \
    --run_name finetune_with_lora \
    --validation_split_percentage 0 \
    --logging_steps 20 \
    --do_train \
    --ddp_timeout 72000 \
    --save_steps 5000 \
    --dataloader_num_workers 1 \
    --do_eval \
    --eval_dataset_path data/example_dataset/train \
    --eval_steps 50 \
    --evaluation_strategy "steps" \
    --fp16 \
    --use_int8 True \
    --torch_dtype float16 \
    --ddp_find_unused_parameters False \
    --disable_group_texts True \
    --arch_type encoder_decoder \
    --lora_target_modules dense_h_to_4h dense_4h_to_h query_key_value\
    | tee ${log_dir}/train.log \
    2> ${log_dir}/train.err

    # --preprocessing_num_workers 8 \
    # --deepspeed configs/ds_config_zero2.json \