#!/bin/bash

CUDA_VISIBLE_DEVICES=0 \
    deepspeed examples/evaluation.py \
    --answer_type medmcqa \
    --model_name_or_path /root/fan.yang/model/chinese-llm \
    --dataset_path data/example_dataset/test \
    --deepspeed examples/ds_config.json \
    --inference_batch_size_per_device 1 \
    --metric accuracy \
    --use_int8 1
