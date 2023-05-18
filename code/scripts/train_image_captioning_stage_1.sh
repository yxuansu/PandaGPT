#!/bin/bash

deepspeed --include localhost:0,1,2,3,4,5,6,7 --master_addr 127.0.0.1 --master_port 28457 train_sft.py \
    --model openllama_peft \
    --stage 1\
    --data_path  ../data/stage_1/pandagpt4_stage_1_data.json\
    --image_root_path ../data/stage_1/images/\
    --imagebind_ckpt_path ../pretrained_ckpt/imagebind_ckpt/\
    --vicuna_ckpt_path ../pretrained_ckpt/vicuna_ckpt/7b_v0/\
    --max_tgt_len 128\
    --save_path  ./ckpt/vicuna_7b_v0_peft/image_captioning_stage_1/\
    --log_path ./ckpt/vicuna_7b_v0_peft/image_captioning_stage_1/log_rest/
