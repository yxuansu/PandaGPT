#!/bin/bash

deepspeed --include localhost:0,1,2,3,4,5,6,7 --master_addr 127.0.0.1 --master_port 28457 train_sft.py \
    --model openllama_peft \
    --stage 1\
    --data_path  ../data/pandagpt4_visual_instruction_data.json\
    --image_root_path ../data/images/\
    --imagebind_ckpt_path ../pretrained_ckpt/imagebind_ckpt/\
    --vicuna_ckpt_path ../pretrained_ckpt/vicuna_ckpt/13b_v0/\
    --max_tgt_len 400\
    --save_path  ./ckpt/pandagpt_13b_v0_peft/\
    --log_path ./ckpt/pandagpt_13b_v0_peft/log_rest/
