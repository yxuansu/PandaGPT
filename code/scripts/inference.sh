#!/bin/bash

CUDA_VISIBLE_DEVICES=0 python inference_model.py \
    --imagebind_ckpt_path ../pretrained_ckpt/imagebind_ckpt/ \
    --vicuna_ckpt_path ../pretrained_ckpt/vicuna_ckpt/7b_v0/ \
    --delta_ckpt_path ./ckpt/vicuna_7b_v0_peft/image_captioning_then_visual_instruction_stage_2//pytorch_model.pt \
    --max_tgt_len 128 
