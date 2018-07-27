#!/usr/bin/env bash

# base_101
CUDA_VISIBLE_DEVICES=0,1 python cvpr18_relation/main.py \
    -gpu_id 0 1 \
    -exp_name base_101 \
    -dataset mini-imagenet \
    -n_way 10 -k_shot 5 -k_query 5 -im_size 512 \
    -batch_sz 4 \

