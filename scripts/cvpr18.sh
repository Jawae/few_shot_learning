#!/usr/bin/env bash

# base_101
CUDA_VISIBLE_DEVICES=2,3 python cvpr18_relation/main.py \
    -device_id 2 3 \
    -exp_name base_101 \
    -dataset mini-imagenet \
    -n_way 5 -k_shot 5 -k_query 1 -im_size 128 \
    -batch_sz 2 \

