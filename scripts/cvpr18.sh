#!/usr/bin/env bash

CUDA_VISIBLE_DEVICES=2,3 python cvpr18_relation/relation.py \
    -gpu_id 2 3 \
    -exp_name base_201 \
    -dataset tier-imagenet \
    -n_way 5 -k_shot 5 -k_query 5 -im_size 84 \
    -batch_sz 8 \

