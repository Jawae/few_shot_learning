#!/usr/bin/env bash

CUDA_VISIBLE_DEVICES=0,1 python /home/hongyang/work/few_shot_learning/cvpr18_relation/main.py -im_size 152 -gpu_id 0 1 -n_way 10 -k_shot 5 -batchsz 2
