#!/usr/bin/env bash

CUDA_VISIBLE_DEVICES=1,2 python cvpr18/train_relation.py -gpu_id 3 -n_way 10 -k_shot 5 -batchsz 2
