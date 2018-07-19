#!/usr/bin/env bash

CUDA_VISIBLE_DEVICES=1 python iclr18_gnn/main.py -gpu_id 1 --dataset omniglot --exp_name default
