#!/usr/bin/env bash

CUDA_VISIBLE_DEVICES=2 python iclr18_gnn/main.py -gpu_id 2 --dataset omniglot --exp_name default_omni
