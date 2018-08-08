#!/usr/bin/env bash

CUDA_VISIBLE_DEVICES=2 python iclr18_gnn/gnn.py -gpu_id 2 --dataset omniglot --exp_name default_omni
