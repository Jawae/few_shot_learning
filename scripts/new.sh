#!/usr/bin/env bash

python src/train.py --gpu_id 0 -nsTr 5 -nsVa 5 -cVa 5 -suffix _cos --distance cosine
