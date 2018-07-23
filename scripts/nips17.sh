#!/usr/bin/env bash

python nips17_proto/train.py --gpu_id 3 -nsTr 5 -nsVa 5 -cVa 20 -suffix _cos --distance cosine
