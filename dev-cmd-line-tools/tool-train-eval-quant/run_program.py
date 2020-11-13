#!/usr/bin/env bash

# =============================================== #
# Script: run_program.sh
# Used it for launching a run for training 
# a deep learning model based on Siren-like 
# Architecture for retrieving model's weights
# that all together represent Cameramen compressed
# image.
# =============================================== #

CUDA_VISIBLE_DEVICES=0 python main.py \
  --logging_root '../../../../results/cameramen' \
  --experiment_name 'train' \
  --sidelength 256 \
  --num_epochs 500000 \
  --n_hf 16 \
  --n_hl 5 6 7 8 9 10 11 12 13 14 \
  --seed 0 \
  --evaluate \
  --dynamic_quant qint8 float16 \
  --verbose 1

exit 0