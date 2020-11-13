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
  --num_epochs 100 \
  --n_hf 16 \
  --n_hl 5 6 \
  --seed 0 1 \
  --cuda \
  --train \
  --evaluate \
  --dynamic_quant qint8 qfloat16 \
  --verbose 0

exit 0