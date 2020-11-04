#!/usr/bin/env bash

# =============================================== #
# Script: run_program.sh
# Used it for launching a run for evaluation of 
# a deep learning model based on Siren-like 
# Architecture for retrieving model's performance
# for representing Cameramen compressed
# image, when post-training quantization technique
# is employed.
# =============================================== #

# CUDA_VISIBLE_DEVICES=0 python main_extended_compare.py \
python post_training_static_quantization.py \
  --logging_root '../../../../results/quantization/post_training/cameramen' \
  --experiment_name 'train' \
  --log_models '/home/chiarlo/siren-project/data/input.csv'

# --model_files ''
# --model_dirs '../../../../results/cameramen'

exit 0
