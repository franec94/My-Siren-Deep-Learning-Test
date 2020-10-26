#!/usr/bin/env bash

# =============================================== #
# Script: run_program.sh
# Used it for launching a run for training 
# a deep learning model based on Siren-like 
# Architecture for retrieving model's weights
# that all together represent Cameramen compressed
# image.
# =============================================== #

CUDA_VISIBLE_DEVICES=0 python main_extended_compare.py \
  --logging_root '../../../../results/cameramen' \
  --experiment_name 'train' \
  --sidelength 256 \
  --num_epochs 100000 \
  --hidden_features 85 \
  --num_attempts 1 \
  --seeds 0 42 123 \
  --hidden_layers 10 11 12 13 \
  --resume_from 0 \
  --end_to  5 \
  --enable_tensorboard_logging \
  --verbose 1 \
  --show_number_of_trials \

exit 0
