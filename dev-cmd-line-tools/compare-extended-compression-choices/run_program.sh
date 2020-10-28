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
  --hidden_features 90 \
  --num_attempts 1 \
  --seeds 0 42 123 \
  --hidden_layers 3 4 5 6 7 8 9 \
  --resume_from 0 \
  --end_to  1 \
  --enable_tensorboard_logging \
  --verbose 1 \
  --show_number_of_trials \

exit 0
