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
  --num_epochs 500000 \
  --hidden_features 16 \
  --num_attempts 1 \
  --seeds 0 \
  --hidden_layers 5 6 7 8 9 10 11 12 13 14\
  --resume_from 0 \
  --steps_til_summary 4 \
  --end_to  1 \
  --enable_tensorboard_logging \
  --verbose 1 \
  --show_number_of_trials \

exit 0
