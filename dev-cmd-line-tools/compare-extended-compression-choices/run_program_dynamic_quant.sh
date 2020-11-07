#!/usr/bin/env bash

# =============================================== #
# Script: run_program.sh
# Used it for launching a run for training 
# a deep learning model based on Siren-like 
# Architecture for retrieving model's weights
# that all together represent Cameramen compressed
# image.
# Quantization technique adopted:
# - Dynamic Quantization
# =============================================== #

CUDA_VISIBLE_DEVICES=0 python main_extended_compare.py \
  --logging_root '../../../../results/dynamic_quant/cameramen' \
  --experiment_name 'train' \
  --sidelength 256 \
  --num_epochs 200000 \
  --hidden_features 32 \
  --num_attempts 1 \
  --seeds 0 42 123 \
  --hidden_layers 5 6 7 8 9 10 \
  --quantization_enabled dynamic \
  --resume_from 0 \
  --end_to  1 \
  --steps_til_summary 3 \
  --enable_tensorboard_logging \
  --verbose 1 \
  --show_number_of_trials \

exit 0