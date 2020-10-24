#!/usr/bin/env bash

CUDA_VISIBLE_DEVICES=0 python main_extended_compare.py \
  --logging_root '../../../../results/cameramen' \
  --experiment_name 'train' \
  --num_epochs 100000 \
  --hidden_features 64 \
  --num_attempts 1 \
  --seeds 0 42 123 \
  --hidden_layers 5 6 7 \
  --resume_from 0 \
  --end_to  3 \
  --enable_tensorboard_logging \
  --verbose 1 \
  --show_number_of_trials \

exit 0
