#!/usr/bin/env bash

CUDA_VISIBLE_DEVICES=0 python main_extended_compare.py \
  --logging_root '../../../../results' \
  --experiment_name 'train' \
  --num_epochs 100000 \
  --hidden_features 64 \
  --num_attempts 1 \
  --seeds 0 42 123 \
  --hidden_layers 10 11 12 13 \
  --resume_from 0 \
  --end_to  1 \
  --verbose 1 \

# --show_timetable_estimate \
exit 0
