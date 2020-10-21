#!/usr/bin/env bash

CUDA_VISIBLE_DEVICES=0 python main_extended_compare.py \
  --logging_root '../../../../results' \
  --experiment_name 'train' \
  --resume_from 0 \
  --end_to  1 \
  --num_epochs 500 \
  --hidden_features 110 \
  --num_attempts 3 \
  --seeds 0 42 123 \
  --hidden_layers 5 7 8 9 \
  --show_timetable_estimate \
  --verbose 1 \

exit 0
