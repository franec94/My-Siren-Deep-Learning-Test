#!/usr/bin/env bash

python3 \
  compute_analysis_jpeg_compression.py \
  --input_image ../../testsets/BSD68/test068.png \
  --output_path test068/output \
  --logging_path test068/log

exit 0
