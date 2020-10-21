#!/usr/bin/env bash

clear
echo "Compute analysis for Camera (default) image:"
python3 \
  compute_analysis_jpeg_compression.py \
  --output_path camera/output \
  --logging_path camera/log

echo "Compute analysis for test068.png image:"
python3 \
  compute_analysis_jpeg_compression.py \
  --input_image ../../testsets/BSD68/test068.png \
  --output_path test068/output \
  --logging_path test068/log

exit 0
