:: =============================================== ::
:: Script: run_program.sh
:: Used it for launching a run for evaluation of 
:: a deep learning model based on Siren-like 
:: Architecture for retrieving model's performance
:: for representing Cameramen compressed
:: image, when post-training quantization technique
:: is employed.
:: =============================================== ::

@cls

python post_training_static_quantization.py ^
  --model_files D:\data\data_thesys\dynamic_quant\cameramen\1604779421-808068\train\arch_no_0\trial_no_0\checkpoints\model_final.pth ^
  --hf 32 ^
  --hl 5 ^
  --sidelength 256 ^
  --logging_root '/d/data/quantization_results/posterior_quantization/cameramen' ^
  --experiment_name 'train'
:: ^
::  --model_dirs '/d/data/cameramen'
