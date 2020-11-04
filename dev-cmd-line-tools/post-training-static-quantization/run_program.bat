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
  --logging_root '/d/data/quantization_results/posterior_quantization/cameramen' ^
  --experiment_name 'train' ^
  --model_dirs '/d/data/cameramen'
