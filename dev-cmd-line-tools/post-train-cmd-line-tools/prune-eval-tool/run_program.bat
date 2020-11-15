@cls

@python main.py 
  --logging_root './results/cameramen' ^
  --experiment_name 'train' ^
  --models_filepath ^
  --sidelength 256 ^
  --n_hf 8 ^
  --n_hl 5 ^
  --global_pruning_techs 'L1Unstructured' 'RandomUnstructured' ^
  --global_pruning_rates .01 .02 .03 .04 .05 .06 .07 .08 .1 .2 .3 .4 .5 .6 .7 .8 .9 ^
  --global_pruning_abs 10 20 30 40 50 60 70 80 90 100 150 200 ^
  --dynamic_quant qint8 qfloat16 ^
  --verbose 0

:: Other options
::  --cuda ^
::  --run_dash ^
