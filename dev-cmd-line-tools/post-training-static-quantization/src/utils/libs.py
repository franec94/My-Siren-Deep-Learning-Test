from __future__ import print_function
from __future__ import division

# --------------------------------------------- #
# Standard Library, plus some Third Party Libraries
# --------------------------------------------- #

import logging
from pprint import pprint
from PIL import Image
from tqdm import tqdm
from typing import Union, Tuple

import configargparse
from functools import partial

import copy
import collections
import datetime
import functools
import h5py
import math
import os
import operator
import pickle
import random
import shutil
import sys
import re
import time
# import visdom

# --------------------------------------------- #
# Data Science and Machine Learning Libraries
# --------------------------------------------- #
import matplotlib
import matplotlib.pyplot as plt
matplotlib.style.use('ggplot')

import numpy as np
import pandas as pd
import sklearn

from sklearn.model_selection import ParameterGrid
from sklearn.model_selection import train_test_split

# --------------------------------------------- #
# Torch
# --------------------------------------------- #
try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    import torch.optim as optim
    from torch.utils.data import DataLoader, Dataset
    import torch.quantization
except:
    print("torch not available!")
    pass


# --------------------------------------------- #
# Import: TorchVision
# --------------------------------------------- #
try:
    import torchvision
    from torchvision import datasets
    from torchvision import transforms
    from torchvision.transforms import Resize, Compose, ToTensor, Normalize
    from torchvision.utils import save_image
except:
    print("torchvision library not available!")
    pass

# --------------------------------------------- #
# Import: skimage
# --------------------------------------------- #
try:
    import skimage
    import skimage.metrics as skmetrics
    from skimage.metrics import peak_signal_noise_ratio as psnr
    from skimage.metrics import structural_similarity as ssim
    from skimage.metrics import mean_squared_error
except:
    print("skimage library not available!")
    pass


# --------------------------------------------- #
# Import: custom, from this project
# --------------------------------------------- #
from src.utils.custom_argparser import get_cmd_line_opts

from src.utils.functions import check_quantization_tech_provided, check_frequencies
from src.utils.functions import get_input_image, get_root_level_logger, log_parser
from src.utils.functions import filter_model_files_opt_args, filter_model_files_csv_opt_args
from src.utils.functions import map_filter_model_dirs_opt_args, create_train_logging_dir

from src.utils.evalute_quantized_model_from_files import evaluate_models_from_files

from src.utils.evalute_quantized_model_from_csv import evaluate_plain_model
from src.utils.evalute_quantized_model_from_csv import evaluate_post_train_quantized_models_by_csv, evaluate_post_train_posterion_quantized_models_by_csv_list
from src.utils.evalute_quantized_model_from_csv import evaluate_post_train_models_by_csv, evaluate_post_train_models_by_csv_list

from src.utils.quant_utils.quant_utils_functions import get_dynamic_quantization_model
from src.utils.quant_utils.quant_utils_functions import get_paszke_quant_model
from src.utils.quant_utils.quant_utils_functions import get_post_training_quantization_model
from src.utils.quant_utils.quant_utils_functions import get_quantization_aware_training
from src.utils.quant_utils.quant_utils_functions import get_static_quantization_model

from src.utils.quant_utils.quant_utils_functions import _evaluate_model
from src.utils.quant_utils.quant_utils_functions import _prepare_data_loaders
from src.utils.quant_utils.quant_utils_functions import prepare_model
