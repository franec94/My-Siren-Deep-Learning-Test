from __future__ import print_function
from __future__ import division

# --------------------------------------------- #
# Standard Library, plus some Third Party Libraries
# --------------------------------------------- #

from PIL import Image
from functools import partial
from pprint import pprint
from tqdm import tqdm
from typing import Tuple, Union


import configargparse
import copy
import collections
import datetime
import functools
import h5py
import logging
import math
import os
import operator
import pickle
import random
import shutil
import sys
import re
import tabulate 
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

# ARCHS sub-module (1)
# --------------------------------------------- #
from src.utils.archs.siren_dynamic_quantization import get_dynamic_quantization_model        # Done.
from src.utils.archs.siren_dynamic_quantization import get_static_quantization_model         # Done.
from src.utils.archs.siren_dynamic_quantization import get_post_training_quantization_model  # Done.

# ARCHS sub-module (2)
# --------------------------------------------- #
from src.utils.archs.siren import Siren                                   # Done.
from src.utils.archs.siren_quantized import SirenQuantized                # Done.
from src.utils.archs.siren_quantized_post_training import SirenQPT        # Done.
from src.utils.archs.siren_quantizatin_aware_train import SineLayerQAT    # Done.

# EVAL_FUNCTIONS sub-module
# --------------------------------------------- #
# import src.utils.eval_functions.evaluate as evaluate                    # Done.

# GENERIC_SRC sub-module (1)
# --------------------------------------------- #
from src.utils.generic_src.custom_argparser import get_cmd_line_opts      # Done.
import src.utils.generic_src.dataio as dataio                             # Done.
import src.utils.generic_src.graphics as graphics                         # Done.

# GENERIC_SRC sub-module (2)
# --------------------------------------------- #
import src.utils.generic_src.utils as utils                               # Done.
import src.utils.generic_src.loss_functions as loss_functions             # Done.
import src.utils.generic_src.modules as modules                           # Done.

# GENERIC_SRC sub-module (3)
# --------------------------------------------- #
from src.utils.generic_src.functions import create_train_logging_dir, get_root_level_logger, check_quantization_tech_provided
from src.utils.generic_src.functions import get_input_image, get_arch_hyperparams, show_number_of_trials
from src.utils.generic_src.functions import log_parser, show_model_summary, set_hyperparams_to_be_tested, check_frequencies

# TRAIN_FUNCTIONS sub-module
# --------------------------------------------- #
import src.utils.train_functions.train_extended_compare as train_extended_compare