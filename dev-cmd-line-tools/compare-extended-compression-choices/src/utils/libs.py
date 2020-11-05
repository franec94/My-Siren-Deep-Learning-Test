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
import src.utils.dataio as dataio
import src.utils.evaluate as evaluate
import src.utils.graphics as graphics

import src.utils.loss_functions as loss_functions
import src.utils.modules as modules
from src.utils.siren import Siren
import src.utils.train_extended_compare as train_extended_compare

import src.utils.utils as utils

from src.utils.functions import create_train_logging_dir, get_root_level_logger, check_quantization_tech_provided
from src.utils.functions import get_input_image, get_arch_hyperparams, show_number_of_trials
from src.utils.functions import log_parser, show_model_summary, set_hyperparams_to_be_tested

from src.utils.siren_dynamic_quantization import get_dynamic_quantization_model, get_static_quantization_model, get_post_training_quantization_model
