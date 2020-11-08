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
# skimage
# --------------------------------------------- #
import skimage
import skimage.metrics as skmetrics
from skimage.metrics import peak_signal_noise_ratio as psnr
from skimage.metrics import structural_similarity as ssim
from skimage.metrics import mean_squared_error

# --------------------------------------------- #
# Torch + Torchvision
# --------------------------------------------- #
import torch
import torchvision

import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset

from torchvision import datasets
from torchvision import transforms
from torchvision.transforms import Resize, Compose, ToTensor, Normalize

# --------------------------------------------- #
# Import: custom, from this project
# --------------------------------------------- #

import src.utils.dataio as dataio
from src.utils.archs.siren import Siren
from src.utils.archs.siren_quantized import SirenQuantized
from src.utils.archs.siren_quantized_post_training import SirenQPT
from src.utils.archs.siren_quantizatin_aware_train import SirenQAT

from src.utils.quant_utils.quant_utils_functions import get_dynamic_quantization_model
from src.utils.quant_utils.quant_utils_functions import get_paszke_quant_model
from src.utils.quant_utils.quant_utils_functions import get_post_training_quantization_model
from src.utils.quant_utils.quant_utils_functions import get_quantization_aware_training
from src.utils.quant_utils.quant_utils_functions import get_static_quantization_model

from src.utils.quant_utils.quant_utils_functions import _evaluate_model
from src.utils.quant_utils.quant_utils_functions import _prepare_data_loaders
from src.utils.quant_utils.quant_utils_functions import prepare_model


def evaluate_models_from_files(opt):
    tuple_data = [opt.model_files, opt.hl, opt.hf, opt.sidelength]
    InfoModel = collections.namedtuple('InfoModel', "filename,hidden_layers,hidden_features,sidelength")
    for a_model_conf in list(map(InfoModel._make, zip(*tuple_data))):
        pprint(a_model_conf)
    pass