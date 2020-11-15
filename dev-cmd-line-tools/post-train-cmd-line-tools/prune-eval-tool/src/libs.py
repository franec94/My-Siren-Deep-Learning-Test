from __future__ import print_function
from __future__ import division

SHOW_RESULTS_BY_TABS = True
DASH_TEMPLATES_LIST = ["plotly", "plotly_white", "plotly_dark", "ggplot2", "seaborn", "simple_white", "none"]

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

# Plotly imports.
# ----------------------------------------------- #
import chart_studio.plotly as py
import plotly.figure_factory as ff
import plotly.express as px

# Dash imports.
# ----------------------------------------------- #
import dash
import dash_bootstrap_components as dbc
import dash_core_components as dcc
import dash_html_components as html
from dash.dependencies import Input, Output

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

from src.generic.utils import prepare_model, get_size_of_model, get_data_for_train, set_seeds, set_device_and_backend_for_torch
from src.generic.custom_argparser import get_cmd_line_opts
from src.generic.functions import create_train_logging_dir, get_input_image, log_parser, get_root_level_logger, check_quantization_tech_provided, check_quant_size_for_dynamic_quant
from src.eval.eval_model import evaluate_model
from src.archs.siren_compute_quantization import compute_quantization_dyanmic_mode
from src.generic.dataio import Implicit2DWrapper
from src.dash_app.custom_dash_app import get_dash_app

from src.prune_techs.prune_utils import compute_pruning_evaluation, compute_prune_unstructured_results