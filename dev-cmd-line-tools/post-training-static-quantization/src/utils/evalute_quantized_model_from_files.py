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

from src.utils.functions import get_input_image


def _evaluate_model_local(image_dataset, model_conf, quant_tech = None, device = 'cpu'):

    eval_scores = []
    if quant_tech == None:
        try:
            print("Try eval plain model on cuda device...")
            eval_dataloader = _prepare_data_loaders(image_dataset, model_conf)
            model = prepare_model(opt = model_conf, arch_hyperparams=model_conf._asdict(), device='cuda', model_weights_file = model_conf.model_filename)
            eval_scores = _evaluate_model(model, evaluate_dataloader=eval_dataloader, device='cuda')
        except:
            print("No cuda device available, switching to cpu.")
            print("Try eval plain model on cpu device...")
            eval_dataloader = _prepare_data_loaders(image_dataset, model_conf)
            model = prepare_model(opt = model_conf, arch_hyperparams=model_conf._asdict(), device='cpu')
            eval_scores = _evaluate_model(model, evaluate_dataloader=eval_dataloader, device='cpu')
        pass

    return eval_scores

def evaluate_models_from_files(opt):

    # Prepare named-tuple for model's detail data.
    tuple_data = [opt.model_files, opt.hl, opt.hf, opt.sidelength]
    InfoModel = collections.namedtuple('InfoModel', "model_filename,hidden_layers,hidden_features,sidelength")

    tuple_data = [opt.model_files, opt.hl, opt.hf, opt.sidelength]
    InfoModel2 = collections.namedtuple('InfoModel', "model_filename,hidden_layers,hidden_features,sidelength, quantization_enabled")

    fields_name = "model_filename,hidden_layers,hidden_features,sidelength,quant_tech,mse,psnr,ssim".split(",")
    InfoResults = collections.namedtuple('InfoResults', fields_name)

    if opt.quantization_enabled != None:
        if isinstance(opt.quantization_enabled, str):
            quant_tech_list = [opt.quantization_enabled]
        else:
            quant_tech_list = opt.quantization_enabled
    else:
        quant_tech_list = []

    records_list = []
    for a_model_conf in list(map(InfoModel._make, zip(*tuple_data))):
        image_dataset, _, _ = get_input_image(opt = opt)
        pprint(a_model_conf)

        a_model_conf_2 = InfoModel2._make(list(a_model_conf._asdict().values()) + [None])

        eval_scores = _evaluate_model_local(image_dataset = image_dataset, model_conf = a_model_conf_2, quant_tech = None, device = 'cuda')

        a_list = list(a_model_conf_2._asdict().values()) + list(eval_scores)
        record_eval_scores = InfoResults._make(a_list)
        records_list.append(record_eval_scores)
        
        for a_tech in quant_tech_list:
            pass
        pass

    if len(records_list) != 0:
        table = tabulate.tabulate(tabular_data=records_list, headers=fields_name)
        print(table)
        pass
    pass