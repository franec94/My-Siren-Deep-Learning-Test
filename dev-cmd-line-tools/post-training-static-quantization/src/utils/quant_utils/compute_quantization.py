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
# Custom Project Imports
# --------------------------------------------- #
import src.utils.dataio as dataio
from src.utils.siren import Siren
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

# ----------------------------------------------------------------------------------------------- #
# Functions
# ----------------------------------------------------------------------------------------------- #

# --------------------------------------------- #
# Compute Quantization By Tech
# --------------------------------------------- #

def compute_quantization_paszke_quant_mode(model_path, arch_hyperparams, img_dataset, opt, fuse_modules = None, device = 'cpu', qconfig = 'fbgemm', model_fp32 = None):
    """Compute post train scores by means of Paszek Quant Mode."""
    eval_scores = None
    input_fp32 = _prepare_data_loaders(img_dataset, opt)

    model_int8 = get_paszke_quant_model(
        metadata_model_dict = arch_hyperparams,
        model_path = model_path,
        fuse_modules = fuse_modules,
        device = f'{device}',
        qconfig = f'{qconfig}',
        model_fp32 = model_fp32)
    
    eval_scores = _evaluate_model(model = model_int8, evaluate_dataloader = input_fp32, loss_fn = nn.MSELoss(), device = f'{device}')
    return eval_scores


def compute_quantization_dyanmic_mode(model_path, arch_hyperparams, img_dataset, opt, fuse_modules = None, device = 'cpu', qconfig = 'fbgemm', model_fp32 = None):
    """Evaluate PyTorch model already trained by means of dynamic quantization.
    Return:
    ------
    :eval_scores: np.ndarray object containing mse, psnr, and ssim scores
    """
    
    input_fp32 = _prepare_data_loaders(img_dataset, opt)
    """model_fp32 = Siren(
        in_features=2,
        out_features=1,
        hidden_features=int(arch_hyperparams['hidden_features']),
        hidden_layers=int(arch_hyperparams['hidden_layers']),
        # outermost_linear=True).to(device=device)
        outermost_linear=True)"""
    model_int8 = get_dynamic_quantization_model(metadata_model_dict = arch_hyperparams, set_layers = {torch.nn.Linear}, device = 'cpu', qconfig = 'fbgemm', model_fp32 = model_fp32)
    eval_scores = _evaluate_model(model = model_int8, evaluate_dataloader = input_fp32, loss_fn = nn.MSELoss(), device = 'cpu')
    return eval_scores


def compute_quantization_static_mode(model_path, arch_hyperparams, img_dataset, opt, fuse_modules = None, device = 'cpu', qconfig = 'fbgemm', model_fp32 = None):
    """Evaluate PyTorch model already trained by means of static quantization.
    Return:
    ------
    :eval_scores: np.ndarray object containing mse, psnr, and ssim scores
    """
    input_fp32 = _prepare_data_loaders(img_dataset, opt)
    model_fp32_prepared = get_static_quantization_model(
        model_path = model_path, metadata_model_dict = arch_hyperparams,
        fuse_modules = fuse_modules, device = f"{device}", qconfig = f"{qconfig}", model_fp32 = model_fp32)
    # Calibrate model
    _ = _evaluate_model(model = model_fp32_prepared, evaluate_dataloader = input_fp32, loss_fn = nn.MSELoss(), device = 'cpu')

    # Evaluate quantized int8 model
    model_int8 = torch.quantization.convert(model_fp32_prepared)
    input_fp32 = _prepare_data_loaders(img_dataset, opt)
    eval_scores = _evaluate_model(model = model_int8, evaluate_dataloader = input_fp32, loss_fn = nn.MSELoss(), device = 'cpu')
    return eval_scores


def compute_quantization_post_train_mode(model_path, arch_hyperparams, img_dataset, opt, fuse_modules = None, device = 'cpu', qconfig = 'fbgemm', model_fp32 = None):
    """Evaluate PyTorch model already trained by means of post train quantization.
    Return:
    ------
    :eval_scores: np.ndarray object containing mse, psnr, and ssim scores
    """
    
    input_fp32 = _prepare_data_loaders(img_dataset, opt)
    model_fp32_prepared = get_post_training_quantization_model(
        model_path = model_path, metadata_model_dict = arch_hyperparams,
        fuse_modules = fuse_modules, device = f"{device}", qconfig = f"{qconfig}", model_fp32 = model_fp32)
    
    # Calibrate model
    _ = _evaluate_model(model = model_fp32_prepared, evaluate_dataloader = input_fp32, loss_fn = nn.MSELoss(), device = 'cpu')

    # Evaluate quantized int8 model
    model_int8 = torch.quantization.convert(model_fp32_prepared)
    input_fp32 = _prepare_data_loaders(img_dataset, opt)
    eval_scores = _evaluate_model(model = model_int8, evaluate_dataloader = input_fp32, loss_fn = nn.MSELoss(), device = 'cpu')
    return eval_scores


def compute_quantization_aware_train_mode(model_path, arch_hyperparams, img_dataset, opt, fuse_modules = None, device = 'cpu', qconfig = 'fbgemm', model_fp32 = None):
    """Evaluate PyTorch model already trained by means of quantization aware train.
    Return:
    ------
    :eval_scores: np.ndarray object containing mse, psnr, and ssim scores
    """
    
    input_fp32 = _prepare_data_loaders(img_dataset, opt)
    model = get_quantization_aware_training(model_path = model_path, metadata_model_dict = arch_hyperparams, fuse_modules = fuse_modules, device = device, qconfig = qconfig, model_fp32 = model_fp32)
    eval_scores = _evaluate_model(model = model, evaluate_dataloader = input_fp32, loss_fn = nn.MSELoss(), device = 'cpu')

    model_int8 = torch.quantization.convert(model)
    input_fp32 = _prepare_data_loaders(img_dataset, opt)
    
    eval_scores = _evaluate_model(model = model_int8, evaluate_dataloader = input_fp32, loss_fn = nn.MSELoss(), device = 'cpu')
    return eval_scores


# --------------------------------------------- #
# Compute Quantization
# --------------------------------------------- #

def compute_quantization(img_dataset, opt, model_path = None, arch_hyperparams = None, fuse_modules = None, device = 'cpu', qconfig = 'fbgemm'):
    """Compute quantized results."""

    eval_scores = None
    if opt.quantization_enabled != None:
        # --- Dynamic Quantization: TODO test it.
        if opt.quantization_enabled == 'dynamic':
            eval_scores = compute_quantization_dyanmic_mode(
                model_path,
                arch_hyperparams,
                img_dataset,
                opt, fuse_modules = None, device = 'cpu', qconfig = 'fbgemm', model_fp32 = None)
            pass
        
        # --- Static Quantization: TODO test it.
        elif opt.quantization_enabled == 'static':
            """eval_scores = compute_quantization_static_mode(
                model_path,
                arch_hyperparams,
                img_dataset,
                opt,
                fuse_modules = None, device = 'cpu', qconfig = 'fbgemm', model_fp32 = None)"""
            pass

        # --- Post Train Quantization: TODO test it.
        elif opt.quantization_enabled == 'post_train':
            eval_scores = compute_quantization_post_train_mode(
                model_path,
                arch_hyperparams,
                img_dataset,
                opt,
                fuse_modules = None, device = 'cpu', qconfig = f"{qconfig}", model_fp32 = None)
            pass

        # --- Quantization Aware Training: TODO test it.
        elif opt.quantization_enabled =='quantization_aware_training':
            """compute_quantization_aware_train_mode(
                model_path,
                arch_hyperparams,
                img_dataset,
                opt,
                fuse_modules = None, device = 'cpu', qconfig = 'fbgemm', model_fp32 = None)"""
            pass
        elif opt.quantization_enabled =='paszke_quant':
            eval_scores = compute_quantization_paszke_quant_mode(
                model_path,
                arch_hyperparams,
                img_dataset,
                opt,
                fuse_modules = None, device = 'cpu', qconfig = f"{qconfig}", model_fp32 = None)
            pass
        else:
            raise Exception(f"Error: {opt.quantization_enabled} not allowed!")
    return eval_scores
