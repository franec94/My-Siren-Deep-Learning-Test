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
from src.archs.siren import Siren
from src.archs.siren_quantized import SirenQuantized
from src.archs.siren_quantized_post_training import SirenQPT
from src.archs.siren_quantizatin_aware_train import SirenQAT
from src.archs.siren_custom_quant import SirenCQ

import src.generic.dataio as dataio

# ----------------------------------------------------------------------------------------------- #
# Functions
# ----------------------------------------------------------------------------------------------- #

# --------------------------------------------- #
# Getter functions: MODELS
# --------------------------------------------- #

def get_custom_quant_model(metadata_model_dict = None, model_path = None, set_layers = {torch.nn.Linear}, device = 'cpu', qconfig = 'fbgemm', model_fp32 = None):
    """Get custom quant model
    Params:
    -------
    `metadata_model_dict` - dict python object, metadata or hyper-params abot model's to be built.\n
    `model_path` - str python object, either None when we do not desire to load weights, or a file path to model's weights.\n
    `set_layers` - collection of modules to quantize.\n
    `device` - str object, kind of device upon which model's weigths and computation will be done.\n
    `qconfig` - str object, kind of device backedn for QuantizedCPU either FBGEMM for x86 server, or QNNAM for mobile devices.\n
    `model_fp32` - PyTorch model to be quantized if diverse from None, otherwise a new model will be created from scratch and then assigning to it weigths gotten from train phase.\n
    Return:
    ------
    `model_fp32` - PyTroch liek object representing DNN model to be quantized by means of Custom quantizing technique.\n
    """
    if model_fp32 == None:
        # print('Creating model...')
        if metadata_model_dict == None: raise Exception(f"metadata_model_dict is None!")
        model_fp32 = SirenCQ(
            in_features=2,
            out_features=1,
            quant = metadata_model_dict['quant'],
            num_bits = metadata_model_dict['num_bits'],
            quant_sym=metadata_model_dict['quant_sym'],
            hidden_features=int(metadata_model_dict['hidden_features']),
            hidden_layers=int(metadata_model_dict['hidden_layers']),
            # outermost_linear=True).to(device=device)
            outermost_linear=True)
        pass
    model_fp32 = check_device_and_weigths_to_laod(model_fp32 = model_fp32, device = f'{device}', model_path = model_path)
    return model_fp32


def get_dynamic_quantization_model(metadata_model_dict = None, model_path = None, set_layers = {torch.nn.Linear}, device = 'cpu', qconfig = 'fbgemm', model_fp32 = None):
    """Get dynamic quantization Siren model.
    Params:
    -------
    `metadata_model_dict` - dict python object, metadata or hyper-params abot model's to be built.\n
    `model_path` - str python object, either None when we do not desire to load weights, or a file path to model's weights.\n
    `set_layers` - collection of modules to quantize.\n
    `device` - str object, kind of device upon which model's weigths and computation will be done.\n
    `qconfig` - str object, kind of device backedn for QuantizedCPU either FBGEMM for x86 server, or QNNAM for mobile devices.\n
    `model_fp32` - PyTorch model to be quantized if diverse from None, otherwise a new model will be created from scratch and then assigning to it weigths gotten from train phase.\n
    Return:
    ------
    `model_quantized` - PyTroch liek object representing DNN model quantized by means of Post Train dynamic quantizing technique.\n
    """

    if model_fp32 == None:
        # print('Creating model...')
        if metadata_model_dict == None: raise Exception(f"metadata_model_dict is None!")
        model_fp32 = Siren(
            in_features=2,
            out_features=1,
            hidden_features=int(metadata_model_dict['hidden_features']),
            hidden_layers=int(metadata_model_dict['hidden_layers']),
            # outermost_linear=True).to(device=device)
            outermost_linear=True)
        pass
    model_fp32 = check_device_and_weigths_to_laod(model_fp32 = model_fp32, device = f'{device}', model_path = model_path)

    if 'dtype' not in metadata_model_dict.keys():
        # print('Default qint8 adopted')
        dtype = torch.qint8
    else:
        dtype = metadata_model_dict['dtype']
        # print(f'Adopted {dtype}')
        pass
    # raise Exception("Debugging kind of quat for Dynamic Technique...")
    # print('Quantize model...')
    model_quantized = torch.quantization.quantize_dynamic(
        model_fp32,         # the original model
        set_layers,         # a set of layers to dynamically quantize
        dtype=dtype)  # the target dtype for quantized weights
    if model_quantized == None: Exception(f"model_quantized is None, when quantization is Dynamic!")
    return model_quantized


def get_static_quantization_model(metadata_model_dict = None, model_path = None, fuse_modules = None, device = 'cpu', qconfig = 'fbgemm', model_fp32 = None):
    """Get static quantization Siren model.
    Params:
    -------
    `metadata_model_dict` - dict python object, metadata or hyper-params abot model's to be built.\n
    `model_path` - str python object, either None when we do not desire to load weights, or a file path to model's weights.\n
    `fuse_modules` - collection of modules to fuse if input model support fuse method.\n
    `device` - str object, kind of device upon which model's weigths and computation will be done.\n
    `qconfig` - str object, kind of device backedn for QuantizedCPU either FBGEMM for x86 server, or QNNAM for mobile devices.\n
    `model_fp32` - PyTorch model to be quantized if diverse from None, otherwise a new model will be created from scratch and then assigning to it weigths gotten from train phase.\n
    Return:
    ------
    `model_fp32_prepared` - PyTroch liek object representing DNN model quantized by means of Post Train Static quantizing technique.\n
    """

    if model_fp32 == None:
        if metadata_model_dict == None: raise Exception(f"metadata_model_dict is None!")
        model_fp32 = SirenQuantized(
            in_features=2,
            out_features=1,
            hidden_features=int(metadata_model_dict['hidden_features']),
            hidden_layers=int(metadata_model_dict['hidden_layers']),
            # outermost_linear=True).to(device=device)
            outermost_linear=True)
        pass
    model_fp32 = check_device_and_weigths_to_laod(model_fp32 = model_fp32, device = f'{device}', model_path = model_path)
    
    if fuse_modules != None:
        model_fp32.qconfig = torch.quantization.get_default_qconfig(f'{qconfig}')
        model_fp32_fused = torch.quantization.fuse_modules(model_fp32, fuse_modules)
        model_fp32_prepared = torch.quantization.prepare(model_fp32_fused)
    else:
        model_fp32.qconfig = torch.quantization.get_default_qconfig(f'{qconfig}')
        model_fp32_prepared = torch.quantization.prepare(model_fp32)
        pass

    if model_fp32_prepared == None: Exception(f"model_fp32_prepared is None, when quantization is Static!")
    return model_fp32_prepared


def get_post_training_quantization_model(metadata_model_dict, model_path = None, fuse_modules = None, device = 'cpu', qconfig = 'fbgemm', model_fp32 = None):
    """Get post-train quantization Siren model.
    Params:
    -------
    `metadata_model_dict` - dict python object, metadata or hyper-params abot model's to be built.\n
    `model_path` - str python object, either None when we do not desire to load weights, or a file path to model's weights.\n
    `fuse_modules` - collection of modules to fuse if input model support fuse method.\n
    `device` - str object, kind of device upon which model's weigths and computation will be done.\n
    `qconfig` - str object, kind of device backedn for QuantizedCPU either FBGEMM for x86 server, or QNNAM for mobile devices.\n
    `model_fp32` - PyTorch model to be quantized if diverse from None, otherwise a new model will be created from scratch and then assigning to it weigths gotten from train phase.\n
    Return:
    ------
    `model_fp32_prepared` - PyTroch liek object representing DNN model quantized by means of Post Train quantizing technique.\n
    """

    # Create model if not exist.
    if model_fp32 == None:
        model_fp32 = SirenQPT(
            in_features=2,
            out_features=1,
            hidden_features=int(metadata_model_dict['hidden_features']),
            hidden_layers=int(metadata_model_dict['hidden_layers']),
            quantize=True,
            outermost_linear=True) # outermost_linear=True).to(device=device)
        pass
    # Set model to cpu device.
    if device != 'cpu':
        raise Exception("Post Train quantization do not support CUDA/GPU backend for computations!")
    model_fp32 = check_device_and_weigths_to_laod(model_fp32 = model_fp32, device = f'{device}', model_path = model_path)
    
    # Set backend for Quantization computations.
    model_fp32.qconfig = torch.quantization.get_default_qconfig(f'{qconfig}')

    # Check if modules should be fused.
    if fuse_modules != None:
        # model_fp32_fused = torch.quantization.fuse_modules(model_fp32, fuse_modules)
        # torch.quantization.prepare(model_fp32_fused, inplace=True)
        model_fp32.fuse_model()
        model_fp32_prepared = torch.quantization.prepare(model_fp32)
    else:
        # torch.quantization.prepare(model_fp32, inplace=True)
        model_fp32.fuse_model()
        model_fp32_prepared = torch.quantization.prepare(model_fp32)
        pass

    if model_fp32_prepared == None: Exception(f"model_fp32_prepared is None, when quantization is Post Training!")
    return model_fp32_prepared


def get_quantization_aware_training(metadata_model_dict, model_path = None, fuse_modules = None, device = 'cpu', qconfig = 'fbgemm', model_fp32 = None):
    """Get quantization aware Siren model.
    Params:
    -------
    `metadata_model_dict` - dict python object, metadata or hyper-params abot model's to be built.\n
    `model_path` - str python object, either None when we do not desire to load weights, or a file path to model's weights.\n
    `fuse_modules` - collection of modules to fuse if input model support fuse method.\n
    `device` - str object, kind of device upon which model's weigths and computation will be done.\n
    `qconfig` - str object, kind of device backedn for QuantizedCPU either FBGEMM for x86 server, or QNNAM for mobile devices.\n
    `model_fp32` - PyTorch model to be quantized if diverse from None, otherwise a new model will be created from scratch and then assigning to it weigths gotten from train phase.\n
    Return:
    ------
    `model_fp32_prepared` - PyTroch liek object representing DNN model quantized by means of Post Train quantizing technique.\n
    """

    if device == 'cpu':
        model_fp32 = SirenQuantized(
            in_features=2,
            out_features=1,
            hidden_features=int(metadata_model_dict['hidden_features']),
            hidden_layers=int(metadata_model_dict['hidden_layers']),
            outermost_linear=True)
        if model_path != None:
            state_dict = torch.load(model_path)
            # model.load_state_dict(state_dict).to('cpu')
            model_fp32.load_state_dict(state_dict)
            pass
        model_fp32.to(device=device)
    else:
        model_fp32 = SirenQuantized(
            in_features=2,
            out_features=1,
            hidden_features=int(metadata_model_dict['hidden_features']),
            hidden_layers=int(metadata_model_dict['hidden_layers']),
            outermost_linear=True)
        if model_path != None:
            state_dict = torch.load(model_path)
            # model.load_state_dict(state_dict).to('cpu')
            model_fp32.load_state_dict(state_dict)
            pass
        model_fp32.load_state_dict(state_dict).cuda()
        pass
    if fuse_modules != None:
        model_fp32_fused = torch.quantization.fuse_modules(model_fp32, fuse_modules)
        model_fp32_prepared = torch.quantization.prepare_qat(model_fp32_fused)
        return model_fp32_prepared
    
    model_fp32_prepared = torch.quantization.prepare_qat(model_fp32)
    return model_fp32_prepared


def get_paszke_quant_model(metadata_model_dict, model_path = None, fuse_modules = None, device = 'cpu', qconfig = 'fbgemm', model_fp32 = None):
    """Build model for Paszek quant evaluation.
    Params:
    -------
    `metadata_model_dict` - dict python object, metadata or hyper-params abot model's to be built.\n
    `model_path` - str python object, either None when we do not desire to load weights, or a file path to model's weights.\n
    `fuse_modules` - collection of modules to fuse if input model support fuse method.\n
    `device` - str object, kind of device upon which model's weigths and computation will be done.\n
    `qconfig` - str object, kind of device backedn for QuantizedCPU either FBGEMM for x86 server, or QNNAM for mobile devices.\n
    `model_fp32` - PyTorch model to be quantized if diverse from None, otherwise a new model will be created from scratch and then assigning to it weigths gotten from train phase.\n
    Return:
    ------
    `model_fp32_prepared` - PyTroch liek object representing DNN model quantized by means of Post Train quantizing technique.\n
    """
    
    if device == 'cpu':
        model_fp32 = Siren(
            in_features=2,
            out_features=1,
            hidden_features=int(metadata_model_dict['hidden_features']),
            hidden_layers=int(metadata_model_dict['hidden_layers']),
            outermost_linear=True)
        model_fp32.to(device='cpu')
        if model_path != None:
            state_dict = torch.load(model_path)
            # model.load_state_dict(state_dict).to('cpu')
            model_fp32.load_state_dict(state_dict)
            pass
    else:
        model_fp32 = Siren(
            in_features=2,
            out_features=1,
            hidden_features=int(metadata_model_dict['hidden_features']),
            hidden_layers=int(metadata_model_dict['hidden_layers']),
            outermost_linear=True)
        model_fp32.load_state_dict(state_dict).cuda()
        if model_path != None:
            state_dict = torch.load(model_path)
            # model.load_state_dict(state_dict).to('cpu')
            model_fp32.load_state_dict(state_dict)
            pass
        pass
    if fuse_modules != None:
        pass
    model_fp32 = quantize_model(model = model_fp32, frequency = metadata_model_dict['frequency'])
    return model_fp32


# --------------------------------------------- #
# Compute Quantization: CHOICES
# --------------------------------------------- #

def compute_quantization_custom_quant_mode(model_path, arch_hyperparams, img_dataset, opt, fuse_modules = None, device = 'cpu', qconfig = 'fbgemm', model_fp32 = None):
    """Compute post train scores by means of Custom Quant Mode.
    Params:
    -------
    `model_path` - str python object, either None when we do not desire to load weights, or a file path to model's weights.\n
    `arch_hyperparams` - dict python object, metadata or hyper-params abot model's to be built.\n
    `image_dataset` - PyTorch Dataset object.\n
    `opt` - Namespace python object.\n
    `fuse_modules` - collection of modules to fuse if input model support fuse method.\n
    `device` - str object, kind of device upon which model's weigths and computation will be done.\n
    `qconfig` - str object, kind of device backedn for QuantizedCPU either FBGEMM for x86 server, or QNNAM for mobile devices.\n
    `model_fp32` - PyTorch model to be quantized if diverse from None, otherwise a new model will be created from scratch and then assigning to it weigths gotten from train phase.\n
    Return:
    ------
    `eval_scores` np.ndarray object containing mse, psnr, and ssim scores
    """
    eval_scores = None
    input_fp32 = _prepare_data_loaders(img_dataset, opt)

    model = get_custom_quant_model(
        metadata_model_dict = arch_hyperparams,
        model_path = model_path,
        # fuse_modules = fuse_modules,
        device = f'{device}',
        qconfig = f'{qconfig}',
        model_fp32 = model_fp32)
    
    model.gather_stats(input_fp32)
    size_model = get_size_of_model(model)
    
    eval_scores, eta_eval = _evaluate_model(model = model, evaluate_dataloader = input_fp32, loss_fn = nn.MSELoss(), device = f'{device}')
    return eval_scores, eta_eval, size_model 


def compute_quantization_paszke_quant_mode(model_path, arch_hyperparams, img_dataset, opt, fuse_modules = None, device = 'cpu', qconfig = 'fbgemm', model_fp32 = None):
    """Compute post train scores by means of Paszek Quant Mode.
    Params:
    -------
    `model_path` - str python object, either None when we do not desire to load weights, or a file path to model's weights.\n
    `arch_hyperparams` - dict python object, metadata or hyper-params abot model's to be built.\n
    `image_dataset` - PyTorch Dataset object.\n
    `opt` - Namespace python object.\n
    `fuse_modules` - collection of modules to fuse if input model support fuse method.\n
    `device` - str object, kind of device upon which model's weigths and computation will be done.\n
    `qconfig` - str object, kind of device backedn for QuantizedCPU either FBGEMM for x86 server, or QNNAM for mobile devices.\n
    `model_fp32` - PyTorch model to be quantized if diverse from None, otherwise a new model will be created from scratch and then assigning to it weigths gotten from train phase.\n
    Return:
    ------
    `eval_scores` np.ndarray object containing mse, psnr, and ssim scores
    """
    eval_scores = None
    input_fp32 = _prepare_data_loaders(img_dataset, opt)

    model_int8 = get_paszke_quant_model(
        metadata_model_dict = arch_hyperparams,
        model_path = model_path,
        fuse_modules = fuse_modules,
        device = f'{device}',
        qconfig = f'{qconfig}',
        model_fp32 = model_fp32)

    size_model = get_size_of_model(model_int8)
    
    eval_scores, eta_eval = _evaluate_model(model = model_int8, evaluate_dataloader = input_fp32, loss_fn = nn.MSELoss(), device = f'{device}')
    return eval_scores, eta_eval, size_model 


def compute_quantization_dyanmic_mode(model_path, arch_hyperparams, img_dataset, opt, fuse_modules = None, device = 'cpu', qconfig = 'fbgemm', model_fp32 = None):
    """Evaluate PyTorch model already trained by means of dynamic quantization.
    Params:
    -------
    `model_path` - str python object, either None when we do not desire to load weights, or a file path to model's weights.\n
    `arch_hyperparams` - dict python object, metadata or hyper-params abot model's to be built.\n
    `image_dataset` - PyTorch Dataset object.\n
    `opt` - Namespace python object.\n
    `fuse_modules` - collection of modules to fuse if input model support fuse method.\n
    `device` - str object, kind of device upon which model's weigths and computation will be done.\n
    `qconfig` - str object, kind of device backedn for QuantizedCPU either FBGEMM for x86 server, or QNNAM for mobile devices.\n
    `model_fp32` - PyTorch model to be quantized if diverse from None, otherwise a new model will be created from scratch and then assigning to it weigths gotten from train phase.\n
    Return:
    ------
    `eval_scores` np.ndarray object containing mse, psnr, and ssim scores
    """
    
    input_fp32 = _prepare_data_loaders(img_dataset, opt)

    model_int8 = get_dynamic_quantization_model(
        model_path = model_path,
        metadata_model_dict = arch_hyperparams, set_layers = {torch.nn.Linear}, device = 'cpu', qconfig = 'fbgemm', model_fp32 = model_fp32)

    size_model = get_size_of_model(model_int8)

    eval_scores, eta_eval = _evaluate_model(model = model_int8, evaluate_dataloader = input_fp32, loss_fn = nn.MSELoss(), device = 'cpu')
    return eval_scores, eta_eval, size_model 


def compute_quantization_static_mode(model_path, arch_hyperparams, img_dataset, opt, fuse_modules = None, device = 'cpu', qconfig = 'fbgemm', model_fp32 = None):
    """Evaluate PyTorch model already trained by means of static quantization.
    Params:
    -------
    `model_path` - str python object, either None when we do not desire to load weights, or a file path to model's weights.\n
    `arch_hyperparams` - dict python object, metadata or hyper-params abot model's to be built.\n
    `image_dataset` - PyTorch Dataset object.\n
    `opt` - Namespace python object.\n
    `fuse_modules` - collection of modules to fuse if input model support fuse method.\n
    `device` - str object, kind of device upon which model's weigths and computation will be done.\n
    `qconfig` - str object, kind of device backedn for QuantizedCPU either FBGEMM for x86 server, or QNNAM for mobile devices.\n
    `model_fp32` - PyTorch model to be quantized if diverse from None, otherwise a new model will be created from scratch and then assigning to it weigths gotten from train phase.\n
    Return:
    ------
    `eval_scores` -  np.ndarray object containing mse, psnr, and ssim scores
    """
    input_fp32 = _prepare_data_loaders(img_dataset, opt)
    model_fp32_prepared = get_static_quantization_model(
        model_path = model_path, metadata_model_dict = arch_hyperparams,
        fuse_modules = fuse_modules, device = f"{device}", qconfig = f"{qconfig}", model_fp32 = model_fp32)
    # Calibrate model
    _ = _evaluate_model(model = model_fp32_prepared, evaluate_dataloader = input_fp32, loss_fn = nn.MSELoss(), device = 'cpu')

    # Evaluate quantized int8 model
    model_int8 = torch.quantization.convert(model_fp32_prepared)
    size_model = get_size_of_model(model_int8)

    input_fp32 = _prepare_data_loaders(img_dataset, opt)
    eval_scores, eta_eval = _evaluate_model(model = model_int8, evaluate_dataloader = input_fp32, loss_fn = nn.MSELoss(), device = 'cpu')
    return eval_scores, eta_eval, size_model 


def compute_quantization_post_train_mode(model_path, arch_hyperparams, img_dataset, opt, fuse_modules = None, device = 'cpu', qconfig = 'fbgemm', model_fp32 = None):
    """Evaluate PyTorch model already trained by means of post train quantization.
    Params:
    -------
    `model_path` - str python object, either None when we do not desire to load weights, or a file path to model's weights.\n
    `arch_hyperparams` - dict python object, metadata or hyper-params abot model's to be built.\n
    `image_dataset` - PyTorch Dataset object.\n
    `opt` - Namespace python object.\n
    `fuse_modules` - collection of modules to fuse if input model support fuse method.\n
    `device` - str object, kind of device upon which model's weigths and computation will be done.\n
    `qconfig` - str object, kind of device backedn for QuantizedCPU either FBGEMM for x86 server, or QNNAM for mobile devices.\n
    `model_fp32` - PyTorch model to be quantized if diverse from None, otherwise a new model will be created from scratch and then assigning to it weigths gotten from train phase.\n
    Return:
    ------
    `eval_scores` -  np.ndarray object containing mse, psnr, and ssim scores
    """
    
    input_fp32 = _prepare_data_loaders(img_dataset, opt)
    model_fp32_prepared = get_post_training_quantization_model(
        model_path = model_path, metadata_model_dict = arch_hyperparams,
        fuse_modules = fuse_modules, device = f"{device}", qconfig = f"{qconfig}", model_fp32 = model_fp32)
    
    # Calibrate model
    _ = _evaluate_model(model = model_fp32_prepared, evaluate_dataloader = input_fp32, loss_fn = nn.MSELoss(), device = 'cpu')

    # Evaluate quantized int8 model
    model_int8 = torch.quantization.convert(model_fp32_prepared)
    size_model = get_size_of_model(model_int8)

    input_fp32 = _prepare_data_loaders(img_dataset, opt)
    eval_scores, eta_eval = _evaluate_model(model = model_int8, evaluate_dataloader = input_fp32, loss_fn = nn.MSELoss(), device = 'cpu')
    return eval_scores, eta_eval, size_model 


def compute_quantization_aware_train_mode(model_path, arch_hyperparams, img_dataset, opt, fuse_modules = None, device = 'cpu', qconfig = 'fbgemm', model_fp32 = None):
    """Evaluate PyTorch model already trained by means of quantization aware train.
    Params:
    -------
    `model_path` - str python object, either None when we do not desire to load weights, or a file path to model's weights.\n
    `arch_hyperparams` - dict python object, metadata or hyper-params abot model's to be built.\n
    `image_dataset` - PyTorch Dataset object.\n
    `opt` - Namespace python object.\n
    `fuse_modules` - collection of modules to fuse if input model support fuse method.\n
    `device` - str object, kind of device upon which model's weigths and computation will be done.\n
    `qconfig` - str object, kind of device backedn for QuantizedCPU either FBGEMM for x86 server, or QNNAM for mobile devices.\n
    `model_fp32` - PyTorch model to be quantized if diverse from None, otherwise a new model will be created from scratch and then assigning to it weigths gotten from train phase.\n
    Return:
    ------
    `eval_scores` -  np.ndarray object containing mse, psnr, and ssim scores
    """
    
    input_fp32 = _prepare_data_loaders(img_dataset, opt)
    model = get_quantization_aware_training(model_path = model_path, metadata_model_dict = arch_hyperparams, fuse_modules = fuse_modules, device = device, qconfig = qconfig, model_fp32 = model)
    eval_scores, eta_eval = _evaluate_model(model = model, evaluate_dataloader = input_fp32, loss_fn = nn.MSELoss(), device = 'cpu')

    model_int8 = torch.quantization.convert(model)
    size_model = get_size_of_model(model_int8)

    input_fp32 = _prepare_data_loaders(img_dataset, opt)
    
    eval_scores, eta_eval = _evaluate_model(model = model_int8, evaluate_dataloader = input_fp32, loss_fn = nn.MSELoss(), device = 'cpu')
    return eval_scores, eta_eval, size_model 


# --------------------------------------------- #
# Compute Quantization: SWITCH
# --------------------------------------------- #
def compute_quantization(img_dataset, opt, model_path = None, arch_hyperparams = None, fuse_modules = None, device = 'cpu', qconfig = 'fbgemm'):
    """Compute quantized results.
    Params:
    -------
    `image_dataset` - PyTorch Dataset object.\n
    `opt` - Namespace python object.\n
    `model_path` - str python object, either None when we do not desire to load weights, or a file path to model's weights.\n
    `arch_hyperparams` - dict python object, metadata or hyper-params abot model's to be built.\n
    `fuse_modules` - collection of modules to fuse if input model support fuse method.\n
    `device` - str object, kind of device upon which model's weigths and computation will be done.\n
    `qconfig` - str object, kind of device backedn for QuantizedCPU either FBGEMM for x86 server, or QNNAM for mobile devices.\n
    Return:
    -------
    `eval_scores` - scores or metrices retrived, that are: MSE, PSNR and SSIM.\n
    `eta_eval` - elapsed time for evaluating input data.\n
    `size_model` - model's size in byte.\n
    """

    eval_scores = None
    if opt.quantization_enabled != None:
        # --- Dynamic Quantization: TODO test it.
        if model_path == None: raise Exception('Model path is None!')
        if opt.quantization_enabled == 'dynamic':
            eval_scores, eta_eval, size_model = compute_quantization_dyanmic_mode(
                model_path,
                arch_hyperparams,
                img_dataset,
                opt, fuse_modules = None, device = 'cpu', qconfig = 'fbgemm', model_fp32 = None)
            pass
        
        # --- Static Quantization: TODO test it.
        elif opt.quantization_enabled == 'static':
            """eval_scores, eta_eval, size_model = compute_quantization_static_mode(
                model_path,
                arch_hyperparams,
                img_dataset,
                opt,
                fuse_modules = None, device = 'cpu', qconfig = 'fbgemm', model_fp32 = None)"""
            pass

        # --- Post Train Quantization: TODO test it.
        elif opt.quantization_enabled == 'post_train':
            eval_scores, eta_eval, size_model = compute_quantization_post_train_mode(
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
            eval_scores, eta_eval, size_model = compute_quantization_paszke_quant_mode(
                model_path,
                arch_hyperparams,
                img_dataset,
                opt,
                fuse_modules = None, device = 'cpu', qconfig = f"{qconfig}", model_fp32 = None)
            pass
        elif opt.quantization_enabled =='custom_quant':
            eval_scores, eta_eval, size_model = compute_quantization_custom_quant_mode(
                model_path,
                arch_hyperparams,
                img_dataset,
                opt,
                fuse_modules = None, device = 'cpu', qconfig = f"{qconfig}", model_fp32 = None)
            pass
        else:
            raise Exception(f"Error: {opt.quantization_enabled} not allowed!")
    return eval_scores, eta_eval, size_model 


# --------------------------------------------- #
# Utils Functions
# --------------------------------------------- #
def check_device_and_weigths_to_laod(model_fp32, device = 'cpu', model_path = None):
    """Check to which device load a PyTorch model and fi it is necessary to also laod weights for that model.
    Params
    ------
    `model_fp32` - PyTorch model to be fetched to a proper device .\n
    `device` - str object, kind of device upon which model's weigths and computation will be done. Allowed CPU,GPU,CUDA.\n
    `model_path` - str python object, either None when we do not desire to load weights, or a file path to model's weights.\n
    Return
    ------
    `model_fp32` - PyTorch model loaded to a proper device .\n
    """
    if device == 'cpu':
        # print('Load Model to cpu device!')
        model_fp32 = model_fp32.to('cpu')
        if model_path != None:
            # print('Load Model weigths!')
            state_dict = torch.load(model_path, map_location=torch.device('cpu'))
            model_fp32.load_state_dict(state_dict)
            pass
        pass
    else:
        try:
            model_fp32 = model_fp32.cuda()
            # print('Load Model to cuda device!')
            if model_path != None:
                # print('Load Model weigths!')
                state_dict = torch.load(model_path, map_location=torch.device('cuda'))
                model_fp32.load_state_dict(state_dict)
                pass
            pass
        except:
            model_fp32 = model_fp32.to('cpu')
            if model_path != None:
                # print('Load Model weigths!')
                state_dict = torch.load(model_path, map_location=torch.device('cpu'))
                model_fp32.load_state_dict(state_dict)
                pass
            pass
        pass
    return model_fp32


def quantize_model(model, frequency):
    """Quantize pytorch model, after training, that is made from sine functions as activations.
    The code provided for function implementation has been picked from works done
    by PyTorch Paszke et al. (2017:
     - A. Paszke, S. Gross, S. Chintala, G. Chanan, E. Yang, Z. DeVito, Z. Lin, A. Desmaison, L. Antiga, and A. Lerer. Automatic differentiation in PyTorch. Proc. Neural Information Processing Systems, 2017.
       https://openreview.net/forum?id=BJJsrmfCZ
    Other works that cited the code snippet exploited here for implementing such a function:
     - On Periodic Functions as Regularizers for Quantization of Neural Networks
       https://www.researchgate.net/publication/329207332_On_Periodic_Functions_as_Regularizers_for_Quantization_of_Neural_Networks
    Params:
    -------
    `model` Pytorch model with sine or trigonometric functions used as activation functions.\n
    `frequency` number that is used for tuning quantization.\n

    Return:
    -------
    `model` updated for quantization.
    """
    def quantize_weights(m):
        if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
            c = m.weight.abs().max().data
            m.weight.data.mul_(frequency/c)
            m.weight.data.round_()
            m.weight.data.mul_(c/frequency)
    model.apply(quantize_weights)
    return model


def prepare_model(opt, arch_hyperparams = None, device = 'cpu'):
    """Prepare Siren model, either non-quantized or dynamic/static/posteriorn quantized model.
    Params
    ------
    `opt` - Namespace python like object with attributes necessary to run the evaluation tasks required.\n
    `arch_hyperparams` - dictionary python object.\n
    `device` - str object, kind of device upon which model will be loaded, allowed only CPU, since PyTorch framework supports just that setup, by now.\n
    Return
    ------
    `model` - PyTorch like model instance.\n
    """
    model = None
    if opt.quantization_enabled != None:
        if opt.quantization_enabled in 'dynamic,paszke_quant'.split(","):
            model = Siren(
                in_features=2,
                out_features=1,
                hidden_features=int(arch_hyperparams['hidden_features']),
                hidden_layers=int(arch_hyperparams['hidden_layers']),
                # outermost_linear=True).to(device=device)
                outermost_linear=True).cuda()
            # model = get_dynamic_quantization_model(metadata_model_dict = arch_hyperparams, set_layers = {torch.nn.Linear}, device = 'cpu', qconfig = 'fbgemm', model_fp32 = model)
        elif opt.quantization_enabled == 'static':
            raise Exception(f"{opt.quantization_enabled} - option not supported yet.")
            """
            model = Siren(
                in_features=2,
                out_features=1,
                hidden_features=int(arch_hyperparams['hidden_features']),
                hidden_layers=int(arch_hyperparams['hidden_layers']),
                # outermost_linear=True).to(device=device)
                outermost_linear=True)"""
            # model = get_static_quantization_model(metadata_model_dict = arch_hyperparams, fuse_modules = None, device = 'cpu', qconfig = 'fbgemm', model_fp32 = model)
            pass
        elif opt.quantization_enabled == 'post_train':
            model = SirenQPT(
                in_features=2,
                out_features=1,
                hidden_features=int(arch_hyperparams['hidden_features']),
                hidden_layers=int(arch_hyperparams['hidden_layers']),
                # outermost_linear=True).to(device=device)
                outermost_linear=True,
                quantize=False
            ).cuda()
            pass
        else:
            raise Exception(f"Error: {opt.quantization_enabled} not allowed!")
    else:
        model = Siren(
            in_features=2,
            out_features=1,
            hidden_features=int(arch_hyperparams['hidden_features']),
            hidden_layers=int(arch_hyperparams['hidden_layers']),
            # outermost_linear=True).to(device=device)
            outermost_linear=True).cuda()
        pass
    return model


def _prepare_data_loaders(img_dataset, opt):
    """Prepare data loader from which fetching data in order to feed models.
    Params
    ------
    `img_dataset` - PyTorch's DataSet like object representing the data against which evaluate models(base model and quantized models, if any).\n
    `opt` - Namespace python like object with attributes necessary to run the evaluation tasks required.\n
    Return
    ------
    `a_dataloader` - PyTorch DataLoader instance.\n
    """

    coord_dataset = dataio.Implicit2DWrapper(
                img_dataset, sidelength=opt.sidelength, compute_diff=None)
    a_dataloader = DataLoader(
                coord_dataset,
                shuffle=False,
                batch_size=1,
                pin_memory=True, num_workers=0)
    return a_dataloader


def _evaluate_model(model, evaluate_dataloader, loss_fn = nn.MSELoss(), device = 'cpu'):
    """ Evalaute model."""

    # --- Evaluate model's on validation data.
    eval_scores = None
    model.eval()
    with torch.no_grad():
        # -- Get data from validation loader.
        val_input, val_gt = next(iter(evaluate_dataloader))

        if device == 'cpu':
            val_input = val_input['coords'].to('cpu')
            val_gt = val_gt['img'].to('cpu')
        else:
            val_input = val_input['coords'].cuda() # .to(device)
            val_gt = val_gt['img'].cuda() # .to(device)
            pass

        # --- Compute estimation.
        start_time = time.time()
        val_output, _ = model(val_input)
        eta_eval = time.time() - start_time

        # --- Prepare data for calculating metrices scores.
        # sidelenght = int(math.sqrt(val_output.size()[1]))
        sidelenght = val_output.size()[1]

        arr_gt = val_gt.cpu().view(sidelenght).detach().numpy()
        arr_gt = (arr_gt / 2.) + 0.5                

        arr_output = val_output.cpu().view(sidelenght).detach().numpy()
        arr_output = (arr_output / 2.) + 0.5
        arr_output = np.clip(arr_output, a_min=0., a_max=1.)
        
        # --- Calculate metrices scores.
        # Metric: MSE
        train_loss = loss_fn(val_output, val_gt)

        # Other Metrics: PSNR, SSIM
        val_psnr = psnr(arr_gt, arr_output, data_range=1.)
        val_mssim = ssim(arr_gt, arr_output, data_range=1.)
        
        # --- Record results.
        eval_scores = np.array([train_loss.item(), val_psnr, val_mssim])
        pass
    return eval_scores, eta_eval


def print_size_of_model(model):
    """Display model size as file size corresponding to model's state dictionary when saved temporarily to 
    disk.
    Params
    ------
    `model` - PyTorch like model.\n
    """
    torch.save(model.state_dict(), "temp.p")
    print('Size (MB):', os.path.getsize("temp.p")/1e6)
    os.remove('temp.p')
    pass


def get_size_of_model(model):
    """Return model size as file size corresponding to model's state dictionary when saved temporarily to 
    disk.
    Params
    ------
    `model` - PyTorch like model.\n
    Return
    ------
    `model_size` - int python object, size of state dictionary expressed in byte.\n
    """
    torch.save(model.state_dict(), "temp.p")
    # print('Size (MB):', os.path.getsize("temp.p")/1e6)
    model_size = os.path.getsize("temp.p")
    os.remove('temp.p')
    return model_size
