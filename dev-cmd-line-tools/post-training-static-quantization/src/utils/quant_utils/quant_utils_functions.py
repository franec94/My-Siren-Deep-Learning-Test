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
from src.utils.archs.siren import Siren
from src.utils.archs.siren_quantized import SirenQuantized
from src.utils.archs.siren_quantized_post_training import SirenQPT
from src.utils.archs.siren_quantizatin_aware_train import SirenQAT

# ----------------------------------------------------------------------------------------------- #
# Functions
# ----------------------------------------------------------------------------------------------- #

# --------------------------------------------- #
# Getter functions
# --------------------------------------------- #
def get_dynamic_quantization_model(metadata_model_dict = None, model_path = None, set_layers = {torch.nn.Linear}, device = 'cpu', qconfig = 'fbgemm', model_fp32 = None):
    """Get dynamic quantization Siren model."""

    if model_fp32 == None:
        if metadata_model_dict == None: raise Exception(f"metadata_model_dict is None!")
        model_fp32 = Siren(
            in_features=2,
            out_features=1,
            hidden_features=int(metadata_model_dict['hidden_features']),
            hidden_layers=int(metadata_model_dict['hidden_layers']),
            # outermost_linear=True).to(device=device)
            outermost_linear=True)
        pass
    if model_path != None:
        state_dict = torch.load(model_path)
        model_fp32.load_state_dict(state_dict)
        pass
    if device == 'cpu':
        model_fp32 = model_fp32.to('cpu')
    else:
        model_fp32 = model_fp32.cuda()
        pass
    model_int8 = torch.quantization.quantize_dynamic(
        model_fp32,         # the original model
        set_layers,         # a set of layers to dynamically quantize
        dtype=torch.qint8)  # the target dtype for quantized weights
    if model_int8 == None: Exception(f"model_int8 is None, when quantization is Dynamic!")
    return model_int8


def get_static_quantization_model(metadata_model_dict = None, model_path = None, fuse_modules = None, device = 'cpu', qconfig = 'fbgemm', model_fp32 = None):
    """Get static quantization Siren model."""

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
    if model_path != None:
        state_dict = torch.load(model_path)
        model_fp32.load_state_dict(state_dict)
        pass
    if device == 'cpu':
        model_fp32 = model_fp32.to('cpu')
    else:
        model_fp32 = model_fp32.cuda()
        pass
    
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
    """Get post-train quantization Siren model."""

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
    if device == 'cpu':
        model_fp32 = model_fp32.to('cpu')
    else:
        raise Exception("Post Train quantization do not support CUDA/GPU backend for computations!")
    # Load weights if requested.
    if model_path != None:
        state_dict = torch.load(model_path)
        model_fp32.load_state_dict(state_dict)
        pass
    
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
    """Get quantization aware Siren model. """
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
    """Build model for Paszek quant evaluation."""
    
    if device == 'cpu':
        model_fp32 = Siren(
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
        model_fp32.to(device='cpu')
    else:
        model_fp32 = Siren(
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
        pass
    model_fp32 = quantize_model(model = model_fp32, frequency = metadata_model_dict['frequency'])
    return model_fp32

# --------------------------------------------- #
# Utils Functions
# --------------------------------------------- #

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
    :model: Pytorch model with sine or trigonometric functions used as activation functions.\n
    :frequency: number that is used for tuning quantization.\n

    Return:
    -------
    :model: updated for quantization.
    """
    def quantize_weights(m):
        if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
            c = m.weight.abs().max().data
            m.weight.data.mul_(frequency/c)
            m.weight.data.round_()
            m.weight.data.mul_(c/frequency)
    model.apply(quantize_weights)
    return model


def prepare_model(opt, arch_hyperparams = None, device = 'cpu', model_weights_file = None):
    """Prepare Siren model, either non-quantized or dynamic/static/posteriorn quantized model."""
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
        if device == 'cpu':
            if model_weights_file == None:
                model = Siren(
                    in_features=2,
                    out_features=1,
                    hidden_features=int(arch_hyperparams['hidden_features']),
                    hidden_layers=int(arch_hyperparams['hidden_layers']),
                    # outermost_linear=True).to(device=device)
                    outermost_linear=True).to('cpu')
                weigths_dict = torch.load(model_weights_file)
                model.load_state_dict(weigths_dict)
                model = model.to('cpu')
            else:
                model = Siren(
                    in_features=2,
                    out_features=1,
                    hidden_features=int(arch_hyperparams['hidden_features']),
                    hidden_layers=int(arch_hyperparams['hidden_layers']),
                    # outermost_linear=True).to(device=device)
                    outermost_linear=True).to('cpu')
        else:
            if model_weights_file != None:
                model = Siren(
                    in_features=2,
                    out_features=1,
                    hidden_features=int(arch_hyperparams['hidden_features']),
                    hidden_layers=int(arch_hyperparams['hidden_layers']),
                    # outermost_linear=True).to(device=device)
                    outermost_linear=True).cuda()
                weigths_dict = torch.load(model_weights_file)
                model.load_state_dict(weigths_dict)
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
    """Prepare data loader from which fetching data in order to feed models."""

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
        val_output, _ = model(val_input)

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
    return eval_scores