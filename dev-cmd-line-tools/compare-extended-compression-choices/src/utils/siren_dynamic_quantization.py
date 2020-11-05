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
from src.utils.siren import Siren

# --------------------------------------------- #
# Functions
# --------------------------------------------- #

def get_dynamic_quantization_model(metadata_model_dict = None, set_layers = {torch.nn.Linear}, device = 'cpu', qconfig = 'fbgemm', model_fp32 = None):
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


def get_static_quantization_model(metadata_model_dict = None, fuse_modules = None, device = 'cpu', qconfig = 'fbgemm', model_fp32 = None):
    """Get static quantization Siren model."""

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


def get_post_training_quantization_model(model_path, metadata_model_dict, fuse_modules = None, device = 'cpu', qconfig = 'fbgemm'):
    """Get posterior quantization Siren model."""

    model_fp32 = Siren(
        in_features=2,
        out_features=1,
        hidden_features=int(metadata_model_dict['hidden_features']),
        hidden_layers=int(metadata_model_dict['hidden_layers']),
        # outermost_linear=True).to(device=device)
        outermost_linear=True)
    if device == 'cpu':
        model_fp32 = model_fp32.to('cpu')
    else:
        raise Exception("Posterior quantization do not support CUDA/GPU backend for computations!")
    
    model_fp32.qconfig = torch.quantization.get_default_qconfig(f'{qconfig}')

    if fuse_modules != None:
        model_fp32_fused = torch.quantization.fuse_modules(model_fp32, fuse_modules)
        torch.quantization.prepare(model_fp32_fused, inplace=True)

        model_fp32_prepared = torch.quantization.prepare(model_fp32_fused)
    else:
        torch.quantization.prepare(model_fp32, inplace=True)

        model_fp32_prepared = torch.quantization.prepare(model_fp32)
        pass

    if model_fp32_prepared == None: Exception(f"model_fp32_prepared is None, when quantization is Post Training!")
    return model_fp32_prepared


def get_quantization_aware_training(model_path, metadata_model_dict, fuse_modules = None, device = 'cpu', qconfig = 'fbgemm'):
    """Get quantization aware Siren model. """
    raise Exception('Not yet implemented')
