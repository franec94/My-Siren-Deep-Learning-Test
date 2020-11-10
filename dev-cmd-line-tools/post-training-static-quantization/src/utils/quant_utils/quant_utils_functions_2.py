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
import tabulate
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
# ----------------------------------------------------------------------------------------------- #
# Imports
# ----------------------------------------------------------------------------------------------- #
from collections import namedtuple
import torch
import torch.nn as nn
import torch.nn.functional as F

# from src.utils.archs.siren_custom_quant import SineLayerCQ

# ----------------------------------------------------------------------------------------------- #
# Objects
# ----------------------------------------------------------------------------------------------- #
QTensor = namedtuple('QTensor', ['tensor', 'scale', 'zero_point'])


# ----------------------------------------------------------------------------------------------- #
# Functions
# ----------------------------------------------------------------------------------------------- #
def calc_scale_zero_point(min_val, max_val, num_bits=8):
  """Calc Scale and zero point of next."""

  qmin = 0.
  qmax = 2.**num_bits - 1.

  scale = (max_val - min_val) / (qmax - qmin)

  initial_zero_point = qmin - min_val / scale
  
  zero_point = 0
  if initial_zero_point < qmin:
      zero_point = qmin
  elif initial_zero_point > qmax:
      zero_point = qmax
  else:
      zero_point = initial_zero_point

  zero_point = int(zero_point)

  return scale, zero_point


def quantize_tensor(x, num_bits=8, min_val=None, max_val=None):
    """Quantize Tensor."""
    
    if not min_val and not max_val: 
      min_val, max_val = x.min(), x.max()

    qmin = 0.
    qmax = 2.**num_bits - 1.

    scale, zero_point = calc_scale_zero_point(min_val, max_val, num_bits)
    q_x = zero_point + x / scale
    q_x.clamp_(qmin, qmax).round_()
    q_x = q_x.round().byte()
    
    return QTensor(tensor=q_x, scale=scale, zero_point=zero_point)


def dequantize_tensor(q_x):
    """Dequantize tensor."""

    return q_x.scale * (q_x.tensor.float() - q_x.zero_point)

# --------------------------------------------- #
# Functions
# --------------------------------------------- #

def quantize_layer(x, layer, stat, scale_x, zp_x):
    """
    Quantize Layer.
    """
    # for both conv and linear layers
    W = layer.weight.data
    B = layer.bias.data

    # scale_x = x.scale
    # zp_x = x.zero_point
    w = quantize_tensor(layer.weight.data) 
    b = quantize_tensor(layer.bias.data)

    layer.weight.data = w.tensor.float()
    layer.bias.data = b.tensor.float()

    scale_w = w.scale
    zp_w = w.zero_point
  
    scale_b = b.scale
    zp_b = b.zero_point
  
    scale_next, zero_point_next = calc_scale_zero_point(min_val=stat['min'], max_val=stat['max'])

    # Perparing input by shifting
    X = x.float() - zp_x
    layer.weight.data = (scale_x * scale_w/scale_next)*(layer.weight.data - zp_w)
    layer.bias.data = (scale_b/scale_next)*(layer.bias.data + zp_b)

    # All int

    x = layer(X) + zero_point_next
    
    # x = F.relu(x)
    # x = torch.sin(x)

    # Reset
    layer.weight.data = W
    layer.bias.data = B
  
    return x, scale_next, zero_point_next


# --------------------------------------------- #
# Functions
# --------------------------------------------- #

def update_stats(x, stats, key):
    """Update Statistics for a given layer with Neural Network model, providing input data to be processed.
    Params:
    -------
    :x: input data fetched from a Pytorch dataloader.\n
    :stats: python dictionary object, containing pairs (layer name,stats values).\n
    :key: python string object, name for getting and updating pait (layer name,stats values).\n

    Return:
    -------
    :stats: dictionary with updated values.
    """
    max_val, _ = torch.max(x, dim=1)
    min_val, _ = torch.min(x, dim=1)
  
  
    if key not in stats:
        stats[key] = {"max": max_val.sum(), "min": min_val.sum(), "total": 1}
    else:
        stats[key]['max'] += max_val.sum().item()
        stats[key]['min'] += min_val.sum().item()
        stats[key]['total'] += 1
  
    return stats

def gather_activation_stats(model, x, stats):
    """Gather Statistics for a given layer with Neural Network model, providing input data to be processed.
    Params:
    -------
    :model: PyTorch like neural network model.\n
    :x: input data fetched from a Pytorch dataloader.\n
    :stats: python dictionary object, that will contain pairs (layer name,stats values).\n
    
    Return:
    -------
    :stats: dictionary with updated values.
    """
    n = len(model.net.named_modules())
    for ii, (name_module, module_obj) in enumerate(model.net.named_modules()):
        if type(module_obj) == nn.Module or type(module_obj) == nn.Linear:
            print('Yes module:', name_module, type(module_obj))
            stats = update_stats(x.clone().view(x.shape[0], -1), stats, f'{name_module}')
            x = module_obj(x)
            if ii + 1 != n:
                x = torch.sin(x)
            pass
        else:
            print('No module:', name_module, type(module_obj))
        pass

    return stats


def gather_stats(model, test_loader, device = 'cpu'):
    """Gather Statistics for a given layer with Neural Network model, providing input data to be processed.
    Params:
    -------
    :model: PyTorch like neural network model.\n
    :test_loader: input data fetched from a Pytorch dataloader.\n
    :device: options allowed [cpu,gpu, or cuda].\n
    
    Return:
    -------
    :stats: dictionary with updated values.
    """
    
    model.eval()
    # test_loss = 0
    # correct = 0
    stats = {}
    with torch.no_grad():
        for data, target in test_loader:
            if device == 'cpu':
                data = data['coords'].to('cpu')
                target = target['img'].to('cpu')
            else:
                data = data['coords'].cuda() # .to(device)
                target = target['img'].cuda() # .to(device)
                pass
            stats = gather_activation_stats(model, data, stats)
    
    final_stats = {}
    for key, value in stats.items():
      final_stats[key] = { "max" : value["max"] / value["total"], "min" : value["min"] / value["total"] }
    return copy.deepcopy(final_stats)
