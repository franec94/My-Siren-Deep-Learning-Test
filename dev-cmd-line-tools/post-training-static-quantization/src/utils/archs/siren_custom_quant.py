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

import torch
import torchvision

import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset

from torchvision import datasets
from torchvision import transforms
from torchvision.transforms import Resize, Compose, ToTensor, Normalize


from collections import OrderedDict
import matplotlib
import matplotlib.pyplot as plt
import numpy as np

from src.utils.quant_utils.quant_utils_functions_2 import gather_stats as gather_stats_inner
from src.utils.quant_utils.quant_utils_functions_2 import quantize_tensor, quantize_layer, QTensor, dequantize_tensor

class SineLayerCQ(nn.Module):
    # See paper sec. 3.2, final paragraph, and supplement Sec. 1.5 for discussion of omega_0.
    
    # If is_first=True, omega_0 is a frequency factor which simply multiplies the activations before the 
    # nonlinearity. Different signals may require different omega_0 in the first layer - this is a 
    # hyperparameter.
    
    # If is_first=False, then the weights will be divided by omega_0 so as to keep the magnitude of 
    # activations constant, but boost gradients to the weight matrix (see supplement Sec. 1.5)
    
    def __init__(self, in_features, out_features, bias=True,
                 is_first=False, omega_0=30, prev_layer_name = None, succ_layer_name = None, quant = False):
        super().__init__()
        self.omega_0 = omega_0
        self.quant = quant
        self.prev_layer_name = prev_layer_name
        self.succ_layer_name = succ_layer_name
        self.is_first = is_first
        
        self.in_features = in_features
        self.linear = nn.Linear(in_features, out_features, bias=bias)
        
        self.init_weights()
        pass
    
    def init_weights(self):
        with torch.no_grad():
            if self.is_first:
                self.linear.weight.uniform_(-1 / self.in_features, 
                                             1 / self.in_features)      
            else:
                self.linear.weight.uniform_(-np.sqrt(6 / self.in_features) / self.omega_0, 
                                             np.sqrt(6 / self.in_features) / self.omega_0)
        pass
        
    def forward(self, input):
        return torch.sin(self.omega_0 * self.linear(input))
    
    def forward_with_intermediate(self, input): 
        # For visualization of activation distributions
        intermediate = self.omega_0 * self.linear(input)
        return torch.sin(intermediate), intermediate
    pass
    
    
class SirenCQ(nn.Module):
    def __init__(self, in_features, hidden_features, hidden_layers, out_features, outermost_linear=False, 
                 first_omega_0=30, hidden_omega_0=30., quant = False, num_bits = 8):
        super().__init__()
        
        self.net = []
        self.quant = quant
        self.num_bits = num_bits
        self.stats = {}
        a_layer = SineLayerCQ(in_features, hidden_features, \
                                  quant=quant, \
                                  is_first=True, omega_0=first_omega_0)
        self.net.append(a_layer)
        
        for i in range(hidden_layers):
            curr_layer = SineLayerCQ(hidden_features, hidden_features, \
                                      quant=quant, \
                                      is_first=False, omega_0=hidden_omega_0, prev_layer_name=None)
            self.net.append(curr_layer)

        if outermost_linear:
            final_linear = nn.Linear(hidden_features, out_features)
            
            with torch.no_grad():
                final_linear.weight.uniform_(-np.sqrt(6 / hidden_features) / hidden_omega_0, 
                                              np.sqrt(6 / hidden_features) / hidden_omega_0)
                
            self.net.append(final_linear)
        else:
            self.net.append(SineLayerCQ(hidden_features, out_features, 
                                      is_first=False, omega_0=hidden_omega_0))
        
        prev_module, prev_name = None, None
        self.net = nn.Sequential(*self.net)
        for a_name, a_module in self.net.named_modules():
            print(a_name)
            if type(a_module) == SineLayerCQ:
                if a_module.is_first:
                    prev_name, prev_module = f"{a_name}", a_module
                else:
                    prev_module.succ_layer_name = f"{a_name}"
                    a_module.prev_layer_name = prev_name
                    prev_name, prev_module = f"{a_name}", a_module
                    pass
                pass
            pass
        pass
    

    def forward(self, coords):
        coords = coords.clone().detach().requires_grad_(True) # allows to take derivative w.r.t. input
        if self.quant == False:
            output = self.net(coords)
        else:
            output = self._forward_quantized(coords)
            pass
        return output, coords        
    

    def _forward_quantized(self, x):
        stats = self.stats
        for module_name, a_module in self.net.named_modules():
            if type(a_module) == SineLayerCQ:
                if a_module.is_first:
                    x = quantize_tensor(x,
                        min_val=stats[f'{module_name}']['min'],
                        max_val=stats[f'{module_name}']['max'],
                        num_bits = self.num_bits)
                    pass
                succ_layer_name = a_module.succ_layer_name
                x, scale_next, zero_point_next = \
                    quantize_layer(x.tensor,
                    a_module,
                    stats[f'{succ_layer_name}'], x.scale, x.zero_point)
                pass
            if type(a_module) == nn.Linear:
                x = dequantize_tensor(QTensor(tensor=x, scale=scale_next, zero_point=zero_point_next))
                x = a_module(x)
                pass
            pass
        return x


    def forward_with_activations(self, coords, retain_grad=False):
        '''Returns not only model output, but also intermediate activations.
        Only used for visualizing activations later!'''
        activations = OrderedDict()

        activation_count = 0
        x = coords.clone().detach().requires_grad_(True)
        activations['input'] = x
        for i, layer in enumerate(self.net):
            if isinstance(layer, SirenCQ):
                x, intermed = layer.forward_with_intermediate(x)
                
                if retain_grad:
                    x.retain_grad()
                    intermed.retain_grad()
                    
                activations['_'.join((str(layer.__class__), "%d" % activation_count))] = intermed
                activation_count += 1
            else: 
                x = layer(x)
                
                if retain_grad:
                    x.retain_grad()
                    
            activations['_'.join((str(layer.__class__), "%d" % activation_count))] = x
            activation_count += 1

        return activations


    def gather_stats(self, test_loader, device = 'cpu'):
        """Gather stats for quantizing computations, inferences.
        Params:
        -------
        :test_loader: DatasetLoader from Pytorch framework.\n
        Return:
        -------
        :stats_collected: python dictionary object,copied and retrieved with statistics collected.\n
        """
        stats_collected = gather_stats_inner(self, test_loader, device)
        self.stats = stats_collected
        return copy.deepcopy(stats_collected)
    pass
