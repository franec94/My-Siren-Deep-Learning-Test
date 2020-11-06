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


def get_data_ready_for_model(model_input, gt, quantization_enabled = None, device = 'cpu'):
    """Setup data to be feeded into the model, as the latter will expect."""

    if device == 'cpu':
        model_input = model_input['coords'].to('cpu')
        gt = gt['img'].to('cpu')
        if quantization_enabled  != None:
            model_input = torch.quantize_per_tensor(model_input, 0.01, 0, torch.qint8)
            gt = torch.quantize_per_tensor(gt, 0.01, 0, torch.qint8)    
            pass
    else:
        if quantization_enabled  != None:
            model_input = torch.quantize_per_tensor(model_input, 0.01, 0, torch.qint8).cuda()
            gt = torch.quantize_per_tensor(gt, 0.01, 0, torch.qint8).cuda()
        else:
            model_input = model_input['coords'].cuda()
            gt = gt['img'].cuda()
            pass
        pass
    return model_input, gt

def compute_desired_metrices(model_output, gt):
    """Compute PSNR and SSIM scores."""

    sidelenght = model_output.size()[1]

    arr_gt = gt.cpu().view(sidelenght).detach().numpy()
    arr_gt = (arr_gt / 2.) + 0.5

    arr_output = model_output.cpu().view(sidelenght).detach().numpy()
    arr_output = (arr_output / 2.) + 0.5
    arr_output = np.clip(arr_output, a_min=0., a_max=1.)

    val_psnr = psnr(arr_gt, arr_output,data_range=1.)
    val_mssim = ssim(arr_gt, arr_output,data_range=1.)
    return val_psnr, val_mssim

def save_data_to_file(root_dir, model, train_scores):
    """Save recorded data, i.e. weights and train scores, during training into a file location."""
    try:
        tmp_file_path = os.path.join(root_dir, 'model_final.pth')
        torch.save(model.state_dict(),
                  tmp_file_path)
        
        tmp_file_path = os.path.join(root_dir, 'train_losses_final.txt')
        np.savetxt(tmp_file_path,
                   np.array(train_scores))
    except Exception as _:
                raise Exception(f"Error when saving file: filename={tmp_file_path} .")
    pass

def show_quantized_computed_scores():
    pass