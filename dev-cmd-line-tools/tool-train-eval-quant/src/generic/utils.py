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


def cond_mkdir(path):
    if not os.path.exists(path):
        os.makedirs(path)


def get_data_ready_for_model(model_input, gt, quantization_enabled = None, device = 'cpu'):
    """Setup data to be feeded into the model, as the latter will expect.
    Params:
    -------
    `model_input` - input to be processed by PyTorch model\n
    `gt` - reference data\n
    `quantization_enabled` - str object, quantization technique name, allowed values: [dynamic,static,post_train,quantization_aware_training]\n
    `device` - str object, allowed values: 'cpu', 'gpu', 'cuda'\n

    Return:
    -------
    `model_input, gt` - data ready to be feeded into PyTorch model
    """
    if device == 'cpu':
        model_input = model_input['coords'].to('cpu')
        gt = gt['img'].to('cpu')
        if quantization_enabled  != None:
            pass
    else:
        model_input = model_input['coords'].cuda()
        gt = gt['img'].cuda()
        if quantization_enabled  != None:
            pass
        pass
    return model_input, gt


def compute_desired_metrices(model_output, gt, data_range=1.):
    """Compute PSNR and SSIM scores.
    Params:
    -------
    `model_output` - output produced by a Pytorch model\n
    `gt` - reference data\n
    `data_range` - int, range of input data\n

    Return:
    -------
    `val_psnr, val_mssim` - scores from metrices PSNR, and SSIM
    """

    sidelenght = model_output.size()[1]

    arr_gt = gt.cpu().view(sidelenght).detach().numpy()
    arr_gt = (arr_gt / 2.) + 0.5

    arr_output = model_output.cpu().view(sidelenght).detach().numpy()
    arr_output = (arr_output / 2.) + 0.5
    arr_output = np.clip(arr_output, a_min=0., a_max=1.)

    val_psnr = psnr(arr_gt, arr_output,data_range=data_range)
    val_mssim = ssim(arr_gt, arr_output,data_range=data_range)
    return val_psnr, val_mssim


def save_data_to_file(root_dir, model, train_scores):
    """Save recorded data, i.e. weights and train scores, during training into a file location.
    Params:
    -------
    `root_dir` - str, dir within save model's weights and training scores\n
    `model` - PyThorch like object representing a model\n
    `train_scores` - np.array like object representing training scores accumulated at training time\n

    Return:
    -------
    `mode_weight_path` - str object, file path to model's weights.\n
    `train_scores_path` - str object, file path to model's train scores.\n
    """
    mode_weight_path, train_scores_path = None, None
    try:
        tmp_file_path = os.path.join(root_dir, 'model_final.pth')
        mode_weight_path = str(tmp_file_path)
        torch.save(model.state_dict(),
                  tmp_file_path)
        
        tmp_file_path = os.path.join(root_dir, 'train_losses_final.txt')
        train_scores_path = str(tmp_file_path)
        np.savetxt(tmp_file_path,
                   np.array(train_scores))
    except Exception as _:
                raise Exception(f"Error when saving file: filename={tmp_file_path} .")
    return mode_weight_path, train_scores_path
