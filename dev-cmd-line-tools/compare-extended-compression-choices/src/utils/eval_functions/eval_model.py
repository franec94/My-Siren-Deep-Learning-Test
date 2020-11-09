'''Module containing fucntions that implement evaluation mode for Deep Learning models encode via Pytorch Framework.
'''
from __future__ import print_function
from __future__ import division


# ----------------------------------------------------------------------------------------------- #
# Globals
# ----------------------------------------------------------------------------------------------- #

# ----------------------------------------------------------------------------------------------- #
# Imports
# ----------------------------------------------------------------------------------------------- #

# --------------------------------------------- #
# Standard Library | Third Party Libraries
# --------------------------------------------- #
from torch.utils.data import DataLoader, Dataset
from torch.utils.tensorboard import SummaryWriter
from tqdm.autonotebook import tqdm

import collections
import logging
import os
import random
import math
import shutil
import tabulate
import time

import numpy as np
from sklearn.preprocessing import MinMaxScaler

# --------------------------------------------- #
# torch
# --------------------------------------------- #
import torch
import torch.nn as nn

# --------------------------------------------- #
# skimage
# --------------------------------------------- #
import skimage
import skimage.metrics as skmetrics
from skimage.metrics import peak_signal_noise_ratio as psnr
from skimage.metrics import structural_similarity as ssim
from skimage.metrics import mean_squared_error

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
# Custom Imports
# --------------------------------------------- #

from src.utils.train_functions.training_utils_funcs import get_data_ready_for_model, compute_desired_metrices, save_data_to_file


# ----------------------------------------------------------------------------------------------- #
# Functions
# ----------------------------------------------------------------------------------------------- #

def evaluate_model_old_mode(model, val_dataloader, quantization_enabled = None, device = 'cpu', loss_fn = nn.MSELoss(), verbose = 0):
    train_scores = None
    model.eval()
    with torch.no_grad():
        # -- Get data from validation loader.
        val_input, val_gt = next(iter(val_dataloader))
        if device == 'cpu':
            val_input = val_input['coords'].to('cpu')
            val_gt = val_gt['img'].to('cpu')
            if quantization_enabled  != None:
                val_input = torch.quantize_per_tensor(val_input, 0.01, 0, torch.qint8)
                val_gt = torch.quantize_per_tensor(val_gt, 0.01, 0, torch.qint8)    
                pass
        else:
            if quantization_enabled  != None:
                model_input = torch.quantize_per_tensor(model_input, 0.01, 0, torch.qint8).cuda()
                gt = torch.quantize_per_tensor(gt, 0.01, 0, torch.qint8).cuda()
            else:
                val_input = val_input['coords'].cuda() # .to(device) 
                val_gt = val_gt['img'].cuda() # .to(device)
                pass
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
        # train_scores = np.array([[train_loss, val_psnr, val_mssim]])
        train_scores = np.array([train_loss.item(), val_psnr, val_mssim])
    return train_scores, eta_eval

def evaluate_model(model, eval_dataloader, device = 'cpu', loss_fn = nn.MSELoss(), quantization_enabled = None, verbose = 0, logging_flag = False, tqdm = None):
    """Evaluate model, computing: loss score, PSNR and MSSI metrices, when model swithced in eval mode..
    Params:
    -------
    :model: PyTorch based model\n
    :eval_dataloader: PyTorch DataLoader like object\n
    :device: str object, allowed values: 'cpu', 'gpu', 'cuda'\n
    :loss_fn: Pytorch like Loss Function object\n
    :quantization_enabled: str object, quantization technique name, allowed values: [dynamic,static,posterior,quantization_aware_training]\n
    :verbose: int, verobose mode allowed values: 0,1,2
    :logging_flag: bool, if True enabling logging info for elapsed time in evaluating input data, result logged in info level.
    :tqdm: tqdm object, if not None write to it.

    Return:
    -------
    :eval_scores: np.array object, containing loss, psnr, mssi scores compute when model swithced in eval mode.
    :eta_eval: float representing time necessary for evaluating the model against the provided input
    """
    eval_scores = None # Define a priori to use later.
    model.eval()
    with torch.no_grad():
        # -- Get data from validation loader.
        eval_input, eval_gt = next(iter(eval_dataloader))
        eval_input, eval_gt = \
            get_data_ready_for_model(model_input = eval_input,
            gt = eval_gt,
            quantization_enabled=quantization_enabled,
            device = device)
        
        # --- Compute estimation.
        start_time = time.time()
        eval_output, _ = model(eval_input)
        eta_eval = time.time() - start_time

        # --- Prepare data for calculating metrices scores.
        # sidelenght = int(math.sqrt(val_output.size()[1]))
        eval_loss = loss_fn(eval_output, eval_gt)
        eval_psnr, eval_mssim = compute_desired_metrices(
            model_output = eval_output,
            gt = eval_gt)
        
        # --- Record results.
        # train_scores = np.array([[train_loss, val_psnr, val_mssim]])
        eval_scores = np.array([eval_loss.item(), eval_psnr, eval_mssim])
        pass
    return eval_scores, eta_eval
