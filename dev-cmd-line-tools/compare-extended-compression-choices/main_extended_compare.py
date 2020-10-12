from __future__ import print_function
from __future__ import division

# Standard Library, plus some Third Party Libraries
from pprint import pprint
from PIL import Image
from tqdm import tqdm
from typing import Union, Tuple

import configargparse
from functools import partial

import copy
import datetime
import h5py
import math
import os
import random
import shutil
import sys
import time
# import visdom

# Data Science and Machine Learning Libraries
import matplotlib
import matplotlib.pyplot as plt
matplotlib.style.use('ggplot')

import numpy as np
import pandas as pd
import sklearn

from sklearn.model_selection import ParameterGrid
from sklearn.model_selection import train_test_split

# Torch
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset

# from piq import ssim
# from piq import psnr

# TorchVision
"""
import torchvision
from torchvision import datasets
from torchvision import transforms
from torchvision.transforms import Resize, Compose, ToTensor, Normalize
from torchvision.utils import save_image
"""

# import torchsummary

# skimage
import skimage
import skimage.metrics as skmetrics
from skimage.metrics import peak_signal_noise_ratio as psnr
from skimage.metrics import structural_similarity as ssim
from skimage.metrics import mean_squared_error


from src.utils.custom_argparser import get_cmd_line_opts
from src.utils.siren import Siren

import src.utils.dataio as dataio
import src.utils.evaluate as evaluate
import src.utils.loss_functions as loss_functions
import src.utils.modules as modules
import src.utils.train_extended_compare as train_extended_compare
import src.utils.utils as utils
import src.utils.graphics as graphics

"""
class Config:  
    def __init__(self, **kwargs):
      for key, value in kwargs.items():
          setattr(self, key, value)
      pass
    pass

class PlotConfig(Config):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        pass
    pass


device, opt, parser = None, None, None

config_plot_loss = PlotConfig(
    image_path = None, # loss_image_path,
    figsize = (10, 7),
    title = "Train - Loss vs Epochs",
    xlabel = "Epochs",
    ylabel = "Loss",
    label = "train loss",
    color = "orange",
    show_min_max = True,
    ax = None
)

config_plot_psnr = PlotConfig(
    image_path = None, # psnr_image_path,
    figsize = (10, 7),
    title = "Train - PSNR (db) vs Epochs",
    xlabel = "Epochs",
    ylabel = "PSNR (db)",
    label = "train PSNR (db)",
    color = "green",
    show_min_max = True,
    ax = None
)

config_plot_ssim = PlotConfig(
    image_path = None, # psnr_image_path,
    figsize = (10, 7),
    title = "Train - SSIM vs Epochs",
    xlabel = "Epochs",
    ylabel = "SSIM",
    label = "train SSIM",
    color = "red",
    show_min_max = True,
    ax = None
)
"""


def get_arch_hyperparams(opt, image_resolution):
    param_grid = None
    sidelength = min(image_resolution)

    seeds_list = opt.seeds
    hidden_layers_list = opt.hidden_layers
    num_hidden_features = opt.num_hidden_features

    start_hf = int(2 ** (np.log2(sidelength)))
    if start_hf == sidelength:
        start_hf = int(2 ** (np.log2(sidelength)-1))
    hidden_features_arr = np.linspace(start_hf, sidelength, num=num_hidden_features, dtype=np.int)

    # pprint(hidden_features_arr)

    param_grid = {
        'seeds': seeds_list,
        'hidden_layers': hidden_layers_list,
        'hidden_features': hidden_features_arr
    }

    return list(ParameterGrid(param_grid))

def check_cmd_line_options():

    global opt, parser
    
    print(opt)
    print("----------")
    print(parser.format_help())
    print("----------")
    print(parser.format_values())    # useful for logging where different settings came from.
    pass


def show_model_summary(model):
    # Print model's state_dict
    print("Model's state_dict:")
    for param_tensor in model.state_dict():
        print(param_tensor, "\t", model.state_dict()[param_tensor].size())
    # print()

    """
    # Print optimizer's state_dict
    print("Optimizer's state_dict:")
    for var_name in optimizer.state_dict():
        print(var_name, "\t", optimizer.state_dict()[var_name])
    """
    pass


def main():

    # Get cmd line options and parser objects.
    global device, opt, parser
    # check_cmd_line_options()

    # Get input image to be compressed.
    if opt.image_filepath is None:
        img_dataset = dataio.Camera()
        # coord_dataset = dataio.Implicit2DWrapper(img_dataset, sidelength=512, compute_diff='all')
        image_resolution = (512, 512)
    else:
        img_dataset =  dataio.ImageFile(opt.image_filepath)
        img = Image.open(opt.image_filepath)
        image_resolution = img.size
        # if opt.sidelength is None:
        #     opt.sidelength = min(img.size)
        # coord_dataset = dataio.Implicit2DWrapper(img_dataset, sidelength=opt.sidelength, compute_diff='all')
        # image_resolution = (opt.sidelength, opt.sidelength)

        """
        fig = plt.figure()
        Image.open(opt.image_filepath).show()
        plt.show()
        """
        pass

    grid_arch_hyperparams = get_arch_hyperparams(opt, image_resolution)
    if opt.show_timetable_estimate:    
        print(f'Total number of trials (with 1 attempts per arch):', len(grid_arch_hyperparams))
    
        tot_trials = len(grid_arch_hyperparams) * opt.num_attempts
        print(f'Total number of trials (with {opt.num_attempts} attempts per arch):', tot_trials)

        estimated_time_30s = time.strftime("%H:%M:%S", time.gmtime(tot_trials * 30))
        estimated_time_1m = time.strftime("%H:%M:%S", time.gmtime(tot_trials * 60))
        estimated_time_3m = time.strftime("%H:%M:%S", time.gmtime(tot_trials * 60 * 3))
        estimated_time_5m = time.strftime("%H:%M:%S", time.gmtime(tot_trials * 60 * 5))

        print("Overall Estimated timetable H:M:S):")
        print(f"Estimated time (considering 30 seconds and {opt.num_attempts} attempts per arch):", estimated_time_30s)
        print(f"Estimated time (considering 1 minute and {opt.num_attempts} attempts per arch):", estimated_time_1m)
        print(f"Estimated time (considering 3 minutes and {opt.num_attempts} attempts per arch):", estimated_time_3m)
        print(f"Estimated time (considering 5 minutes and {opt.num_attempts} attempts per arch):", estimated_time_5m)
        pass

    
    train_extended_compare.train_extended_protocol_compare_archs(
        grid_arch_hyperparams=grid_arch_hyperparams,
        img_dataset=img_dataset,
        opt=opt,
        verbose=2,
    )
    
    pass


if __name__ == "__main__":

    # Initialize option and parser objects.
    opt, parser = get_cmd_line_opts()
    
    # Set seeds for experiment re-running.
    if hasattr(opt, 'seed'): seed = opt.seed
    else: seed = 0
    
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    
    # Set device upon which compute model's fitting
    # or evaluation, depending on the current desired task.
    device = None
    try:
        device = (torch.device('cuda:0') if torch.cuda.is_available()
        else torch.device('gpu'))
    except:
        device = torch.device('cpu')
        pass
    print(f"Training on device {device}.")
    print(f"# cuda device: {torch.cuda.device_count()}")
    if torch.cuda.device_count() > 0:
        print(f"Id current device: {torch.cuda.current_device()}")

    # Run training.
    main()
    pass