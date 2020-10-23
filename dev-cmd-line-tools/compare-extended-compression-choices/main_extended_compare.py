#!/usr/bin/env python3
 # -*- coding: utf-8 -*-

from __future__ import print_function
from __future__ import division

# Standard Library, plus some Third Party Libraries

import logging
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
import re
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


# ----------------------------------------------------------------------------------------------- #
# Globals
# ----------------------------------------------------------------------------------------------- #

opt, parser, device = None, None, None

# ----------------------------------------------------------------------------------------------- #
# Util Functions for Main
# ----------------------------------------------------------------------------------------------- #

def get_arch_hyperparams(opt, image_resolution):
    param_grid = None
    sidelength = min(image_resolution)

    seeds_list = opt.seeds
    hidden_layers_list = opt.hidden_layers
    hidden_features_list = opt.hidden_features
    """
    num_hidden_features = opt.num_hidden_features

    start_hf = int(2 ** int(np.log2(sidelength)))
    if start_hf == sidelength:
        start_hf = int(2 ** int(np.log2(sidelength)-1))
    hidden_features_arr = np.linspace(start_hf, sidelength, num=num_hidden_features, dtype=np.int)
    """

    # pprint(hidden_features_arr)
    pprint(hidden_features_list)

    param_grid = {
        'seeds': seeds_list,
        'hidden_layers': hidden_layers_list,
        # 'hidden_features': hidden_features_arr
        'hidden_features': hidden_features_list
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


def get_input_image(opt):
    if opt.image_filepath is None:
        img_dataset = dataio.Camera()
        img = Image.fromarray(skimage.data.camera())
        image_resolution = img.size
        if opt.sidelength is None:
            opt.sidelength = image_resolution
            # opt.sidelength = 256
            pass
    else:
        img_dataset =  dataio.ImageFile(opt.image_filepath)
        img = Image.open(opt.image_filepath)
        image_resolution = img.size
        if opt.sidelength is None:
            opt.sidelength = image_resolution
            # opt.sidelength = image_resolution
            pass
        pass

    return img_dataset, img, image_resolution


def create_train_logging_dir(opt):
    p = re.compile(r'\.')
    curr_time = datetime.datetime.now()
    curr_date = curr_time.strftime("%d-%m-%Y")

    curr_time_str = str(curr_time.timestamp())
    curr_timestamp = p.sub('-', curr_time_str)

    root_path = os.path.join(opt.logging_root,
        curr_date,
        curr_timestamp,
        opt.experiment_name)
    
    try: os.makedirs(root_path)
    except: pass

    return root_path, curr_date, curr_timestamp


def  get_root_level_logger(root_path):
    log_filename = os.path.join(root_path, 'train.log')
    logging.basicConfig(filename=f'{log_filename}', level=logging.INFO)
    pass


def main():

    # --- Get cmd line options and parser objects.
    global device
    global opt
    global parser
    # check_cmd_line_options()

    # --- Get input image to be compressed.
    img_dataset, img, image_resolution = \
        get_input_image(opt)

    # --- Get Hyper-params list.
    grid_arch_hyperparams = get_arch_hyperparams(opt, image_resolution)

    # --- Check verbose style.
    if opt.verbose not in [0, 1, 2]:
        raise ValueError(f"opt.verbose = {opt.verbose} not allowed!")

    # --- Set Hyper-params to be tested.
    if opt.end_to is None:
        opt.end_to = len(grid_arch_hyperparams)
    if opt.end_to > len(grid_arch_hyperparams):
        raise ValueError(f'opt.end_to = {opt.end_to} not allowed!')
    if opt.resume_from < 0 or opt.resume_from > opt.end_to:
        raise ValueError(f'opt.resume_from = {opt.resume_from} not allowed!')


    num_seeds = len(opt.seeds)
    num_hidden_layers = len(opt.hidden_layers)

    pos_start = opt.resume_from * (num_seeds * num_hidden_layers)
    pos_end = opt.end_to * (num_seeds * num_hidden_layers)

    # --- Create logging dirs.
    root_path, curr_date, curr_timestamp = \
        create_train_logging_dir(opt)
    
    # --- Create root logger.
    get_root_level_logger(root_path)

    # --- Log parsed cmd args.
    parser_logged = os.path.join(root_path, 'parser_logged.txt')
    with open(parser_logged, "w") as f:
        f.write(parser.format_values())
        pass
    logging.info(parser.format_values())

    # --- Show overall number of trials.
    if opt.show_number_of_trials:    
        logging.info(f'Total number of archs: {len(grid_arch_hyperparams)}')
        print(f'Total number of archs:', len(grid_arch_hyperparams))
    
        tot_trials = len(grid_arch_hyperparams) * opt.num_attempts
        logging.info(f'Total number of trials (with {opt.num_attempts} attempts per arch): {tot_trials}')
        print(f'Total number of trials (with {opt.num_attempts} attempts per arch):', tot_trials)
        pass


    # --- Set device upon which compute model's fitting
    # or evaluation, depending on the current desired task.
    try:
        device = (torch.device('cuda:0') if torch.cuda.is_available()
        else torch.device('gpu'))
    except:
        device = torch.device('cpu')
        pass
    print(f"Training on device {device}.")
    logging.info(f"Training on device {device}.")

    print(f"# cuda device: {torch.cuda.device_count()}")
    logging.info(f"# cuda device: {torch.cuda.device_count()}")

    if torch.cuda.device_count() > 0:
        print(f"Id current device: {torch.cuda.current_device()}")
        logging.info(f"Id current device: {torch.cuda.current_device()}")
        pass

    # --- Start training.
    start_time = time.time()
    print(f"Start training [{curr_date}][timestamp={curr_timestamp}] ...")
    logging.info(f"Start training [{curr_date}][timestamp={curr_timestamp}] ...")

    train_extended_compare.train_extended_protocol_compare_archs(
        grid_arch_hyperparams=grid_arch_hyperparams[pos_start:pos_end],
        img_dataset=img_dataset,
        opt=opt,
        model_dir=root_path,
        verbose=opt.verbose,
    )

    print(f"End training [{curr_date}][timestamp={curr_timestamp}] eta: {time.time() - start_time} seconds.")
    logging.info(f"End training [{curr_date}][timestamp={curr_timestamp}] eta: {time.time() - start_time} seconds.")
    
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

    # Run training.
    main()
    pass