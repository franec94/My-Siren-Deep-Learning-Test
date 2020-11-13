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

import torch

# --------------------------------------------- #
# Import: custom, from this project
# --------------------------------------------- #

import src.generic.dataio as dataio

from src.generic.custom_argparser import DYNAMIC_QUAT_SIZES


def show_model_summary(model):
    """Show Model summary"""
    
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


def get_arch_hyperparams(opt, image_resolution):
    """ Setup combinations to be tested."""

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

    if opt.quantization_enabled != None:
        if opt.quantization_enabled.lower() == "paszke_quant":
            param_grid['frequency'] = opt.frequences
            pass
        pass
    
    return list(ParameterGrid(param_grid))


def get_input_image(opt):
    """Get input image, if none image is provided, then Cameramen default image will be fetched."""
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
    """Create train logging directory."""

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
    """Get root logger"""
    log_filename = os.path.join(root_path, 'train.log')
    logging.basicConfig(filename=f'{log_filename}', level=logging.INFO)
    pass


def log_parser(root_path, parser, opt, debug_mode = False):
    """Log parser by means of plain .txt file and .pickle file.
    If debug_mode is True, no data will be stored.
    """
    if debug_mode is False:
        parser_logged = os.path.join(root_path, 'parser_logged.txt')
        with open(parser_logged, "w") as f:
            f.write(parser.format_values())
            pass
    
        parser_pickled = os.path.join(root_path, 'options.pickle')
        with open(parser_pickled, "wb") as f:
            # pickleopt = pickle.dumps(opt)
            pickle.dump(opt, f)
            pass
        pass
    pass


def set_hyperparams_to_be_tested(opt, grid_arch_hyperparams):
    """Set Hyper-parameters to be tested."""

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
    return opt, pos_start, pos_end


def show_number_of_trials(opt, grid_arch_hyperparams, via_tabulate = False):
    """
    Show number of trials.
    """
    if opt.show_number_of_trials:    
        logging.info(f'Total number of archs: {len(grid_arch_hyperparams)}')
        print(f'Total number of archs:', len(grid_arch_hyperparams))
    
        tot_trials = len(grid_arch_hyperparams) * opt.num_attempts
        logging.info(f'Total number of trials (with {opt.num_attempts} attempts per arch): {tot_trials}')
        print(f'Total number of trials (with {opt.num_attempts} attempts per arch):', tot_trials)
        pass
    pass


def check_quantization_tech_provided(opt):
    """Check quantization technique provided for training a Siren based model:
    - allowed techniques: [dynamic,static,posterior,quantization_aware_training]
    If none model is provided the default value will be None.
    """
    if opt.quantization_enabled == None: return opt

    quant_tech = opt.quantization_enabled.lower()
    if quant_tech not in "dynamic,static,post_train,paszke_quant,quantization_aware_training".split(","):
        raise Exception(f"Error: {quant_tech} not allowed!")

    opt.quantization_enabled = quant_tech
    return opt


def check_frequencies(opt):
    if opt.quantization_enabled == None: return
    if opt.quantization_enabled == 'paszke_quant':
        if opt.frequences == None:
            raise Exception('Error no frequences provided for Pazke Quant Tech.')
        for f in opt.frequences:
            if f < 0:
                raise Exception(f'Error frequence {f} value is not allowed.')
        pass
    pass


def check_quant_size_for_dynamic_quant(opt):
    """Check whether dynamic quant size provided by user from cmd line option is allowed."""

    if isinstance(opt.dynamic_quant, str):
        a_dynamic_size = opt.dynamic_quant.lower()
        if a_dynamic_size in DYNAMIC_QUAT_SIZES:
            if a_dynamic_size == 'qint8':
                opt.dynamic_quant = torch.qint8
            elif a_dynamic_size == 'float16':
                opt.dynamic_quant = torch.float16
        else:
            raise Exception(f"Dynamic quant size '{a_dynamic_size}' provided is not allowed.")
        opt.dynamic_quant = [opt.dynamic_quant]
    else:
        dynamic_size_list = []
        for a_dynamic_size in opt.dynamic_quant:
            a_dynamic_size = a_dynamic_size.lower()
            if a_dynamic_size in DYNAMIC_QUAT_SIZES:
                if a_dynamic_size == 'qint8':
                    dynamic_size_list.append(torch.qint8)
                elif a_dynamic_size == 'float16':
                    dynamic_size_list.append(torch.float16)
            else:
                raise Exception(f"Dynamic quant size '{a_dynamic_size}' provided is not allowed.")
            opt.dynamic_quant = dynamic_size_list
    return opt
