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


def log_parser(root_path, parser, debug_mode = False):
    if debug_mode is False:
        parser_logged = os.path.join(root_path, 'parser_logged.txt')
        with open(parser_logged, "w") as f:
            f.write(parser.format_values())
            pass
    
        parser_pickled = os.path.join(root_path, 'parser.pickle')
        with open(parser_pickled, "w") as f:
            pickle.dump(parser, f)
            pass
        pass
    pass


def set_hyperparams_to_be_tested(opt, grid_arch_hyperparams):
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
    if opt.show_number_of_trials:    
        logging.info(f'Total number of archs: {len(grid_arch_hyperparams)}')
        print(f'Total number of archs:', len(grid_arch_hyperparams))
    
        tot_trials = len(grid_arch_hyperparams) * opt.num_attempts
        logging.info(f'Total number of trials (with {opt.num_attempts} attempts per arch): {tot_trials}')
        print(f'Total number of trials (with {opt.num_attempts} attempts per arch):', tot_trials)
        pass
    pass