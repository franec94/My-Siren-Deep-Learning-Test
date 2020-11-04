
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
import datetime
import h5py
import math
import os
import pickle
import random
import shutil
import sys
import re
import time

# --------------------------------------------- #
# Import: skimage
# --------------------------------------------- #
import skimage
import skimage.metrics as skmetrics
from skimage.metrics import peak_signal_noise_ratio as psnr
from skimage.metrics import structural_similarity as ssim
from skimage.metrics import mean_squared_error

# --------------------------------------------- #
# Import: custom, from this project
# --------------------------------------------- #
import src.utils.dataio as dataio

def check_cmd_line_options(opt, parser):
    
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
        pass
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
        img_dataset = dataio.ImageFile(opt.image_filepath)
        img = Image.open(opt.image_filepath)
        image_resolution = img.size
        if opt.sidelength is None:
            opt.sidelength = image_resolution
            # opt.sidelength = image_resolution
            pass
        pass

    return img_dataset, img, image_resolution


def create_train_logging_dir(opt, debug_mode = False):
    p = re.compile(r'\.')
    curr_time = datetime.datetime.now()
    curr_date = curr_time.strftime("%d-%m-%Y")

    curr_time_str = str(curr_time.timestamp())
    curr_timestamp = p.sub('-', curr_time_str)

    root_path = os.path.join(opt.logging_root,
        curr_date,
        curr_timestamp,
        opt.experiment_name)
    if debug_mode is False:
        try: os.makedirs(root_path)
        except: pass
    else:
        print('DEBUG MODE: root_path - not created!')

    return root_path, curr_date, curr_timestamp


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

def  get_root_level_logger(root_path, debug_mode = False):
    if debug_mode is True:
        logging.basicConfig(filemode=sys.stdout, level=logging.INFO)
    else:
        log_filename = os.path.join(root_path, 'train.log')
    logging.basicConfig(filename=f'{log_filename}', level=logging.INFO)
    pass

def filter_model_files_csv_opt_args(args, logger = None, ext = [".csv"]):
    def is_file_filter(a_file, logger = logger):
        is_file_res = os.path.isfile(f"{a_file}")
        if logger != None:
            logger.info(f"{a_file} is file? A: {is_file_res}")
            pass
        return is_file_res
    def is_file_valid(a_file, logger = logger, ext = ext):
        _, extension = os.path.splitext(f"{a_file}")
        is_valid_res = extension in ext
        if logger != None:
            logger.info(f"{a_file} is valid ({ext})? A: {is_valid_res}")
            pass
        return is_valid_res

    args.log_models = list(map(is_file_valid, map(is_file_filter, args.log_models)))
    return args


def filter_model_files_opt_args(args, logger = None, ext = [".ph"]):
    def is_file_filter(a_file, logger = logger):
        is_file_res = os.path.isfile(f"{a_file}")
        if logger != None:
            logger.info(f"{a_file} is file? A: {is_file_res}")
            pass
        return is_file_res
    def is_file_valid(a_file, logger = logger, ext = ext):
        _, extension = os.path.splitext(f"{a_file}")
        is_valid_res = extension in ext
        if logger != None:
            logger.info(f"{a_file} is valid ({ext})? A: {is_valid_res}")
            pass
        return is_valid_res

    args.model_files = list(map(is_file_valid, map(is_file_filter, args.model_files)))
    return args

def map_filter_model_dirs_opt_args(args, logger = None, ext = [".ph"]):
    def is_dir_valid(a_dir, logger = logger, ext = ext):
        is_valid_res = os.path.isdir(f"{a_dir}")
        if logger != None:
            logger.info(f"{a_dir} is valid ({ext})? A: {is_valid_res}")
            pass
        return is_valid_res

    args.model_dirs = list(map(is_dir_valid, args.model_dirs))
    return args
