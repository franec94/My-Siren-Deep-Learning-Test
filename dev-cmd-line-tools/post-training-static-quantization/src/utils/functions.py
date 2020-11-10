
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
from src.utils.custom_argparser import QUANT_TECHS

def check_cmd_line_options(opt, parser):
    
    print(opt)
    print("----------")
    print(parser.format_help())
    print("----------")
    print(parser.format_values())    # useful for logging where different settings came from.
    pass


def show_model_summary(model):
    """Show model summary."""
    
    # Print model's state_dict
    print("Model's state_dict:")
    for param_tensor in model.state_dict():
        print(param_tensor, "\t", model.state_dict()[param_tensor].size())
        pass
    pass


def get_input_image(opt):
    """Get input image, either provided as path to file from command line. If no
    input image file will be provided from command line, default cameramen image will be fetched from
    skimage python library as default image.

    Return:
    -------
    :img_dataset, img, image_resolution: where:\n
    - img_dataset: instance from Pytorch DataSet\n
    - img: PIL.image instance\n
    - image_resolution: (int, int) size of the image
    """
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
    """Create dir where all results will be stored, within local file system.
    If debug_mode is False no dir will be created and therefore, no data will be saved.
    """
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


def log_parser(root_path, parser, opt, debug_mode = False):
    """Save parser information and options loaded from it as .txt and pickle files respectively.
    If debug_mode is False no data will be saved.
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
def  get_root_level_logger(root_path, debug_mode = False):
    if debug_mode is True:
        logging.basicConfig(filemode=sys.stdout, level=logging.INFO)
    else:
        log_filename = os.path.join(root_path, 'train.log')
        logging.basicConfig(filename=f'{log_filename}', level=logging.INFO)
        pass
    pass

def filter_model_files_csv_opt_args(args, logger = None, ext = [".csv"]):
    """Check wheter list of input .csv files contains .csv files, and filter out those which do not are .csv files."""

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

    args.log_models = list(filter(is_file_valid, filter(is_file_filter, args.log_models)))
    return args


def filter_model_files_opt_args(args, logger = None, ext = [".pth"]):
    """Check wheter list of input files contains .pth files, and filter out those which do not are .pth files."""

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

    args.model_files = list(filter(is_file_valid, filter(is_file_filter, args.model_files)))
    return args

def map_filter_model_dirs_opt_args(args, logger = None, ext = [".pth"]):
    """Check wheter list of input dirs contains dirs, and filter out those which do not are dirs."""

    def is_dir_valid(a_dir, logger = logger, ext = ext):
        is_valid_res = os.path.isdir(f"{a_dir}")
        if logger != None:
            logger.info(f"{a_dir} is valid? A: {is_valid_res}")
            pass
        return is_valid_res

    args.model_dirs = list(filter(is_dir_valid, args.model_dirs))
    return args


def check_quantization_tech_provided(opt):
    """Check quantization technique provided for training a Siren based model:
    - allowed techniques: [dynamic,static,posterior,quantization_aware_training]
    If none model is provided the default value will be None.
    """
    if opt.quantization_enabled == None: return opt
 
    # quant_techs = "dynamic,static,post_train,paszke_quant,quantization_aware_training".split(",")
    quant_techs = QUANT_TECHS 
    if isinstance(opt.quantization_enabled, str):
        quant_tech = opt.quantization_enabled.lower()
        if quant_tech not in quant_techs:
            raise Exception(f"Error: {quant_tech} not allowed!")
        opt.quantization_enabled = quant_tech
    else:
        quant_tech_list = []
        for quant_tech in opt.quantization_enabled:
            quant_tech = quant_tech.lower()
            if quant_tech not in "dynamic,static,post_train,paszke_quant,quantization_aware_training".split(","):
                raise Exception(f"Error: {quant_tech} not allowed!")
            quant_tech_list.append(quant_tech)
        opt.quantization_enabled = quant_tech_list
    return opt


def check_frequencies(opt):
    """Check Frequencies only if quant tech has been considered paszke_quant."""

    if opt.quantization_enabled == None: return
    if isinstance(opt.quantization_enabled, str):
        if opt.quantization_enabled == 'paszke_quant':
            if opt.frequences == None:
                raise Exception('Error no frequences provided for Pazke Quant Tech.')
            for f in opt.frequences:
                if f < 0:
                    raise Exception(f'Error frequence {f} value is not allowed.')
            pass
    else:
        if 'paszke_quant' in opt.quantization_enabled:
            if opt.frequences == None:
                raise Exception('Error no frequences provided for Pazke Quant Tech.')
            for f in opt.frequences:
                if f < 0:
                    raise Exception(f'Error frequence {f} value is not allowed.')
            pass
    pass

def check_sidelength(opt):
    """Check Sidelength ."""
    if opt.hf == None and opt.hl == None:
        return opt
    if opt.hf == None or opt.hl == None:
        raise Exception('Missing input arguments to keep on checking sidelength, since either opt.hf or opt.hl are None')

    def check_and_map_sl(a_sl):
        a_sl = eval(a_sl)
        if isinstance(a_sl, str):
            a_sl = int(a_sl)
            if a_sl <= 0: raise Exception("A sidelength provided is negative, which is not allowed!")
        if isinstance(a_sl, int):
            if a_sl <= 0: raise Exception("A sidelength provided is negative, which is not allowed!")
        if isinstance(a_sl, list) or isinstance(a_sl, tuple):
            if len(a_sl) > 2: raise Exception("To many values provided for defining sidelength")
            if len(a_sl) == 0: raise Exception("No values provided for defining sidelength")
            if len(a_sl) == 1:
                a_sl = (int(a_sl[0]), int(a_sl[0]))
            
            a_sl = (int(a_sl[0]), int(a_sl[1]))
            val_1, val_2 = a_sl
            if val_1 <= 0 or val_2: raise Exception("A sidelength provided is negative, which is not allowed!")
        return a_sl

    opt.sidelength = list(map(check_and_map_sl, opt.sidelength))
    return opt
