
from __future__ import print_function
from __future__ import division

# --------------------------------------------- #
# Import: Standard Library
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
import collections
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
# Data Science and Machine Learning Libraries
# --------------------------------------------- #
import numpy as np
import pandas as pd

# --------------------------------------------- #
# Import: skimage
# --------------------------------------------- #
import skimage
import skimage.metrics as skmetrics
from skimage.metrics import peak_signal_noise_ratio as psnr
from skimage.metrics import structural_similarity as ssim
from skimage.metrics import mean_squared_error

# --------------------------------------------- #
# Import: torch
# --------------------------------------------- #
from torch.utils.data import DataLoader

import torch
import torch
import torch.nn as nn


# --------------------------------------------- #
# Import: torchvision
# --------------------------------------------- #
from torchvision import datasets

import torchvision
import torchvision.transforms as transforms

import torch.quantization

# --------------------------------------------- #
# Import: custom, from this project
# --------------------------------------------- #
from src.utils.archs.siren import Siren
from src.utils.archs.siren_quantized import SirenQuantized
import src.utils.dataio as dataio

from src.utils.quant_utils.quant_utils_functions import _evaluate_model
from src.utils.quant_utils.quant_utils_functions import _prepare_data_loaders
from src.utils.quant_utils.quant_utils_functions import prepare_model

from src.utils.functions import get_input_image
from src.utils.quant_utils.compute_quantization import compute_quantization

from src.utils.quant_utils.quant_utils_functions import _evaluate_model as _evaluate_model_2
from src.utils.quant_utils.compute_quantization import get_size_of_model


def _evaluate_model_local(image_dataset, model_conf, quant_tech = None, device = 'cpu'):

    eval_scores = []
    if quant_tech == None:
        if torch.cuda.is_available():
            print("Try eval plain model on cuda device...")
            eval_dataloader = _prepare_data_loaders(image_dataset, model_conf)
            model = prepare_model(opt = model_conf, arch_hyperparams=model_conf._asdict(), device='cuda', model_weights_file = model_conf.model_filename)
            size_model = get_size_of_model(model)
            eval_scores, eval_time = _evaluate_model_2(model, evaluate_dataloader=eval_dataloader, device='cuda')
        else:
            print("No cuda device available, switching to cpu.")
            print("Try eval plain model on cpu device...")
            eval_dataloader = _prepare_data_loaders(image_dataset, model_conf)
            model = prepare_model(opt = model_conf, arch_hyperparams=model_conf._asdict(), device='cpu',  model_weights_file = model_conf.model_filename)
            size_model = get_size_of_model(model)
            eval_scores, eval_time = _evaluate_model_2(model, evaluate_dataloader=eval_dataloader, device='cpu')
        pass
    else:
        print('Eval:', quant_tech.upper())
        eval_scores, eta_eval, size_model = compute_quantization(img_dataset = image_dataset, opt = model_conf, model_path = model_conf.model_filename, arch_hyperparams = model_conf._asdict(), fuse_modules = None, device = 'cpu', qconfig = 'fbgemm')
        pass

    return eval_scores, eta_eval, size_model


def _print_size_of_model(model):
    """Print model's size."""

    torch.save(model.state_dict(), "temp.p")
    print('Size (MB):', os.path.getsize("temp.p")/1e6)
    os.remove('temp.p')
    pass


def _prepare_post_training_model(model_path, model_params, is_quantized = False, device = 'cpu', verbose = 0):
    """Prepare model for posterior quantization technique."""

    if is_quantized is False:
        if device == 'cpu':
            model = Siren(
                in_features=2,
                out_features=1,
                hidden_features=int(model_params['hidden_features']),
                hidden_layers=int(model_params['hidden_layers']),
                outermost_linear=True)
            state_dict = torch.load(model_path)
            # model.load_state_dict(state_dict).to('cpu')
            model.load_state_dict(state_dict)
            model.to(device=device)
        else:
            model = Siren(
                in_features=2,
                out_features=1,
                hidden_features=int(model_params['hidden_features']),
                hidden_layers=int(model_params['hidden_layers']),
                outermost_linear=True)
            state_dict = torch.load(model_path)
            model.load_state_dict(state_dict).cuda()
            pass
        print(f"Size of model loaded to device = {device}")
        _print_size_of_model(model)
    else:
        model = SirenQuantized(
            in_features=2,
            out_features=1,
            hidden_features=int(model_params['hidden_features']),
            hidden_layers=int(model_params['hidden_layers']),
            outermost_linear=True)
        state_dict = torch.load(model_path)
        model.load_state_dict(state_dict)
        model.to(device=device)
        if verbose > 1:
            print("Size of model Before quantization")
            _print_size_of_model(model)
            pass

        # set quantization config for server (x86)
        # model.qconfig = torch.quantization.default_qconfig
        model.eval()
        model.qconfig = torch.quantization.get_default_qconfig('fbgemm')
        # model.qconfig = torch.quantization.get_default_qat_qconfig('qnnpack')
        torch.quantization.prepare(model, inplace=True)
        # model = torch.quantization.prepare_qat(model, inplace=True)
        pass

    return model


def _prepare_data_loaders(img_dataset, opt):
    """Prepare data loader from which fetching data in order to feed models."""

    coord_dataset = dataio.Implicit2DWrapper(
                img_dataset, sidelength=opt.sidelength, compute_diff=None)
    a_dataloader = DataLoader(
                coord_dataset,
                shuffle=False,
                batch_size=1,
                pin_memory=True, num_workers=0)
    return a_dataloader


def _evaluate_model(model, loss_fn, evaluate_dataloader, device = 'cpu'):
    """ Evalaute model."""

    # --- Evaluate model's on validation data.
    eval_scores = None
    model.eval()
    with torch.no_grad():
        # -- Get data from validation loader.
        val_input, val_gt = next(iter(evaluate_dataloader))

        if device == 'cpu':
            val_input = val_input['coords'].to(device)
            val_gt = val_gt['img'].to(device)
        else:
            val_input = val_input['coords'].cuda() # .to(device)
            val_gt = val_gt['img'].cuda() # .to(device)
            pass

        # --- Compute estimation.
        val_output, _ = model(val_input)

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
        eval_scores = np.array([train_loss.item(), val_psnr, val_mssim])
        pass
    return eval_scores


def _evaluate_quantized_model(model_path, model_params, img_dataset, opt, loss_fn = nn.MSELoss(), device = 'cpu', model = None, verbose = 0):
    """Evaluate model by means of posterior quantization model."""

    if model is None:
        print("Create model for quantization...")
        model = \
            _prepare_post_training_model(
                model_path,
                model_params,
                is_quantized = True,
                verbose = verbose
        )
        pass
    else:
        print("Model for quantization already created.")
        pass

    eval_dataloader = \
        _prepare_data_loaders(
            img_dataset,
            opt
    )
    eval_scores = \
        _evaluate_model(
            model,
            loss_fn, device = device, evaluate_dataloader = eval_dataloader
    )
    return eval_scores, model


def evaluate_plain_model(model_path, model_params, img_dataset, opt, loss_fn = nn.MSELoss(), device = 'cpu', verbose = 0):
    """Evaluate plain model."""

    model = \
        _prepare_post_training_model(
            model_path,
            model_params,
            device = device,
            verbose = verbose
    )
    eval_dataloader = \
        _prepare_data_loaders(
            img_dataset,
            opt
    )
    eval_scores = \
        _evaluate_model(
            model,
            loss_fn,
            device = device,
            evaluate_dataloader = eval_dataloader
    )
    return eval_scores, model


def evaluate_post_train_quantized_models_by_csv_2(a_file_csv, args, device = 'cpu', verbose = 0):

    cropped_images_df = _read_csv_data(a_file_csv)

    if cropped_images_df.shape[0] == 0:
        return []

    # - Sort data
    attrs_for_sorting = "timestamp,hf,hl".split(",")
    cropped_images_df = cropped_images_df.sort_values(by = attrs_for_sorting)

    Columns = collections.namedtuple('Columns', cropped_images_df.columns)
    Options = collections.namedtuple('Options', "image_filepath,sidelength".split(","))
    EvalScores = collections.namedtuple('EvalScores', "mse,psnr,ssim".split(","))

    field_names = list(cropped_images_df.columns) + "mse,psnr,ssim".split(",")
    RecordTuple = collections.namedtuple('RecordTuple', field_names)

    # fields_name = "model_filename,hidden_layers,hidden_features,sidelength,quant_tech,mse,psnr,ssim".split(",")
    fields_name = "model_filename,hidden_layers,hidden_features,sidelength,quant_tech,mse,psnr,ssim,eta_seconds,model_size".split(",")
    InfoResults = collections.namedtuple('InfoResults', fields_name)

    records_list = []
    files_not_found = []

    if args.quantization_enabled != None:
        print('opt.quantization_enabled != None')
        if isinstance(args.quantization_enabled, str):
            quant_tech_list = [args.quantization_enabled]
        else:
            quant_tech_list = args.quantization_enabled
    else:
        print('opt.quantization_enabled = None')
        quant_tech_list = []

    records_list = []
    for row in cropped_images_df[:].values:
        vals = Columns._make(row)

        if os.path.exists(vals.path) is False or os.path.isfile(vals.path) is False:
            files_not_found.append(vals.path)
            continue

        model_params = dict(hidden_features=int(vals.hf),
            hidden_layers=int(vals.hl),
            quantization_enabled=None,
            model_filename=vals.path, sidelength=int(vals.cropped_width))
        opt = Options._make([args.image_filepath, int(vals.cropped_width)])

        model_conf = collections.namedtuple('ModelConf', list(model_params.keys()))._make(list(model_params.values()))
        pprint(model_conf)

        # --- Get input image to be evaluated.
        # img_dataset, img, image_resolution = \
        img_dataset, _, _ = \
            get_input_image(opt)
        
        eval_scores, eta_eval, size_model = _evaluate_model_local(image_dataset = img_dataset, model_conf = model_conf, quant_tech = None, device = 'cuda')
        pprint(eval_scores)

        vals_r = [vals.path, int(vals.hl), int(vals.hf), opt.sidelength, 'None'] + list(eval_scores) + [eta_eval, size_model]
        a_record = InfoResults._make(vals_r)
        records_list.append(a_record)

        for a_tech in quant_tech_list:
            print('Eval quant tech:', a_tech)
            model_params = dict(hidden_features=int(vals.hf),
                hidden_layers=int(vals.hl),
                quantization_enabled=a_tech,
                model_filename=vals.path, sidelength=int(vals.cropped_width))
            model_conf = collections.namedtuple('ModelConf', list(model_params.keys()))._make(list(model_params.values()))
            eval_scores, eta_eval, size_model = _evaluate_model_local(image_dataset = img_dataset, model_conf = model_conf, quant_tech = a_tech, device = 'cpu')
            pprint(eval_scores)
            vals_r = [vals.path, int(vals.hl), int(vals.hf), opt.sidelength, a_tech] + list(eval_scores) + [eta_eval, size_model]
            a_record = InfoResults._make(vals_r)
            records_list.append(a_record)
            pass
        pass

    return records_list, files_not_found

def evaluate_post_train_quantized_models_by_csv(a_file_csv, args, device = 'cpu', verbose = 0):
    """
    Evaluate posterior quantized models fetching data and weigths from information gotten reading a .csv file.
    """
    # - Read data from src file

    cropped_images_df = _read_csv_data(a_file_csv)

    if cropped_images_df.shape[0] == 0:
        return []

    # - Sort data
    attrs_for_sorting = "timestamp,hf,hl".split(",")
    cropped_images_df = cropped_images_df.sort_values(by = attrs_for_sorting)

    Columns = collections.namedtuple('Columns', cropped_images_df.columns)
    Options = collections.namedtuple('Options', "image_filepath,sidelength".split(","))
    EvalScores = collections.namedtuple('EvalScores', "mse,psnr,ssim".split(","))

    field_names = list(cropped_images_df.columns) + "mse,psnr,ssim".split(",")
    RecordTuple = collections.namedtuple('RecordTuple', field_names)

    records_list = []
    files_not_found = []
    for row in cropped_images_df[:].values:
        vals = Columns._make(row)

        if os.path.exists(vals.path) is False or os.path.isfile(vals.path) is False:
            files_not_found.append(vals.path)
            continue

        model_params = dict(hidden_features=int(vals.hf), hidden_layers=int(vals.hl))
        opt = Options._make([args.image_filepath, int(vals.cropped_width)])

        # --- Get input image to be evaluated.
        # img_dataset, img, image_resolution = \
        img_dataset, _, _ = \
            get_input_image(opt)

        # Calibrate first
        print('Post Training Quantization Prepare: Inserting Observers')

        # Calibrate with the training set
        eval_scores, model = _evaluate_quantized_model(
            model_path = vals.path,
            model_params = model_params,
            img_dataset = img_dataset,
            opt = opt,
            loss_fn = nn.MSELoss(),
            device = device,
            model = None,
            verbose = 0)
        print('Post Training Quantization: Calibration done')
        print(eval_scores)

        img_dataset, _, _ = \
            get_input_image(opt)
        
        # Convert to quantized model
        torch.quantization.convert(model, inplace=True)
        print('Post Training Quantization: Convert done')
        if verbose > 1:
            print("Size of model After quantization")
            _print_size_of_model(model)
            pass
        eval_scores, model = _evaluate_quantized_model(
            model_path = vals.path,
            model_params = model_params,
            img_dataset = img_dataset,
            opt = opt,
            loss_fn = nn.MSELoss(),
            device = device,
            model = model,
            verbose = 0)

        print(eval_scores)
        
        a_record = EvalScores._make(eval_scores)

        all_vals = vals + a_record
        record_tuple = RecordTuple._make(all_vals)

        records_list.append(record_tuple)
        pass


    # - Add columns for better working
    return records_list, files_not_found


def _read_csv_data(a_file_csv):
    """Read input csv data."""

    # - Read data from src file
    runs_df = pd.read_csv(a_file_csv)

    if 'Unnamed: 0' in runs_df.columns:
        runs_df = runs_df.drop(['Unnamed: 0'], axis = 1)
        pass
    cropped_images_df = runs_df[~runs_df['cropped_heigth'].isna()][~runs_df['cropped_width'].isna()]

    # - Sort data
    attrs_for_sorting = "timestamp,hf,hl".split(",")
    cropped_images_df = cropped_images_df.sort_values(by = attrs_for_sorting)

    return cropped_images_df


def evaluate_post_train_models_by_csv(a_file_csv, args, device = 'cpu'):
    """
    Evaluate plain models fetching data and weigths from information gotten reading a .csv file.
    """

    cropped_images_df = _read_csv_data(a_file_csv)

    if cropped_images_df.shape[0] == 0:
        return []
    
    Columns = collections.namedtuple('Columns', cropped_images_df.columns)
    Options = collections.namedtuple('Options', "image_filepath,sidelength".split(","))
    EvalScores = collections.namedtuple('EvalScores', "mse,psnr,ssim".split(","))

    field_names = list(cropped_images_df.columns) + "mse,psnr,ssim".split(",")
    RecordTuple = collections.namedtuple('RecordTuple', field_names)

    records_list = []
    files_not_found = []
    for row in cropped_images_df[:].values:
        vals = Columns._make(row)

        if os.path.exists(vals.path) is False or os.path.isfile(vals.path) is False:
            files_not_found.append(vals.path)
            continue

        model_params = dict(hidden_features=int(vals.hf), hidden_layers=int(vals.hl))
        opt = Options._make([args.image_filepath, int(vals.cropped_width)])

        # --- Get input image to be evaluated.
        # img_dataset, img, image_resolution = \
        img_dataset, _, _ = \
            get_input_image(opt)

        eval_scores, _ = evaluate_plain_model(
            model_path = vals.path,
            model_params = model_params,
            img_dataset = img_dataset,
            opt = opt,
            loss_fn = nn.MSELoss(),
            device = device,
            verbose = 0)

        print(eval_scores)
        
        a_record = EvalScores._make(eval_scores)

        all_vals = vals + a_record
        record_tuple = RecordTuple._make(all_vals)

        records_list.append(record_tuple)
        pass


    # - Add columns for better working
    return records_list, files_not_found


def evaluate_post_train_models_by_csv_list(file_csv_list, args, device = 'cpu'):
    """
    Evaluate models fetching data and weigths from information gotten reading .csv file.
    """

    if file_csv_list is None or len(file_csv_list) == 0:
        return []
    
    records_list = []
    files_not_found = []
    for a_file_csv in file_csv_list:
        records_list_tmp, files_not_found_tmp = \
            evaluate_post_train_models_by_csv(
                a_file_csv,
                args,
                device = device)
        records_list.extend(records_list_tmp)
        files_not_found.extend(files_not_found_tmp)
        pass
    
    return records_list, files_not_found


def evaluate_post_train_posterion_quantized_models_by_csv_list(file_csv_list, args, device = 'cpu'):
    """
    Evaluate posterior quantization models fetching data and weigths from information gotten reading .csv files.
    """

    if file_csv_list is None or len(file_csv_list) == 0:
        return []
    
    records_list = []
    files_not_found = []
    for a_file_csv in file_csv_list:
        records_list_tmp, files_not_found_tmp = \
            evaluate_post_train_quantized_models_by_csv_2(
                a_file_csv,
                args,
                device = device)
        records_list.extend(records_list_tmp)
        files_not_found.extend(files_not_found_tmp)
        pass
    
    return records_list, files_not_found
