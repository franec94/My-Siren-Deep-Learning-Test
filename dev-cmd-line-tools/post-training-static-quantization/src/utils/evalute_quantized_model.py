
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
from src.utils.siren import Siren
from src.utils.siren_quantized import SirenQuantized
import src.utils.dataio as dataio
from src.utils.functions import get_input_image

def _print_size_of_model(model):
    torch.save(model.state_dict(), "temp.p")
    print('Size (MB):', os.path.getsize("temp.p")/1e6)
    os.remove('temp.p')
    pass


def _prepare_post_training_model(model_path, model_params, is_quantized = False, device = 'cpu', verbose = 0):

    if is_quantized is False:
        if model == 'cpu':
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
        torch.quantization.prepare(model, inplace=True)
        pass

    return model


def _prepare_data_loaders(img_dataset, opt):
    coord_dataset = dataio.Implicit2DWrapper(
                img_dataset, sidelength=opt.sidelength, compute_diff=None)
    a_dataloader = DataLoader(
                coord_dataset,
                shuffle=False,
                batch_size=1,
                pin_memory=True, num_workers=0)
    return a_dataloader


def _evaluate_model(model, loss_fn, evaluate_dataloader, device = 'cpu'):
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
    if model is None:
        model = \
            _prepare_post_training_model(
                model_path,
                model_params,
                is_quantized = True,
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
            loss_fn, device = device, evaluate_dataloader = eval_dataloader
    )
    return eval_scores, model


def evaluate_plain_model(model_path, model_params, img_dataset, opt, loss_fn = nn.MSELoss(), device = 'cpu', verbose = 0):
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


def evaluate_post_train_quantized_models_by_csv(a_file_csv, args, device = 'cpu', verbose = 0):
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
            verbose = 0)
        print('Post Training Quantization: Calibration done')

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
        records_list.append(a_record)
        pass


    # - Add columns for better working
    return records_list, files_not_found


def _read_csv_data(a_file_csv):
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

    cropped_images_df = _read_csv_data(a_file_csv)

    if cropped_images_df.shape[0] == 0:
        return []
    
    Columns = collections.namedtuple('Columns', cropped_images_df.columns)
    Options = collections.namedtuple('Options', "image_filepath,sidelength".split(","))
    EvalScores = collections.namedtuple('EvalScores', "mse,psnr,ssim".split(","))

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
        records_list.append(a_record)
        pass


    # - Add columns for better working
    return records_list, files_not_found


def evaluate_post_train_models_by_csv_list(file_csv_list, args, device = 'cpu'):

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

    if file_csv_list is None or len(file_csv_list) == 0:
        return []
    
    records_list = []
    files_not_found = []
    for a_file_csv in file_csv_list:
        records_list_tmp, files_not_found_tmp = \
            evaluate_post_train_quantized_models_by_csv(
                a_file_csv,
                args,
                device = device)
        records_list.extend(records_list_tmp)
        files_not_found.extend(files_not_found_tmp)
        pass
    
    return records_list, files_not_found

