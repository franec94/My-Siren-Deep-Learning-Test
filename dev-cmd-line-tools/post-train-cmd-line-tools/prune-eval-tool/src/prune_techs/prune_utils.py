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
import torch.nn.utils.prune as prune
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.nn.utils.prune as prune
from torch.utils.data import DataLoader, Dataset
import torch.quantization

# --------------------------------------------- #
# Import: custom, from this project
# --------------------------------------------- #

import src.generic.dataio as dataio
from src.eval.eval_model import evaluate_model

from src.generic.dataio import Implicit2DWrapper
from src.generic.utils import prepare_model, get_data_ready_for_model, get_data_for_train
from src.generic.utils import get_size_of_model, check_device_and_weigths_to_laod, log_infos
from src.generic.custom_argparser import DYNAMIC_QUAT_SIZES

# ---------------------------------------------- #
# Util Functions
# ---------------------------------------------- #
def get_params_to_prune(model, module_set = {torch.nn.Linear}) -> tuple:
    """Get params to be pruned.\n
    Params
    ------
    `model` - torch.nn.Module, DNN architecture.\n
    `module_set` - set of torch.nn.* to be pruned.\n
    Return
    ------
    `parameters_to_prune` - tuple of pairs (torch.nn.Module, 'weight').\n
    """
    parameters_to_prune = list()
    for name, module in model.named_modules():
        for a_kind_module in module_set:
            if isinstance(module, a_kind_module):
                parameters_to_prune.append([module, 'weight'])
    parameters_to_prune = tuple(map(tuple, parameters_to_prune))
    return parameters_to_prune


def local_prunening_remove_action(model, module_set = {torch.nn.Linear}) -> tuple:
    """Apply prune.remove(<module>, <attribute name>) to proper modules within model.\n
    Params
    ------
    `model` - torch.nn.Module, DNN architecture.\n
    `module_set` - set of torch.nn.* to be pruned.\n
    Return
    ------
    `model` - torch.nn.Module, DNN architecture, updated.\n
    """
    parameters_to_prune = list()
    for name, module in model.named_modules():
        for a_kind_module in module_set:
            if isinstance(module, a_kind_module):
                prune.remove(module, 'weigth')
    return model


def show_model_sparsity(model: torch.nn.Module) -> None:
    """Show PyTorch's model sparsity.\n
    Params
    ------
    `model` - torch.nn.Module, DNN architecture.
    """
    zero_elemenets, n_elements = 0, 0
    for name, module in model.named_modules():
        if isinstance(module, torch.nn.Linear):
            print("Local sparsity({}): {:.2f}%".format(name, 100. * float(torch.sum(module.weight == 0)) / float(module.weight.nelement())))
            zero_elemenets += torch.sum(module.weight == 0)
            n_elements += module.weight.nelement()
    print("Global sparsity: {:.2f}%".format(100. * float(zero_elemenets)/ float(n_elements)))
    pass


def compute_pruning_evaluation(
    model: torch.nn.Module,
    arch_hyperparams: dict,
    image_dataset,
    amount = 0.2,
    pruning_method = prune.L1Unstructured,
    number_trials: int = 10, device: str = 'cpu') -> list:
    """Compute pruning compression technique on a given model a given number of times.
    Params
    ------
    `model` - torch.nn.Module.\n
    `arch_hyperparams` - python dictionary instance.\n
    `image_dataset` - PyTorch Dataset.\n
    `amount` - float percentage of weigths to be pruned randomly from each layer.\n
    `pruning_method` - pruning technique to be adopted.\n
    `number_trials` - number of times repeating the calculation.\n
    `device` - str = 'cpu'.\n
    Return
    ------
    `eval_info_list` - list object containing results.\n
    """
    eval_info_list = []
    name_pruning_method = str(pruning_method).split(" ")[1].split(".")[-1].replace('>', '').replace("'","")
    for trial_no in range(number_trials):
        _set_seeds(seed = trial_no)
        model_copied = copy.deepcopy(model)
        parameters_to_prune = get_params_to_prune(model_copied, module_set = {torch.nn.Linear})
        prune.global_unstructured(
            parameters_to_prune,
            pruning_method=pruning_method,
            amount=amount,
        )
        # prune.remove(module, 'weight')
        # model_copied = remove_to_prune(model = model_copied)
        if isinstance(amount, int):
            model_type = f'{name_pruning_method}_{amount:.0f}'
        else:
            model_type = f'{name_pruning_method}_{amount:.2f}'
        arch_hyperparams['model_type'] = model_type
        
        OptionModel = collections.namedtuple('OptionModel', arch_hyperparams.keys())
        opt = OptionModel._make(arch_hyperparams.values())
        res_evaluation = _evaluate_model_wrapper(
            model = model_copied,
            opt = opt,
            img_dataset = image_dataset,
            model_name = f'model.{opt.n_hf}.{opt.n_hl}.{trial_no}',
            model_weight_path = None,
            logging=None,
            tqdm=None,
            verbose=0)

        eval_info_list.extend(res_evaluation)
        pass
    return eval_info_list

# ---------------------------------------------- #
# Local Util Functions
# ---------------------------------------------- #
def _evaluate_model_wrapper(model, opt, img_dataset, model_name, model_weight_path = None, logging=None, tqdm=None, verbose=0):
    """Evaluate model after training.
    Params
    ------
    `model` - PyTorch like object representing a Neural Network model.\n
    `opt` - Namespace python like object with attributes necessary to run the evaluation tasks required.\n
    `img_dataset` - PyTorch's DataSet like object representing the data against which evaluate models(base model and quantized models, if any).\n
    `model_name` - str like object, representing a identifier with which referring to the current trial to be evaluated.\n
    `model_weight_path` - str like object representing local file path for model's weights to be exploited when evaluating quantized models.\n
    `logging` - logging python's std library object for logging reasons to a log file.\n
    `tqdm` - tqdm instance for logging data to stdout keeping order with which informations are displayed.\n
    `verbose` - int python object, for deciding verbose strategy, available options: 0 = no info displayed to tqdm, 1 = info displayed to tqdm object.\n
    Return
    ------
    `eval_info_list` - python list object containing collections.namedtuple instances with results from different evaluations.\n
    """

    eval_dataloader, _ = \
        _get_data_for_train(img_dataset, sidelength=opt.sidelength, batch_size=opt.batch_size)

    eval_field_names = "model_name,model_type,mse,psnr_db,ssim,eta_seconds,footprint_byte,footprint_percent".split(",")
    EvalInfos = collections.namedtuple("EvalInfos", eval_field_names)
    eval_info_list = []

    # tot_weights_model = sum(p.numel() for p in model.parameters())
    eval_scores, eta_eval = \
        evaluate_model(
            model=model,
            eval_dataloader=eval_dataloader,
            device='cpu')
    
    basic_model_size = _get_size_of_model(model)
    # eval_info = EvalInfos._make(['Basic'] + list(eval_scores) + [eta_eval, tot_weights_model * 4, 100.0])
    eval_info = EvalInfos._make([model_name, opt.model_type] + list(eval_scores) + [eta_eval, basic_model_size, 100.0])
    eval_info_list.append(eval_info)
    """
    if opt.dynamic_quant != []:
        for a_dynamic_type in opt.dynamic_quant:
            eval_scores, eta_eval, model_size = \
                _evaluate_dynamic_quant(
                    opt,
                    dtype=a_dynamic_type,
                    img_dataset=img_dataset,
                    model = copy.deepcopy(model),
                    model_weight_path = model_weight_path,
                    device = 'cpu',
                    qconfig = 'fbgemm')
            eval_info = EvalInfos._make([model_name, f'Quant-{str(a_dynamic_type)}'] + list(eval_scores) + [eta_eval, model_size, model_size / basic_model_size * 100])
            eval_info_list.append(eval_info)
            pass
        pass
    
    table_vals = list(map(operator.methodcaller("values"), map(operator.methodcaller("_asdict"), eval_info_list)))
    table = tabulate.tabulate(table_vals, headers=eval_field_names)
    _log_infos(info_msg = f"{table}", header_msg = None, logging=logging, tqdm=tqdm, verbose=verbose)
    """
    return eval_info_list


def _read_csv_data(a_file_csv):
    """Read input csv data.
    Param
    -----
    `a_file_csv` - str object python, input csv file path.\n
    Return
    ------
    `cropped_images_df` - pd.DatFrame.\n
    """

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


def _set_seeds(seed):
    """Set seeds for torch, np.random and random std python library.
    Params
    ------
    `seed` - int object, seed for starting pseudo-random series.\n
    """
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    pass


def _get_size_of_model(model):
    """Return model size as file size corresponding to model's state dictionary when saved temporarily to 
    disk.
    Params
    ------
    `model` - PyTorch like model.\n
    Return
    ------
    `model_size` - int python object, size of state dictionary expressed in byte.\n
    """
    torch.save(model.state_dict(), "temp.p")
    # print('Size (MB):', os.path.getsize("temp.p")/1e6)
    model_size = os.path.getsize("temp.p")
    os.remove('temp.p')
    return model_size


def _get_data_ready_for_model(model_input, gt, quantization_enabled = None, device = 'cpu'):
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


def _get_data_for_train(img_dataset, sidelength, batch_size):
    """Get data ready to be feed into a DNN model as input data for training and evaluating phase, respectively.
    Params
    ------
    `img_dataset` - PyTorch's DataSet like object representing the data against which evaluate models(base model and quantized models, if any).\n
    `sidelength` - eithr int object or lsit,tuple, representing width and height for center cropping input image.\n
    `batch_size` - int object for dividing input data into several batches.\n
    Return
    ------
    `train_dataloader` - PyTorch DataLoader instance.\n
    `val_dataloader` - PyTorch DataLoader instance.\n
    """
    coord_dataset = Implicit2DWrapper(
        img_dataset, sidelength=sidelength, compute_diff=None)

    # --- Prepare dataloaders for train and eval phases.
    train_dataloader = DataLoader(
        coord_dataset,
        shuffle=True,
        batch_size=batch_size,
        pin_memory=True, num_workers=0)

    val_dataloader = DataLoader(
        coord_dataset,
        shuffle=False,
        batch_size=batch_size,
        pin_memory=True, num_workers=0)
    
    return train_dataloader, val_dataloader


# ---------------------------------------------- #
# Main Functions
# ---------------------------------------------- #
def compute_prune_unstructured_results(opt, image_dataset, verbose = 0):
    """
    Compute pruning technique following Unstructured approach.\n
    Params
    ------
    `opt` - namespace python object.\n
    `image_dataset` - PyTorch Dataset.\n
    `verbose` - int python object, for deciding verbose strategy, available options: 0 = no info displayed to tqdm, 1 = info displayed to tqdm object.\n
    Return
    ------
    `eval_info_list` - list python object.\n
    `df` - pd.DataFrame.\n
    """
    opt_dict = collections.OrderedDict(
        n_hf=opt.n_hf,
        n_hl=opt.n_hl,
        # lr=opt.lr,
        # epochs=opt.num_epochs,
        # seed=opt.seed,
        dynamic_quant=[opt.dynamic_quant],
        sidelength=opt.sidelength,
        batch_size=opt.batch_size,
        verbose=[opt.verbose]
    )
    
    eval_info_list = []
    if len(opt.models_filepath) == 0: eval_info_list, None
    
    opt_hyperparm_list = list(ParameterGrid(opt_dict))
    n = len(opt_hyperparm_list)
    
    HyperParams = collections.namedtuple('HyperParams', "n_hf,n_hl,dynamic_quant,sidelength,batch_size,verbose".split(","))
    eval_field_names = "model_name,model_type,mse,psnr_db,ssim,eta_seconds,footprint_byte,footprint_percent".split(",")

    n = n * (len(opt.global_pruning_rates) * len(opt.global_pruning_techs) + len(opt.global_pruning_abs) * len(opt.global_pruning_techs))
    with tqdm(total=n) as pbar:
        for arch_no, hyper_param_dict in enumerate(opt_hyperparm_list):
            # --- Get hyperparams as Namedtuple
            hyper_param_list = []
            for a_key in opt_dict.keys():
                hyper_param_list.append(hyper_param_dict[f'{a_key}'])
            hyper_param_opt = HyperParams._make(hyper_param_list)
            
            model = prepare_model(arch_hyperparams=hyper_param_dict, device='cpu')
            model = check_device_and_weigths_to_laod(model_fp32=model, device='cpu', model_path=opt.models_filepath[arch_no])

            log_infos(info_msg = 'global_pruning_rates evalauting...', header_msg = None, logging = None, tqdm = tqdm, verbose = 1)
            for a_rate in opt.global_pruning_rates:
                for a_prune_tech in opt.global_pruning_techs:
                    # log_infos(info_msg = 'global_pruning_techs evalauting...', header_msg = None, logging = None, tqdm = tqdm, verbose = 1)
                    tmp_res = compute_pruning_evaluation(
                        model=copy.deepcopy(model),
                        amount=a_rate,
                        number_trials = opt.global_pruning_number_trials,
                        pruning_method=copy.deepcopy(a_prune_tech),
                        arch_hyperparams=hyper_param_dict,
                        image_dataset=image_dataset,

                    )
                    eval_info_list.extend(tmp_res)
                    pbar.update(len(opt.global_pruning_techs))
                    pass
            log_infos(info_msg = 'global_pruning_abs evalauting...', header_msg = None, logging = None, tqdm = tqdm, verbose = 1)
            for a_rate in opt.global_pruning_abs:
                for a_prune_tech in opt.global_pruning_techs:
                    # log_infos(info_msg = 'global_pruning_rates evalauting...', header_msg = None, logging = None, tqdm = tqdm, verbose = 1)
                    tmp_res = compute_pruning_evaluation(
                        model=copy.deepcopy(model),
                        amount=a_rate,
                        number_trials = opt.global_pruning_number_trials,
                        pruning_method=copy.deepcopy(a_prune_tech),
                        arch_hyperparams=hyper_param_dict,
                        image_dataset=image_dataset,

                    )
                    eval_info_list.extend(tmp_res)
                    pbar.update(len(opt.global_pruning_techs))
                    pass
            pass
        pass

    if eval_info_list == []: return eval_info_list, None

    data = list(map(operator.methodcaller("_asdict"), eval_info_list))
    df = pd.DataFrame(data = data)

    def model_size_to_bpp(model_footprint, w = 256, h = 256):
        return model_footprint * 4 / (w * h)
    df['bpp'] = list(map(model_size_to_bpp, df['footprint_byte'].values))

    def model_type_to_quant_tech(model_type):
        return model_type.split("_")[0]
    df['quant_tech'] = list(map(model_type_to_quant_tech, df['model_type'].values))

    def model_type_to_quant_tech_2(model_type):
        if model_type == 'Basic': return model_type
        quant_tech_2 = model_type.split("_")[0]
        value = int(float(model_type.split("_")[1]))
        if value != 0:
            return quant_tech_2 + "_" + "abs"
        return quant_tech_2 + "_" + "rate"
    df['quant_tech_2'] = list(map(model_type_to_quant_tech_2, df['model_type'].values))

    def model_type_to_prune_amount(model_type):
        if model_type == 'Basic':
            return np.nan
        return float(model_type.split("_")[1])
    df['prune_amount'] = list(map(model_type_to_prune_amount, df['model_type'].values))

    df = df.drop(["model_type"], axis = 1)
    return eval_info_list, df


def compute_prune_unstructured_results_from_csv_list(opt, image_dataset, verbose = 0):
    """
    Compute pruning technique following Unstructured approach.\n
    Params
    ------
    `opt` - namespace python object.\n
    `image_dataset` - PyTorch Dataset.\n
    `verbose` - int python object, for deciding verbose strategy, available options: 0 = no info displayed to tqdm, 1 = info displayed to tqdm object.\n
    Return
    ------
    `eval_info_list` - list python object.\n
    `df` - pd.DataFrame.\n
    """

    eval_info_list = []
    opt_copy = copy.deepcopy(opt)
    for a_file_csv in opt.csv_files:
        df_image = _read_csv_data(a_file_csv=a_file_csv)
        Columns = collections.namedtuple('Columns', df_image.columns)
        for a_row in df_image.values[:1]:
            pprint(a_row)
            a_row_rcrd = Columns._make(a_row)
            opt_copy.n_hf = [int(a_row_rcrd.hf)]
            opt_copy.n_hl = [int(a_row_rcrd.hl)]
            eval_info_list_tmp, _ = compute_prune_unstructured_results(opt, image_dataset = image_dataset, verbose = 0)
            eval_info_list.extend(eval_info_list_tmp)
            pass
        pass

    data = list(map(operator.methodcaller("_asdict"), eval_info_list))
    df = pd.DataFrame(data = data)

    print(df.head(5))

    def model_size_to_bpp(model_footprint, w = 256, h = 256):
        return model_footprint * 4 / (w * h)
    df['bpp'] = list(map(model_size_to_bpp, df['footprint_byte'].values))

    def model_type_to_quant_tech(model_type):
        return model_type.split("_")[0]
    df['quant_tech'] = list(map(model_type_to_quant_tech, df['model_type'].values))

    def model_type_to_quant_tech_2(model_type):
        if model_type == 'Basic': return model_type
        quant_tech_2 = model_type.split("_")[0]
        value = int(float(model_type.split("_")[1]))
        if value != 0:
            return quant_tech_2 + "_" + "abs"
        return quant_tech_2 + "_" + "rate"
    df['quant_tech_2'] = list(map(model_type_to_quant_tech_2, df['model_type'].values))

    def model_type_to_prune_amount(model_type):
        if model_type == 'Basic':
            return np.nan
        return float(model_type.split("_")[1])
    df['prune_amount'] = list(map(model_type_to_prune_amount, df['model_type'].values))

    df = df.drop(["model_type"], axis = 1)

    return eval_info_list, df