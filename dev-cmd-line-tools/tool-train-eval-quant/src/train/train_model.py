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

import copy
import collections
import logging
import os
import operator
import random
import math
import shutil
import tabulate
import time

import pytorch_model_summary as pms

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

from sklearn.model_selection import ParameterGrid

# --------------------------------------------- #
# Custom Imports
# --------------------------------------------- #

from src.generic.utils import get_data_ready_for_model, compute_desired_metrices, save_data_to_file, cond_mkdir
from src.eval.eval_model import evaluate_model
from src.generic.dataio import Implicit2DWrapper
from src.archs.siren_compute_quantization import compute_quantization_dyanmic_mode
from src.archs.siren import Siren
from src.generic.functions import log_data_via_pickle

# --------------------------------------------- #
# Global variables
# --------------------------------------------- #

HyperParams = collections.namedtuple('HyperParams', "n_hf,n_hl,lr,epochs,seed,dynamic_quant,sidelength,lambda_L_1,weight_decay,batch_size,verbose".split(","))

# ----------------------------------------------------------------------------------------------- #
# Functions
# ----------------------------------------------------------------------------------------------- #

# --------------------------------------------- #
# Local Utils
# --------------------------------------------- #

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


def _evaluate_dynamic_quant(opt, dtype, img_dataset, model = None, model_weight_path = None, device = 'cpu', qconfig = 'fbgemm'):
    """
    Evaluate model exploiting PyTorch built-it dynamic quant mode, a.k.a. Post training dynamic quantization mode.
    
    Params
    ------
    `opt` - Namespace python like object with attributes necessary to run the evaluation tasks required.\n
    `dtype` - either torch.qint8 or torch.qfloat16 instances, for quantizing model's weigths.\n
    `img_dataset` - PyTorch's DataSet like object representing the data against which evaluate models(base model and quantized models, if any).\n
    `model` - PyTorch like object representing a Neural Network model.\n
    `model_weight_path` - str like object representing local file path for model's weights to be exploited when evaluating quantized models.\n
    `device` - str object, kind of device upon which model will be loaded, allowed only CPU, since PyTorch framework supports just that setup, by now.\n
    `qconfig` - str object, quantization backed type, allowed fbgemm for x86 server architectures, or QNNAM for mobile architectures.\n
    Return
    ------
    `eval_scores, eta_eval, size_model` - np.ndarray object with values related to the following scores: MSE, PSNR, MSSi.\n
    `eta_eval` - python float object, representig time elapsed when evaluation was carried out, expressed in seconds.\n
    `size_model` - python int object representing model' size expressed in Bytes.\n
    """

    arch_hyperparams = dict(
        hidden_layers=opt.n_hl,
        hidden_features=opt.n_hf,
        sidelength=opt.sidelength,
        dtype=dtype
    )
    eval_scores, eta_eval, size_model = \
        compute_quantization_dyanmic_mode(
                model_path = model_weight_path,
                arch_hyperparams = arch_hyperparams,
                img_dataset = img_dataset,
                opt = opt,
                fuse_modules = None,
                device = f'{device}',
                qconfig = f'{qconfig}',
                model_fp32 = model)

    return eval_scores, eta_eval, size_model


def _evaluate_model(model, opt, img_dataset, model_name, model_weight_path = None, logging=None, tqdm=None, verbose=0):
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
            device='cuda')
    
    basic_model_size = _get_size_of_model(model)
    # eval_info = EvalInfos._make(['Basic'] + list(eval_scores) + [eta_eval, tot_weights_model * 4, 100.0])
    eval_info = EvalInfos._make([model_name, 'Basic'] + list(eval_scores) + [eta_eval, basic_model_size, 100.0])
    eval_info_list.append(eval_info)

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
    return eval_info_list


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


def _show_summary_model(model, logging=None, tqdm=None, verbose = 0):
    """Log model's architecture detail.
    Params
    ------
    `logging` - logging python's std library object for logging reasons to a log file.\n
    `tqdm` - tqdm instance for logging data to stdout keeping order with which informations are displayed.\n
    `verbose` - int python object, for deciding verbose strategy, available options: 0 = no info displayed to tqdm, 1 = info displayed to tqdm object.\n
    """
    try:
        model_summary_str = pms.summary(model, torch.Tensor((1, 2)).cuda(), show_input=False, show_hierarchical=True)
        # logging.info(f"{model_summary_str}"); tqdm.write(f"{model_summary_str}")    
        _log_infos(model_summary_str, logging=logging, tqdm=tqdm, verbose=verbose)
    except: pass
    pass


def _prepare_model(arch_hyperparams, device = 'cpu', empty_cache_flag = False, verbose = 0):
    """Prepare plain siren model.
    Params
    ------
    `arch_hyperparams` - python dictionary object, containing model's hyper-params with which build the final architecture.\n
    `device` - str object, kind of device upon which model will be loaded, allowed CPU, GPU, CUDA.\n
    `empty_cache_flag` - bool python object, if true function attempts to free cache from previous runs.\n
    `verbose` - int python object, for deciding verbose strategy, available options: 0 = no info displayed to tqdm, 1 = info displayed to tqdm object.\n
    Return
    ------
    `model` - PyTorch like object representing DNN architecture.\n
    """

    if device != 'cpu' and device != 'gpu':
        if empty_cache_flag:
            torch.cuda.empty_cache()
        pass
    model = Siren(
        in_features=2,
        out_features=1,
        hidden_features=int(arch_hyperparams['n_hf']),
        hidden_layers=int(arch_hyperparams['n_hl']),
        outermost_linear=True)
    if device == 'cpu':
        model = model.to('cpu')
    elif device == 'cuda':
        try:
            model = model.cuda()
        except:
            model = model.to('cpu')
    elif device == 'gpu':
        try:
            model = model.to('gpu')
        except:
            model = model.to('cpu')
            pass
        pass
    return model


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


def _log_infos(info_msg, header_msg = None, logging = None, tqdm = None, verbose = 0):
    """Log information messages to logging and tqdm objects.
    Params:
    -------
    `info_msg` either a str object or list of objects to be logged.\n
    `header_msg` str object used as header or separator, default None means no header will be shown.\n
    `logging` logging python's std lib object, if None no information will be logged via logging.\n
    `tqdm` tqdm python object, if None no information will be logged via tqdm.\n
    Return:
    -------
    None
    """

    if not isinstance(info_msg , list) :
        info_msg = [info_msg]
        pass
    
    if logging != None:
        if header_msg != None:
            logging.info(f"{header_msg}")
            pass
        for a_msg in info_msg:
            logging.info(f"{a_msg}")
            pass
        pass
    if tqdm != None:
        if verbose == 0: return
        if header_msg != None:
            tqdm.write(f"{header_msg}")
            pass
        for a_msg in info_msg:
            tqdm.write(f"{a_msg}")
            pass
        pass
    pass


# --------------------------------------------- #
# Train functions
# --------------------------------------------- #

def _train_loop(
    model, train_dataloader,
    epochs,
    lr,
    optim=None,
    loss_fn=nn.MSELoss(),
    clip_grad=False,
    device='cpu',
    data_range = 255,
    model_dir=None,
    calc_metrices = False,
    steps_til_summary=None,
    epochs_til_checkpoint=None,
    log_tensorboard_flag=False,
    save_results_flag=False,
    weight_decay = 0.0,
    lambda_L_1 = None,
    ):
    """
    Performe training on a given input model, specifing onto which device the training process will be done.
    """
    # - Set model into train mode, for safe reasons.
    model.train()

    if optim == None:
        optim = torch.optim.Adam(lr=lr, params=model.parameters(), weight_decay = weight_decay)
    
    # - Local variable.
    train_scores = []
    writer_tb = None

    # - Root dir for current attempt of a given arch.
    if save_results_flag:
        cond_mkdir(model_dir)

        # Where to collect checkpoints.
        checkpoints_dir = os.path.join(model_dir, 'checkpoints')
        cond_mkdir(checkpoints_dir)

        # Where to store results for tensorboard.
        if log_tensorboard_flag:
            tensorboard_dir = os.path.join(model_dir, 'summary-tensorboard')
            cond_mkdir(tensorboard_dir)
            writer_tb = SummaryWriter(tensorboard_dir)
            pass
        pass

    # Train model on input data
    for _, (model_input, gt) in enumerate(train_dataloader):
        # Get data to be used for feeding in model during training.
        model_input, gt = \
            get_data_ready_for_model(model_input, gt, device = device)
        for epoch in range(epochs):
            optim.zero_grad()

            
            model_output, _ = model(model_input)
            if lambda_L_1 != 0:
                regularization_loss = 0
                for param in model.parameters():
                    regularization_loss += torch.sum(torch.abs(param))
                train_loss, _ = loss_fn(model_output, gt) + lambda_L_1 * regularization_loss
            else:
                train_loss = loss_fn(model_output, gt)    
                pass
            
            if calc_metrices:    
                val_psnr, val_mssim = compute_desired_metrices(model_output, gt)
                train_scores.append([train_loss.item(), val_psnr, val_mssim])
                """
                tqdm.write("Epoch %d loss=%0.6f, PSNR=%0.6f, SSIM=%0.6f, iteration time=%0.6f"
                        % (epoch, train_scores[0], train_scores[1], train_scores[2], stop_time))
                """
                if save_results_flag and log_tensorboard_flag:
                    writer_tb.add_scalar('train_mse', train_loss.item(), epoch)
                    writer_tb.add_scalar('train_psnr',val_psnr, epoch)
                    writer_tb.add_scalar('train_ssim', val_mssim, epoch)
                    pass
            else:
                train_scores.append(train_loss.item())
                # tqdm.write("Epoch %d loss=%0.6f" % (epoch,train_loss.item()))
                if save_results_flag and log_tensorboard_flag:
                    writer_tb.add_scalar('train_mse', train_loss.item(), epoch)
                    pass
                pass

            # Backward pass.
            train_loss.backward()
            if clip_grad:
                if isinstance(clip_grad, bool):
                    torch.nn.utils.clip_grad_norm_(
                        model.parameters(), max_norm=1.)
                else:
                    torch.nn.utils.clip_grad_norm_(
                        model.parameters(), max_norm=clip_grad)
                    pass
                pass
            optim.step()

            pass # end inner loop
        pass # end outer loop
    
    # Return 'model', and 'model_weight_path', 'train_scores_path'
    # after data have been saved, if 'save_results_flag' results is true
    if save_results_flag:
        model_weight_path, train_scores_path = \
            save_data_to_file(root_dir = checkpoints_dir, model = model, train_scores = train_scores)
        return model, model_dir, model_weight_path, train_scores_path
    
    # Return model
    return model


def train_model(opt, image_dataset, model_dir = '.', save_results_flag = False):

    # ---  Setup logger.
    log_filename = os.path.join(model_dir, 'train.log')
    logging.basicConfig(filename=f'{log_filename}', level=logging.INFO)

    stop_times = []
    eval_results_list = []

    opt_dict = collections.OrderedDict(
        n_hf=opt.n_hf,
        n_hl=opt.n_hl,
        lr=opt.lr,
        epochs=opt.num_epochs,
        seed=opt.seed,
        dynamic_quant=[opt.dynamic_quant],
        sidelength=opt.sidelength,
        lambda_L_1=opt.lambda_L_1,
        weight_decay=opt.lambda_L_2,
        batch_size=opt.batch_size,

        verbose=[opt.verbose]
    )
    opt_hyperparm_list = list(ParameterGrid(opt_dict))

    # -- Processing Bar to control the workout.
    n = len(opt_hyperparm_list)
    with tqdm(total=n) as pbar:
        for arch_no, hyper_param_dict in enumerate(opt_hyperparm_list):

            if opt.evaluate and n > 1:
                sep_str_arch_no = "=" * 50 + f" ARCH {arch_no} " + "=" * 50
                header_arch = '_' * len(sep_str_arch_no)
                _log_infos(info_msg=sep_str_arch_no, header_msg=header_arch, logging=logging, tqdm=tqdm, verbose=opt.verbose)
                pass

            # --- Get hyperparams as Namedtuple
            hyper_param_list = []
            for a_key in opt_dict.keys():
                hyper_param_list.append(hyper_param_dict[f'{a_key}'])
            hyper_param_opt = HyperParams._make(hyper_param_list)

            # --- Show some infos from main function.
            table_vals = list(hyper_param_opt._asdict().items())
            table = tabulate.tabulate(table_vals, headers="Hyper-param,Value".split(","))
            _log_infos(info_msg = f"{table}", header_msg = f'{"-" * 25} Model Details {"-" * 25}', logging=logging, tqdm=tqdm, verbose=opt.verbose)

            train_h = "-" * 25 + " Train " + "-" * 25
            if opt.evaluate and n > 1:
                info_msg = [f"[*] Train Mode: On", f"[*] Train Device: cuda", f"- Arch no: {arch_no} running..."]
                _log_infos(info_msg = info_msg, header_msg = train_h, logging=logging, tqdm=tqdm, verbose=opt.verbose)
            else:
                info_msg = [f"[*] Train Mode: On", f"[*] Train Device: cuda", f"- Train running..."]
                _log_infos(info_msg = info_msg, header_msg = train_h, logging=logging, tqdm=tqdm, verbose=opt.verbose)
                pass
            
            # --- Set seed.
            _set_seeds(hyper_param_opt.seed)

            # --- Get ready the model for training phase.
            model = _prepare_model(
                arch_hyperparams = hyper_param_opt._asdict(),
                device = 'cuda',
                empty_cache_flag = True)
            # tot_weights_model = sum(p.numel() for p in model.parameters())

            _show_summary_model(model, logging=logging, tqdm=tqdm, verbose=opt.verbose)

            # --- Get data for training.
            train_dataloader, _ = \
                _get_data_for_train(
                    img_dataset = image_dataset,
                    sidelength = hyper_param_opt.sidelength,
                    batch_size = hyper_param_opt.batch_size)

            # --- Train model, recording eta.
            # tmp_model_dir = os.path.join(model_dir, f"arch_no_{arch_no_tmp}", f"trial_no_{trial_no}")
            tmp_model_dir = os.path.join(model_dir, f"arch_no_{arch_no}")

            start_time_to = time.time()
            model, model_dir_train, model_weight_path, train_scores_path = \
                _train_loop(
                    model=model,
                    train_dataloader=train_dataloader,
                    epochs=hyper_param_opt.epochs,
                    lr=hyper_param_opt.lr,
                    loss_fn=nn.MSELoss(),
                    clip_grad=False,
                    device='cuda',
                    data_range=1.0,
                    calc_metrices=True,
                    model_dir=tmp_model_dir,
                    steps_til_summary=None,
                    epochs_til_checkpoint=None,
                    log_tensorboard_flag=True,
                    lambda_L_1=hyper_param_opt.lambda_L_1,
                    weight_decay=hyper_param_opt.weight_decay,
                    save_results_flag=save_results_flag)
            stop_time = time.time() - start_time_to
            stop_times.append(stop_time)
            log_data_via_pickle(hyper_param_opt, model_dir_train, 'hyper-params-model')
            _log_infos(info_msg = "- Train total time (seconds): {0:.1f}".format(stop_time), header_msg = None, logging=logging, tqdm=tqdm, verbose=opt.verbose)


            # --- Evaluate model's on validation data.
            if opt.evaluate and n > 1:
                eval_h = "-" * 25 + " Eval " + "-" * 25; info_msg = [f"[*] Eval Mode: On", f"[*] Eval devices: CUDA(Basic) | CPU(Quantized)"]
                _log_infos(info_msg = info_msg, header_msg=eval_h, logging=logging, tqdm=tqdm, verbose = opt.verbose)
                
                model_name = f"arch-{arch_no}.{hyper_param_opt.n_hf}.{hyper_param_opt.n_hl}.{hyper_param_opt.seed}.{hyper_param_opt.sidelength}"
                eval_info_list = \
                    _evaluate_model(model=model, model_name = f'{model_name}', opt=hyper_param_opt, img_dataset=image_dataset, model_weight_path = model_weight_path, logging=logging, tqdm=tqdm, verbose = opt.verbose)
                eval_results_list.extend(eval_info_list)
                pass

            pass # end opt_hyperparam_list loop
        pbar.update(1)
        pass # end tqdm
    
    if save_results_flag and n == 1:
        return model, model_weight_path, train_scores_path
    
    return eval_results_list
