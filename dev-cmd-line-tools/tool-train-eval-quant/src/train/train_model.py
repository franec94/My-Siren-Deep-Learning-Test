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

# --------------------------------------------- #
# Global variables
# --------------------------------------------- #

HyperParams = collections.namedtuple('HyperParams', "n_hf,n_hl,lr,epochs,seed,sidelength,batch_size,verbose".split(","))

# ----------------------------------------------------------------------------------------------- #
# Functions
# ----------------------------------------------------------------------------------------------- #

# --------------------------------------------- #
# Local Utils
# --------------------------------------------- #

def _evaluate_dynamic_quant(opt, dtype, img_dataset, model = None, model_weight_path = None, device = 'cpu', qconfig = 'fbgemm'):
    arch_hyperparams = dict(
        hidden_layers=opt.n_hl[0],
        hidden_features=opt.n_hf[0],
        sidelength=opt.sidelength[0],
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


def _evaluate_model(model, opt, img_dataset, model_weight_path = None, logging=None, tqdm=None):

    eval_dataloader, _ = \
        _get_data_for_train(img_dataset, sidelength=opt.sidelength, batch_size=opt.batch_size)

    eval_field_names = "model_type,mse,psnr,ssim,eta,footprint_byte,footprint_percent".split(",")
    EvalInfos = collections.namedtuple("EvalInfos", eval_field_names)
    eval_info_list = []

    tot_weights_model = sum(p.numel() for p in model.parameters())
    eval_scores, eta_eval = \
        evaluate_model(
            model=model,
            eval_dataloader=eval_dataloader,
            device='cuda')
    eval_info = EvalInfos._make(['Basic'] + list(eval_scores) + [eta_eval, tot_weights_model * 4, 100.0])
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
            eval_info = EvalInfos._make([f'Quant-{str(a_dynamic_type)}'] + list(eval_scores) + [eta_eval, model_size, model_size / tot_weights_model * 4])
            eval_info_list.append(eval_info)
            pass
        pass

    table_vals = list(map(operator.methodcaller("items"), map(operator.methodcaller("_asdict"), eval_info_list)))
    table = tabulate.tabulate(table_vals, headers=eval_field_names)
    _log_infos(info_msg = f"{table}", header_msg = None, logging=logging, tqdm=tqdm)
    pass

def _get_data_for_train(img_dataset, sidelength, batch_size):
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
    try:
        model_summary_str = pms.summary(model, torch.Tensor((1, 2)).cuda(), show_input=False, show_hierarchical=True)
        # logging.info(f"{model_summary_str}"); tqdm.write(f"{model_summary_str}")    
        _log_infos(model_summary_str, logging=logging, tqdm=tqdm, verbose=verbose)
    except: pass
    pass


def _prepare_model(arch_hyperparams, device = 'cpu', empty_cache_flag = False, verbose = 0):

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
        model = model.cuda()
    elif device == 'gpu':
        model = model.to('gpu')
    
    return model


def _set_seeds(seed):
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    pass


def _log_infos(info_msg, header_msg = None, logging = None, tqdm = None, verbose = 0):
    """Log information messages to logging and tqdm objects.
    Params:
    -------
    :info_msg: either a str object or list of objects to be logged.\n
    :header_msg: str object used as header or separator, default None means no header will be shown.\n
    :logging: logging python's std lib object, if None no information will be logged via logging.\n
    :tqdm: tqdm python object, if None no information will be logged via tqdm.\n
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
    ):
    """
    Performe training on a given input model, specifing onto which device the training process will be done.
    """
    # - Set model into train mode, for safe reasons.
    model.train()

    if optim == None:
        optim = torch.optim.Adam(lr=lr, params=model.parameters())
    
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
            train_loss = loss_fn(model_output, gt)
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
        return model, model_weight_path, train_scores_path
    
    # Return model
    return model


def train_model(opt, image_dataset, model_dir = '.', save_results_flag = False):

    # ---  Setup logger.
    log_filename = os.path.join(model_dir, 'train.log')
    logging.basicConfig(filename=f'{log_filename}', level=logging.INFO)

    stop_times = []

    opt_dict = collections.OrderedDict(
        n_hf=opt.n_hf,
        n_hl=opt.n_hl,
        lr=opt.lr,
        epochs=opt.num_epochs,
        seed=opt.seed,
        sidelength=opt.sidelength,
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
                _log_infos(info_msg=sep_str_arch_no, header_msg=header_arch, logging=logging, tqdm=tqdm, verbose=1)
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
                _log_infos(info_msg = info_msg, header_msg = train_h, logging=logging, tqdm=tqdm, verbose = 1)
            else:
                info_msg = [f"[*] Train Mode: On", f"[*] Train Device: cuda", f"- Train running..."]
                _log_infos(info_msg = info_msg, header_msg = train_h, logging=logging, tqdm=tqdm, verbose = 1)
                pass
            
            # --- Set seed.
            _set_seeds(hyper_param_opt.seed)

            # --- Get ready the model for training phase.
            model = _prepare_model(
                arch_hyperparams = hyper_param_opt._asdict(),
                device = 'cuda',
                empty_cache_flag = True)
            # tot_weights_model = sum(p.numel() for p in model.parameters())

            _show_summary_model(model, logging=logging, tqdm=tqdm, verbose=1)

            # --- Get data for training.
            train_dataloader, _ = \
                _get_data_for_train(
                    img_dataset = image_dataset,
                    sidelength = hyper_param_opt.sidelength,
                    batch_size = hyper_param_opt.batch_size)

            # --- Train model, recording eta.
            start_time_to = time.time()
            model_trained, model_weight_path, train_scores_path = \
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
                    model_dir=None,
                    steps_til_summary=None,
                    epochs_til_checkpoint=None,
                    log_tensorboard_flag=True,
                    save_results_flag=save_results_flag)
            stop_time = time.time() - start_time_to
            stop_times.append(stop_time)
            _log_infos(info_msg = "- Train total time (seconds): {0:.1f}".format(stop_time), header_msg = None, logging=logging, tqdm=tqdm, verbose = 1)


            # --- Evaluate model's on validation data.
            if opt.evaluate and n > 1:
                """
                eval_h = "-" * 25 + " Eval " + "-" * 25; info_msg = [f"[*] Eval Mode: On", f"[*] Eval device: cuda"]
                _log_infos(info_msg = info_msg, header_msg = eval_h, logging=logging, tqdm=tqdm, verbose = 1)
                eval_scores, eta_eval = evaluate_model(
                    model = model_trained,
                    eval_dataloader=val_dataloader,
                    device='cuda',
                    loss_fn=nn.MSELoss(),
                    quantization_enabled=None)
                stop_times.append(eta_eval)
                info_eval_quant_stats = ["- Evaluate total time (seconds): {0:.5f}".format(eta_eval),
                                         "- Model Size (Bite): {0:.1f}".format(tot_weights_model * 4),
                                         "- arch_no=%d, loss=%0.6f, PSNR(db)=%0.6f, SSIM=%0.6f" \
                                                % (arch_no, eval_scores[0], eval_scores[1], eval_scores[2])]
                _log_infos(info_msg = info_eval_quant_stats, header_msg=None, logging=logging, tqdm=tqdm, verbose = 1)
                """
                _evaluate_model(model=model, opt=hyper_param_opt, img_dataset=image_dataset, model_weight_path = model_weight_path, logging=logging, tqdm=tqdm)
                pass

            pass # end opt_hyperparam_list loop
        pbar.update(1)
        pass # end tqdm
    
    if save_results_flag and n == 1:
        return model_trained, model_weight_path, train_scores_path
    
    return None, None, None