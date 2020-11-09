'''Implements a generic training loop for comparing different architectures.
'''
from __future__ import print_function
from __future__ import division


# --------------------------------------------- #
# Globals
# --------------------------------------------- #

FILE_PATH = None

# --------------------------------------------- #
# Standard Library | Third Party Libraries
# --------------------------------------------- #
from torch.utils.data import DataLoader, Dataset
from torch.utils.tensorboard import SummaryWriter
from tqdm.autonotebook import tqdm

import collections
import logging
import os
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
# Custom Imports
# --------------------------------------------- #
import src.utils.generic_src.utils as utils
import src.utils.generic_src.dataio as dataio
from src.utils.archs.siren import Siren

import pytorch_model_summary as pms

from src.utils.train_functions.training_utils_funcs import get_data_ready_for_model, compute_desired_metrices, save_data_to_file
from src.utils.eval_functions.eval_model import evaluate_model

from src.utils.archs.siren_dynamic_quantization import prepare_model, compute_quantization
from src.utils.archs.siren_dynamic_quantization import get_dynamic_quantization_model, get_static_quantization_model
from src.utils.archs.siren_dynamic_quantization import get_post_training_quantization_model, get_quantization_aware_training

# ----------------------------------------------------------------------------------------------- #
# Functions
# ----------------------------------------------------------------------------------------------- #

# --------------------------------------------- #
# Local Utils
# --------------------------------------------- #

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


def update_avg_stats_arrays(avg_train_scores, global_avg_train_losses, eval_scores):
    """Updated arrays used to show avg stats
    Params:
    -------
    :avg_train_scores: np.ndarray or None, dest of eval_scores data for computing current arch stasts.\n
    :global_avg_train_losses: np.ndarray or None, dest of eval_scores data for computing global stasts.\n
    :eval_scores: np.ndarray like object containing new data to be added for later stats evaluating.\n
    Return:
    -------
    :avg_train_scores, global_avg_train_losses:
    """
    if avg_train_scores is None:
        avg_train_scores =  np.array([eval_scores])
    else:
        avg_train_scores = np.concatenate((avg_train_scores, [eval_scores]), axis=0)
        pass
    
    if global_avg_train_losses is None:
        global_avg_train_losses = np.array([eval_scores])
    else:
        global_avg_train_losses = np.concatenate((global_avg_train_losses, [eval_scores]), axis=0)
    return avg_train_scores, global_avg_train_losses


def _show_table_info_curr_trial(record_info, headers="Info,Detail".split(","), header_msg = None, logging = None, tqdm = None, verbose = 0):
    """Show Current Infos about trial that will be carryed out."""
    table_vals = list(record_info._asdict().items())
    # table = tabulate.tabulate(table_vals, headers="Info,Detail".split(","))
    table = tabulate.tabulate(table_vals, headers=headers)
    # tqdm.write(f"{table}")
    # logging.info(f"{table}")
    _log_infos(info_msg = f"{table}", header_msg = None, logging = logging, tqdm = tqdm, verbose = verbose)
    pass


# --------------------------------------------- #
# Train Protocol Definition
# --------------------------------------------- #


def train_extended_compare_loop(
    model, train_dataloader,
    epochs, lr,
    opt,
    steps_til_summary=None,
    epochs_til_checkpoint=None,
    model_dir=None,
    loss_fn=None,
    summary_fn=None,
    double_precision=False,
    clip_grad=False,
    use_lbfgs=False,
    loss_schedules=None,
    device='cpu',
    data_range = 255,
    log_for_tensorboard=False,
    save_metrices = False):
    """
    Performe training on a given input model, specifing onto which device the training process will be done.
    """

    # --- Set model into train mode, for safe reasons.
    model.train()

    # --- Define some objects necessary to lead the protocol
    # which is in charge of let the model to be trained.
    optim = torch.optim.Adam(lr=lr, params=model.parameters())
    loss_fn = nn.MSELoss()

    # --- Local variables.
    train_scores = []  # used for recording metrices when evaluated.
    writer_tb = None

    # --- Arch's attempt summary dirs.
    # Root dir for current attempt of a given arch.
    try: os.makedirs(model_dir)
    except: pass

    """
    summaries_dir = os.path.join(model_dir, 'summaries')
    utils.cond_mkdir(summaries_dir)
    """
    # Where to collect checkpoints.
    checkpoints_dir = os.path.join(model_dir, 'checkpoints')
    utils.cond_mkdir(checkpoints_dir)

    # Where to store results for tensorboard.
    
    if log_for_tensorboard:
        tensorboard_dir = os.path.join(model_dir, 'summary-tensorboard')
        utils.cond_mkdir(tensorboard_dir)
        writer_tb = SummaryWriter(tensorboard_dir)
        pass

    # --- Number of interation for current image.
    for _, (model_input, gt) in enumerate(train_dataloader):
        # --- Loop for let model's arch be improved, updateding weights values.

        model_input, gt = \
            get_data_ready_for_model(model_input, gt, device = device)


        for epoch in range(epochs):
            # --- Compute forward pass.
            optim.zero_grad()

            model_output, _ = model(model_input)
            train_loss = loss_fn(model_output, gt)

            if save_metrices:
                
                val_psnr, val_mssim = compute_desired_metrices(model_output, gt)
                train_scores.append([train_loss.item(), val_psnr, val_mssim])
                """
                tqdm.write(
                    "Epoch %d loss=%0.6f, PSNR=%0.6f, SSIM=%0.6f, iteration time=%0.6f"
                        % (epoch, train_scores[0], train_scores[1], train_scores[2], stop_time))
                """
                if log_for_tensorboard:
                    writer_tb.add_scalar('train_mse', train_loss.item(), epoch)
                    writer_tb.add_scalar('train_psnr',val_psnr, epoch)
                    writer_tb.add_scalar('train_ssim', val_mssim, epoch)
                    pass
            else:
                train_scores.append(train_loss.item())
                if log_for_tensorboard:
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

    # --- Save overall training results.
    FILE_PATH = os.path.join(checkpoints_dir, 'train_losses_final.txt')
    save_data_to_file(root_dir = checkpoints_dir, model = model, train_scores = train_scores)

    # Return best metrices.
    return model


def train_extended_protocol_compare_archs(grid_arch_hyperparams, img_dataset, opt, model_dir = None, loss_fn=nn.MSELoss(), summary_fn=None, device = 'cpu', steps_til_summary = 1, verbose = 0, save_metrices = False, data_range = 255):
    """
    Protocol set to collect data about different hyper-params combination done between number of hidden features and number of hidden layers.
    """

    # --- Local variables.
    fields_info_models = 'Model_Type,Device_Train,Device_Quant_Eval,Arch_No,Trial_No,Hidden_Features,Hidden_Layers,Seed,No_Weights,Size_Bits'.split(",")
    SomeInfosModel = collections.namedtuple('SomeInfosModel', fields_info_models) # Variable used for storing and displaying infos during training.

    writer_tb = None    # Variable for logging data for displaying them later via Tensorboard.
    history_combs = []  # Variable for recording combinations explored.
    step = 1            # Variable for keep trace of which step we are in.
    arch_step = 0       # Variable for displaying which arch wre are considering.
    is_first_arch = True

    # Variables for most inner loop, here.
    model, global_avg_train_losses = None, None
    train_dataloader, val_dataloader = None, None
    
    # ---  Setup logger.
    log_filename = os.path.join(model_dir, 'train.log')
    logging.basicConfig(filename=f'{log_filename}', level=logging.INFO)

    if opt.enable_tensorboard_logging:
        tensorboard_dir = os.path.join(model_dir, 'summary-avg-tensorboard')
        utils.cond_mkdir(tensorboard_dir)
        writer_tb = SummaryWriter(tensorboard_dir)
        pass

    # --- Processing Bar to control the workout.
    with tqdm(total=len(grid_arch_hyperparams)) as pbar:

        # --- For loop for performing different training depending on the
        # chosen hyper-params.
        # print()
        for arch_no, arch_hyperparams in enumerate(grid_arch_hyperparams):
            # --- Start time: it's the point in time from which the current train
            # begins, when new hyper-params are selected and evaluted in terms of performances.
            if verbose >= 1:
                start_time_ao = time.time()
                pass

            sep_str_arch_no = "=" * 25 + f" ARCH {arch_no + opt.resume_from} " + "=" * 25
            header_arch = '_' * len(sep_str_arch_no)
            _log_infos(info_msg=sep_str_arch_no, header_msg=header_arch, logging=logging, tqdm=tqdm, verbose=1)

            # --- Rescale image to be correctly processed by the net.
            # sidelength = int(arch_hyperparams['hidden_features'])
            coord_dataset = dataio.Implicit2DWrapper(
                img_dataset, sidelength=opt.sidelength, compute_diff=None)

            # --- Prepare dataloaders for train and eval phases.
            train_dataloader = DataLoader(
                coord_dataset,
                shuffle=True,
                batch_size=opt.batch_size,
                pin_memory=True, num_workers=0)

            val_dataloader = DataLoader(
                coord_dataset,
                shuffle=False,
                batch_size=opt.batch_size,
                pin_memory=True, num_workers=0)

            # --- Train with the same configuration n-times (at least one).
            # logging.info("-" * 50); tqdm.write("-" * 50)
            seed = int(arch_hyperparams['seeds'])
            avg_train_scores = None
            stop_times = []
            for trial_no in range(opt.num_attempts):
                arch_no_tmp = arch_no + opt.resume_from
                trial_specifics = [arch_no_tmp, trial_no, arch_hyperparams['hidden_features'], arch_hyperparams['hidden_layers'], seed]

                # --- Create dir for record results for current trial.
                tmp_model_dir = os.path.join(model_dir, f"arch_no_{arch_no_tmp}", f"trial_no_{trial_no}")
                try: os.makedirs(tmp_model_dir)
                except: pass
                
                # --- Set seeds
                torch.manual_seed(seed)
                np.random.seed(seed)
                random.seed(seed)

                # --- Prepare siren model.
                device_tmp = 'cuda'
                if device_tmp != 'cpu' and device_tmp != 'gpu':
                    torch.cuda.empty_cache()
                    pass
                model = prepare_model(opt, arch_hyperparams = arch_hyperparams, device = f'{device_tmp}')

                # print(model)
                tot_weights_model = sum(p.numel() for p in model.parameters())
                trial_specifics.append(tot_weights_model)

                # --- Show infos about model to be tested.
                record_info = SomeInfosModel._make([
                    str(model).split('(')[0],
                    'cuda', 'cpu',
                    arch_no + opt.resume_from, trial_no,
                    arch_hyperparams['hidden_features'], arch_hyperparams['hidden_layers'],
                    seed, tot_weights_model, tot_weights_model*32])
                _show_table_info_curr_trial(record_info, headers="Info,Detail".split(","), header_msg = None, logging = logging, tqdm = tqdm, verbose = 1)

                # --- Show model's architecture and more details.
                # if device != 'cpu' and device != 'gpu':
                try:
                    model_summary_str = pms.summary(model, torch.Tensor((1, 2)).cuda(), show_input=False, show_hierarchical=True)
                    logging.info(f"{model_summary_str}"); tqdm.write(f"{model_summary_str}")
                    # pass
                except: pass

                # --- Train model.
                train_h = "-" * 25 + " Train " + "-" * 25
                info_msg = [f"[*] Train Mode: On", f"[*] Train Device: cuda", f"- Arch no: {arch_no + opt.resume_from} | Trial No: ({trial_no+1}/{opt.num_attempts}) running..."]
                _log_infos(info_msg = info_msg, header_msg = train_h, logging=logging, tqdm=tqdm, verbose = 1)
                
                start_time_to = time.time()
                model_trained = train_extended_compare_loop(
                    model=model,
                    train_dataloader=train_dataloader,
                    epochs=opt.num_epochs,
                    lr=opt.lr,
                    opt=opt,
                    # steps_til_summary=opt.steps_til_summary,
                    epochs_til_checkpoint=opt.epochs_til_ckpt,
                    model_dir=tmp_model_dir,
                    loss_fn=loss_fn,
                    device='cuda',
                    summary_fn=summary_fn,
                    save_metrices = save_metrices,
                    log_for_tensorboard=opt.enable_tensorboard_logging,
                    data_range = data_range)
                stop_time = time.time() - start_time_to
                stop_times.append(stop_time)
                _log_infos(info_msg = "- Train total time (seconds): {0:.1f}".format(stop_time), header_msg = None, logging=logging, tqdm=tqdm, verbose = 1)
                
                # --- Evaluate model's on validation data.
                eval_h = "-" * 25 + " Eval " + "-" * 25; info_msg = [f"[*] Eval Mode: On", f"[*] Eval device: cuda"]
                _log_infos(info_msg = info_msg, header_msg = eval_h, logging=logging, tqdm=tqdm, verbose = 1)
                
                # eval_start_time = time.time()
                eval_scores, eta_eval = evaluate_model(
                    model = model_trained, eval_dataloader=val_dataloader,
                    device='cuda', loss_fn=loss_fn,
                    quantization_enabled=opt.quantization_enabled)
                stop_times.append(eta_eval)
                # eval_duration_time = time.time() - eval_start_time

                info_eval_quant_stats = ["- arch_no=%d, trial_no=%d, loss=%0.6f, PSNR(db)=%0.6f, SSIM=%0.6f" \
                                                % (arch_no, trial_no, eval_scores[0], eval_scores[1], eval_scores[2]),
                                        "- Evaluate total time (seconds): {0:.1f}".format(eta_eval)]
                _log_infos(info_msg = info_eval_quant_stats, header_msg=None, logging=logging, tqdm=tqdm, verbose = 1)

                # --- Record eval_scores for later average computations.
                avg_train_scores, global_avg_train_losses = \
                    update_avg_stats_arrays(avg_train_scores, global_avg_train_losses, eval_scores)

                # --- Show quantized scores if necessary.
                eval_quantized = []
                if opt.quantization_enabled != None:
                    info_msg = [f"[*] Evaluating Quant. Tech.: {opt.quantization_enabled.upper()}", f"[*] Eval device: cpu"]
                    _log_infos(info_msg = info_msg, header_msg = None, logging=logging, tqdm=tqdm, verbose = 1)
                    
                    # eval_start_time = time.time()
                    eval_quantized, eta_quant, size_model_quant = compute_quantization(
                        img_dataset=img_dataset,
                        opt=opt,
                        model_path=FILE_PATH, arch_hyperparams=arch_hyperparams, device='cpu')
                    stop_times.append(eta_quant)
                    trial_specifics.appen(size_model_quant)
                    # eval_duration_time = time.time() - eval_start_time

                    info_eval_quant_stats = ["- arch_no=%d, trial_no=%d, loss=%0.6f, PSNR(db)=%0.6f, SSIM=%0.6f" \
                                                % (arch_no, trial_no, eval_quantized[0], eval_quantized[1], eval_quantized[2]),
                                            "- Evaluate total time (seconds): {0:.1f}".format(eta_quant)]
                    _log_infos(info_msg = info_eval_quant_stats, header_msg=None, logging=logging, tqdm=tqdm, verbose = 1)
                    pass

                # --- Record performance metrices for later investigations.
                history_combs.append(
                    np.concatenate( (trial_specifics, eval_scores, eval_quantized, stop_times), axis=None)
                )
                pass
            
            # --- Show AVG stats per Arch.
            stats_h = "-" * 25 + " Stats " + "-" * 25                                                              # Header employed
            avg_train_scores = avg_train_scores.mean(axis = 0)                                                     # Average computed
            info_stats = ["- Per Arch stats: arch_no=%d, loss(avg)=%0.6f, PSNR(avg-db)=%0.6f, SSIM(avg)=%0.6f" \
                                % (arch_step, avg_train_scores[0], avg_train_scores[1], avg_train_scores[2]),      # Data to show
                          "- Global stats: loss(avg)=%0.6f, PSNR(avg-db)=%0.6f, SSIM(avg)=%0.6f" \
                                % (avg_train_scores[0], avg_train_scores[1], avg_train_scores[2])]
            _log_infos(info_msg = info_stats, header_msg=stats_h, logging=logging, tqdm=tqdm, verbose = verbose)   # Log requested data

            if opt.enable_tensorboard_logging:
                writer_tb.add_scalar('train_mse_avg', avg_train_scores[0], step)
                writer_tb.add_scalar('train_psnr_avg', avg_train_scores[1], step)
                writer_tb.add_scalar('train_ssim_avg', avg_train_scores[2], step)
                pass
            
            # --- Save data following step strategy.
            if steps_til_summary > 0:
                if arch_no % steps_til_summary == 0:
                    # Save into output file recorded metrices across different trials.
                    tqdm.write(f"[*] Reached {arch_no}. Saving data...")
                    if is_first_arch != True:
                        try:
                            """
                            path_result_comb_train = f'/content/result_comb_train_{arch_step + opt.resume_from}.txt'
                            result = np.array(history_combs)
                            np.savetxt(path_result_comb_train, result)
                            """
                            path_result_comb_train = os.path.join(model_dir, f'result_comb_train_{arch_step + opt.resume_from}.txt')
                            result = np.array(history_combs)
                            np.savetxt(path_result_comb_train,result)
                            history_combs = []
                        except Exception as _:
                            raise Exception(f"Error when saving file: filename={path_result_comb_train} .")
                        arch_step += 1
                    else:
                        is_first_arch = False
                        pass
                    pass
                pass
            step += 1
            # --- Update counter used to handle processing bar.
            pbar.update(1)
            pass
        pass

    # --- Save into output file recorded metrices across different trials.
    try:
        path_result_comb_train = os.path.join(model_dir, 'result_comb_train.txt')
        result = np.array(history_combs)
        np.savetxt(
        path_result_comb_train,
        result
        )

        """
        path_result_comb_train = '/content/result_comb_train.txt'
        result = np.array(history_combs)
        np.savetxt(
            path_result_comb_train,
            result
        )
        """
    except Exception as _:
        raise Exception(f"Error when saving file: filename={path_result_comb_train} .")

    pass