'''Implements a generic training loop for comparing different architectures.
'''
from __future__ import print_function
from __future__ import division

# --------------------------------------------- #
# Standard Library | Third Party Libraries
# --------------------------------------------- #

import logging

import torch
import src.utils.utils as utils
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from torch.utils.tensorboard import SummaryWriter
from tqdm.autonotebook import tqdm
import time
import numpy as np
import os
import random
import math
import shutil

# --------------------------------------------- #
# skimage
# --------------------------------------------- #
import skimage
import skimage.metrics as skmetrics
from skimage.metrics import peak_signal_noise_ratio as psnr
from skimage.metrics import structural_similarity as ssim
from skimage.metrics import mean_squared_error

import src.utils.dataio as dataio
from src.utils.siren import Siren

from sklearn.preprocessing import MinMaxScaler

import pytorch_model_summary as pms

from src.utils.siren_dynamic_quantization import get_dynamic_quantization_model, get_static_quantization_model, get_post_training_quantization_model

# --------------------------------------------- #
# Functions
# --------------------------------------------- #

def train_extended_compare_loop(
    model, train_dataloader,
    epochs, lr,
    steps_til_summary=None,
    epochs_til_checkpoint=None,
    model_dir=None,
    loss_fn=None,
    summary_fn=None,
    val_dataloader=None,
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
        model_input = model_input['coords'].cuda()
        gt = gt['img'].cuda()


        for epoch in range(epochs):
            # --- Compute forward pass.
            optim.zero_grad()

            model_output, _ = model(model_input)
            train_loss = loss_fn(model_output, gt)

            if save_metrices:
                sidelenght = model_output.size()[1]

                arr_gt = gt.cpu().view(sidelenght).detach().numpy()
                arr_gt = (arr_gt / 2.) + 0.5

                arr_output = model_output.cpu().view(sidelenght).detach().numpy()
                arr_output = (arr_output / 2.) + 0.5
                arr_output = np.clip(arr_output, a_min=0., a_max=1.)

                val_psnr = psnr(arr_gt, arr_output,data_range=1.)
                val_mssim = ssim(arr_gt, arr_output,data_range=1.)

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
    try:
        tmp_file_path = os.path.join(checkpoints_dir, 'model_final.pth')
        torch.save(model.state_dict(),
                  tmp_file_path)
        
        tmp_file_path = os.path.join(checkpoints_dir, 'train_losses_final.txt')
        np.savetxt(tmp_file_path,
                   np.array(train_scores))
    except Exception as _:
                raise Exception(f"Error when saving file: filename={tmp_file_path} .")

    
    # --- Evaluate model's on validation data.
    train_scores = None
    model.eval()
    with torch.no_grad():
        # -- Get data from validation loader.
        val_input, val_gt = next(iter(val_dataloader))

        val_input = val_input['coords'].cuda() # .to(device)
        val_gt = val_gt['img'].cuda() # .to(device)

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
        # train_scores = np.array([[train_loss, val_psnr, val_mssim]])
        train_scores = np.array([train_loss.item(), val_psnr, val_mssim])
        pass

    # Return best metrices.
    return train_scores


def prepare_model(opt, arch_hyperparams = None):
    if opt.quantization_enabled != None:
        if opt.quantization_enabled == 'dynamic':
            model = Siren(
                in_features=2,
                out_features=1,
                hidden_features=int(arch_hyperparams['hidden_features']),
                hidden_layers=int(arch_hyperparams['hidden_layers']),
                # outermost_linear=True).to(device=device)
                outermost_linear=True)
            model = get_dynamic_quantization_model(metadata_model_dict = arch_hyperparams, set_layers = {torch.nn.Linear}, device = 'cpu', qconfig = 'fbgemm', model_fp32 = model)
        elif opt.quantization_enabled =='static':
            model = Siren(
                in_features=2,
                out_features=1,
                hidden_features=int(arch_hyperparams['hidden_features']),
                hidden_layers=int(arch_hyperparams['hidden_layers']),
                # outermost_linear=True).to(device=device)
                outermost_linear=True)
            model = get_static_quantization_model(metadata_model_dict = arch_hyperparams, fuse_modules = None, device = 'cpu', qconfig = 'fbgemm', model_fp32 = model)
            pass
        else:
            raise Exception(f"Error: {opt.quantization_enabled} not allowed!")
    else:
        model = Siren(
            in_features=2,
            out_features=1,
            hidden_features=int(arch_hyperparams['hidden_features']),
            hidden_layers=int(arch_hyperparams['hidden_layers']),
            # outermost_linear=True).to(device=device)
            outermost_linear=True).cuda()
            
        pass
    return model

def train_extended_protocol_compare_archs(grid_arch_hyperparams, img_dataset, opt, model_dir = None, loss_fn=nn.MSELoss(), summary_fn=None, device = 'cpu', verbose = 0, save_metrices = False, data_range = 255):
    """
    Protocol set to collect data about different hyper-params combination done between number of hidden features and number of hidden layers.
    """

    # --- Local variables.
    writer_tb = None
    history_combs = []
    step = 1
    arch_step = 0
    # steps_til_summary = len(opt.seeds) * len(opt.hidden_layers)
    steps_til_summary = 1
    model, train_dataloader, val_dataloader = None, None, None
    global_avg_train_losses = None

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

            torch.cuda.empty_cache()
            # --- Start time: it's the point in time from which the current train
            # begins, when new hyper-params are selected and evaluted in terms of performances.
            if verbose >= 1:
                start_time_ao = time.time()
            # print(hidden_features, hidden_layers)

            # --- Print hyperparams to be tested.
            # logging.info("_" * 50); logging.info("_" * 50)
            # tqdm.write("_" * 50); tqdm.write("_" * 50)

            sep_str_arch_no = "=" * 25 + f" Arch no.={arch_no + opt.resume_from} " + "=" * 25
            logging.info(sep_str_arch_no)
            tqdm.write(sep_str_arch_no)

            arch_hyperparams_str = '\n'.join([f"{str(k)}: {str(v)}" for k,v in arch_hyperparams.items()])
            tqdm.write(f"{arch_hyperparams_str}")
            logging.info(arch_hyperparams_str)

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
            logging.info("-" * 50); tqdm.write("-" * 50)
            seed = int(arch_hyperparams['seeds'])
            avg_train_losses = None
            for trial_no in range(opt.num_attempts):
                # --- Create dir for record results for current trial.
                tmp_model_dir = os.path.join(model_dir, f"arch_no_{arch_no + opt.resume_from}", f"trial_no_{trial_no}")
                try: os.makedirs(tmp_model_dir)
                except: pass
                
                # --- Set seeds
                torch.manual_seed(seed)
                np.random.seed(seed)
                random.seed(seed)

                # --- Prepare siren model.
                model = prepare_model(opt, arch_hyperparams = arch_hyperparams)

                # print(model)
                tot_weights_model = sum(p.numel() for p in model.parameters())
                tqdm.write(f"Model's size (# parameters): {tot_weights_model} | Model's size (# bits, 1 weight = 32 bits): {tot_weights_model * 32}")
                logging.info(f"Model's size (# parameters): {tot_weights_model} | Model's size (# bits, 1 weight = 32 bits): {tot_weights_model * 32}")
                logging.info("-" * 50); tqdm.write("-" * 50)

                model_summary_str = pms.summary(model, torch.Tensor((1, 2)).cuda(), show_input=False, show_hierarchical=True)
                logging.info(f"{model_summary_str}"); tqdm.write(f"{model_summary_str}")

                # --- Train model.
                # Set start time and show messages.
                start_time_to = time.time()
                tqdm.write(f"Arch no.={arch_no + opt.resume_from} | trial no.=({trial_no+1}/{opt.num_attempts}) running...")
                logging.info(f"Arch no.={arch_no + opt.resume_from} | trial no.=({trial_no+1}/{opt.num_attempts}) running...")

                train_scores = train_extended_compare_loop(
                    model=model,
                    train_dataloader=train_dataloader,
                    epochs=opt.num_epochs,
                    lr=opt.lr,
                    val_dataloader=val_dataloader,
                    # steps_til_summary=opt.steps_til_summary,
                    epochs_til_checkpoint=opt.epochs_til_ckpt,
                    model_dir=tmp_model_dir,
                    loss_fn=loss_fn,
                    device=device,
                    summary_fn=summary_fn,
                    save_metrices = True,
                    log_for_tensorboard=opt.enable_tensorboard_logging,
                    data_range = data_range)
                
                stop_time = time.time() - start_time_to
                tqdm.write(f"Arch no.={arch_no + opt.resume_from} | trial no.=({trial_no+1}/{opt.num_attempts}) | eta: {stop_time}")
                logging.info(f"Arch no.={arch_no + opt.resume_from} | trial no.=({trial_no+1}/{opt.num_attempts}) | eta: {stop_time}")
                logging.info("-" * 50); tqdm.write("-" * 50)

                # --- Record train_loss for later average computations.
                if avg_train_losses is None:
                    avg_train_losses =  np.array([train_scores])
                else:
                    avg_train_losses = np.concatenate((avg_train_losses, [train_scores]), axis=0)
                    pass
                if global_avg_train_losses is None:
                    global_avg_train_losses = np.array([train_scores])
                else:
                    global_avg_train_losses = np.concatenate((global_avg_train_losses, [train_scores]), axis=0)
                # --- Show some output per arch per trial.
                if verbose == 1:
                    tqdm.write(
                        "Arch no.=%d, Trial no.=%d, loss=%0.6f, PSNR=%0.6f, SSIM=%0.6f, eta=%0.6f"
                        % (arch_no, trial_no, train_scores[0], train_scores[1], train_scores[2], stop_time))
                    pass
                logging.info(
                        "Arch no.=%d, Trial no.=%d, loss=%0.6f, PSNR=%0.6f, SSIM=%0.6f, eta=%0.6f"
                        % (arch_no, trial_no, train_scores[0], train_scores[1], train_scores[2], stop_time)
                    )
                

                # --- Record performance metrices for later investigations.
                # history_combs.append(np.concat(train_scores, [stop_time]))
                history_combs.append(
                    np.concatenate(
                        (
                            [tot_weights_model, seed, arch_hyperparams['hidden_layers'], arch_hyperparams['hidden_features']],
                            train_scores,
                            [stop_time]
                        ),
                        axis=None)
                )
                pass
            
            # --- Show AVG stats per Arch.
            if verbose >= 1:
                stop_time = time.time() - start_time_ao
                # Show Average stats about current arch
                avg_train_losses = avg_train_losses.mean(axis = 0)
                tqdm.write(
                        "[*] --> Arch no.=%d, loss(avg)=%0.6f, PSNR(avg)=%0.6f, SSIM(avg)=%0.6f, eta=%0.6f"
                        % (arch_step, avg_train_losses[0], avg_train_losses[1], avg_train_losses[2], stop_time))
                # Show Global Average stats about training process.
                avg_train_losses = global_avg_train_losses.mean(axis = 0)
                tqdm.write(
                        "[*] --> Global stats: loss(avg)=%0.6f, PSNR(avg)=%0.6f, SSIM(avg)=%0.6f, eta=%0.6f"
                        % (avg_train_losses[0], avg_train_losses[1], avg_train_losses[2], stop_time))
                pass
            logging.info("[*] --> Arch no.=%d stats, loss(avg)=%0.6f, PSNR(avg)=%0.6f, SSIM(avg)=%0.6f, eta=%0.6f"
                        % (arch_step, avg_train_losses[0], avg_train_losses[1], avg_train_losses[2], stop_time))
            logging.info("[*] --> Global stats: loss(avg)=%0.6f, PSNR(avg)=%0.6f, SSIM(avg)=%0.6f, eta=%0.6f"
                        % (avg_train_losses[0], avg_train_losses[1], avg_train_losses[2], stop_time))
            if opt.enable_tensorboard_logging:
                writer_tb.add_scalar('train_mse_avg', avg_train_losses[0], step)
                writer_tb.add_scalar('train_psnr_avg', avg_train_losses[1], step)
                writer_tb.add_scalar('train_ssim_avg', avg_train_losses[2], step)
                pass
            
            # --- Save data following step strategy.
            if step // steps_til_summary == step:
                # Save into output file recorded metrices across different trials.
                try:
                    """
                    path_result_comb_train = f'/content/result_comb_train_{arch_step + opt.resume_from}.txt'
                    result = np.array(history_combs)
                    np.savetxt(
                        path_result_comb_train,
                        result
                    )
                    """
                    path_result_comb_train = os.path.join(model_dir, f'result_comb_train_{arch_step + opt.resume_from}.txt')
                    result = np.array(history_combs)
                    np.savetxt(
                        path_result_comb_train,
                        result
                    )
                except Exception as _:
                    raise Exception(f"Error when saving file: filename={path_result_comb_train} .")
                step = 0
                arch_step += 1
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