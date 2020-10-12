'''Implements a generic training loop for comparing different architectures.
'''

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

# skimage
import skimage
import skimage.metrics as skmetrics
from skimage.metrics import peak_signal_noise_ratio as psnr
from skimage.metrics import structural_similarity as ssim
from skimage.metrics import mean_squared_error

import src.utils.dataio as dataio
from src.utils.siren import Siren


def train_extended_compare_loop(model, train_dataloader, epochs, lr, steps_til_summary=None, epochs_til_checkpoint=None, model_dir=None, loss_fn=None,
                        summary_fn=None, val_dataloader=None, double_precision=False, clip_grad=False, use_lbfgs=False, loss_schedules=None, device='cpu', debug_mode = False):
    """
    Performe training on a given input model, specifing onto which device the training process will be done.
    """

    # Set model into train mode, for safe reasons.
    model.train()

    # Define some objects necessary to lead the protocol
    # which is in charge of let the model to be trained.
    optim = torch.optim.Adam(lr=lr, params=model.parameters())
    loss_fn = nn.MSELoss()

    # copy settings from Raissi et al. (2019) and here
    # https://github.com/maziarraissi/PINNs
    if use_lbfgs:
        optim = torch.optim.LBFGS(
            lr=lr,
            params=model.parameters(),
            max_iter=50000,
            max_eval=50000,
            history_size=50, line_search_fn='strong_wolfe')
        pass

    # Local variables.
    train_losses = []  # used for recording metrices when evaluated.

    # Number of interation for current image.
    for epoch in range(epochs):
        # Loop for let model's arch be improved, updateding weights values.
        for _, (model_input, gt) in enumerate(train_dataloader):
            if debug_mode:
                start_time = time.time()
            # Get input data and set it to desired device
            # for computation reasons.
            model_input = model_input['coords'].to(device)
            gt = gt['img'].to(device)

            # sidelenght = int(math.sqrt(model_input.size()[1]))

            if use_lbfgs:
                def closure():
                    optim.zero_grad()
                    model_output = model(model_input)
                    losses = loss_fn(model_output, gt)
                    train_loss = 0.
                    for _, loss in losses.items():
                        train_loss += loss.mean()
                    train_loss.backward()
                    return train_loss
                optim.step(closure)

            # Compute forward pass.
            model_output, _ = model(model_input)
            # losses = loss_fn(model_output, gt)
            train_loss = loss_fn(model_output, gt)

            if debug_mode:
                stop_time = time.time() - start_time
                sidelenght = int(math.sqrt(model_output.size()[1]))
                val_psnr = \
                    psnr(
                        model_output.cpu().view(sidelenght, sidelenght).detach().numpy(),
                        gt.cpu().view(sidelenght, sidelenght).detach().numpy(),
                        data_range=1.0)
                # running_psnr += batch_psnr

                # Metric: SSIM
                # skmetrics.structural_similarity(
                val_mssim = \
                        ssim(
                        model_output.cpu().view(sidelenght, sidelenght).detach().numpy(),
                        gt.cpu().view(sidelenght, sidelenght).detach().numpy(),
                        data_range=1.0)
                train_losses = [train_loss, val_psnr, val_mssim]
                tqdm.write(
                    "Epoch %d loss=%0.6f, PSNR=%0.6f, SSIM=%0.6f, iteration time=%0.6f"
                        % (epoch, train_losses[0], train_losses[1], train_losses[2], stop_time))
                pass

            # Backward pass.
            if not use_lbfgs:
                optim.zero_grad()
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
                pass
                optim.step()
            pass
        pass

    # Evaluate model's on validation data
    model.eval()
    with torch.no_grad():
        val_input, val_gt = next(iter(val_dataloader))

        val_input = val_input['coords'].to(device)
        val_gt = val_gt['img'].to(device)

        val_output, _ = model(val_input)

        sidelenght = int(math.sqrt(val_output.size()[1]))

        train_loss = loss_fn(val_output, val_gt)

        val_psnr = \
            psnr(
                val_output.cpu().view(sidelenght, sidelenght).detach().numpy(),
                val_gt.cpu().view(sidelenght, sidelenght).detach().numpy(),
                data_range=1.0)
            # running_psnr += batch_psnr

        # Metric: SSIM
        # skmetrics.structural_similarity(
        val_mssim = \
            ssim(
                val_output.cpu().view(sidelenght, sidelenght).detach().numpy(),
                val_gt.cpu().view(sidelenght, sidelenght).detach().numpy(),
                data_range=1.0)
        # train_losses = np.array([[train_loss, val_psnr, val_mssim]])
        train_losses = np.array([train_loss, val_psnr, val_mssim])
        pass

    # Return best metrices.
    return train_losses


def train_extended_protocol_compare_archs(grid_arch_hyperparams, img_dataset, opt, loss_fn=nn.MSELoss(), summary_fn=None, root_path = None, device = 'cpu', verbose = 0):
    """
    Protocol set to collect data about different hyper-params combination done between number of hidden features and number of hidden layers.
    """

    # Local variables.
    history_combs = []
    step = 0
    arch_step = 0
    steps_til_summary = len(opt.seeds) * len(opt.hidden_layers)
    model, train_dataloader, val_dataloader = None, None, None

    # Processing Bar to control the workout.
    with tqdm(total=len(grid_arch_hyperparams)) as pbar:

        # For loop for performing different training depending on the
        # chosen hyper-params.
        # print()
        for arch_no, arch_hyperparams in enumerate(grid_arch_hyperparams):
            # Start time: it's the point in time from which the current train
            # begins, when new hyper-params are selected and evaluted in terms of performances.
            if verbose >= 1:
                start_time_ao = time.time()
            # print(hidden_features, hidden_layers)

            # Rescale image to be correctly processed by the net.
            sidelength = int(arch_hyperparams['hidden_features'])
            coord_dataset = dataio.Implicit2DWrapper(
                img_dataset, sidelength=sidelength, compute_diff='all')

            # Prepare dataloaders for train and eval phases.
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

            seed = int(arch_hyperparams['seeds'])
            avg_train_losses = None
            for trial_no in range(opt.num_attempts):
                tqdm.write(f"Arch no.={arch_no} | trial no.={trial_no} running...")
                start_time_to = time.time()
                torch.manual_seed(seed)
                np.random.seed(seed)
                random.seed(seed)
                # Prepare siren model.
                model = Siren(
                    in_features=2,
                    out_features=1,
                    hidden_features=int(arch_hyperparams['hidden_features']),
                    hidden_layers=int(arch_hyperparams['hidden_layers']),
                    outermost_linear=True).to(device=device)
            
                tot_weights_model = sum(p.numel() for p in model.parameters())
                # print(model)

                # Train model.
                train_losses = train_extended_compare_loop(
                    model=model,
                    train_dataloader=train_dataloader,
                    epochs=opt.num_epochs,
                    lr=opt.lr,
                    val_dataloader=val_dataloader,
                    # steps_til_summary=opt.steps_til_summary,
                    # epochs_til_checkpoint=opt.epochs_til_ckpt,
                    model_dir=root_path,
                    loss_fn=loss_fn,
                    device=device,
                    summary_fn=summary_fn)
                
                # record train_loss for later average computations.
                if avg_train_losses is None:
                    avg_train_losses = train_losses
                else:
                    avg_train_losses = np.concatenate(([avg_train_losses], [train_losses]), axis=0)
                
                stop_time = time.time() - start_time_to

                # Show some output per arch per trial.
                if verbose == 1:
                    tqdm.write(
                        "Arch no.=%d, Trial no.=%d, loss=%0.6f, PSNR=%0.6f, SSIM=%0.6f, iteration time=%0.6f"
                        % (arch_no, trial_no, train_losses[0], train_losses[1], train_losses[2], stop_time))
                    pass

                # Record performance metrices for later investigations.
                # history_combs.append(np.concat(train_losses, [stop_time]))
                history_combs.append(
                    np.concatenate(
                        (
                            [tot_weights_model, seed, arch_hyperparams['hidden_layers'], arch_hyperparams['hidden_features']],
                            train_losses,
                            [stop_time]
                        ),
                        axis=None)
                )
                pass
            
            # Show AVG stats per Arch.
            if verbose >= 1:
                stop_time = time.time() - start_time_ao
                avg_train_losses = avg_train_losses.mean(axis = 0)
                tqdm.write(
                        "Arch no.=%d, avg_loss=%0.6f, avg_PSNR=%0.6f, avg_SSIM=%0.6f, iteration time=%0.6f"
                        % (arch_step, avg_train_losses[0], avg_train_losses[1], avg_train_losses[2], stop_time))
                pass
            
            if step == steps_til_summary:
                # Save into output file recorded metrices across different trials.
                path_result_comb_train = f'/content/result_comb_train_{arch_step}.txt'
                result = np.array(history_combs)
                np.savetxt(
                    path_result_comb_train,
                    result
                )
                step = -1
                arch_step += 1
                pass
            step += 1
            # Update counter used to handle processing bar.
            pbar.update(1)
            pass
        pass

    # Save into output file recorded metrices across different trials.
    path_result_comb_train = '/content/result_comb_train.txt'
    result = np.array(history_combs)
    np.savetxt(
        path_result_comb_train,
        result
    )

    pass