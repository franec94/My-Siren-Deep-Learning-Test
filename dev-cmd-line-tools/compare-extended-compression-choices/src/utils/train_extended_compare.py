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

from sklearn.preprocessing import MinMaxScaler


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
    save_metrices = False):
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
    try: os.makedirs(model_dir)
    except: pass

    """
    summaries_dir = os.path.join(model_dir, 'summaries')
    utils.cond_mkdir(summaries_dir)
    """

    checkpoints_dir = os.path.join(model_dir, 'checkpoints')
    utils.cond_mkdir(checkpoints_dir)

    # Number of interation for current image.
    for epoch in range(epochs):
        # Save partial results as checkpoints.
        if not epoch % epochs_til_checkpoint and epoch:
            try:
                model_name_path = os.path.join(checkpoints_dir, 'model_epoch_%04d.pth' % epoch)
                tmp_file_path = model_name_path
                torch.save(model.state_dict(),
                           model_name_path)
                
                data_name_path = os.path.join(checkpoints_dir, 'train_losses_epoch_%04d.txt' % epoch)
                tmp_file_path = data_name_path
                np.savetxt(data_name_path,
                           np.array(train_losses))
            except Exception as _:
                raise Exception(f"Error when saving file: filename={tmp_file_path} .")
        # Loop for let model's arch be improved, updateding weights values.
        for _, (model_input, gt) in enumerate(train_dataloader):
            # if save_metrices: start_time = time.time()
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

            sidelenght = model_output.size()[1]
            if save_metrices:
                # stop_time = time.time() - start_time
                # sidelenght = int(math.sqrt(model_output.size()[1]))
                sidelenght = model_output.size()[1]

                arr_gt = gt.cpu().view(sidelenght).detach().numpy()
                # arr_gt = np.array([(xi/2+0.5)*255 for xi in arr_gt])
                scaler = MinMaxScaler(feature_range=(0, 255))
                arr_gt = \
                    scaler.fit_transform(arr_gt.reshape(-1, 1)).flatten().astype(np.int8)

                arr_output = model_output.cpu().view(sidelenght).detach().numpy()
                # arr_output = np.array([(xi/2+0.5)*255 for xi in arr_output])
                scaler = MinMaxScaler(feature_range=(0, 255))
                arr_output = \
                    scaler.fit_transform(arr_output.reshape(-1, 1)).flatten().astype(np.int8)
                val_psnr = \
                    psnr(
                        # model_output.cpu().view(sidelenght, sidelenght).detach().numpy(),
                        # gt.cpu().view(sidelenght, sidelenght).detach().numpy(),
                        # gt.cpu().view(sidelenght).detach().numpy(),
                        # model_output.cpu().view(sidelenght).detach().numpy(),
                        arr_gt, arr_output,
                        data_range=data_range)
                # running_psnr += batch_psnr

                # Metric: SSIM
                # skmetrics.structural_similarity(
                val_mssim = \
                        ssim(
                        # model_output.cpu().view(sidelenght, sidelenght).detach().numpy(),
                        # gt.cpu().view(sidelenght, sidelenght).detach().numpy(),
                        # gt.cpu().view(sidelenght).detach().numpy(),
                        # model_output.cpu().view(sidelenght).detach().numpy(),
                        arr_gt, arr_output,
                        data_range=data_range)
                train_losses.append([train_loss, val_psnr, val_mssim])
                """
                tqdm.write(
                    "Epoch %d loss=%0.6f, PSNR=%0.6f, SSIM=%0.6f, iteration time=%0.6f"
                        % (epoch, train_losses[0], train_losses[1], train_losses[2], stop_time))
                """
            else:
                train_losses.append(train_loss)
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

    # Save overall training results.
    try:
        tmp_file_path = os.path.join(checkpoints_dir, 'model_final.pth')
        torch.save(model.state_dict(),
                  tmp_file_path)
        tmp_file_path = os.path.join(checkpoints_dir, 'train_losses_final.txt')
        np.savetxt(tmp_file_path,
                   np.array(train_losses))
    except Exception as _:
                raise Exception(f"Error when saving file: filename={tmp_file_path} .")

    
    # Evaluate model's on validation data.
    model.eval()
    with torch.no_grad():
        val_input, val_gt = next(iter(val_dataloader))

        val_input = val_input['coords'].to(device)
        val_gt = val_gt['img'].to(device)

        val_output, _ = model(val_input)

        # sidelenght = int(math.sqrt(val_output.size()[1]))
        sidelenght = val_output.size()[1]

        train_loss = loss_fn(val_output, val_gt)

        arr_gt = val_gt.cpu().view(sidelenght).detach().numpy()
        # arr_gt = np.array([(xi/2+0.5)*255 for xi in arr_gt])
        scaler = MinMaxScaler(feature_range=(0, 255))
        arr_gt = \
            scaler.fit_transform(arr_gt.reshape(-1, 1)).flatten().astype(np.int8)

        arr_output = val_output.cpu().view(sidelenght).detach().numpy()
        scaler = MinMaxScaler(feature_range=(0, 255))
        arr_output = \
          scaler.fit_transform(arr_output.reshape(-1, 1)).flatten().astype(np.int8)
        # arr_output = np.array([(xi/2+0.5)*255 for xi in arr_output])

        val_psnr = \
            psnr(
                # val_gt.cpu().view(sidelenght, sidelenght).detach().numpy(),
                # val_output.cpu().view(sidelenght, sidelenght).detach().numpy(),
                arr_gt, arr_output,
                data_range=data_range)
            # running_psnr += batch_psnr

        # Metric: SSIM
        # skmetrics.structural_similarity(
        val_mssim = \
            ssim(
                # val_gt.cpu().view(sidelenght, sidelenght).detach().numpy(),
                # val_output.cpu().view(sidelenght, sidelenght).detach().numpy(),
                arr_gt, arr_output,
                data_range=data_range)
        # train_losses = np.array([[train_loss, val_psnr, val_mssim]])
        train_losses = np.array([train_loss, val_psnr, val_mssim])
        pass

    # Return best metrices.
    return train_losses


def train_extended_protocol_compare_archs(grid_arch_hyperparams, img_dataset, opt, model_dir = None, loss_fn=nn.MSELoss(), summary_fn=None, device = 'cpu', verbose = 0, save_metrices = False, data_range = 255):
    """
    Protocol set to collect data about different hyper-params combination done between number of hidden features and number of hidden layers.
    """

    # Local variables.
    history_combs = []
    step = 1
    arch_step = 0
    # steps_til_summary = len(opt.seeds) * len(opt.hidden_layers)
    steps_til_summary = 1
    model, train_dataloader, val_dataloader = None, None, None
    global_avg_train_losses = None

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

            arch_hyperparams_str = '\n'.join([f"{str(k)}: {str(v)}" for k,v in arch_hyperparams.items()])
            tqdm.write(f"{arch_hyperparams_str}")

            # Rescale image to be correctly processed by the net.
            sidelength = int(arch_hyperparams['hidden_features'])
            coord_dataset = dataio.Implicit2DWrapper(
                img_dataset, sidelength=opt.sidelength, compute_diff=None)

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
                
                # Create dir for record results for current trial.
                tmp_model_dir = os.path.join(model_dir, f"arch_no_{arch_no + opt.resume_from}", f"trial_no_{trial_no}")
                try: os.makedirs(tmp_model_dir)
                except: pass
                
                # Set seeds
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
                # Set start time and show messages.
                start_time_to = time.time()
                tqdm.write(f"Arch no.={arch_no + opt.resume_from} | trial no.=({trial_no+1}/{opt.num_attempts}) running...")
                train_losses = train_extended_compare_loop(
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
                    save_metrices = save_metrices,
                    data_range = data_range)
                stop_time = time.time() - start_time_to
                tqdm.write(f"Arch no.={arch_no + opt.resume_from} | trial no.=({trial_no+1}/{opt.num_attempts}) | eta: {stop_time}")
                
                # record train_loss for later average computations.
                if avg_train_losses is None:
                    avg_train_losses =  np.array([train_losses])
                else:
                    avg_train_losses = np.concatenate((avg_train_losses, [train_losses]), axis=0)
                    pass
                if global_avg_train_losses is None:
                    global_avg_train_losses = np.array([train_losses])
                else:
                    global_avg_train_losses = np.concatenate((global_avg_train_losses, [train_losses]), axis=0)
                # Show some output per arch per trial.
                if verbose == 1:
                    tqdm.write(
                        "Arch no.=%d, Trial no.=%d, loss=%0.6f, PSNR=%0.6f, SSIM=%0.6f, eta=%0.6f"
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
                # Show Average stats about current arch
                avg_train_losses = avg_train_losses.mean(axis = 0)
                tqdm.write(
                        "[*] Arch no.=%d, loss(avg)=%0.6f, PSNR(avg)=%0.6f, SSIM(avg)=%0.6f, eta=%0.6f"
                        % (arch_step, avg_train_losses[0], avg_train_losses[1], avg_train_losses[2], stop_time))
                # Show Global Average stats about training process.
                avg_train_losses = global_avg_train_losses.mean(axis = 0)
                tqdm.write(
                        "[*] Global stats: loss(avg)=%0.6f, PSNR(avg)=%0.6f, SSIM(avg)=%0.6f, eta=%0.6f"
                        % (avg_train_losses[0], avg_train_losses[1], avg_train_losses[2], stop_time))
                pass
            
            # Save data following step strategy.
            if step // steps_til_summary == step:
                # Save into output file recorded metrices across different trials.
                try:
                    path_result_comb_train = f'/content/result_comb_train_{arch_step + opt.resume_from}.txt'
                    result = np.array(history_combs)
                    np.savetxt(
                        path_result_comb_train,
                        result
                    )
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
            # Update counter used to handle processing bar.
            pbar.update(1)
            pass
        pass

    # Save into output file recorded metrices across different trials.
    try:
        path_result_comb_train = os.path.join(model_dir, 'result_comb_train.txt')
        result = np.array(history_combs)
        np.savetxt(
        path_result_comb_train,
        result
        )

        path_result_comb_train = '/content/result_comb_train.txt'
        result = np.array(history_combs)
        np.savetxt(
            path_result_comb_train,
            result
        )
    except Exception as _:
        raise Exception(f"Error when saving file: filename={path_result_comb_train} .")

    pass