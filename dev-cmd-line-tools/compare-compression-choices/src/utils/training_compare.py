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


class LinearDecaySchedule():
    def __init__(self, start_val, final_val, num_steps):
        self.start_val = start_val
        self.final_val = final_val
        self.num_steps = num_steps

    def __call__(self, iter):
        return self.start_val + (self.final_val - self.start_val) * min(iter / self.num_steps, 1.)


def train_compare_archs(model, train_dataloader, epochs, lr, steps_til_summary=None, epochs_til_checkpoint=None, model_dir=None, loss_fn=None,
                        summary_fn=None, val_dataloader=None, double_precision=False, clip_grad=False, use_lbfgs=False, loss_schedules=None, device='cpu'):
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
        optim = torch.optim.LBFGS(lr=lr, params=model.parameters(), max_iter=50000, max_eval=50000,
                                  history_size=50, line_search_fn='strong_wolfe')

    # Local variables.
    # total_steps = 0 # used for show output depending on 'steps_til_summary' function's input argument.
    train_losses = []  # used for recording metrices when evaluated.

    # Processing Bar to control the workout.
    # with tqdm(total=len(train_dataloader) * epochs) as pbar:

    # Number of interation for current image.
    for epoch in range(epochs):
        # Loop for let model's arch be improved, updateding weights values.
        for step, (model_input, gt) in enumerate(train_dataloader):
            """
            # Start time: it's the point in time from which the current train
            # start_time = time.time()
            
            # model_input = {key: value.cuda() for key, value in model_input.items()}
            # gt = {key: value.cuda() for key, value in gt.items()}

            if double_precision:
            # model_input = {key: value.double() for key, value in model_input.items()}
            # gt = {key: value.double() for key, value in gt.items()}
            pass
            """
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
                    for loss_name, loss in losses.items():
                        train_loss += loss.mean()
                    train_loss.backward()
                    return train_loss
                optim.step(closure)

            # Compute forward pass.
            model_output, coords = model(model_input)
            # losses = loss_fn(model_output, gt)
            train_loss = loss_fn(model_output, gt)

            """
            # Other metrices.
            batch_psnr = \
                psnr(
                    model_output.cpu().view(sidelenght,sidelenght).detach().numpy(),
                    gt.cpu().view(sidelenght,sidelenght).detach().numpy(),
                data_range=1.0)
            # running_psnr += batch_psnr

            # Metric: SSIM
            # skmetrics.structural_similarity(
            batch_mssim = \
                ssim(
                model_output.cpu().view(sidelenght,sidelenght).detach().numpy(),
                    gt.cpu().view(sidelenght,sidelenght).detach().numpy(),
                    data_range=1.0)
            # running_ssim += batch_mssim
            """

            """
            # Record metrices.
            # writer.add_scalar("total_train_ssim", batch_mssim, total_steps)
            # train_losses.append(np.array([train_loss, batch_psnr, batch_mssim]))
            """

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

            # Update counter used to handle processing bar.
            # pbar.update(1)

            """
            # Show some output.
            if not total_steps % steps_til_summary:
                # tqdm.write("Epoch %d, Total loss %0.6f, Total PSNR %0.6f, Total SSIM %0.6f, iteration time %0.6f" % (epoch, train_loss, batch_psnr, batch_mssim, time.time() - start_time))
                pass
            # total_steps += 1
            """
            pass
        # pass
        pass

    # Evaluate model's on validation data
    model.eval()
    with torch.no_grad():
        (val_input, val_gt) = next(iter(val_dataloader))
        model_input = val_input['coords'].to(device)
        val_gt = val_gt['img'].to(device)
        val_output, val_coords = model(val_input)

        sidelenght = int(math.sqrt(val_output.size()[1]))

        train_loss = loss_fn(val_output, val_gt)

        val_psnr = \
            psnr(
                val_output.cpu().view(sidelenght, sidelenght).detach().numpy(),
                gt.cpu().view(sidelenght, sidelenght).detach().numpy(),
                data_range=1.0)
            # running_psnr += batch_psnr

        # Metric: SSIM
        # skmetrics.structural_similarity(
        val_mssim = \
            ssim(
                val_output.cpu().view(sidelenght, sidelenght).detach().numpy(),
                gt.cpu().view(sidelenght, sidelenght).detach().numpy(),
                data_range=1.0)
        train_losses = np.array([[train_loss, val_psnr, val_mssim]])
        pass

    # Return best metrices.
    return train_losses[-1]


def train_protocol_compare_archs(arch_hyperparams, img_dataset, opt, loss_fn=nn.MSELoss(), summary_fn=None, root_path = None, device = 'cpu'):
    """
    Protocol set to collect data about different hyper-params combination done between number of hidden features and number of hidden layers.
    """

    # Blueprint code.
    """
    dataloader = DataLoader(
        coord_dataset,
        shuffle=True,
        batch_size=opt.batch_size,
        pin_memory=True, num_workers=0)
    """

    """
    model = Siren(
        in_features = 2,
        out_features = 1,
        hidden_features = image_resolution[0], 
        hidden_layers = 3,
        outermost_linear=True)
    """

    # Local variables.
    history_combs = []
    model, train_dataloader, val_dataloader = None, None, None

    # Processing Bar to control the workout.
    with tqdm(total=len(arch_hyperparams)) as pbar:

        # For loop for performing different training depending on the
        # chosen hyper-params.
        for arch_no, (hidden_layers, hidden_features) in enumerate(arch_hyperparams):
            # Start time: it's the point in time from which the current train
            # begins, when new hyper-params are selected and evaluted in terms of performances.
            start_time = time.time()
            # print(hidden_features, hidden_layers)

            # Rescale image to be correctly processed by the net.
            sidelength = int(hidden_features)
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

            # Prepare siren model.
            model = Siren(
                in_features=2,
                out_features=1,
                hidden_features=int(hidden_features),
                hidden_layers=int(hidden_layers),
                outermost_linear=True).to(device=device)

            # print(model)

            # Train model.
            train_losses = train_compare_archs(
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

            # Show some output.
            tqdm.write(
                "Arch no. %d, Total loss %0.6f, Total PSNR %0.6f, Total SSIM %0.6f, iteration time %0.6f"
                % (arch_no, train_losses[0], train_losses[1], train_losses[2], time.time() - start_time))

            # Record performance metrices for later investigations.
            history_combs.append(train_losses)
            pass

        # Update counter used to handle processing bar.
        pbar.update(1)
        pass

    # Save into output file recorded metrices across different trials.
    path_result_comb_train = '/content/result_comb_train.txt'
    result = np.array(history_combs)
    np.savetxt(
        path_result_comb_train,
        result
    )

    pass
