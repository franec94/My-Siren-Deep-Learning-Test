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

from src.utils.siren import Siren


class LinearDecaySchedule():
    def __init__(self, start_val, final_val, num_steps):
        self.start_val = start_val
        self.final_val = final_val
        self.num_steps = num_steps

    def __call__(self, iter):
        return self.start_val + (self.final_val - self.start_val) * min(iter / self.num_steps, 1.)


def train_compare_archs(model, train_dataloader, epochs, lr, steps_til_summary, epochs_til_checkpoint, model_dir, loss_fn,
          summary_fn, val_dataloader=None, double_precision=False, clip_grad=False, use_lbfgs=False, loss_schedules=None, device = 'cpu'):

    optim = torch.optim.Adam(lr=lr, params=model.parameters())
    loss_fn = nn.MSELoss()

    # copy settings from Raissi et al. (2019) and here 
    # https://github.com/maziarraissi/PINNs
    if use_lbfgs:
        optim = torch.optim.LBFGS(lr=lr, params=model.parameters(), max_iter=50000, max_eval=50000,
                                  history_size=50, line_search_fn='strong_wolfe')

    total_steps = 0
    train_losses = []
    with tqdm(total=len(train_dataloader) * epochs) as pbar:
        
        for epoch in range(epochs):

            for step, (model_input, gt) in enumerate(train_dataloader):
                start_time = time.time()
            
                # model_input = {key: value.cuda() for key, value in model_input.items()}
                # gt = {key: value.cuda() for key, value in gt.items()}

                if double_precision:
                    # model_input = {key: value.double() for key, value in model_input.items()}
                    # gt = {key: value.double() for key, value in gt.items()}
                    pass

                model_input = model_input['coords'].to(device)
                gt = gt['img'].to(device)

                sidelenght = int(math.sqrt(model_input.size()[1]))

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

                model_output, coords = model(model_input)
                # losses = loss_fn(model_output, gt)
                train_loss = loss_fn(model_output, gt)

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
                
                # writer.add_scalar("total_train_ssim", batch_mssim, total_steps)

                train_losses.append(np.array([train_loss, batch_psnr, batch_mssim]))

                if not use_lbfgs:
                    optim.zero_grad()
                    train_loss.backward()

                    if clip_grad:
                        if isinstance(clip_grad, bool):
                            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.)
                        else:
                            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=clip_grad)

                    optim.step()

                pbar.update(1)

                if not total_steps % steps_til_summary:
                    # tqdm.write("Epoch %d, Total loss %0.6f, Total PSNR %0.6f, Total SSIM %0.6f, iteration time %0.6f" % (epoch, train_loss, batch_psnr, batch_mssim, time.time() - start_time))
                    pass

                total_steps += 1
                pass
            pass
        pass
    return train_losses[-1]


def train_protocol_compare_archs(arch_hyperparams, coord_dataset, opt, loss_fn = None, summary_fn = None, root_path = None, device = 'cpu'):

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

    history_combs = []

    model, dataloader = None, None
    for _, (hidden_features, hidden_layers) in enumerate(arch_hyperparams):
        # print(hidden_features, hidden_layers)
        model = None

        dataloader = DataLoader(
            coord_dataset,
            shuffle=True,
            batch_size=opt.batch_size,
            pin_memory=True, num_workers=0)
        
        model = Siren(
            in_features = 2,
            out_features = 1,
            hidden_features = hidden_features, 
            hidden_layers = hidden_layers,
            outermost_linear=True)
        
        
        train_losses = train_compare_archs(
            model=model,
            train_dataloader=dataloader,
            epochs=opt.num_epochs,
            lr=opt.lr,
            steps_til_summary=opt.steps_til_summary,
            epochs_til_checkpoint=opt.epochs_til_ckpt,
            model_dir=root_path,
            loss_fn=loss_fn,
            device = device,
            summary_fn=summary_fn)
        
        history_combs.append(train_losses)
        pass

    path_result_comb_train = '/content/result_comb_train.txt'
    result = np.array(history_combs)

    np.savetxt(
        path_result_comb_train,
        result
    )

    pass
