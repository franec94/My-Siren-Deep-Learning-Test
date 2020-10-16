'''Implements a generic training loop.
'''

import torch
import src.utils.utils as utils
import torch.nn as nn
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

def train(model, train_dataloader, epochs, lr, steps_til_summary, epochs_til_checkpoint, model_dir, loss_fn,
          summary_fn, val_dataloader=None, double_precision=False, clip_grad=False, use_lbfgs=False, loss_schedules=None, device = 'cpu'):

    optim = torch.optim.Adam(lr=lr, params=model.parameters())
    loss_fn = nn.MSELoss()

    # copy settings from Raissi et al. (2019) and here 
    # https://github.com/maziarraissi/PINNs
    if use_lbfgs:
        optim = torch.optim.LBFGS(lr=lr, params=model.parameters(), max_iter=50000, max_eval=50000,
                                  history_size=50, line_search_fn='strong_wolfe')

    os.makedirs(model_dir)

    summaries_dir = os.path.join(model_dir, 'summaries')
    utils.cond_mkdir(summaries_dir)

    checkpoints_dir = os.path.join(model_dir, 'checkpoints')
    utils.cond_mkdir(checkpoints_dir)

    writer = SummaryWriter(summaries_dir)

    total_steps = 0
    with tqdm(total=len(train_dataloader) * epochs) as pbar:
        train_losses = []
        for epoch in range(epochs):
            if not epoch % epochs_til_checkpoint and epoch:
                torch.save(model.state_dict(),
                           os.path.join(checkpoints_dir, 'model_epoch_%04d.pth' % epoch))
                np.savetxt(os.path.join(checkpoints_dir, 'train_losses_epoch_%04d.txt' % epoch),
                           np.array(train_losses))

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

                # sidelenght = int(math.sqrt(model_input.size()[1]))

                if use_lbfgs:
                    def closure():
                        optim.zero_grad()
                        model_output = model(model_input)
                        losses = loss_fn(model_output, gt)
                        train_loss = 0.
                        for _, loss in losses.items(): # loss_name
                            train_loss += loss.mean() 
                        train_loss.backward()
                        return train_loss
                    optim.step(closure)

                model_output, _ = model(model_input) # coords
                # losses = loss_fn(model_output, gt)
                train_loss = loss_fn(model_output, gt)
                writer.add_scalar("total_train_loss", train_loss.item(), total_steps)

                sidelenght = model_output.size()[1]

                batch_psnr = \
                    psnr(
                        # model_output.cpu().view(sidelenght, sidelenght).detach().numpy(),
                        # gt.cpu().view(sidelenght, sidelenght).detach().numpy(),
                        model_output.cpu().vew(sidelenght).detach().numpy(),
                        gt.cpu().view(sidelenght).detach().numpy(),
                    data_range=1.0)
                # running_psnr += batch_psnr
                writer.add_scalar("total_train_psnr", batch_psnr, total_steps)

                # Metric: SSIM
                # skmetrics.structural_similarity(
                batch_mssim = \
                    ssim(
                        model_output.cpu().vew(sidelenght).detach().numpy(),
                        gt.cpu().view(sidelenght).detach().numpy(),
                        data_range=1.0)
                # running_ssim += batch_mssim
                writer.add_scalar("total_train_ssim", batch_mssim, total_steps)

                """
                train_loss = 0.
                for loss_name, loss in losses.items():
                    single_loss = loss.mean()

                    if loss_schedules is not None and loss_name in loss_schedules:
                        writer.add_scalar(loss_name + "_weight", loss_schedules[loss_name](total_steps), total_steps)
                        single_loss *= loss_schedules[loss_name](total_steps)

                    writer.add_scalar(loss_name, single_loss, total_steps)
                    train_loss += single_loss

                train_losses.append(train_loss.item())
                writer.add_scalar("total_train_loss", train_loss, total_steps)
                """

                train_losses.append(np.array([train_loss, batch_psnr, batch_mssim]))

                if not total_steps % steps_til_summary:
                    torch.save(model.state_dict(),
                               os.path.join(checkpoints_dir, 'model_current.pth'))
                    # summary_fn(model, model_input, gt, model_output, writer, total_steps)

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
                    tqdm.write("Epoch %d, Total loss %0.6f, Total PSNR %0.6f, Total SSIM %0.6f, iteration time %0.6f" % (epoch, train_loss, batch_psnr, batch_mssim, time.time() - start_time))

                    """
                    if val_dataloader is not None:
                        print("Running validation set...")
                        model.eval()
                        with torch.no_grad():
                            val_losses = []
                            for (model_input, gt) in val_dataloader:
                                model_output = model(model_input)
                                val_loss = loss_fn(model_output, gt)
                                val_losses.append(val_loss)

                            writer.add_scalar("val_loss", np.mean(val_losses), total_steps)
                        model.train()
                    """

                total_steps += 1
                pass
            pass

        torch.save(model.state_dict(),
                   os.path.join(checkpoints_dir, 'model_final.pth'))
        np.savetxt(os.path.join(checkpoints_dir, 'train_losses_final.txt'),
                   np.array(train_losses))
        pass
    pass


class LinearDecaySchedule():
    def __init__(self, start_val, final_val, num_steps):
        self.start_val = start_val
        self.final_val = final_val
        self.num_steps = num_steps

    def __call__(self, iter):
        return self.start_val + (self.final_val - self.start_val) * min(iter / self.num_steps, 1.)