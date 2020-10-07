from __future__ import print_function
from __future__ import division

# Standard Library, plus some Third Party Libraries
from pprint import pprint
from PIL import Image
from tqdm import tqdm
from typing import Union, Tuple

import configargparse
from functools import partial

import copy
import datetime
import h5py
import math
import os
import random
import sys
import time
# import visdom

# Data Science and Machine Learning Libraries
import matplotlib
import matplotlib.pyplot as plt
matplotlib.style.use('ggplot')

import numpy as np
import pandas as pd
import sklearn

from sklearn.model_selection import train_test_split

# Torch
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset

# from piq import ssim
# from piq import psnr

# TorchVision
import torchvision
from torchvision import datasets
from torchvision import transforms
from torchvision.transforms import Resize, Compose, ToTensor, Normalize
from torchvision.utils import save_image

import torchsummary

# skimage
import skimage
import skimage.metrics as skmetrics
from skimage.metrics import peak_signal_noise_ratio as psnr
from skimage.metrics import structural_similarity as ssim
from skimage.metrics import mean_squared_error


from src.utils.custom_argparser import get_cmd_line_opts
from src.utils.siren import Siren

import src.utils.dataio as dataio
import src.utils.evaluate as evaluate
import src.utils.loss_functions as loss_functions
import src.utils.modules as modules
import src.utils.training as training
import src.utils.utils as utils
import src.utils.graphics as graphics


class Config:  
    def __init__(self, **kwargs):
      for key, value in kwargs.items():
          setattr(self, key, value)
      pass
    pass

class PlotConfig(Config):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        pass
    pass


device, opt, parser = None, None, None

config_plot_loss = PlotConfig(
    image_path = None, # loss_image_path,
    figsize = (10, 7),
    title = "Train - Loss vs Epochs",
    xlabel = "Epochs",
    ylabel = "Loss",
    label = "train loss",
    color = "orange",
    show_min_max = True,
    ax = None
)

config_plot_psnr = PlotConfig(
    image_path = None, # psnr_image_path,
    figsize = (10, 7),
    title = "Train - PSNR (db) vs Epochs",
    xlabel = "Epochs",
    ylabel = "PSNR (db)",
    label = "train PSNR (db)",
    color = "green",
    show_min_max = True,
    ax = None
)

config_plot_ssim = PlotConfig(
    image_path = None, # psnr_image_path,
    figsize = (10, 7),
    title = "Train - SSIM vs Epochs",
    xlabel = "Epochs",
    ylabel = "SSIM",
    label = "train SSIM",
    color = "red",
    show_min_max = True,
    ax = None
)

def check_cmd_line_options():

    global opt, parser
    
    print(opt)
    print("----------")
    print(parser.format_help())
    print("----------")
    print(parser.format_values())    # useful for logging where different settings came from.
    pass


def show_model_summary(model):
    # Print model's state_dict
    print("Model's state_dict:")
    for param_tensor in model.state_dict():
        print(param_tensor, "\t", model.state_dict()[param_tensor].size())
    # print()

    """
    # Print optimizer's state_dict
    print("Optimizer's state_dict:")
    for var_name in optimizer.state_dict():
        print(var_name, "\t", optimizer.state_dict()[var_name])
    """
    pass


def main():

    # Get cmd line options and parser objects.
    global device, opt, parser
    # check_cmd_line_options()

    # Get input image to be compressed.
    if opt.image_filepath is None:
        img_dataset = dataio.Camera()
        coord_dataset = dataio.Implicit2DWrapper(img_dataset, sidelength=512, compute_diff='all')
        image_resolution = (512, 512)
    else:
        img_dataset =  dataio.ImageFile(opt.image_filepath)
        img = Image.open(opt.image_filepath)
        if opt.sidelength is None:
            opt.sidelength = min(img.size)
        coord_dataset = dataio.Implicit2DWrapper(img_dataset, sidelength=opt.sidelength, compute_diff='all')
        image_resolution = (opt.sidelength, opt.sidelength)

        fig = plt.figure()
        Image.open(opt.image_filepath).show()
        plt.show()

    dataloader = DataLoader(coord_dataset, shuffle=True, batch_size=opt.batch_size, pin_memory=True, num_workers=0)

    # Define the model.
    if opt.model_type == 'sine' or opt.model_type == 'relu' or opt.model_type == 'tanh' or opt.model_type == 'selu' or opt.model_type == 'elu'\
        or opt.model_type == 'softplus':
        model = modules.SingleBVPNet(type=opt.model_type, mode='mlp', sidelength=image_resolution)
        # raise NotImplementedError
    elif opt.model_type == 'rbf' or opt.model_type == 'nerf':
        model = modules.SingleBVPNet(type='relu', mode=opt.model_type, sidelength=image_resolution)
    elif opt.model_type == 'siren':
        model = Siren(
            in_features = 2,
            out_features = 1,
            hidden_features = image_resolution[0], 
            hidden_layers = 3,
            outermost_linear=True)
    else:
        raise NotImplementedError

    # model.cuda()
    model = model.to(device)
    show_model_summary(model)

    root_path = os.path.join(opt.logging_root, opt.experiment_name)

    # Define the loss.
    loss_fn = partial(loss_functions.image_mse, None)
    summary_fn = partial(utils.write_image_summary, image_resolution)

    # Performe training.
    print('Train on device: ', device)
    if os.path.exists(root_path):
        if opt.y is None and opt.n is None:
            val = input("The model directory %s exists. Overwrite? (y/n)" % root_path)
            if val == 'y':
                shutil.rmtree(root_path)
        elif opt.y is True:
            shutil.rmtree(model_dir)
            pass
    training.train(
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

    if opt.evaluate:
        predicted_image_path = os.path.join(root_path, 'predicted_images.png')

        # Initialize the model
        print('Evaluate on device: ', device)
        model = Siren(
            in_features = 2,
            hidden_features = image_resolution[0],
            hidden_layers = 3,
            out_features = 1,
            outermost_linear = True, 
            first_omega_0 = 30,
            hidden_omega_0 = 30.
        )
        print(model)

        model_state_path = os.path.join(root_path, 'checkpoints', 'model_final.pth')
        model = model.to(device)
        model.load_state_dict(torch.load(model_state_path, map_location=device))
        
        test_dataloader = DataLoader(coord_dataset, shuffle=True, batch_size=opt.batch_size, pin_memory=True, num_workers=0)
        #predicted_image, ground_thruth, predicted_image, predicted_grad_image, predicted_laplacian_image = evaluate.eval(model, test_dataloader)
        predicted_image, ground_thruth, predicted_image, _, _ = evaluate.eval(model, test_dataloader, device)

        sidelenght = image_resolution[0]
        # Metric: MSE
        val_mse = \
            mean_squared_error(
                ground_thruth.cpu().view(sidelenght, sidelenght).detach().numpy(),
                predicted_image.cpu().view(sidelenght, sidelenght).detach().numpy())

        # Metric: PSNR
        val_psnr = \
            psnr(
                ground_thruth.cpu().view(sidelenght, sidelenght).detach().numpy(),
                predicted_image.cpu().view(sidelenght, sidelenght).detach().numpy(),
                data_range=1.0)

        # Metric: SSIM
        # skmetrics.structural_similarity(
        val_mssim = \
            ssim(
                ground_thruth.cpu().view(sidelenght, sidelenght).detach().numpy(),
                predicted_image.cpu().view(sidelenght, sidelenght).detach().numpy(),
                data_range=1.0)

        image = predicted_image.cpu().view(sidelenght, sidelenght).detach().numpy()

        data = np.array([val_mse, val_psnr, val_mssim])
        columns = [f"predicted_{metric}" for metric in "mse;psnr;mssim".split(";")]

        metrics_txt = '\n'.join([f"{k}: {v:.4f}" for k, v in zip(columns, data)])
        graphics.show_image_with_metrcis_scores(image, sidelenght, metrics_txt,  predicted_image_path)
        pass

    if opt.show_graphics:
        train_metrics_trend = os.path.join(root_path, 'train_metrics_trend.png')
        history_data_path = os.path.join(root_path, 'checkpoints', 'train_losses_final.txt')
        data = np.loadtxt(history_data_path)
        
        history_df = pd.DataFrame(data = data, columns = ['train_loss', 'train_psnr', 'train_ssim'])
        fig, axes = plt.subplots(3, figsize=(10, 10))
        fig.tight_layout(pad=5.0)

        config_plot_loss.ax = axes[0]
        config_plot_psnr.ax = axes[1]
        config_plot_ssim.ax = axes[2]

        graphics.plot_series_graphic_by_config(history_df['train_loss'].values, config_plot_loss)
        graphics.plot_series_graphic_by_config(history_df['train_psnr'].values, config_plot_psnr)
        graphics.plot_series_graphic_by_config(history_df['train_ssim'].values, config_plot_ssim)
        plt.savefig('train_metrics_trend')
        pass
    pass


if __name__ == "__main__":

    # Initialize option and parser objects.
    opt, parser = get_cmd_line_opts()
    
    # Set seeds for experiment re-running.
    seed = opt.seed
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    device = None
    try:
        device = (torch.device('cuda:0') if torch.cuda.is_available()
        else torch.device('gpu'))
    except:
        device = torch.device('cpu')
    print(f"Training on device {device}.")
    print(f"# cuda device: {torch.cuda.device_count()}")
    if torch.cuda.device_count() > 0:
        print(f"Id current device: {torch.cuda.current_device()}")

    # Run training.
    main()
    pass