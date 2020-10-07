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
import src.utils.loss_functions as loss_functions
import src.utils.modules as modules
import src.utils.training as training
import src.utils.utils as utils



device, opt, parser = None, None, None

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
    check_cmd_line_options()

    # Get input image to be compressed.
    if opt.image_filepath is None:
        img_dataset = dataio.Camera()
        coord_dataset = dataio.Implicit2DWrapper(img_dataset, sidelength=512, compute_diff='all')
        image_resolution = (512, 512)
    else:
        img_dataset =  np.asarray(Image.open(opt.image_filepath))
        if opt.sidelength is None:
            opt.sidelength = min(img_dataset.shape)
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