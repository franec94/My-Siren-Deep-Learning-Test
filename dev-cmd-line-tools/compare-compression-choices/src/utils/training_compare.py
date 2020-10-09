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


def train_protocol_compare_archs(arch_hyperparams, coord_dataset, opt):

    dataloader = DataLoader(
        coord_dataset,
        shuffle=True,
        batch_size=opt.batch_size,
        pin_memory=True, num_workers=0)

    """
    model = Siren(
        in_features = 2,
        out_features = 1,
        hidden_features = image_resolution[0], 
        hidden_layers = 3,
        outermost_linear=True)
    """

    for _, (_, hidden_features, hidden_layers) in enumerate(arch_hyperparams):
        print(hidden_features, hidden_layers)
        """
        model = Siren(
            in_features = 2,
            out_features = 1,
            hidden_features = hidden_features, 
            hidden_layers = hidden_layers,
            outermost_linear=True)
        """
        pass

    pass
