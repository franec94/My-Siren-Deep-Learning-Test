# ============================================= #
# Standard Imports
# ============================================= #
from PIL import Image

import copy
import skimage

import matplotlib.pyplot as plt

# ============================================= #
# Torch Imports
# ============================================= #

import torch
import torchvision

import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset

from torchvision import datasets
from torchvision import transforms
from torchvision.transforms import Resize, Compose, ToTensor, Normalize


from utils.functions import divergence, get_mgrid, gradient, laplace

# ============================================= #
# Utils Classes and Functions
# ============================================= #

class ImageFitting(Dataset):
    def __init__(self, sidelength):
        super().__init__()
        img = get_cameraman_tensor(sidelength)
        self.pixels = img.permute(1, 2, 0).view(-1, 1)
        self.coords = get_mgrid(sidelength, 2)
        pass

    def __len__(self):
        return 1

    def __getitem__(self, idx):    
        if idx > 0: raise IndexError
            
        return self.coords, self.pixels
    pass


def get_cameraman_tensor(sidelength):
    img = Image.fromarray(skimage.data.camera())        
    transform = Compose([
        Resize(sidelength),
        ToTensor(),
        Normalize(torch.Tensor([0.5]), torch.Tensor([0.5]))
    ])
    img = transform(img)
    return img


def show_images(model_output, coords, sidelength):
    img_grad = gradient(model_output, coords)
    img_laplacian = laplace(model_output, coords)

    fig, axes = plt.subplots(1,3, figsize=(18,6))
    axes[0].imshow(model_output.cpu().view(sidelength, sidelength).detach().numpy())
    axes[1].imshow(img_grad.norm(dim=-1).cpu().view(sidelength, sidelength).detach().numpy())
    axes[2].imshow(img_laplacian.cpu().view(sidelength, sidelength).detach().numpy())
    plt.show()
    pass

# ============================================= #
# basic_traininig_loop
# ============================================= #

def basic_traininig_loop(optimizer, model, model_input, ground_truth, total_steps, sidelength, steps_til_summary = 10):
    
    train_acc_history = [] # val_acc_history = []
    train_loss_history = [] # val_loss_history = []

    best_model_wts = copy.deepcopy(model.state_dict())

    model.train()  # Set model to training mode

    for step in range(total_steps):
        # print('Epoch {}/{}'.format(step, total_steps - 1))
        # print('-' * 10)

        model_output, coords = model(model_input)    
        loss = ((model_output - ground_truth)**2).mean()
    
        if not step % steps_til_summary:
            print("Step %d, Total loss %0.6f" % (step, loss))
            show_images(model_output, coords, sidelength)
            pass
        
        train_loss_history.append(loss)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        best_model_wts = copy.deepcopy(model.state_dict())
        pass
    

    # load best model weights
    model.load_state_dict(best_model_wts)

    # create history as python dictionary
    keys_history_list = "train_loss".split(",")
    values_history_list = [train_loss_history]

    history = dict(zip(keys_history_list, values_history_list))

    return model, history

# ============================================= #
# plane_traininig_loop
# ============================================= #

def plane_traininig_loop(optimizer, criterion, model, model_input, ground_truth, total_steps):
    
    train_acc_history = [] # val_acc_history = []
    train_loss_history = [] # val_loss_history = []

    best_model_wts = copy.deepcopy(model.state_dict())

    model.train()  # Set model to training mode

    phase = 'train'
    for step in range(total_steps):
        print('Epoch {}/{}'.format(step, total_steps - 1))
        print('-' * 10)

        model_output, coords = model(model_input)    
        loss = criterion(model_output, ground_truth)
        # loss = ((model_output - ground_truth)**2).mean()
        

        print('{} Loss: {:.4f}'.format(phase, loss))
        train_loss_history.append(loss)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        best_model_wts = copy.deepcopy(model.state_dict())
        pass
    

    # load best model weights
    model.load_state_dict(best_model_wts)

    # create history as python dictionary
    keys_history_list = "train_loss".split(",")
    values_history_list = [train_loss_history]

    history = dict(zip(keys_history_list, values_history_list))

    return model, history
