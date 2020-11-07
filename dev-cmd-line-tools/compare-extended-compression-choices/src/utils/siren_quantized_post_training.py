import torch
import torchvision

import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from torch.quantization import QuantStub, DeQuantStub
from torch.nn.quantized import QFunctional

from torchvision import datasets
from torchvision import transforms
from torchvision.transforms import Resize, Compose, ToTensor, Normalize


from collections import OrderedDict
import matplotlib
import matplotlib.pyplot as plt
import numpy as np

class SineLayerQPT(nn.Module):
    """
    SineLayerQPT PyTorch's module for costructing Siren-like Neuarl Net Architectures.
    """
    # See paper sec. 3.2, final paragraph, and supplement Sec. 1.5 for discussion of omega_0.
    
    # If is_first=True, omega_0 is a frequency factor which simply multiplies the activations before the 
    # nonlinearity. Different signals may require different omega_0 in the first layer - this is a 
    # hyperparameter.
    
    # If is_first=False, then the weights will be divided by omega_0 so as to keep the magnitude of 
    # activations constant, but boost gradients to the weight matrix (see supplement Sec. 1.5)
    
    def __init__(self, in_features, out_features, bias=True,
                 is_first=False, omega_0=30):
        """
        Constructor for SineLayerQPT PyTorch's module for costructing Siren-like Neuarl Net Architectures.
        """
        super().__init__()

        self.omega_0 = omega_0
        self.is_first = is_first

        self.in_features = in_features
        self.linear = nn.Linear(in_features, out_features, bias=bias)
        self.relu = nn.ReLU()
        # self.linear = nn.quantized.dynamic.modules.Linear(in_features, out_features, bias_=bias)
        
        self.init_weights()
        # self.quant = QuantStub()
        # self.dequant = DeQuantStub()
        self.mult_x_omega0 = nn.quantized.FloatFunctional()
        pass
    
    def init_weights(self):
        with torch.no_grad():
            if self.is_first:
                self.linear.weight.uniform_(-1 / self.in_features, 
                                             1 / self.in_features)      
            else:
                self.linear.weight.uniform_(-np.sqrt(6 / self.in_features) / self.omega_0, 
                                             np.sqrt(6 / self.in_features) / self.omega_0)
        pass
        
    def forward(self, input):
        """Compute SineLlayerPTQ forward pass."""
        x = self.linear(input)
        # return torch.sin(self.omega_0 * x)
        x = self.mult_x_omega0.mul_scalar(x, self.omega_0)
        # return self.relu(x) # Relu works, but still problems for understanding how to exploit fuse_modules option, in fact yet not correctly understood.
        return torch.sin(x)
    
    def forward_with_intermediate(self, input):
        input_quant = self.quant(input)
        # For visualization of activation distributions
        intermediate = self.omega_0 * self.linear(input_quant)
        return torch.sin(intermediate), intermediate
    pass
    
    
class SirenQPT(nn.Module):
    """
    SirenQPT class implements a Siren based Neural Network Architecture for quantization applied post trainining.
    """
        
    def __init__(self, in_features, hidden_features, hidden_layers, out_features, outermost_linear=False, 
                 first_omega_0=30, hidden_omega_0=30., engine = 'fbgemm', quantize=False):
        """
        Initialize a new instance from SirenQPT class, which define and implement a Siren based model for Post-Training Quantization.
        """
        super().__init__()
        
        self.net = []
        self.net.append(SineLayerQPT(in_features, hidden_features, 
                                  is_first=True, omega_0=first_omega_0))

        for i in range(hidden_layers):
            a_layer = SineLayerQPT(hidden_features, hidden_features, 
                                      is_first=False, omega_0=hidden_omega_0)
            a_layer.qconfig = torch.quantization.get_default_qat_qconfig(f'{engine}')
            a_layer = torch.quantization.prepare(a_layer)
            self.net.append(a_layer)

        if outermost_linear:
            final_linear = nn.Linear(hidden_features, out_features)
            
            with torch.no_grad():
                final_linear.weight.uniform_(-np.sqrt(6 / hidden_features) / hidden_omega_0, 
                                              np.sqrt(6 / hidden_features) / hidden_omega_0)
                
            self.net.append(final_linear)
        else:
            self.net.append(SineLayerQPT(hidden_features, out_features, 
                                      is_first=False, omega_0=hidden_omega_0))
        
        self.net = nn.Sequential(*self.net)
        self.quantize = quantize
        self.engine = engine
        if quantize:
            self.quant = torch.quantization.QuantStub()
            self.dequant = torch.quantization.DeQuantStub()
            pass
        pass
    
    def forward(self, coords):
        """Forward pass, if quantize was passed as True then quantization and dequantazing operations are done respectively
        before net computation starts and before returning model's output.
        """
        coords = coords.clone().detach().requires_grad_(True) # allows to take derivative w.r.t. input
        if self.quantize:
            coords_quanted = self.quant(coords)
            x = self.net(coords_quanted)
            output = self.dequant(x)
        else:
            output = self.net(coords)
            pass
        return output, coords        

    def forward_with_activations(self, coords, retain_grad=False):
        '''Returns not only model output, but also intermediate activations.
        Only used for visualizing activations later!'''
        activations = OrderedDict()

        activation_count = 0
        x = coords.clone().detach().requires_grad_(True)
        activations['input'] = x
        for i, layer in enumerate(self.net):
            if isinstance(layer, SineLayerQPT):
                x, intermed = layer.forward_with_intermediate(x)
                
                if retain_grad:
                    x.retain_grad()
                    intermed.retain_grad()
                    
                activations['_'.join((str(layer.__class__), "%d" % activation_count))] = intermed
                activation_count += 1
            else: 
                x = layer(x)
                
                if retain_grad:
                    x.retain_grad()
                    
            activations['_'.join((str(layer.__class__), "%d" % activation_count))] = x
            activation_count += 1

        return activations
        
    def fuse_model(self):
        """Call it in order to try fusing module from which net is done, it really takes effect when quantize input paramenter
        was set to True at instantiating time.
        """
        if self.quantize:
            for m in self.modules():
                if type(m) == SineLayerQPT:
                    # torch.quantization.fuse_modules(m, ['linear', 'relu'], inplace=True)
                    pass
                pass
            pass
        pass
    pass
