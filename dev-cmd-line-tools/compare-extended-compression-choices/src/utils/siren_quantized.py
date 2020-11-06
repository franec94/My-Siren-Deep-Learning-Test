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

class SineLayerQuantized(nn.Module):
    # See paper sec. 3.2, final paragraph, and supplement Sec. 1.5 for discussion of omega_0.
    
    # If is_first=True, omega_0 is a frequency factor which simply multiplies the activations before the 
    # nonlinearity. Different signals may require different omega_0 in the first layer - this is a 
    # hyperparameter.
    
    # If is_first=False, then the weights will be divided by omega_0 so as to keep the magnitude of 
    # activations constant, but boost gradients to the weight matrix (see supplement Sec. 1.5)
    
    def __init__(self, in_features, out_features, bias=True,
                 is_first=False, omega_0=30):
        super().__init__()
        self.omega_0 = omega_0
        self.is_first = is_first
        
        self.in_features = in_features
        self.linear = nn.Linear(in_features, out_features, bias=bias)
        # self.linear = nn.quantized.dynamic.modules.Linear(in_features, out_features, bias_=bias)
        
        self.init_weights()

        # Objetc to handle quantization/unquantizing data.
        self.quant = QuantStub()
        self.dequant = DeQuantStub()
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
        # Quantize input data
        input_quant = self.quant(input)

        # Process quantized data
        x = self.linear(input_quant)

        # Unquantized Processed data
        
        # qmul = QFunctional()
        # scalar_omega_0 = torch.Tensor([self.omega_0])
        # scalar_omega_0 = torch.quantize_per_tensor(scalar_omega_0, 0.01, 0, torch.qint8)
        # x = qmul.mul_scalar(x, scalar_omega_0)
        x = self.omega_0 * x
        x = torch.sin(x)
        
        x = self.dequant(x)
        # Let unquantized data flow through Activation 
        return x
    
    def forward_with_intermediate(self, input):
         # Quantize input data
        input_quant = self.quant(input)
        
        # Process quantized data
        x = self.linear(input_quant)
        qmul = QFunctional()
        scalar_omega_0 = torch.Tensor([self.omega_0])
        scalar_omega_0 = torch.quantize_per_tensor(t, 0.01, 0, torch.qint8)
        x = qmul.mul_scalar(x, scalar_omega_0)

        # Unquantized Processed data
        x_dequant = self.dequant(x)
        # Let unquantized data flow through Activation
        x_out = torch.sin(x_dequant)
        
        return x_out, x_dequant
    pass
    
    
class SirenQuantized(nn.Module):
    def __init__(self, in_features, hidden_features, hidden_layers, out_features, outermost_linear=False, 
                 first_omega_0=30, hidden_omega_0=30.):
        super().__init__()
        
        self.net = []
        self.net.append(SineLayerQuantized(in_features, hidden_features, 
                                  is_first=True, omega_0=first_omega_0))

        for i in range(hidden_layers):
            self.net.append(SineLayerQuantized(hidden_features, hidden_features, 
                                      is_first=False, omega_0=hidden_omega_0))

        if outermost_linear:
            final_linear = nn.Linear(hidden_features, out_features)
            
            with torch.no_grad():
                final_linear.weight.uniform_(-np.sqrt(6 / hidden_features) / hidden_omega_0, 
                                              np.sqrt(6 / hidden_features) / hidden_omega_0)
                
            self.net.append(final_linear)
        else:
            self.net.append(SineLayerQuantized(hidden_features, out_features, 
                                      is_first=False, omega_0=hidden_omega_0))
        
        self.net = nn.Sequential(*self.net)
        # self.quant = QuantStub()
        # self.dequant = DeQuantStub()
        pass
    
    def forward(self, coords):
        # x = self.quant(coords)
        coords = coords.clone().detach().requires_grad_(True) # allows to take derivative w.r.t. input
        # output = self.dequant(self.net(x))
        output = self.net(coords)
        return output, coords        

    def forward_with_activations(self, coords, retain_grad=False):
        '''Returns not only model output, but also intermediate activations.
        Only used for visualizing activations later!'''
        activations = OrderedDict()

        activation_count = 0
        x = coords.clone().detach().requires_grad_(True)
        activations['input'] = x
        for i, layer in enumerate(self.net):
            if isinstance(layer, SineLayerQuantized):
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
    pass
