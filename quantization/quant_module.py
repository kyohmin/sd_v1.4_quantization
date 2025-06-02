import torch
import torch.nn as nn
import torch.nn.functional as F

from quantization.quantizer import Quantizer

class StraightThrough(nn.Module):
    def __init__(self, channel_num: int = 1):
        super().__init__()

    def forward(self, input):
        return input

class QuantModule(nn.Module):
    def __init__(self, org_module, quant_hyperparams):
        super().__init__()
        self.weight_quant_params, self.act_quant_params = quant_hyperparams

        # Forward kwargs and function
        if isinstance(org_module, nn.Conv2d):
            self.fwd_kwargs = dict(stride=org_module.stride,
                                   padding=org_module.padding,
                                   dilation=org_module.dilation,
                                   groups=org_module.groups)
            self.fwd_func = F.conv2d
        elif isinstance(org_module, nn.Linear):
            self.fwd_kwargs = dict()
            self.fwd_func = F.linear

        # Weight
        self.weight = org_module.weight
        self.org_weight = org_module.weight.data.clone()

        # Bias 
        if org_module.bias is not None:
            self.bias = org_module.bias
            self.org_bias = org_module.bias.data.clone()
        else:
            self.bias = None
            self.org_bias = None

        # Turn off quantization by default
        self.use_weight_quant = False
        self.use_act_quant = False

        # Define Quantizers
        self.weight_quantizer = Quantizer(**self.weight_quant_params)
        self.act_quantizer = Quantizer(**self.act_quant_params)

        self.split = 0

        self.activation_function = StraightThrough() # Placeholder
        self.ignore_reconstruction = False

    def forward(self, input: torch.Tensor, split: int = 0):
        if split != 0 and self.split != 0: assert(split == self.split)
        elif split != 0:
            self.split = split
            self.set_split() # Split Weight and Activation
        
        # Split Activation if exist
        if self.use_act_quant:
            if self.split != 0:
                input_0 = self.act_quantizer(input[:, :self.split, :, :])
                input_1 = self.act_quantizer_0(input[:, self.split:, :, :])
                input = torch.cat([input_0, input_1], dim = 1)
            else:
                input = self.act_quantizer(input)
        
        # Split Weight if exist
        if self.use_weight_quant:
            if self.split != 0:
                weight_0 = self.weight_quantizer(self.weight[:, :self.split, ...])
                weight_1 = self.weight_quantizer_0(self.weight[:, self.split:, ...])
                weight = torch.cat([weight_0, weight_1], dim = 1)
            else:
                weight = self.weight_quantizer(self.weight)
            bias = self.bias
        else:
            weight = self.org_weight
            bias = self.org_bias

        out = self.fwd_func(input, weight, bias, **self.fwd_kwargs)
        out = self.activation_function(out)

        return out

    def set_quant_state(self, weight_quant: bool = False, act_quant: bool = False):
        self.use_weight_quant = weight_quant
        self.use_act_quant = act_quant

    def set_split(self):
        self.weight_quantizer_0 = Quantizer(**self.weight_quant_params)
        self.act_quantizer_0 = Quantizer(**self.act_quant_params)

    def set_running_stat(self, running_stat: bool):
        self.act_quantizer.running_stat = running_stat
        if self.split != 0:
            self.act_quantizer_0.running_stat = running_stat
