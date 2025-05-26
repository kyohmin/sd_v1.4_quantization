import torch
import torch.nn as nn
import torch.nn.functional as F
            
from quantizer import Quantizer

class StraightThrough(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, input):
        return input

class QuantModule(nn.Module):
    def __init__(self, original_module: nn.Conv2d | nn.Linear, weight_quant_params: dict = {}, act_quant_params: dict = {}, bit_shift: int = 4):
        super().__init__()
        self.weight_quant_params = weight_quant_params
        self.act_quant_params = act_quant_params
        self.split = 0

        if isinstance(original_module, nn.Conv2d):
            self.forward_kwargs = dict(stride = original_module.stride, padding = original_module.padding, dilation = original_module.dilation, groups = original_module.groups)
            self.forward_function = F.conv2d
        elif isinstance(original_module, nn.Linear):
            self.forward_kwargs = dict()
            self.forward_function = F.linear

        self.weigth = original_module.weight
        self.original_weight = original_module.weight.detach().clone()
        if original_module.bias is not None:
            self.bias = original_module.bias
            self.original_bias = original_module.bias.detach().clone()
        else:
            self.bias = None
            self.original_bias = None

        self.weight_quantizer = Quantizer(**self.weight_quant_params)
        self.act_quantizer = Quantizer(**self.act_quant_params)
        
        self.activation_function = StraightThrough()
        self.ignore_reconstruction = False

    def forward(self, input: torch.Tensor, split: int = 0):
        # if split != 0 and self.split != 0:
        #     assert(split == self.split)
        # elif split != 0:
        #     self.split = split
        #     self.set_split()

        if self.use_act_quant:
            # if self.split != 0:
            #     input_0 = self.act_quantizer(input[:, :self.split, ...])
            #     input_1 = self.act_quantizer_0(input[:, self.split:, ...])
            #     input = torch.cat([input_0, input_1], dim=1)
            # else:
            input = self.act_quantizer(input)

        if self.use_weight_quant:
            # if self.split != 0:
            #     weight_0 = self.weight_quantizer(self.weight[:, :self.split, ...])
            #     weight_1 = self.weight_quantizer_0(self.weight[:, self.split, ...])
            #     weight = torch.cat([weight_0, weight_1], dim=1)
            # else:
            weight = self.weight_quantizer(self.weight)
            bias = self.bias
        else:
            weight = self.original_weight
            bias = self.original_bias

        output = self.forward_function(input, weight, bias, **self.forward_kwargs)
        output = self.activation_function(output)

        return output

    # def set_split(self):
    #     self.weight_quantizer_0 = Quantizer(**self.weight_quant_params)
    #     self.act_quantizer_0 = Quantizer(**self.act_quant_params)

    def set_quant_state(self, use_weight_quant: bool = False, use_act_quant: bool = False):
        self.use_weight_quant = use_weight_quant
        self.use_act_quant = use_act_quant

    def set_running_stat(self, running_stat:bool):
        self.act_quantizer.running_stat = running_stat
        # if self.split != 0: self.act_quantizer_0.running_stat = running_stat
