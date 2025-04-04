import torch
import torch.nn as nn
import torch.nn.functional as F

from quantization.quantization import Quantization

class QuantWrapper(nn.Module):
    def __init__(self, module, weight=None, weight_scale=None, weight_zero=None, activation_zero=0, activation_scale=1.0):
        super().__init__()
        self.module = module
        self.use_weightquant = False
        self.use_activationquant = True

        # Weight Quant Parameters
        self.register_buffer("weight", weight.to(torch.uint8) if weight is not None else None)
        self.register_buffer("weight_scale", torch.tensor(weight_scale, dtype=torch.float32) if weight_scale is not None else None)
        self.register_buffer("weight_zero", torch.tensor(weight_zero, dtype=torch.int32) if weight_zero is not None else None)

        # Activation Quant Parameters
        self.register_buffer("activation_scale", torch.tensor(activation_scale, dtype=torch.float32) if activation_scale is not None else None)
        self.register_buffer("activation_zero", torch.tensor(activation_zero, dtype=torch.int32) if activation_zero is not None else None)

    def update_weight_params(self, weight, scale, zero, dtype=torch.uint8):
        self.weight = torch.tensor(weight, dtype=dtype)
        self.weight_scale = torch.tensor(scale, dtype=torch.float32)
        self.weight_zero = torch.tensor(zero, dtype=torch.float32)
        self.use_weightquant = True

    def update_activation_params(self, act_scale, act_zero):
        self.act_scale = torch.tensor(act_scale, dtype=torch.float32)
        self.act_zero = torch.tensor(act_zero, dtype=torch.float32)
        self.use_activationquant = True

    def forward(self, x):
        # Calculate INT8 x INT8
        if self.use_weightquant and self.use_activationquant:
            quantized_activation, self.activation_scale, self.activation_zero, dtype = Quantization.quantize(x, quantization_mode="asymmetric", range_estimator_type="min_max", bits=8, zero=self.activation_zero, scale=self.activation_scale)
            if isinstance(self.module, nn.Conv2d):
                return F.conv2d(quantized_activation.to(torch.float32), self.weight.to(torch.float32), self.module.bias, self.module.stride, self.module.padding, self.module.dilation, self.module.groups)
            elif isinstance(self.module, nn.Linear):
                output = Quantization.int8_compute(quantized_weight=self.weight.to(torch.float32), quantized_activation=quantized_activation.to(torch.float32), target="linear", weight_scale=self.weight_scale, activation_scale=self.activation_scale, bias=self.module.bias)
                # return F.linear(quantized_activation.to(torch.float32), self.weight.to(torch.float32), self.module.bias)
                return output
            
        # Calculate INT8 -> FP32 x FP32
        elif self.use_weightquant and not self.use_activationquant:
            dequantized_weight = Quantization.dequantize(self.weight, self.weight_zero, self.weight_scale)
            if isinstance(self.module, nn.Conv2d):
                return F.conv2d(x, dequantized_weight, self.module.bias, self.module.stride, self.module.padding, self.module.dilation, self.module.groups)
            elif isinstance(self.module, nn.Linear):
                return F.linear(x, dequantized_weight, self.module.bias)

        else:
            return self.module(x)




