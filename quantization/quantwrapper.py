import torch
import torch.nn as nn
import torch.nn.functional as F

from quantization.quantization import Quantization

class QuantWrapper(nn.Module):
    def __init__(self, module, weight=None, weight_scale=None, weight_zero=None, activation_zero=None, activation_scale=None):
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

    def update_activation_params(self, activation_scale, activation_zero):
        self.activation_scale = torch.tensor(activation_scale, dtype=torch.float32)
        self.activation_zero = torch.tensor(activation_zero, dtype=torch.float32)
        self.use_activationquant = True

    def forward(self, x):
        # Calculate INT8 x INT8
        if self.use_weightquant and self.use_activationquant:
            quantized_activation, scale, zero, dtype = Quantization.quantize(x, quantization_mode="asymmetric", range_estimator_type="min_max", bits=8)
            self.update_activation_params(scale, zero)
            quantized_activation = quantized_activation.to(torch.float32) - self.activation_zero
            quantized_weight = self.weight.to(torch.float32) - self.weight_zero
            # print("act", quantized_activation)
            # print("weight", quantized_weight)
            if isinstance(self.module, nn.Conv2d):
                output_32 = F.conv2d(quantized_activation,quantized_weight, bias=None, stride=self.module.stride, padding=self.module.padding, dilation=self.module.dilation, groups=self.module.groups)
                output_32 = output_32 * (self.weight_scale * self.activation_scale)
                if self.module.bias is not None:
                    output_32 = output_32 + self.module.bias.view(1,-1,1,1)

                if torch.any(torch.isnan(output_32)) or torch.any(torch.isinf(output_32)):
                    print("Original - CONV2D")
                    dequantized_weight = Quantization.dequantize(self.weight, self.weight_zero, self.weight_scale)
                    return F.conv2d(x, dequantized_weight, self.module.bias, self.module.stride, self.module.padding, self.module.dilation, self.module.groups)
                print("INT8 - CONV2D")
                return output_32
            
            elif isinstance(self.module, nn.Linear):
                output_32 = F.linear(quantized_activation, quantized_weight, bias = None)
                output_32 = output_32 * (self.weight_scale * self.activation_scale)

                if self.module.bias is not None:
                    output_32 = output_32 + self.module.bias

                if torch.any(torch.isnan(output_32)) or torch.any(torch.isinf(output_32)):
                    print("Original - LINEAR")
                    dequantized_weight = Quantization.dequantize(self.weight, self.weight_zero, self.weight_scale)
                    return F.linear(x, dequantized_weight, self.module.bias)

                print("INT8 - LINEAR")
                return output_32
            
        # Calculate INT8 -> FP32 x FP32
        elif self.use_weightquant and not self.use_activationquant:
            dequantized_weight = Quantization.dequantize(self.weight, self.weight_zero, self.weight_scale)
            if isinstance(self.module, nn.Conv2d):
                return F.conv2d(x, dequantized_weight, self.module.bias, self.module.stride, self.module.padding, self.module.dilation, self.module.groups)
            elif isinstance(self.module, nn.Linear):
                return F.linear(x, dequantized_weight, self.module.bias)

        else:
            return self.module(x)




