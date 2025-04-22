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
        self.failed = False

        # Remove when optimizing model
        self.dict = {"quantization_mode":"asymmetric", "range_estimator_type":"min_max", "bits":8}

        # Weight Quant Parameters
        self.register_buffer("weight", weight.to(torch.uint8) if weight is not None else None) # Change it to be tensor or list
        self.register_buffer("weight_scale", torch.tensor(weight_scale, dtype=torch.float16) if weight_scale is not None else None) # Change it to be tensor or list
        self.register_buffer("weight_zero", torch.tensor(weight_zero, dtype=torch.int32) if weight_zero is not None else None) # Change it to be tensor or list

        # Activation Quant Parameters
        self.register_buffer("activation_scale", torch.tensor(activation_scale, dtype=torch.float16) if activation_scale is not None else None) # Change it to be tensor or list (Must match the length of weight channel)
        self.register_buffer("activation_zero", torch.tensor(activation_zero, dtype=torch.int32) if activation_zero is not None else None) # Change it to be tensor or list (Must match the length of weight channel)

    def update_weight_params(self, weight, scale, zero, dtype=torch.uint8):
        self.weight = torch.tensor(weight, dtype=dtype)
        self.weight_scale = torch.tensor(scale, dtype=torch.float16)
        self.weight_zero = torch.tensor(zero, dtype=torch.float16)
        self.use_weightquant = True

    def _update_activation_params(self, activation_scale, activation_zero):
        self.activation_scale = torch.tensor(activation_scale, dtype=torch.float16)
        self.activation_zero = torch.tensor(activation_zero, dtype=torch.float16)
        self.use_activationquant = True

    def update_dict(self, dict):
        self.dict = dict

    def forward(self, x):
        # Calculate INT8 x INT8
        if self.use_weightquant and self.use_activationquant:
            quantized_activation, scale, zero, dtype = Quantization.quantize(x, quantization_mode=self.dict["quantization_mode"], range_estimator_type=self.dict["range_estimator_type"], bits=self.dict["bits"])
            self._update_activation_params(scale, zero)

            shift_bits = 4
            quantized_activation = (quantized_activation.to(torch.float32) - self.activation_zero) >> shift_bits
            quantized_weight = (self.weight.to(torch.float32) - self.weight_zero) >> shift_bits

            if isinstance(self.module, nn.Conv2d):
                output_16 = F.conv2d(quantized_activation,quantized_weight, bias=None, stride=self.module.stride, padding=self.module.padding, dilation=self.module.dilation, groups=self.module.groups)
                output_16 = output_16 * (self.weight_scale * self.activation_scale) * (1 << (2*shift_bits))
                if self.module.bias is not None:
                    output_16 = output_16 + self.module.bias.view(1,-1,1,1)

                if torch.any(torch.isnan(output_16)) or torch.any(torch.isinf(output_16)):
                    # print("Original - CONV2D")
                    self.failed = True
                    dequantized_weight = Quantization.dequantize(self.weight, self.weight_zero, self.weight_scale)
                    return F.conv2d(x, dequantized_weight, self.module.bias, self.module.stride, self.module.padding, self.module.dilation, self.module.groups)
                # print("INT8 - CONV2D")
                return output_16
            
            elif isinstance(self.module, nn.Linear):
                output_16 = F.linear(quantized_activation, quantized_weight, bias = None)
                output_16 = output_16 * (self.weight_scale * self.activation_scale) * (1 << (2*shift_bits))

                if self.module.bias is not None:
                    output_16 = output_16 + self.module.bias

                if torch.any(torch.isnan(output_16)) or torch.any(torch.isinf(output_16)):
                    # print("Original - LINEAR")
                    self.failed = True
                    dequantized_weight = Quantization.dequantize(self.weight, self.weight_zero, self.weight_scale)
                    return F.linear(x, dequantized_weight, self.module.bias)

                # print("INT8 - LINEAR")
                return output_16
            
        # Calculate INT8 -> FP32 x FP32
        elif self.use_weightquant and not self.use_activationquant:
            dequantized_weight = Quantization.dequantize(self.weight, self.weight_zero, self.weight_scale)
            if isinstance(self.module, nn.Conv2d):
                return F.conv2d(x, dequantized_weight, self.module.bias, self.module.stride, self.module.padding, self.module.dilation, self.module.groups)
            elif isinstance(self.module, nn.Linear):
                return F.linear(x, dequantized_weight, self.module.bias)

        else:
            return self.module(x)




