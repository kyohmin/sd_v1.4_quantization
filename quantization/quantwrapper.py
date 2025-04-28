import torch
import torch.nn as nn
import torch.nn.functional as F

class Quantization:
    @staticmethod
    def quantize(original_tensor, quantization_mode = "asymmetric", range_estimator_type = "min_max", bits=8, zero = None, scale = None):
        if quantization_mode =="asymmetric": min_q, max_q, dtype = 0, 2**bits - 1, torch.uint8
        elif quantization_mode =="symmetric": min_q, max_q, dtype = -(2**(bits-1) - 1), (2**(bits-1) - 1), torch.int8

        # Activation Quantization (scale and zero already given)
        if zero is not None and scale is not None:
            quantized_tensor =  torch.clamp(torch.round(original_tensor / scale + zero), min_q, max_q).to(dtype)
            return quantized_tensor, scale, zero, dtype

        # Set Range Estimator type
        if range_estimator_type == "min_max":
            min_v = original_tensor.min().item()
            max_v = original_tensor.max().item()

        # Find zero and scale value
        if quantization_mode == "asymmetric":
            span = max_v - min_v
            scale = span / (max_q - min_q) if span != 0 else 1.0
            zero = round(min_q - min_v / scale)
            zero = int(max(min_q, min(max_q, zero)))

        elif quantization_mode == "symmetric":
            S = max(abs(min_v), abs(max_v))
            scale = S / max_q if S != 0 else 1.0
            zero = 0

        quantized_tensor = torch.round(original_tensor / scale + zero)
        quantized_tensor = torch.clamp(quantized_tensor, min_q, max_q).to(dtype)
        return quantized_tensor, scale, zero, dtype

    @staticmethod
    def dequantize(quantized_tensor, zero, scale):
        dequantized_tensor = (quantized_tensor.float() - zero) * scale
        return dequantized_tensor

class QuantWrapper(nn.Module):
    def __init__(self, module, weight=None, weight_scale=None, weight_zero=None, activation_zero=None, activation_scale=None):
        super().__init__()
        self.module = module
        self.use_weightquant = False
        self.use_activationquant = True
        self.failed = False
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

            # Bit Shift
            shift_bits = 4 # 4 bit shift to fit the range for fp16
            quantized_activation = (quantized_activation.to(torch.float32) - self.activation_zero) >> shift_bits
            quantized_weight = (self.weight.to(torch.float32) - self.weight_zero) >> shift_bits

            # Conv2d 
            if isinstance(self.module, nn.Conv2d):
                output_16 = F.conv2d(quantized_activation, quantized_weight, bias=None, stride=self.module.stride, padding=self.module.padding, dilation=self.module.dilation, groups=self.module.groups) 
                output_16 = output_16 * (self.weight_scale * self.activation_scale) * (1 << (2*shift_bits))
                if self.module.bias is not None:
                    output_16 = output_16 + self.module.bias.view(1,-1,1,1)

                # If failed, compute original conv2d
                if torch.any(torch.isnan(output_16)) or torch.any(torch.isinf(output_16)):
                    self.failed = True
                    dequantized_weight = Quantization.dequantize(self.weight, self.weight_zero, self.weight_scale)
                    return F.conv2d(x, dequantized_weight, self.module.bias, self.module.stride, self.module.padding, self.module.dilation, self.module.groups)
                return output_16

            # Linear  
            elif isinstance(self.module, nn.Linear):
                output_16 = F.linear(quantized_activation, quantized_weight, bias = None)
                output_16 = output_16 * (self.weight_scale * self.activation_scale) * (1 << (2*shift_bits))
                if self.module.bias is not None:
                    output_16 = output_16 + self.module.bias

                # If failed, compute original linear
                if torch.any(torch.isnan(output_16)) or torch.any(torch.isinf(output_16)):
                    self.failed = True
                    dequantized_weight = Quantization.dequantize(self.weight, self.weight_zero, self.weight_scale)
                    return F.linear(x, dequantized_weight, self.module.bias)

                return output_16
            
        # For dequantize quantized wegiht for original computation
        elif self.use_weightquant and not self.use_activationquant:
            dequantized_weight = Quantization.dequantize(self.weight, self.weight_zero, self.weight_scale)
            if isinstance(self.module, nn.Conv2d):
                return F.conv2d(x, dequantized_weight, self.module.bias, self.module.stride, self.module.padding, self.module.dilation, self.module.groups)
            elif isinstance(self.module, nn.Linear):
                return F.linear(x, dequantized_weight, self.module.bias)

        else:
            return self.module(x)