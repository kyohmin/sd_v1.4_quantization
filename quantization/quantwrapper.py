import torch
import torch.nn as nn
import torch.nn.functional as F

class QuantWrapper(nn.Module):
    def __init__(self, module, weight=None, scale=None, zero=None):
        super().__init__()
        self.module = module
        self.use_quant = False

        if weight is not None:
            self.register_buffer("weight", weight.to(torch.uint8))
        else:
            self.register_buffer("weight", None)

        if scale is not None:
            self.register_buffer("scale", torch.tensor(scale, dtype=torch.float32))
        else:
            self.register_buffer("scale", None)

        if zero is not None:
            self.register_buffer("zero", torch.tensor(zero, dtype=torch.int32))
        else:
            self.register_buffer("zero", None)

    def quantize(self, original_tensor=None, uniform_type=None, calibration=None, bits=None):
        from quantization.quantization import Quantization

        if original_tensor is None:
            original_tensor = self.module.weight.data

        if original_tensor.dtype == torch.float32:
            weight_fp32 = self.module.weight.data
            quantized_weight, scale, zero = Quantization.quantize(weight_fp32, uniform_type=uniform_type, calibration=calibration, bits=bits)

            self.register_buffer("weight", quantized_weight.to(torch.uint8))
            self.register_buffer("scale", torch.tensor(scale, dtype=torch.float32))
            self.register_buffer("zero", torch.tensor(zero, dtype=torch.int32))
            self.use_quant = True

            if 'weight' in self.module._parameters:
                del self.module._parameters['weight']

    def forward(self, x):
        from quantization.quantization import Quantization
        # print("---> forward saved dtype:",self.weight.data.dtype)
        if self.use_quant: w = Quantization.dequantize(self.weight, self.zero, self.scale)
        else: w = None

        # print("---> forward process dtype:",w.data.dtype)
        if isinstance(self.module, nn.Conv2d):
            print("Conv2d")
            return F.conv2d(x, w, self.module.bias,self.module.stride,self.module.padding,self.module.dilation,self.module.groups)
        elif isinstance(self.module, nn.Linear):
            print("Linear")
            return F.linear(x, w, self.module.bias)
