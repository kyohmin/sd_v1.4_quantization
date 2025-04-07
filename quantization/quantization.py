import torch
import torch.nn as nn
import numpy as np
from ldm.util import instantiate_from_config
from omegaconf import OmegaConf

class Quantization:
    @staticmethod
    def quantize(original_tensor, quantization_mode = "asymmetric", range_estimator_type = "min_max", bits=8, zero = None, scale = None):
        
        if quantization_mode =="asymmetric": min_q, max_q, dtype = 0, 2**bits - 1, torch.uint8
        elif quantization_mode =="symmetric": min_q, max_q, dtype = -(2**(bits-1) - 1), (2**(bits-1) - 1), torch.int8

        # Activation Quantization (scale and zero already given)
        # if zero is not None and scale is not None:
        #     quantized_tensor =  torch.clamp(torch.round(original_tensor / scale + zero), min_q, max_q).to(dtype)
        #     return quantized_tensor, scale, zero, dtype

        # Set Range Estimator type
        if range_estimator_type == "min_max":
            min_v = original_tensor.min().item()
            max_v = original_tensor.max().item()
        elif range_estimator_type == "percentile":
            numpy_tensor = original_tensor.detach().cpu().numpy()
            min_v = np.percentile(numpy_tensor, 1)
            max_v = np.percentile(numpy_tensor, 99)
        
        # Find zero and scale value
        if quantization_mode == "asymmetric":
            if (max_v - min_v) == 0:
                scale = 1.0 # Avoid zero division error
            else:
                scale = (max_v - min_v) / (max_q - min_q)
            zero = min_q - (min_v / scale)
            zero = int(round(zero))
        elif quantization_mode == "symmetric":
            max_abs = max(abs(min_v), abs(max_v))
            if max_abs == 0:
                scale = 1.0
            else:
                scale = max_abs / max_q
            zero = 0

        quantized_tensor = torch.round(original_tensor / scale + zero)
        quantized_tensor = torch.clamp(quantized_tensor, min_q, max_q).to(dtype)

        return quantized_tensor, scale, zero, dtype

    @staticmethod
    def dequantize(quantized_tensor, zero, scale):
        dequantized_tensor = (quantized_tensor.float() - zero) * scale
        return dequantized_tensor


    @staticmethod
    def quantization_metric(original_tensor, dequantized_tensor, option = "mse"):
        if option == "mse":
            loss = ((original_tensor - dequantized_tensor) ** 2).mean()
        elif option == "l1":
            loss = (abs(original_tensor - dequantized_tensor)).mean()

        return loss





    
