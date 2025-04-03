import torch
import torch.nn as nn
from ldm.util import instantiate_from_config
from omegaconf import OmegaConf

from quantization.quantwrapper import QuantWrapper

# CUSTOM Wrapper ============
class Quantization:
    @staticmethod
    def quantize(original_tensor,uniform_type = "asymmetric", calibration = "min_max", bits=8):
        # Calibration Type
        if calibration == "min_max":
            min_v, max_v = original_tensor.min().item(), original_tensor.max().item()
        # tensor does not support percentile (use tensor -> numpy -> tensor)
        # elif calibration == "percentile":
            # min_v, max_v = np.percentile(original_tensor, 1), np.percentile(original_tensor, 99)
        
        if uniform_type == "asymmetric":
            min_q, max_q = 0, 2**bits - 1
            range_v = max_v - min_v
            if range_v == 0:
                scale = 1.0
            else:
                scale = range_v / (max_q - min_q)
            zero = min_q - (min_v / scale)
            zero = int(round(zero))
            dtype = torch.uint8

        elif uniform_type == "symmetric":
            min_q, max_q = -(2**(bits-1) - 1), (2**(bits-1) - 1)
            max_abs = max(abs(min_v), abs(max_v))
            if max_abs == 0:
                scale = 1.0
            else:
                scale = max_abs / max_q

            zero = 0
            dtype = torch.int8

        quantized_tensor = torch.round(original_tensor / scale + zero)
        quantized_tensor = torch.clip(quantized_tensor, min_q, max_q).to(dtype)

        return quantized_tensor, scale, zero

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

    @staticmethod
    def weight_quantization(model, layer_types=(nn.Conv2d, nn.Linear),uniform_type = "asymmetric",calibration_type = "min_max",bits = 8):
        for name, module in model.named_modules():
            if isinstance(module, QuantWrapper) and isinstance(module.module, layer_types):
                with torch.no_grad():
                    module.quantize(uniform_type=uniform_type, calibration=calibration_type, bits=bits)
                    print("WORKED")

        return model

    @staticmethod
    def activation_quantization(self):
        pass
    
    @staticmethod
    def quantized_ckpt(model, name="quantized.ckpt")
        pass