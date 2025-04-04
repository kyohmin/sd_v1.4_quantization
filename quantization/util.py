import torch

from quantization.quantization import Quantization
from quantization.quantwrapper import QuantWrapper

class Util:
    # MODEL WRAPPING =====
    @staticmethod
    def rewrap(model, layer_types=(torch.nn.Conv2d, torch.nn.Linear)):
        Util.add_wrapper(model, layer_types)
        return model

    @staticmethod
    def add_wrapper(module, layer_types=(torch.nn.Conv2d, torch.nn.Linear)):
        for name, child in module.named_children():
            if isinstance(child, layer_types):
                setattr(module, name, QuantWrapper(child))
            else:
                # Recursively wrap inner children
                Util.add_wrapper(child, layer_types)
        return module
    
    # QUANTIZE WRAPPED MODEL =====
    @staticmethod
    def quantize_model_weights(
        model,
        layer_types=(torch.nn.Conv2d, torch.nn.Linear),
        uniform_type="asymmetric",
        calibration_type = "min_max",
        bits = 8
        ):
        for name, module in model.named_modules():
            if isinstance(module, QuantWrapper) and isinstance(module.module, layer_types):
                with torch.no_grad():
                    if module.weight == None: weight = module.module.weight.data # PROBLEM
                    else: weight = module.weight
                    quantized_tensor, scale, zero, dtype = Quantization.quantize(weight)
                    module.update_weight_params(quantized_tensor, scale, zero)
                    if 'weight' in module.module._parameters:
                        del module.module._parameters['weight']

        return model

    # Activation Calibration
    @staticmethod
    def calibrate_activation_parameters(model, ):
        pass
