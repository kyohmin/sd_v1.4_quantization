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
    def add_wrapper(module, layer_types=(torch.nn.Conv2d, torch.nn.Linear), skip_keys=()):
        for name, child in module.named_children():
            if any(skip_key in name for skip_key in skip_keys):
                continue
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
        layer_types=(torch.nn.Linear, torch.nn.Conv2d),
        uniform_type="asymmetric",
        calibration_type = "min_max",
        bits = 8
        ):
        for name, module in model.model.diffusion_model.named_modules():
            if isinstance(module, QuantWrapper) and isinstance(module.module, layer_types):
                with torch.no_grad():
                    if module.weight == None: weight = module.module.weight.data 
                    else: weight = module.weight
                    quantized_tensor, scale, zero, dtype = Quantization.quantize(weight, uniform_type, calibration_type, bits)
                    module.update_weight_params(quantized_tensor, scale, zero, dtype)
                    module.update_dict({"quantization_mode":uniform_type, "range_estimator_type":calibration_type, "bits":bits})
                    if 'weight' in module.module._parameters:
                        del module.module._parameters['weight']

        return model
    
    # @staticmethod
    # def quantize_model_weights(
    #     model,
    #     layer_types=(torch.nn.Linear, torch.nn.Conv2d),
    #     uniform_type="asymmetric",
    #     calibration_type="min_max",
    #     bits=8
    # ):
    #     for name, module in model.model.diffusion_model.named_modules():
    #         if isinstance(module, QuantWrapper) and isinstance(module.module, layer_types):
    #             with torch.no_grad():
    #                 # grab either the already‑stored quant weight or the raw float weight
    #                 weight = module.weight if module.weight is not None else module.module.weight.data

    #                 # use per‑channel quantization
    #                 q_w, scales, zero_points, dtype = Quantization.quantize_by_channel(
    #                     weight,
    #                     uniform_type,
    #                     calibration_type,
    #                     bits
    #                 )

    #                 # update the wrapper with the new params
    #                 module.update_weight_params(q_w, scales, zero_points, dtype)
    #                 module.update_dict({
    #                     "quantization_mode": uniform_type,
    #                     "range_estimator_type": calibration_type,
    #                     "bits": bits
    #                 })

    #                 # remove the original float parameter so only q_w remains
    #                 if 'weight' in module.module._parameters:
    #                     del module.module._parameters['weight']

    #     return model

    # Activation Calibration
    @staticmethod
    def calibrate_activation_parameters(model, ):
        pass

    @staticmethod
    def check_failure(
        model,
        layer_types=(torch.nn.Linear, torch.nn.Conv2d),
    ):
        from torch.nn import Module

        seen_ids = set()
        original, success, failed = 0, 0, 0
        linear_success, conv2d_success = 0, 0
        linear_failed, conv2d_failed = 0, 0

        def scan(module: Module, prefix=""):
            nonlocal original, success, failed
            nonlocal linear_success, conv2d_success, linear_failed, conv2d_failed

            for name, child in module.named_children():
                full_name = f"{prefix}.{name}" if prefix else name

                if id(child) in seen_ids:
                    continue
                seen_ids.add(id(child))

                # Case 1: QuantWrapper with target module
                if isinstance(child, QuantWrapper) and isinstance(child.module, layer_types):
                    if getattr(child, "failed", None) is False:
                        print(f"Success  - {full_name}", child.module)
                        success += 1
                        if isinstance(child.module, torch.nn.Linear):
                            linear_success += 1
                        elif isinstance(child.module, torch.nn.Conv2d):
                            conv2d_success += 1
                    elif getattr(child, "failed", None) is True:
                        print(f"Failed   - {full_name}", child.module)
                        failed += 1
                        if isinstance(child.module, torch.nn.Linear):
                            linear_failed += 1
                        elif isinstance(child.module, torch.nn.Conv2d):
                            conv2d_failed += 1

                    continue

                # Case 2: Unwrapped Linear/Conv2d
                elif isinstance(child, layer_types):
                    print(f"Original - {full_name}", child)
                    original += 1

                # Recurse into other children
                else:
                    scan(child, prefix=full_name)

        scan(model.model.diffusion_model)

        print(f"Original (unwrapped): {original}")
        print(f"Quantized Success   : {success}")
        print(f"Quantized Failed    : {failed}")
        print(f"Success - Linear    : {linear_success}")
        print(f"Success - Conv2d    : {conv2d_success}")
        print(f"Failed - Linear     : {linear_failed}")
        print(f"Failed - Conv2d     : {conv2d_failed}")