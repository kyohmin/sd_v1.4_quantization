import torch

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
        elif range_estimator_type == "percentile": # Not Working (also the tensor -> numpy)
            min_v = original_tensor.quantile(0.01).item()
            max_v = original_tensor.quantile(0.99).item()
        
        # Find zero and scale value
        if quantization_mode == "asymmetric":
            span = max_v - min_v
            scale = span / (max_q - min_q) if span != 0 else 1.0
            zero = round(min_q - min_v / scale)
            zero = int(max(min_q, min(max_q, zero)))

        else:  # symmetric
            S = max(abs(min_v), abs(max_v))
            scale = S / max_q if S != 0 else 1.0
            zero = 0

        quantized_tensor = torch.round(original_tensor / scale + zero)
        quantized_tensor = torch.clamp(quantized_tensor, min_q, max_q).to(dtype)

        return quantized_tensor, scale, zero, dtype

    # @staticmethod
    # def quantize_by_channel(original_tensor, quantization_mode = "asymmetric", range_estimator_type = "min_max", bits=8, zero = None, scale = None):
    #     qmin, qmax = 0, 2**bits - 1
    #     dtype = torch.uint8

    #     # 1) figure out which dims to reduce:
    #     #    keep axis 0 (out‑channels or rows), collapse the rest
    #     dims_to_reduce = tuple(i for i in range(original_tensor.ndim) if i != 0)

    #     # 2) min/max per “channel” (axis 0)
    #     min_c = original_tensor.amin(dim=dims_to_reduce)
    #     max_c = original_tensor.amax(dim=dims_to_reduce)

    #     # 3) compute scale & zero_point (shape: (C_out,))
    #     scale = (max_c - min_c) / float(qmax - qmin)
    #     zero_point = torch.round(qmin - min_c / scale).clamp(qmin, qmax).to(torch.int32)

    #     # 4) reshape for broadcasting back onto the full tensor
    #     #    build a shape of [C_out, 1, 1, …] matching original_tensor.ndim
    #     bc_shape = [1] * original_tensor.ndim
    #     bc_shape[0] = original_tensor.shape[0]      # C_out
    #     scale_bc = scale.view(bc_shape)
    #     zero_bc  = zero_point.view(bc_shape)

    #     # 5) quantize
    #     q = torch.round(original_tensor / scale_bc + zero_bc)
    #     q = torch.clamp(q, qmin, qmax).to(dtype)

    #     return q, scale, zero_point, dtype

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
