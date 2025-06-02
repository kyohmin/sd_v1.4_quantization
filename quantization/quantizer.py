import torch
import torch.nn as nn

def round_ste(x: torch.Tensor):
        return (x.round() - x).detach() + x

def lp_loss(pred, tgt, p=2.0, reduction='none'):
    if reduction == 'none':
        return (pred-tgt).abs().pow(p).sum(1).mean()
    else:
        return (pred-tgt).abs().pow(p).mean()

class Quantizer(nn.Module):
    def __init__(self, n_bits: int = 8, channel_wise: bool = True, leaf_param: bool = False, scale_method: str = "min_max", always_zero: bool = False):
        super().__init__()
        self.n_bits = n_bits
        self.q_max = 2 ** self.n_bits - 1  # Asymmetric-only
        self.scale = None
        self.scale_method = scale_method
        self.zero = None
        self.inited = False
        self.channel_wise = channel_wise
        self.always_zero = always_zero

        # For Activation
        self.running_stat = False
        self.leaf_param = leaf_param
        if self.leaf_param:
            self.x_min = None
            self.x_max = None

    def forward(self, x: torch.Tensor):
        if not self.inited:
            self.scale, self.zero = self.new_get_scale_zero(x, self.channel_wise)
            if self.leaf_param:
                self.scale = torch.nn.Parameter(self.scale)
            self.inited = True

        if self.running_stat:
            self.act_momentum_update(x)

        # Fake Quantization
        x_quant = self.quantize(x, self.scale, self.zero)
        x_dequant = self.dequantize(x_quant, self.scale, self.zero)
        return x_dequant
    
    def quantize(self, x: torch.Tensor, scale: torch.Tensor, zero: torch.Tensor):
        x_quant = torch.clamp(self.round_ste(x / self.scale) + self.zero, 0, self.q_max)
        return x_quant

    def dequantize(self, x: torch.Tensor, scale: torch.Tensor, zero: torch.Tensor):
        x_dequant = (x - zero) * self.scale
        return x_dequant

    def new_get_scale_zero(self, x: torch.Tensor, channel_wise: bool = True):
        if channel_wise:
            n_channels = x.shape[0]
            scale = torch.zeros(n_channels, dtype=x.dtype, device=x.device)
            zero = torch.zeros(n_channels, dtype=x.dtype, device=x.device)

            if self.leaf_param:
                self.x_min = torch.zeros(n_channels, dtype=x.dtype, device=x.device)
                self.x_max = torch.zeros(n_channels, dtype=x.dtype, device=x.device)

            for c in range(n_channels):
                x_c = x[c]
                x_min = x_c.min().item()
                x_max = x_c.max().item()
                scale_val = max(x_max - x_min, 1e-8) / self.q_max
                zero_val = round(-x_min / scale_val)

                scale[c] = scale_val
                zero[c] = zero_val

                if self.leaf_param:
                    self.x_min[c] = x_min
                    self.x_max[c] = x_max

            shape = [-1] + [1] * (x.dim() - 1)
            return scale.view(*shape), zero.view(*shape)

        else:
            x_min = x.min().item()
            x_max = x.max().item()

            if self.leaf_param:
                self.x_min = x_min
                self.x_max = x_max

            scale_val = max(x_max - x_min, 1e-8) / self.q_max
            scale = torch.tensor(scale_val, dtype=x.dtype, device=x.device)
            zero = torch.round(torch.tensor((-x_min) / scale, dtype=x.dtype, device=x.device))

            return scale, zero

    def act_momentum_update(self, x: torch.Tensor, act_range_momentum: float = 0.95):
        assert(self.inited)
        assert(self.leaf_param)

        x_min = x.detach().min()
        x_max = x.detach().max()
        self.x_min = self.x_min * act_range_momentum + x_min * (1 - act_range_momentum)
        self.x_max = self.x_max * act_range_momentum + x_max * (1 - act_range_momentum)

        scale = torch.clamp((self.x_max - self.x_min), min=1e-8) / self.q_max
        zero = torch.round(torch.tensor(-x_min / scale.item(), dtype=x.dtype, device=x.device))

        self.scale = torch.nn.Parameter(scale)
        self.zero = zero