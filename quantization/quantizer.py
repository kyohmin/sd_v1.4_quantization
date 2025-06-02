import torch
import torch.nn as nn


class Quantizer(nn.Module):
    def __init__(self, n_bits: int = 8, channel_wise: bool = True, leaf_param: bool = False, scale_method: str = "min_max"):
        super().__init__()
        self.n_bits = n_bits
        self.q_max = 2 ** self.n_bits - 1  # Asymmetric-only
        self.scale = None
        self.scale_method = scale_method
        self.zero = None
        self.inited = False
        self.channel_wise = channel_wise

        # For Activation
        self.running_stat = False
        self.leaf_param = leaf_param
        if self.leaf_param:
            self.x_min = None
            self.x_max = None

    def forward(self, x: torch.Tensor):
        # Initialize scale and zero
        if not self.inited:
            self.scale, self.zero = self.get_scale_zero(x, self.channel_wise)
            if self.leaf_param:
                self.scale = torch.nn.Parameter(self.scale)
            self.inited = True

        # Update min,max,scale,zero
        if self.running_stat: self.act_momentum_update(x)

        # Fake Quantization
        x_quant, _, _ = self.quantize(x)
        x_dequant = self.dequantize(x_quant)
        return x_dequant
    
    def quantize(self, x: torch.Tensor, x_min = None, x_max = None):
        if x_min is None and x_max is None:
            x_int = self.round_ste(x / self.scale) + self.zero
            x_quant = torch.clamp(x_int, 0, self.q_max)
            return x_quant, self.scale, self.zero
        
        scale = (x_max - x_min) / self.q_max
        zero = torch.round(torch.tensor(-x_min / scale, device=x.device, dtype=x.dtype))
        x_int = self.round_ste(x / scale) + zero
        x_quant = torch.clamp(x_int, 0, self.q_max)
        return x_quant, scale, zero

    def dequantize(self, x: torch.Tensor, scale = None, zero = None):
        if scale is None and zero is None:
            return (x - self.zero) * self.scale
        else:
            return (x - zero) * scale

    def get_scale_zero(self, x: torch.Tensor, channel_wise: bool = True):
        if channel_wise:
            n_channels = x.size(0)
            scale_tensor = torch.zeros(n_channels, dtype=x.dtype, device=x.device)
            zero_tensor = torch.zeros(n_channels, dtype=x.dtype, device=x.device)

            if self.leaf_param:
                self.x_min = torch.zeros(n_channels, dtype=x.dtype, device=x.device)
                self.x_max = torch.zeros(n_channels, dtype=x.dtype, device=x.device)

            for c in range(n_channels):
                x_c = x[c]
                c_min = x_c.min().item()
                c_max = x_c.max().item()
                
                if self.scale_method == 'min_max':
                    c_scale = max(c_max - c_min, 1e-8) / self.q_max
                    c_zero  = round(-c_min / c_scale)
                elif self.scale_method == 'mse':
                    best_loss = float('inf')
                    best_scale = None
                    best_zero = None
                    for i in range(90):
                        i_max = c_max * (1.0 - 0.01 * i)
                        i_min = c_min * (1.0 - 0.01 * i)

                        i_quant, i_scale, i_zero = self.quantize(x_c, i_min, i_max)
                        i_dequant = self.dequantize(i_quant, i_scale, i_zero)

                        loss = self.lp_loss(x_c.unsqueeze(0), i_dequant.unsqueeze(0), p=2.4, reduction ='none')
                        if loss < best_loss:
                            best_loss = loss
                            best_scale = i_scale
                            best_zero = i_zero
                        
                    c_scale = best_scale
                    c_zero = best_zero
                
                scale_tensor[c] = c_scale
                zero_tensor[c] = c_zero

                if self.leaf_param:
                    self.x_min[c] = c_min
                    self.x_max[c] = c_max
                
            shape = [n_channels] + [1] * (x.dim() - 1)
            return scale_tensor.view(*shape), zero_tensor.view(*shape)

        else:
            x_min = x.min().item()
            x_max = x.max().item()
            
            if self.scale_method == 'min_max': 
                scale = max(x_max - x_min, 1e-8) / self.q_max
                zero = round(-x_min / scale)

            elif self.scale_method == 'mse':
                best_loss = float('inf')
                best_scale = None
                best_zero = None
                for i in range(90):
                    i_max = x_max * (1.0 - 0.01 * i)
                    i_min = x_min * (1.0 - 0.01 * i)

                    x_quant, i_scale, i_zero = self.quantize(x, i_min, i_max)
                    x_dequant = self.dequantize(x_quant, i_scale, i_zero)
                    loss = self.lp_loss(x.unsqueeze(0), x_dequant.unsqueeze(0), p=2.4, reduction='none')

                    if loss < best_loss:
                        best_loss = loss
                        best_scale = i_scale
                        best_zero = i_zero
                
                scale = best_scale
                zero = best_zero

            if self.leaf_param:
                self.x_min = x_min
                self.x_max = x_max

            return torch.tensor(scale, dtype=x.dtype, device=x.device), torch.tensor(zero, dtype=x.dtype, device=x.device)

    def act_momentum_update(self, x: torch.Tensor, act_range_momentum: float = 0.95):
        assert(self.inited)
        assert(self.leaf_param)

        if self.channel_wise:
            flat = x.detach().permute(1, 0, 2, 3).contiguous().view(x.size(1), -1)
            x_min = flat.min(dim=1)[0]
            x_max = flat.max(dim=1)[0]
        else:
            x_min = x.detach().min()
            x_max = x.detach().max()

        self.x_min = self.x_min * act_range_momentum + x_min * (1.0 - act_range_momentum)
        self.x_max = self.x_max * act_range_momentum + x_max * (1.0 - act_range_momentum)

        span = torch.clamp(self.x_max - self.x_min, min=1e-8)
        new_scale = span / self.q_max
        new_zero = torch.round(-self.x_min / new_scale)

        if isinstance(self.scale, nn.Parameter):
            self.scale.data = new_scale
        else:
            self.scale = nn.Parameter(new_scale)
        self.zero = new_zero

    def bitwidth_refactor(self, refactored_bit: int):
        self.n_bits = refactored_bit
        self.q_max = 2 ** self.n_bits -1
        self.inited = False

    # StraightThrough Estimator
    @staticmethod
    def round_ste(x: torch.Tensor):
        return (x - x.detach()) + x.detach().round()

    # For MSE
    def lp_loss(self, pred: torch.Tensor, tgt: torch.Tensor, p=2.0, reduction='none'):
        diff = (pred - tgt).abs().pow(p)
        if reduction == 'none':
            b = diff.size(0)
            return diff.view(b, -1).sum(dim=1).mean()
        else:
            return diff.mean()
