import torch
import torch.nn as nn

class Quantizer(nn.Module):
    def __init__(self, n_bits: int = 8, channel_wise: bool = True, leaf_param: bool = False, scale_method: str = "min_max"):
        super().__init__()
        # Quantization Params
        self.n_bits = n_bits
        self.q_max = 2 ** self.n_bits - 1  # Asymmetric-only
        self.scale_method = scale_method
        self.inited = False
        self.channel_wise = channel_wise

        # Activation Params
        self.running_stat = False # For tracking min/max for act
        self.leaf_param = leaf_param # For training scale/zero for act
        if self.leaf_param:
            self.register_buffer("x_min", torch.tensor(0.0))
            self.register_buffer("x_max", torch.tensor(0.0))
            self.scale = nn.Parameter(torch.tensor(0.0))
            self.zero = nn.Parameter(torch.tensor(0.0))
        else: # Weight
            self.register_buffer("scale", torch.tensor(0.0))
            self.register_buffer("zero", torch.tensor(0.0))

    def forward(self, x: torch.Tensor):
        # Initialize scale and zero for first-run
        if not self.inited:

            init_scale, init_zero = self._get_scale_zero(x, self.channel_wise)
            with torch.no_grad():
                self.scale.data = init_scale
                self.zero.data = torch.tensor(init_zero, device = x.device)

            self.inited = True

        # For activation, update min, max, scale, and zero
        if self.running_stat: self._act_momentum_update(x)

        # Fake Quantization
        x_quant, _, _ = self._quantize(x) # Use saved scale & zero
        x_dequant = self._dequantize(x_quant) # Use saved scale & zero
        return x_dequant
    
    def bitwidth_refactor(self, refactored_bit: int):
        self.n_bits = refactored_bit
        self.q_max = 2 ** self.n_bits -1
        self.inited = False
    
    def _quantize(self, x: torch.Tensor, x_min=None, x_max=None):
        # pick stored params only if BOTH mins and maxes are missing
        if x_min is None and x_max is None:
            scale = self.scale
            zero  = self.zero
        else:
            scale = (x_max - x_min).clamp(min=1e-8) / self.q_max
            zero  = (-x_min / scale).round()

        # if per-channel vector, reshape to [C,1,1,…]
        if scale.dim() == 1:
            dims = [x.size(0)] + [1] * (x.dim() - 1)
            scale = scale.view(*dims)
            zero  = zero.view(*dims)

        x_int = self._round_ste(x / scale) + zero
        x_q   = x_int.clamp(0, self.q_max)
        return x_q, scale, zero

    def _dequantize(self, x_q: torch.Tensor, scale=None, zero=None) -> torch.Tensor:
        if scale is None and zero is None:
            scale = self.scale
            zero  = self.zero

        # same reshape logic if you ever pass a 1-D scale
        if scale.dim() == 1:
            dims = [x_q.size(0)] + [1] * (x_q.dim() - 1)
            scale = scale.view(*dims)
            zero  = zero.view(*dims)

        return (x_q - zero) * scale

    def _get_scale_zero(self, x: torch.Tensor, channel_wise: bool = True):
        scale, zero = None, None

        if channel_wise:
            x_clone = x.clone().detach()
            x_flat = x_clone.view(x_clone.size(0), -1)
            x_min = x_flat.min(dim=1)[0]
            x_max = x_flat.max(dim=1)[0]

            if "min_max" in self.scale_method:
                scale = (x_max - x_min).clamp(min=1e-8) / self.q_max
                zero = torch.round(-x_min / scale)

            elif "mse" in self.scale_method:
                best_score = torch.full_like(x_min, float('inf'))
                best_scale = torch.zeros_like(x_min)
                best_zero = torch.zeros_like(x_min)

                for i in range(80):
                    factor = 1.0 - (i * 0.01)
                    new_min = x_min * factor
                    new_max = x_max * factor

                    x_q, scale_i, zero_i = self._quantize(x_clone, new_min, new_max)
                    x_dq = self._dequantize(x_q, scale_i, zero_i)

                    score = LossFunction.lp_loss_channel(x_dq, x_clone, p=2.4)
                    improved = score < best_score
                    best_score[improved] = score[improved]
                    best_scale[improved] = scale_i.view(-1)[improved]
                    best_zero[improved] = zero_i.view(-1)[improved]

                scale, zero = best_scale, best_zero

            dims = [x.size(0)] + [1] * (x.dim() - 1)
            scale = scale.view(*dims)
            zero = zero.view(*dims)

        else:
            if self.leaf_param:
                self.x_min = x.data.min()
                self.x_max = x.data.max()

            if 'min_max' in self.scale_method:
                x_min = min(x.min().item(), 0)
                x_max = max(x.max().item(), 0)
                scale = float(x_max - x_min) / self.q_max
                zero = round(-x_min / scale)
                scale = torch.tensor(scale).type_as(x)
                zero = torch.tensor(zero).type_as(x)

            elif 'mse' in self.scale_method:
                x_min_t = torch.clamp(x.min(), max=0.0)
                x_max_t = torch.clamp(x.max(), min=0.0)

                best_score = float('inf')
                best_scale = None
                best_zero = None

                for i in range(80):
                    factor = 1.0 - (i * 0.01)
                    new_min = x_min_t * factor
                    new_max = x_max_t * factor

                    x_q, sc, zp = self._quantize(x, new_min, new_max)
                    x_dq = self._dequantize(x_q, sc, zp)

                    score = LossFunction.lp_loss(x, x_dq, p=2.4, reduction='all')
                    if score < best_score:
                        best_score = score
                        best_scale = sc
                        best_zero = zp

                if best_scale is None:
                    raise RuntimeError("MSE-based scale search failed")

                scale, zero = best_scale, best_zero

        return scale, zero


    def _act_momentum_update(self, x: torch.Tensor, act_range_momentum: float = 0.95):
        if self.channel_wise:
            flat = x.detach().permute(1, 0, 2, 3).contiguous().view(x.size(1), -1)
            x_min = flat.min(dim=1)[0]
            x_max = flat.max(dim=1)[0]
        else:
            x_min = x.detach().min()
            x_max = x.detach().max()

        with torch.no_grad():
            # Safely update buffer
            self.x_min.mul_(act_range_momentum).add_(x_min, alpha=(1.0 - act_range_momentum))
            self.x_max.mul_(act_range_momentum).add_(x_max, alpha=(1.0 - act_range_momentum))

            # Recalculate scale and zero and update
            new_scale = torch.clamp((self.x_max - self.x_min), min=1e-8) / self.q_max
            new_zero = (-self.x_min / new_scale).round()
            self.scale.copy_(new_scale)
            self.zero.copy_(new_zero)

    # StraightThrough Estimator
    @staticmethod
    def _round_ste(x: torch.Tensor):
        return (x - x.detach()) + x.detach().round()

class LossFunction:
    # MSE loss computation
    @staticmethod
    def lp_loss(self, prediction: torch.Tensor, target: torch.Tensor, p=2.0, reduction='none'):
        if reduction == 'none':
            return (target - prediction).abs().pow(p).sum(1).mean()
        else: # all elements
            return (target - prediction).abs().pow(p).mean()

    # MSE loss per channel
    @staticmethod
    def lp_loss_channel(self, prediction: torch.Tensor, target: torch.Tensor, p=2.0):
        loss = (target - prediction).abs().pow(p)

        if loss.dim() == 4:
            score = loss.view(loss.size(0), -1).mean(dim=1)
        elif loss.dim() == 3:
            score = loss.view(loss.size(0), -1).mean(dim=1)
        elif loss.dim() == 2:
            score = loss.mean(dim=1)
        else:
            score = loss

        return score

