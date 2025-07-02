import torch

from quantization.quant_module import QuantModule, StraightThrough
from quantization.quant_block import BaseQuantBlock
from quantization.quant_model import QuantModel
from quantization.quantizer import LossFunction
from quantization.util import Util

def layer_reconstruction(model: QuantModel, layer: QuantModule, cali_data: torch.Tensor,
                         batch_size: int = 32, iters: int = 20000, weight: float = 0.001, opt_mode: str = 'mse',
                         asym: bool = False, include_act_func: bool = True, b_range: tuple = (20, 2),
                         warmup: float = 0.0, act_quant: bool = False, lr: float = 4e-5, p: float = 2.0,
                         multi_gpu: bool = False, cond: bool = False, is_sm: bool = False):
    
    model.set_quant_state(False, False)
    layer.set_quant_state(True, act_quant)
    round_mode = 'learned_hard_sigmoid'

    if not include_act_func:
        org_act_func = layer.activation_function
        layer.activation_function = StraightThrough()

    if not act_quant:
        # AdaRound not yet supported
        if layer.split != 0:
            pass
        else:
            pass
    else:
        opt_params = [layer.act_quantizer.scale]
        if layer.split != 0 and layer.act_quantizer_0.scale is not None: opt_params += [layer.act_quantizer_0.scale]
        
        optimizer = torch.optim.Adam(opt_params, lr=lr)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max= iters, eta_min=0.)

    loss_mode = 'none' if act_quant else 'relaxation'
    rec_loss = opt_mode

    loss_func = QuantLossFunction(layer=layer, round_loss=loss_mode, weight=weight,
                             max_count=iters, rec_loss=rec_loss, b_range=b_range,
                             decay_start=0, warmup=warmup, p=p)

    cached_inps, cached_outs = Util.save_inp_oup_data(model, layer, cali_data, asym, act_quant, 8, keep_gpu=False, cond=cond, is_sm=is_sm)

    if opt_mode != 'mse': cached_grads = Util.save_grad_data(model, layer, cali_data, act_quant, batch_size=batch_size)
    else: cached_grads = None

    device = 'cuda'
    for i in range(iters):
        idx = torch.randperm(cached_inps.size(0))[:batch_size]
        cur_inp = cached_inps[idx].to(device)
        cur_out = cached_outs[idx].to(device)
        cur_grad = cached_grads[idx] if opt_mode != 'mse' else None

        optimizer.zero_grad()
        out_quant = layer(cur_inp)

        err = loss_func(out_quant, cur_out, cur_grad)
        err.backward(retain_graph=True)
        if multi_gpu:
            raise NotImplementedError
            # for p in opt_params:
            #     link.allreduce(p.grad)
        optimizer.step()
        if scheduler:
            scheduler.step()

    torch.cuda.empty_cache()

    # Finish optimization, use hard rounding.
    layer.weight_quantizer.soft_targets = False
    if layer.split != 0:
        layer.weight_quantizer_0.soft_targets = False

    # Reset original activation function
    if not include_act_func:
        layer.activation_function = org_act_func

def block_reconstruction():
    pass

def recon_model(model, params):
    for name, module in model.named_children():
        if isinstance(module, QuantModule):
            if module.ignore_reconstruction is True: continue
            else:
                layer_reconstruction(model, module, **params)
        elif isinstance(module, BaseQuantBlock):
            if module.ignore_reconstruction is True: continue
            else:
                block_reconstruction(model, module, **params)

class QuantLossFunction:
    def __init__(self,
                 block: BaseQuantBlock = None,
                 layer: QuantModule = None,
                 round_loss: str = 'relaxation',
                 weight: float = 1.,
                 rec_loss: str = 'mse',
                 max_count: int = 2000,
                 b_range: tuple = (10, 2),
                 decay_start: float = 0.0,
                 warmup: float = 0.0,
                 p: float = 2.):
        self.layer = layer
        self.block = block
        self.round_loss = round_loss
        self.weight = weight
        self.rec_loss = rec_loss
        self.loss_start = max_count * warmup
        self.p = p

        self.temp_decay = LinearTempDecay(max_count, rel_start_decay=warmup + (1 - warmup) * decay_start,
                                          start_b=b_range[0], end_b=b_range[1])
        self.count = 0

    def __call__(self, pred, tgt, grad=None):
        self.count += 1
        if self.rec_loss == 'mse':
            rec_loss = LossFunction.lp_loss(pred, tgt, p=self.p)
        
        b = self.temp_decay(self.count)
        if self.count < self.loss_start or self.round_loss == 'none':
            b = round_loss = 0
        elif self.round_loss == 'relaxation':
            round_loss = 0
            if self.block is not None:
                # block
                round_loss = 0
                for name, module in self.block.named_modules():
                    if isinstance(module, QuantModule):
                        round_vals = module.weight_quantizer.get_soft_targets()
                        round_loss += self.weight * (1 - ((round_vals - .5).abs() * 2).pow(b)).sum()

            elif self.layer is not None:
                # layer
                round_vals = self.layer.weight_quantizer.get_soft_targets()
                round_loss += self.weight * (1 - ((round_vals - .5).abs() * 2).pow(b).sum())

        total_loss = rec_loss + round_loss
        
        return total_loss
    
class LinearTempDecay:
    def __init__(self, t_max: int, rel_start_decay: float = 0.2, start_b: int = 10, end_b: int = 2):
        self.t_max = t_max
        self.start_decay = rel_start_decay * t_max
        self.start_b = start_b
        self.end_b = end_b

    def __call__(self, t):
        """
        Cosine annealing scheduler for temperature b.
        :param t: the current time step
        :return: scheduled temperature
        """
        if t < self.start_decay:
            return self.start_b
        else:
            rel_t = (t - self.start_decay) / (self.t_max - self.start_decay)
            return self.end_b + (self.start_b - self.end_b) * max(0.0, (1 - rel_t))
