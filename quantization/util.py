import torch

from quantization.quantwrapper import Quantization
from quantization.quantwrapper import QuantWrapper

from quantization.quant_model import QuantModel
from quantization.quant_block import BaseQuantBlock

from tqdm import trange
import numpy as np

class Util:
    @staticmethod
    def round_ste(x: torch.Tensor):
        return (x.round() - x).detach() + x

    # MODEL WRAPPING
    @staticmethod
    def rewrap(model, layer_types=(torch.nn.Conv2d, torch.nn.Linear)):
        Util.add_wrapper(model, layer_types)
        return model

    @staticmethod
    def add_wrapper(module, layer_types=(torch.nn.Conv2d, torch.nn.Linear), skip_keys=()):
        for name, child in module.named_children():
            if any(skip_key in name for skip_key in skip_keys): # Layers to skip wrapping
                continue
            if isinstance(child, layer_types): # Wrap the selected layer
                setattr(module, name, QuantWrapper(child))
            else:
                Util.add_wrapper(child, layer_types) # Recursively go down
        return module
    
    # QUANTIZE (WEIGHT) WRAPPED MODEL
    @staticmethod
    def quantize_model_weights(
        model,
        layer_types=(torch.nn.Linear, torch.nn.Conv2d),
        uniform_type="asymmetric",
        calibration_type = "min_max",
        bits = 8
        ):
        # If quantwrapper, quantize and update the weight
        for _, module in model.model.diffusion_model.named_modules():
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

    # Activation Calibration
    @staticmethod
    def calibrate_activation_parameters(model, ):
        pass

    @staticmethod
    def save_inp_oup_data(model: QuantModel, layer, cali_data: torch.Tensor,
                        asym: bool = False, act_quant: bool = False, batch_size: int = 32, keep_gpu: bool = True,
                        cond: bool = False, is_sm: bool = False):
        
        device = next(model.parameters()).device
        get_inp_out = GetLayerInpOut(model, layer, device=device, asym=asym, act_quant=act_quant)
        cached_batches = []
        cached_inps, cached_outs = None, None
        torch.cuda.empty_cache()

        if not cond:
            cali_xs, cali_ts = cali_data
        else:
            cali_xs, cali_ts, cali_conds = cali_data

        if is_sm:
            if not cond:
                test_inp, test_out = get_inp_out(
                    cali_xs[:1].to(device), 
                    cali_ts[:1].to(device)
                )
            else:
                test_inp, test_out = get_inp_out(
                    cali_xs[:1].to(device), 
                    cali_ts[:1].to(device),
                    cali_conds[:1].to(device)
                )
                
            is_sm = False
            if (isinstance(test_inp, tuple) and test_inp[0].shape[1] == test_inp[0].shape[2]):
                if test_inp[0].shape[1] == 4096:
                    is_sm = True
            if test_out.shape[1] == test_out.shape[2]:
                if test_out.shape[1] == 4096:
                    is_sm = True
                
            if is_sm:
                inds = np.random.choice(cali_xs.size(0), cali_xs.size(0) // 2, replace=False)
        
        
        num = int(cali_xs.size(0) / batch_size)
        if is_sm:
            num //= 2
        l_in_0, l_in_1, l_in, l_out = 0, 0, 0, 0
        for i in trange(num):
            if not cond:
                cur_inp, cur_out = get_inp_out(
                    cali_xs[i * batch_size:(i + 1) * batch_size].to(device), 
                    cali_ts[i * batch_size:(i + 1) * batch_size].to(device)
                ) if not is_sm else get_inp_out(
                    cali_xs[inds[i * batch_size:(i + 1) * batch_size]].to(device), 
                    cali_ts[inds[i * batch_size:(i + 1) * batch_size]].to(device)
                )
            else:
                cur_inp, cur_out = get_inp_out(
                    cali_xs[i * batch_size:(i + 1) * batch_size].to(device), 
                    cali_ts[i * batch_size:(i + 1) * batch_size].to(device),
                    cali_conds[i * batch_size:(i + 1) * batch_size].to(device)
                ) if not is_sm else get_inp_out(
                    cali_xs[inds[i * batch_size:(i + 1) * batch_size]].to(device), 
                    cali_ts[inds[i * batch_size:(i + 1) * batch_size]].to(device),
                    cali_conds[inds[i * batch_size:(i + 1) * batch_size]].to(device)
                )
            if isinstance(cur_inp, tuple):
                cur_x, cur_t = cur_inp
                if not is_sm:
                    cached_batches.append(((cur_x.cpu(), cur_t.cpu()), cur_out.cpu()))
                else:
                    if cached_inps is None:
                        l_in_0 = cur_x.shape[0] * num
                        l_in_1 = cur_t.shape[0] * num
                        cached_inps = [torch.zeros(l_in_0, *cur_x.shape[1:]), torch.zeros(l_in_1, *cur_t.shape[1:])]
                    cached_inps[0].index_copy_(0, torch.arange(i * cur_x.shape[0], (i + 1) * cur_x.shape[0]), cur_x.cpu())
                    cached_inps[1].index_copy_(0, torch.arange(i * cur_t.shape[0], (i + 1) * cur_t.shape[0]), cur_t.cpu())
            else:
                if not is_sm:
                    cached_batches.append((cur_inp.cpu(), cur_out.cpu()))
                else:
                    if cached_inps is None:
                        l_in = cur_inp.shape[0] * num
                        cached_inps = torch.zeros(l_in, *cur_inp.shape[1:])
                    cached_inps.index_copy_(0, torch.arange(i * cur_inp.shape[0], (i + 1) * cur_inp.shape[0]), cur_inp.cpu())
            
            if is_sm:
                if cached_outs is None:
                    l_out = cur_out.shape[0] * num
                    cached_outs = torch.zeros(l_out, *cur_out.shape[1:])
                cached_outs.index_copy_(0, torch.arange(i * cur_out.shape[0], (i + 1) * cur_out.shape[0]), cur_out.cpu())

        if not is_sm:
            if isinstance(cached_batches[0][0], tuple):
                cached_inps = [
                    torch.cat([x[0][0] for x in cached_batches]), 
                    torch.cat([x[0][1] for x in cached_batches])
                ]
            else:
                cached_inps = torch.cat([x[0] for x in cached_batches])
            cached_outs = torch.cat([x[1] for x in cached_batches])
        

        torch.cuda.empty_cache()
        if keep_gpu:
            if isinstance(cached_inps, list):
                cached_inps[0] = cached_inps[0].to(device)
                cached_inps[1] = cached_inps[1].to(device)
            else:
                cached_inps = cached_inps.to(device)
            cached_outs = cached_outs.to(device)
        return cached_inps, cached_outs
    

class GetLayerInpOut:
    def __init__(self, model: QuantModel, layer,
                 device: torch.device, asym: bool = False, act_quant: bool = False):
        self.model = model
        self.layer = layer
        self.asym = asym
        self.device = device
        self.act_quant = act_quant
        self.data_saver = DataSaverHook(store_input=True, store_output=True, stop_forward=True)

    def __call__(self, x, timesteps, context=None):
        self.model.eval()
        self.model.set_quant_state(False, False)

        handle = self.layer.register_forward_hook(self.data_saver)
        with torch.no_grad():
            _ = self.model(x, timesteps, context)

            if self.asym:
                # Recalculate input with network quantized
                self.data_saver.store_output = False
                self.model.set_quant_state(weight_quant=True, act_quant=self.act_quant)
                _ = self.model(x, timesteps, context)
                self.data_saver.store_output = True

        handle.remove()

        self.model.set_quant_state(False, False)
        self.layer.set_quant_state(True, self.act_quant)
        self.model.train()

        if len(self.data_saver.input_store) > 1 and torch.is_tensor(self.data_saver.input_store[1]):
            return (self.data_saver.input_store[0].detach(),  
                self.data_saver.input_store[1].detach()), self.data_saver.output_store.detach()
        else:
            return self.data_saver.input_store[0].detach(), self.data_saver.output_store.detach()

class DataSaverHook:
    """
    Forward hook that stores the input and output of a block
    """
    def __init__(self, store_input=False, store_output=False, stop_forward=False):
        self.store_input = store_input
        self.store_output = store_output
        self.stop_forward = stop_forward

        self.input_store = None
        self.output_store = None

    def __call__(self, module, input_batch, output_batch):
        if self.store_input:
            self.input_store = input_batch
        if self.store_output:
            self.output_store = output_batch