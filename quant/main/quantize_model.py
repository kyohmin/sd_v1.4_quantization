import os
import argparse

import torch
import torch.nn as nn
import numpy as np
from tqdm import trange
from omegaconf import OmegaConf
from pytorch_lightning import seed_everything

from util import load_model
from ldm.models.diffusion.ddim import DDIMSampler
from quantization.quant_model import QuantModel
from quantization.quant_module import QuantModule
from quantization.quant_block import BaseQuantBlock
from quantization.quantizer import Quantizer
from quantization.adaptive_rounding import AdaRoundQuantizer
from quantization.layer_recon import layer_reconstruction
from quantization.block_recon import block_reconstruction


def get_train_samples(opt, sample_data, sampling_step=None):
    """
    Prepare calibration samples. Always returns (xs, ts, conds),
    where `conds` is None if --cond is not set.
    """
    num_samples = 1024
    num_st = opt.cali_divide
    sampling_step = opt.ddim_steps if sampling_step is None else sampling_step

    if num_st == 1:
        xs = sample_data["xs"][0][:num_samples] if isinstance(sample_data, dict) else sample_data[:num_samples]
        ts = torch.ones(num_samples) * 800
    else:
        all_xs = sample_data["xs"]
        all_ts = sample_data["ts"]
        nsteps = len(all_ts)
        assert nsteps >= sampling_step, "Not enough timesteps for calibration"
        timesteps = list(range(0, nsteps, nsteps // num_st))
        xs = torch.cat([all_xs[i][:num_samples] for i in timesteps], dim=0)
        ts = torch.cat([all_ts[i][:num_samples] for i in timesteps], dim=0)

    conds = None
    if opt.cond:
        cs = sample_data["cs"]
        ucs = sample_data["ucs"]
        if num_st == 1:
            conds = cs[:num_samples]
        else:
            conds = torch.cat(
                [cs[i][:num_samples] for i in timesteps] + [ucs[i][:num_samples] for i in timesteps],
                dim=0
            )
    return xs, ts, conds


def get_parser():
    parser = argparse.ArgumentParser(
        description="Quantize and calibrate a Stable Diffusion UNet model"
    )

    # Quantization toggles (enabled by default)
    parser.add_argument(
        "--no-weight-quant",
        dest="weight_quant",
        action="store_false",
        help="disable weight quantization (enabled by default)"
    )
    parser.add_argument(
        "--no-act-quant",
        dest="act_quant",
        action="store_false",
        help="disable activation quantization (enabled by default)"
    )
    parser.set_defaults(weight_quant=True, act_quant=True)

    parser.add_argument("--weight_bit", type=int, default=8,
                        help="bit-width for weight quantization")
    parser.add_argument("--act_bit", type=int, default=8,
                        help="bit-width for activation quantization")
    parser.add_argument("--cond", action="store_true",
                        help="enable conditional (text) guidance during calibration")

    # Paths
    parser.add_argument("--in_config", type=str,
                        default="configs/v1-inference.yaml",
                        help="path to model config YAML")
    parser.add_argument("--in_ckpt", type=str,
                        default="models/model.ckpt",
                        help="path to model checkpoint")
    parser.add_argument("--out_config", type=str,
                        default="configs/calibrated_model.yaml",
                        help="where to save updated config YAML")
    parser.add_argument("--out_ckpt", type=str,
                        default="models/quantized_model",
                        help="directory for saving quantized model")
    parser.add_argument("--in_cali_data", type=str, required=True,
                        help=".pt file containing calibration data dict")

    # Calibration settings
    parser.add_argument("--ddim_steps", type=int, default=50,
                        help="number of DDIM steps in calibration samples")
    parser.add_argument("--cali_divide", type=int, default=1,
                        help="number of evenly spaced time segments for calibration")
    parser.add_argument("--running_stat", action="store_true",
                        help="collect running stats for batchnorm during activation calibration")

    return parser


def recon_model(model, outpath, kwargs):
    """
    Walk the model tree and apply per-layer or per-block reconstruction.
    Save an interim checkpoint if hitting output_blocks or large block index.
    """
    for name, module in model.named_children():
        # optional interim save
        if name == 'output_blocks' or (name.isdigit() and int(name) >= 9):
            torch.save(model.state_dict(), os.path.join(outpath, "ckpt.pth"))

        # quantization reconstruction
        if isinstance(module, QuantModule):
            if not module.ignore_reconstruction:
                layer_reconstruction(model, module, **kwargs)
        elif isinstance(module, BaseQuantBlock):
            if not module.ignore_reconstruction:
                block_reconstruction(model, module, **kwargs)
        else:
            recon_model(module, outpath, kwargs)


def calibrate_weight(model, opt, calibration_data):
    xs, ts, cs = calibration_data
    device = next(model.parameters()).device

    # warm up (run forward once)
    model.set_quant_state(True, False)
    xi = xs[:10].to(device)
    ti = ts[:10].to(device)
    if cs is not None:
        ci = cs[:10].to(device)
        _ = model(xi, ti, ci)
    else:
        _ = model(xi, ti)

    return dict(
        cali_data=calibration_data,
        batch_size=32,
        iters=20000,
        weight=0.01,
        asym=True,
        b_range=(20, 2),
        warmup=0.2,
        act_quant=False,
        opt_mode='mse',
        cond=opt.cond
    )


def calibrate_activation(model, opt, calibration_data):
    # weights must be reconstructed first, so call calibrate_weight externally or ensure order
    xs, ts, cs = calibration_data
    device = next(model.parameters()).device

    model.set_quant_state(True, True)
    model.set_running_stat(False, False)

    # collect statistics
    with torch.no_grad():
        idx = np.random.choice(xs.shape[0], 16, replace=False)
        xi = xs[idx].to(device)
        ti = ts[idx].to(device)
        if cs is not None:
            ci = cs[idx].to(device)
            _ = model(xi, ti, ci)
        else:
            _ = model(xi, ti)

        if opt.running_stat:
            perm = np.random.permutation(xs.shape[0])
            model.set_running_stat(True, False)
            for i in trange(xs.shape[0] // 16):
                batch_idx = perm[i * 16:(i + 1) * 16]
                xi = xs[batch_idx].to(device)
                ti = ts[batch_idx].to(device)
                if cs is not None:
                    ci = cs[batch_idx].to(device)
                    _ = model(xi, ti, ci)
                else:
                    _ = model(xi, ti)

    return dict(
        cali_data=calibration_data,
        batch_size=32,
        iters=5000,
        act_quant=True,
        opt_mode='mse',
        lr=4e-4,
        p=2.4,
        cond=opt.cond
    )


def set_calibrated_params(model, opt):
    for module in model.model.modules():
        if isinstance(module, AdaRoundQuantizer):
            module.zero_point = nn.Parameter(module.zero_point)
            module.scale = nn.Parameter(module.scale)
        elif isinstance(module, Quantizer) and opt.act_quant:
            if module.zero_point is not None:
                zp = module.zero_point
                module.zero_point = nn.Parameter(torch.as_tensor(zp, dtype=torch.float32))
                module.scale = nn.Parameter(torch.as_tensor(module.scale, dtype=torch.float32))


def main():
    seed_everything(42)
    parser = get_parser()
    opt = parser.parse_args()

    os.makedirs(opt.out_ckpt, exist_ok=True)

    # 1. Load model and move to device
    config = OmegaConf.load(opt.in_config)
    model = load_model(config, opt.in_ckpt)
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    model = model.to(device)

    # 2. Prepare UNet quantizer
    weight_q = {'n_bits': opt.weight_bit, 'channel_wise': True, 'scale_method': 'mse'}
    act_q = {'n_bits': opt.act_bit, 'channel_wise': False, 'scale_method': 'mse', 'leaf_param': True}
    quant_unet = QuantModel(
        model=model.model.diffusion_model,
        weight_quant_params=weight_q,
        act_quant_params=act_q
    )

    # 3. Load calibration data and samples
    sample_data = torch.load(opt.in_cali_data)
    calibration_data = get_train_samples(opt, sample_data, opt.ddim_steps)

    # 4. Calibrate weights
    weight_kwargs = calibrate_weight(quant_unet, opt, calibration_data)
    recon_model(quant_unet, opt.out_ckpt, weight_kwargs)

    # 5. Calibrate activations (if requested)
    if opt.act_quant:
        activation_kwargs = calibrate_activation(quant_unet, opt, calibration_data)
        recon_model(quant_unet, opt.out_ckpt, activation_kwargs)

    # 6. Finalize params and save
    set_calibrated_params(quant_unet, opt)
    model.model.diffusion_model = quant_unet
    torch.save(model.state_dict(), os.path.join(opt.out_ckpt, "quantized_model.pth"))
    OmegaConf.save(config, opt.out_config)


if __name__ == "__main__":
    main()
