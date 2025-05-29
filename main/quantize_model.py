import os
import argparse

import torch
import torch.nn as nn
import numpy as np
from tqdm import tqdm, trange
from PIL import Image
import glob
from omegaconf import OmegaConf
from pytorch_lightning import seed_everything

from util import instantiate_from_config

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
                        default="configs/original_model.yaml",
                        help="path to model config YAML")
    parser.add_argument("--in_ckpt", type=str,
                        default="models/original_model.ckpt",
                        help="path to model checkpoint")
    parser.add_argument("--out_config", type=str,
                        default="configs/quantized_model.yaml",
                        help="where to save updated config YAML")
    parser.add_argument("--out_ckpt", type=str,
                        default="models/quantized_model.pth",
                        help="directory for saving quantized model")
    parser.add_argument("--out_sample", type=str,
                        default="samples")
    # parser.add_argument("--in_cali_data", type=str, required=True,
    #                     help=".pt file containing calibration data dict")

    # Calibration settings
    parser.add_argument("--ddim_steps", type=int, default=50,
                        help="number of DDIM steps in calibration samples")
    parser.add_argument("--ddim_eta", type=float, default=1.0)
    parser.add_argument("--cali_divide", type=int, default=1,
                        help="number of evenly spaced time segments for calibration")
    parser.add_argument("--running_stat", action="store_true",
                        help="collect running stats for batchnorm during activation calibration")

    # Generation
    parser.add_argument("--n_samples", type=int, default=50)
    parser.add_argument("--batch_size", type=int, default=5)
    parser.add_argument("-p", "--prompt", type=str, default="a photograph of an astronaut riding a horse")
    parser.add_argument("--scale", type=float, default=7.5)


    return parser

# UTIL
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

# CUSTOM FUNCTIONS
def load_model(config_path: str, ckpt_path: str): # DONE
    state_dict = torch.load(ckpt_path, map_location="cpu")["state_dict"]
    config = OmegaConf.load(f"{config_path}")
    model = instantiate_from_config(config.model)
    model.load_state_dict(state_dict, strict=False)

    model.cuda()
    model.eval()
    return model

def convsample(model, opt):
    input_shape = [4, 64, 64]
    sampler = DDIMSampler(model)
    samples, intermediates = sampler.sample(S=opt.ddim_steps, batch_size=)
    
def custom_to_np(x):
    # saves the batch in adm style as in https://github.com/openai/guided-diffusion/blob/main/scripts/image_sample.py
    sample = x.detach().cpu()
    sample = ((sample + 1) * 127.5).clamp(0, 255).to(torch.uint8)
    sample = sample.permute(0, 2, 3, 1)
    sample = sample.contiguous()
    return sample


@torch.no_grad()
def generate_samples(model, opt):
    """
    Generate and save samples into a single .npz archive named full_{n_samples}_samples.npz.
    """
    # Latent-space shape (C, H, W)
    input_shape = [4, 64, 64]
    total_samples = opt.n_samples
    batch_size = opt.batch_size
    prompt = opt.prompt
    out_dir = opt.out_sample  # directory to save .npz

    # Prepare repeated prompts for each batch
    prompts_list = [prompt] * batch_size

    # Pre-allocate NumPy array for all images in HWC uint8 format
    C, H, W = input_shape
    all_samples = np.empty((total_samples, H, W, C), dtype=np.uint8)
    ptr = 0

    sampler = DDIMSampler(model)

    with torch.autocast("cuda"):
        with model.ema_scope():
            # Loop over full batches
            for _ in trange(total_samples // batch_size, desc="Sampling Batches"):
                # Compute conditioning once per batch
                cond = model.get_learned_conditioning(prompts_list)
                uncond = None
                if opt.scale != 1.0:
                    uncond = model.get_learned_conditioning([""] * batch_size)

                # Sample latents
                samples, _ = sampler.sample(
                    S=opt.ddim_steps,
                    conditioning=cond,
                    batch_size=batch_size,
                    shape=input_shape,
                    verbose=False,
                    unconditional_guidance_scale=opt.scale,
                    unconditional_conditioning=uncond,
                    eta=opt.ddim_eta,
                    x_T=None
                )

                # Decode and normalize to [0,1]
                x = model.decode_first_stage(samples)
                x = torch.clamp((x + 1.0) / 2.0, 0.0, 1.0)

                # Convert to uint8 HWC numpy
                x_np = (x * 255.0).byte().cpu().permute(0, 2, 3, 1).numpy()

                # Store into pre-allocated array
                all_samples[ptr:ptr + batch_size] = x_np
                ptr += batch_size

    # Ensure output directory exists
    os.makedirs(out_dir, exist_ok=True)
    # Construct filename
    filename = f"full_{total_samples}_samples.npz"
    filepath = os.path.join(out_dir, filename)
    # Save all samples
    np.savez(filepath, samples=all_samples)
    print(f"Saved {ptr} samples to {filepath}")

    # Additionally, save each sample as a PNG
    for idx in range(total_samples):
        img = Image.fromarray(all_samples[idx])
        img.save(os.path.join(out_dir, f"sample_{idx:06}.png"))
    print(f"Saved all {total_samples} samples as PNGs in {out_dir}")

def wrap_quant(model, quant_hyperparams):
    weight_param, act_param = quant_hyperparams
    quant_unet = QuantModel(
        model=model.model.diffusion_model,
        weight_quant_params=weight_param,
        act_quant_params=act_param
    )
    model.diffusion_model = quant_unet

    return model

def calibrate(model, opt, params):
    wq_params, aq_params = params

    if opt.resume:
        pass

def generate_cali_data(model, opt):
    # Read Samples
    # Split into Cali
    pass

def main():
    seed_everything(42)
    parser = get_parser()
    opt = parser.parse_args()

    # os.makedirs(opt.out_ckpt, exist_ok=True)

    # 1. Load model - DONE
    model = load_model(config_path=opt.in_config, ckpt_path=opt.in_ckpt)

    # 2. Generate Calibration Data (From Q-Diffusion) - DONE
    generate_samples(model, opt)

    # 3. Define Quant Unet and replace
    wq_params = {'n_bits': opt.weight_bit, 'channel_wise': True, 'scale_method': 'mse'}
    aq_params = {'n_bits': opt.act_bit, 'symmetric': opt.a_sym, 'channel_wise': False, 'scale_method': 'mse', 'leaf_param': opt.quant_act}
    quant_hyperparams = [wq_params, aq_params]
    quantized_model = wrap_quant(model, opt, quant_hyperparams)

    # # 4. Generate Calibration Data
    # generate_cali_data()

    # # 5. Calibrate
    # calibrated_model = calibrate(quant_model, opt, quant_hyperparams)

    # # 6. Save Model and Config
    # save calibrated_model as pth
    # save calibrated_model.state_dict as pth

if __name__ == "__main__":
    main()
