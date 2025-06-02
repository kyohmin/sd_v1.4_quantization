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

from util import *

from ldm.models.diffusion.ddim import DDIMSampler
from quantization.quant_model import QuantModel

from quantization.quant_module import QuantModule
from quantization.quant_block import BaseQuantBlock
from quantization.quantizer import Quantizer


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

@torch.no_grad()
def generate_samples(model, opt):
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

                x = model.decode_first_stage(samples)
                x = torch.clamp((x + 1.0) / 2.0, 0.0, 1.0)

                x_np = (x * 255.0).byte().cpu().permute(0, 2, 3, 1).numpy()

                all_samples[ptr:ptr + batch_size] = x_np
                ptr += batch_size


    os.makedirs(out_dir, exist_ok=True)

    filename = f"full_{total_samples}_samples.npz"
    filepath = os.path.join(out_dir, filename)

    np.savez(filepath, samples=all_samples)
    print(f"Saved {ptr} samples to {filepath}")

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
    model.model.diffusion_model = quant_unet

    return model

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
    quantized_model = wrap_quant(model, opt, (wq_params, aq_params))

    # # 4. Generate Calibration Data
    # generate_cali_data()

    # # 5. Calibrate
    # calibrated_model = calibrate(quant_model, opt, quant_hyperparams)

    # # 6. Save Model and Config
    # save calibrated_model as pth
    # save calibrated_model.state_dict as pth

if __name__ == "__main__":
    main()
