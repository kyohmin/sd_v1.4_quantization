import os, gc
import argparse

import torch
import math
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
from quantization.reconstruction import recon_model


def get_parser():
    parser = argparse.ArgumentParser()

    # Paths
    parser.add_argument("--in_config", type=str,
                        default="configs/original_model.yaml")
    parser.add_argument("--in_original_ckpt", type=str,
                        default="models/original_model.ckpt")
    parser.add_argument("--in_quantized_ckpt", type=str,
                        default="models/quantized_model.ckpt")
    parser.add_argument("--in_txt", type=str,
                        default="samples/quant_data/prompts.txt")
    parser.add_argument("--in_cali_data", type=str,
                        default="samples/cali_data/full_dict.pt")
    parser.add_argument("--out_indiv_samples", type=str,
                        default="samples/quant_data/individual_samples")
    parser.add_argument("--out_grid_samples", type=str,
                        default="samples/quant_data/grid_samples")
    parser.add_argument("--out_npz", type=str,
                        default="samples/quant_data")
    parser.add_argument("--out_ckpt", type=str,
                        default="samples/quant_data")
    
    # Generation
    parser.add_argument("--ddim_steps", type=int, default=50)
    parser.add_argument("--ddim_eta", type=float, default=1.0)
    parser.add_argument("--n_samples", type=int, default=1) # image per prompt
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--scale", type=float, default=7.5)
    parser.add_argument("--n_prompts", type=int, default=1024) # number of prompts to use, only when prompts.txt exists
    parser.add_argument("--use_prompt_list", default=False, action="store_true") # If True, use prompt.txt
    parser.add_argument("--prompt", type=str, default="a photograph of an astronaut riding a horse")

    # Quantization
    parser.add_argument("--no-weight-quant", action="store_false", default=True)
    parser.add_argument("--no-act-quant", action="store_false", default=True)
    parser.add_argument("--weight_bit", type=int, default=8)
    parser.add_argument("--act_bit", type=int, default=8)

    # Calibration
    parser.add_argument("--running_stat", action="store_false", default=True)
    parser.add_argument("--cond", action="store_false", default=True)

    return parser

@torch.no_grad()
def generate_samples(model, opt):
    if opt.use_prompt_list:
        prompts = [opt.prompt]
        total_prompts = 1
    else:
        with open(opt.in_txt, 'r', encoding='utf-8') as f:
            prompts = [line.strip() for line in f if line.strip()]
            total_prompts = opt.n_prompts

    images_per_prompt = opt.n_samples
    total_samples = total_prompts * images_per_prompt
    batch_size = opt.batch_size

    input_shape = [4, 64, 64] # LDM (Stable Diffusion)
    all_samples = None
    ptr = 0
    sampler = DDIMSampler(model)

    num_batches = math.ceil(total_samples / batch_size)
    
    # Prepare prompts in list form
    prompts_list = []
    for prompt in prompts: prompts_list += [prompt] * images_per_prompt

    with torch.autocast("cuda"):
        with model.ema_scope():
            for batch in trange(num_batches, desc="Batches"):
                start = batch * batch_size
                end = min(start + batch_size, total_samples)

                curr_batch_size = end - start

                curr_prompts = prompts_list[start:end]

                cond = model.get_learned_conditioning(curr_prompts)
                uncond = None

                if opt.scale != 1.0:
                    uncond = model.get_learned_conditioning([""]*curr_batch_size)

                samples, _ = sampler.sample(
                    S=opt.ddim_steps,
                    conditioning=cond,
                    batch_size=curr_batch_size,
                    shape=input_shape,
                    verbose=False,
                    unconditional_guidance_scale=opt.scale,
                    unconditional_conditioning=uncond,
                    eta=opt.ddim_eta,
                    log_every_t=1,
                    x_T=None
                )

                x = model.decode_first_stage(samples)
                x = torch.clamp((x + 1.0) / 2.0, 0.0, 1.0)
                x_np = (x * 255).byte().cpu().permute(0, 2, 3, 1).numpy()

                if all_samples is None: # Initialize all_samples
                    _, H, W, C = x_np.shape
                    print("HWC:",H, "", W,"", C) # REMOVE
                    all_samples = np.empty((total_samples, H, W, C), dtype=np.uint8)

                all_samples[start:end] = x_np

                # Save Images
                images = x_np
                save_individual_images(images, opt.out_indiv_samples, start)
                make_image_grid(images, opt.out_grid_samples, batch, 4, 4)

    # Save 16x16 grid images
    make_image_grid(all_samples, opt.out_grid_samples, rows=16, cols=16)

def wrap_quant(model, quant_hyperparams):
    quant_unet = QuantModel(
        u_net=model.model.diffusion_model,
        quant_hyperparams=quant_hyperparams
    )
    model.model.diffusion_model = quant_unet

    return model

def x_to_np(x, )

def save_individual_images(images, path, numbering_index=0):
    for i in range(images.shape[0]):
        img = Image.fromarray(images[i])
        img.save(os.path.join(path, f"sample_{(numbering_index + i):04}.png"))

def make_image_grid(images, path, numbering_index=0, rows=1, cols=1):
    pil_list = [Image.fromarray(arr) for arr in images]
    pil_imgs = []
    for img in pil_list:
        if isinstance(img, np.ndarray):
            pil = Image.fromarray(img)
        else:
            pil = img
        pil_imgs.append(pil.convert("RGB"))

    W, H = pil_imgs[0].size

    grid = Image.new("RGB", (cols * W, rows * H), color = (0,0,0))

    for idx, img in enumerate(pil_imgs):
        if idx >= rows * cols:
            break
        # optionally resize
        tile = img.resize((W, H), Image.LANCZOS)
        r = idx // cols
        c = idx % cols
        grid.paste(tile, (c * W, r * H))

    grid.save(os.path.join(path, f"grid_{numbering_index:02}.png"))

def calibrate(model, opt):
    # 1. Load Calibration Data
    calibration_data = torch.load(opt.in_cali_data)
    samples = get_train_samples(opt, calibration_data)
    cali_xs, cali_ts, cali_context = calibration_data

    del(calibration_data)
    gc.collect()

    # 2. Initialize, Calibrate (Running_Stat), Reconstruct
    _ = model(cali_xs[:10].cuda(), cali_ts[:10].cuda(), cali_context[:10].cuda())
    params = dict(cali_data=samples, batch_size=32, iters=20000, weight=0.01, asym=True, b_range=(20, 2), warmup=0.2, act_quant=False, opt_mode='mse')
    recon_model(model, params)
    model.set_quant_state(weight_quant=True, act_quant=False)

    if opt.act_quant:
        model.set_quant_state(weight_quant=True, act_quant=True)
        with torch.no_grad():
            _ = model(cali_xs[:64].cuda((), cali_ts[:64].cuda(), cali_context[:64].cuda()))
            # Running Stat to collect data for activation initialization
            if opt.running_stat:
                model.set_running_stat(True)
                for i in trange(int(cali_xs.size(0) / 64)):
                    _ = model(cali_xs[i * 64: (i + 1) * 64].cuda(), cali_ts[i * 64: (i + 1) * 64].cuda())
                model.set_running_stat(False)

        params = dict(cali_data=samples, iters=5000, act_quant=True, opt_mode='mse', lr = 4e-4, p = 2.4)
        recon_model(model, params)
        model.set_quant_state(weight_quant=True, act_quant=True)

    # 3. Save Model
    for module in model.model.modules():
        if isinstance(module, Quantizer) and opt.quant_act:
            if module.zero_point is not None:
                if not torch.is_tensor(module.zero_point):
                    module.zero_point = nn.Parameter(torch.tensor(float(module.zero_point)))
                else:
                    module.zero_point = nn.Parameter(module.zero_point)
    torch.save(model.state_dict(), os.path.join(opt.out_, "quantized_model.pth"))

    return model

def save_checkpoint(model, ckpt_path):
    state_dict = model.module.state_dict() if hasattr(model, "module") else model.state_dict()

    os.makedirs(os.path.dirname(ckpt_path), exist_ok=True)
    torch.save(state_dict, ckpt_path)
    print(f"Saved quantized model weights to {ckpt_path}")

def main():
    seed_everything(42)
    parser = get_parser()
    opt = parser.parse_args()


    # 1. Load model - DONE
    model = load_model(config_path=opt.in_config, ckpt_path=opt.in_ckpt)
    model = model.cuda()
    model.eval()

    # 2. Define Quant Unet and Replace
    wq_params = {'n_bits': opt.weight_bit, 'channel_wise': True, 'leaf_param': False,'scale_method': 'mse'}
    aq_params = {'n_bits': opt.act_bit, 'channel_wise': False, 'leaf_param': True, 'scale_method': 'min_max'}
    params = (wq_params, aq_params)

    model = wrap_quant(model, params)
    model.eval()

    # 3. Running_stat & Calibrate
    model = calibrate(model, opt)

    # 5. Test Quantized Generation
    generate_samples(model, opt)


if __name__ == "__main__":
    main()
