import os
import argparse

import torch
import torch.nn as nn

from tqdm import tqdm, trange
import glob
from omegaconf import OmegaConf
from pytorch_lightning import seed_everything

from ldm.models.diffusion.ddim import DDIMSampler
from quantization.quant_model import QuantModel
from main.util import *


def get_parser():
    parser = argparse.ArgumentParser()

    # Paths
    parser.add_argument("--in_config", type=str,
                        default="configs/original_model.yaml")
    parser.add_argument("--in_ckpt", type=str,
                        default="models/original_model.ckpt")
    parser.add_argument("--in_txt", type=str,
                        default="samples/cali_data/prompts.txt")
    parser.add_argument("--out_indiv_samples", type=str,
                        default="samples/cali_data/individual_samples")
    parser.add_argument("--out_grid_samples", type=str,
                        default="samples/cali_data/grid_samples")
    parser.add_argument("--out_npz", type=str,
                        default="samples/cali_data")
    parser.add_argument("--out_pt", type=str,
                        default="samples/cali_data")
    

    # Generation
    parser.add_argument("--ddim_steps", type=int, default=50)
    parser.add_argument("--n_samples", type=int, default=1) # image per prompt
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--scale", type=float, default=7.5)
    parser.add_argument("--ddim_eta", type=float, default=1.0)
    parser.add_argument("--n_prompts", type=int, default=1024)


    return parser

@torch.no_grad()
def generate_calibration_data(model, opt):
    # Open file and load prompts
    with open(opt.in_txt, 'r', encoding='utf-8') as f:
        prompts = [line.strip() for line in f if line.strip()]

    # hyper parameters
    images_per_prompt = opt.n_samples # 1
    total_prompts = opt.n_prompts
    total_samples = total_prompts * images_per_prompt # 1024
    batch_size = opt.batch_size # 16

    # Latent-space shape (C, H, W)
    input_shape = [4, 64, 64]
    all_samples = None
    ptr = 0
    sampler = DDIMSampler(model)

    num_batches = (total_samples) // batch_size
    
    # Prepare prompts for 
    prompts_list = []
    for prompt in prompts: prompts_list += [prompt] * images_per_prompt

    cali_dict = {
        "xs": [], "ts": [], "cs": [], "ucs": [], "prompts": []
    }

    with torch.autocast("cuda"):
        with model.ema_scope():
            for batch in trange(num_batches, desc="Batches"):
                start = batch * batch_size
                end = start + batch_size

                curr_prompts = prompts_list[start:end]

                cond = model.get_learned_conditioning(curr_prompts)
                uncond = None

                if opt.scale != 1.0:
                    uncond = model.get_learned_conditioning([""]*batch_size)

                samples, intermediates = sampler.sample(
                    S=opt.ddim_steps,
                    conditioning=cond,
                    batch_size=batch_size,
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

                if all_samples is None:
                    _, H, W, C = x_np.shape
                    all_samples = np.empty((total_samples, H, W, C), dtype=np.uint8)

                all_samples[ptr:ptr+batch_size] = x_np
                
                # Save intermediates
                x_inter = torch.stack(intermediates["x_inter"][1:],  dim=0)
                ts = torch.stack(intermediates["ts"], dim=0)
                cs = cond.unsqueeze(0).expand(opt.ddim_steps, -1, -1, -1)
                ucs = uncond.unsqueeze(0).expand(opt.ddim_steps, -1, -1, -1) if uncond is not None else torch.zeros_like(cs)

                x_cpu   = x_inter.detach().cpu()
                t_cpu   = ts.detach().cpu()
                cs_cpu  = cs.detach().cpu()
                ucs_cpu = ucs.detach().cpu()
                
                if len(cali_dict['xs']) == 0:
                    cali_dict['xs']  = x_cpu
                    cali_dict['ts']  = t_cpu
                    cali_dict['cs']  = cs_cpu
                    cali_dict['ucs'] = ucs_cpu
                    cali_dict['prompts'] = list(curr_prompts)
                else:
                    cali_dict['xs']  = torch.cat([cali_dict['xs'],  x_cpu],  dim=1)
                    cali_dict['ts']  = torch.cat([cali_dict['ts'],  t_cpu],  dim=1)
                    cali_dict['cs']  = torch.cat([cali_dict['cs'],  cs_cpu], dim=1)
                    cali_dict['ucs'] = torch.cat([cali_dict['ucs'], ucs_cpu], dim=1)
                    cali_dict['prompts'].extend(curr_prompts)

                # Save individual image
                for i in range(batch_size):
                    img = Image.fromarray(all_samples[ptr+i])
                    img.save(os.path.join(opt.out_indiv_samples, f"sample_{(ptr+i):04}.png"))

                # Save grid image
                imgs = all_samples[ptr:ptr+batch_size]
                pil_list = [Image.fromarray(arr) for arr in imgs]
                grid = make_image_grid(pil_list, rows=4, cols=4)
                grid.save(os.path.join(opt.out_grid_samples, f"grid_{batch:02}.png"))

                ptr += batch_size
                
    total_imgs = all_samples
    pil_list = [Image.fromarray(arr) for arr in total_imgs]
    total_grid = make_image_grid(pil_list, rows=16, cols=16)
    total_grid.save(os.path.join(opt.out_grid_samples, f"total_grid.png"))

    np.savez(os.path.join(opt.out_npz, f"full_{total_samples}_samples.npz"), samples=all_samples)
    print(f"Saved {ptr} samples to {opt.out_npz}/full_{total_samples}_samples.npz")

    # Print Final Calibration Dict's Shapes
    print("cali_dict info")
    print("xs: ", cali_dict['xs'].shape)
    print("ts: ", cali_dict['ts'].shape)
    print("cs: ", cali_dict['cs'].shape)
    print("ucs: ", cali_dict['ucs'].shape)
    # Generate pt files
    torch.save(cali_dict, os.path.join(opt.out_pt, "full_dict.pt"))

def main():
    seed_everything(42)
    parser = get_parser()
    opt = parser.parse_args()

    # 1. Load Model
    model = load_model(config_path=opt.in_config, ckpt_path=opt.in_ckpt)
    model = model.cuda()
    model.eval()

    # 2. Generate Calibration Data
    generate_calibration_data(model, opt)

if __name__ == "__main__":
    main()