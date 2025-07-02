import torch
import torch.nn as nn

import os
import argparse
from pytorch_lightning import seed_everything

from PIL import Image
from omegaconf import OmegaConf

from util import instantiate_from_config
from ldm.models.diffusion.ddim import DDIMSampler

def get_parser():
    parser = argparse.ArgumentParser(
        description="Generate Quantized & Calibrated Diffusion Model"
    )
    # Paths
    parser.add_argument("--in_config", type=str,
                        default="configs/original_model.yaml",
                        help="path to model config YAML")
    parser.add_argument("--in_ckpt", type=str,
                        default="models/original_model.ckpt",
                        help="path to model checkpoint")
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

    # Calibration

    # 


def generate_samples():
    pass

def calibrate():
    pass

def main():
    # 1. Load

    # 2. Set Params

    # 3. Sample
    pass

if __name__ == '__main__':
    main()