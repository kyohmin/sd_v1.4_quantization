import torch
import torch.nn as nn

import os
import argparse
from pytorch_lightning import seed_everything

from PIL import Image
from omegaconf import OmegaConf

from util import load_model
from ldm.models.diffusion.ddim import DDIMSampler
from quantization.quant_model import QuantModel

def main():
    # 1. Load

    # 2. Param

    # 3. Generate Samples for different timesteps

    # 3. Run

    pass