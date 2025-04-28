# Stable Diffusion v1.4 Quantization Project

This repository provides a collection of quantization utilities for the Stable Diffusion v1.4 model.

## Overview
The project implements quantization methods using static utility methods that simplify the process of converting model weights and activations from floating point to lower-precision representations. The approach supports both asymmetric and symmetric quantization, along with various calibration methods (e.g., min-max).

## Key Features
**Weight Quantization:** Automatically quantize convolutional and linear layers using a custom quantization wrapper.

**Activation Quantization**: Run-time activation quantization.

**Stateless Utility Functions**: All functions are implemented as static methods, clarifying that they do not rely on class or instance state.

**Modular Design**: Easily integrate with existing PyTorch-based pipelines.

## Installation
Clone the Repository:

```bash
git clone https://github.com/your_username/sd_v1.4_quantization.git
```

## Usage
Import the quantization module and apply it to your Stable Diffusion inference file.

```python
from quantization.util import Util

# Define Quantization Parameters
quantizing_layers = (torch.nn.Conv2d, torch.nn.Linear)
uniform_type = "asymmetric"
calibration_type = "min_max"
bits = 8

# Wrap existing torch.nn.Conv2d and torch.nn.Linear with QuantWrapper
model = Util.rewrap(model, quantizing_layers)

# Quantize the weights of the selected layers accordingly
model = Util.quantize_model_weights(model, quantizing_layers, uniform_type, calibration_type, bits)
```

## References
**GitHub** (Stable Diffusion):
https://github.com/CompVis/stable-diffusion

**Huggingface** (ckpt file):
[https://huggingface.co/CompVis/stable-diffusion-v1-4](https://huggingface.co/CompVis/stable-diffusion-v-1-4-original)
