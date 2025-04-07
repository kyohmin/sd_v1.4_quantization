# Stable Diffusion v1.4 Quantization Project

This repository provides a collection of quantization utilities for the Stable Diffusion v1.4 model.

## Overview
The project implements quantization methods using static utility methods that simplify the process of converting model weights and activations from floating point to lower-precision representations. The approach supports both asymmetric and symmetric quantization, along with various calibration methods (e.g., min-max).

## Key Features
**Weight Quantization:** Automatically quantize convolutional and linear layers using a custom quantization wrapper.

**Activation Quantization**: (Planned) Future support for activation quantization.

**Stateless Utility Functions**: All functions are implemented as static methods, clarifying that they do not rely on class or instance state.

Modular Design: Easily integrate with existing PyTorch-based pipelines.

## Installation
Clone the Repository:

```bash
git clone https://github.com/your_username/sd_v1.4_quantization.git
```

## Usage
Import the quantization module and apply it to your Stable Diffusion inference file.

```python
from quantization.util import Util
```

## References
**GitHub** (Stable Diffusion):
https://github.com/CompVis/stable-diffusion

**Huggingface** (ckpt file):
[https://huggingface.co/CompVis/stable-diffusion-v1-4](https://huggingface.co/CompVis/stable-diffusion-v-1-4-original)
