from setuptools import setup, find_packages

setup(
    name='quant',
    version='0.1',
    description='Quantized Stable Diffusion project with VAE and CLIP support',
    author='Your Name',
    packages=find_packages(include=[
        'ldm', 'ldm.*',
        'quantization', 'quantization.*',
        'main', 'main.*'
    ]),
    install_requires=[
        'torch>=1.11.0',
        'torchvision>=0.12.0',
        'tqdm==4.64.0',
        'numpy',
        'einops==0.3.0',
        'pytorch-lightning==1.4.2',
        'lightning-utilities==0.8.0',
        'omegaconf==2.1.1',
        'transformers==4.22.2',
        'torchmetrics==0.6.0',
        'kornia==0.6.9',
        'imageio==2.9.0',
        'imageio-ffmpeg==0.4.2',
        'opencv-python==4.1.2.30',
        'albumentations==0.4.3',
        'invisible-watermark',
        'test-tube>=0.7.5',
        'streamlit>=0.73.1',
        'pandas==1.4.2',
        'PyYAML==6.0',
        'six==1.16.0',
    ],
    python_requires='>=3.8',
    include_package_data=True,
    zip_safe=False,
)
