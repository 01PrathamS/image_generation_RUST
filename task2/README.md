# DDPM Face Generator (Unconditional)

This project implements a Denoising Diffusion Probabilistic Model (DDPM) using a U-Net backbone for unconditional image generation. Although the original project aimed to generate **128x128 face images from embeddings**, this current version demonstrates **unconditional sampling** of images from noise using a pre-trained model.

## Original Project Goal

> Train a generative model that takes embeddings as input and produces 128x128 human face images.  
> The model should generalize to unseen face embeddings (zero-shot) and be trained within 6 hours using limited compute.  
> Any generative method (e.g., GANs, Diffusion, Flows) can be used.

## Current Implementation

This repo contains an implementation of an **unconditional DDPM** model trained to generate **64x64 images**. It does not currently condition on any embeddings.

## Pretrained Weights

Download the pretrained weights from Hugging Face: 
[ckpt_20.pt](https://huggingface.co/01PrathamS/ddpm-unconditional-model/resolve/main/ckpt_20.pt)

Rename or place it as `ckpt.pt` in your working directory.

##  Usage

```python
from modules import UNet
from ddpm import Diffusion
import torch 
from utils import plot_images

device = "cuda" if torch.cuda.is_available() else "cpu"
model = UNet().to(device)
ckpt = torch.load("ckpt.pt")
model.load_state_dict(ckpt)

diffusion = Diffusion(img_size=64, device=device)
x = diffusion.sample(model, n=16)

plot_images(x)
```
