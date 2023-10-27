import torch
from torch import nn

import torchvision

from diffusers import AutoencoderKL


class StableDiffusionVae:
    device: torch.device

    vae: AutoencoderKL

    transforms: nn.Module

    latent_channels: int

    def __init__(self, stable_diffusion_path: str, device: torch.device):
        self.device = device
        
        self.vae = AutoencoderKL.from_pretrained(stable_diffusion_path, subfolder="vae").to(self.device)
        self.vae.requires_grad_(False)
        
        self.transforms = torchvision.transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        
        self.latent_channels = 4
    
    def calc_latent_size(self, image_size: int):
        return image_size // 8
        
    def encode(self, images):
        images = self.transforms(images)
        images = images.to(dtype=torch.float32, device=self.device)
        latents = self.vae.encode(images).latent_dist.sample()
        latents = latents * 0.18215
        return latents
    
    def decode(self, latents):
        latents = 1 / 0.18215 * latents
        image = self.vae.decode(latents).sample
        image = (image / 2 + 0.5).clamp(0, 1) # 謎
        return image
    