import torch
import PIL.Image
import numpy as np
from diffusers import AutoencoderKL
from diffusers.image_processor import VaeImageProcessor

from lcm_scheduler import LCMScheduler

class VAEProcessor:
    def __init__(self, vae: AutoencoderKL, scheduler: LCMScheduler, device="cuda"):
        self.vae = vae.to(device)
        self.scheduler = scheduler
        self.device = device
        self.vae_scale_factor = 2 ** (len(self.vae.config.block_out_channels) - 1)
        self.image_processor = VaeImageProcessor(vae_scale_factor=self.vae_scale_factor)

    def preprocess_image(self, image):
        """Preprocess image for VAE input"""
        if isinstance(image, torch.Tensor):
            return image
        elif isinstance(image, PIL.Image.Image):
            image = [image]

        if isinstance(image[0], PIL.Image.Image):
            w, h = image[0].size
            w, h = map(lambda x: x - x % 8, (w, h))  # resize to integer multiple of 8

            image = [np.array(i.resize((w, h), resample=PIL.Image.Resampling.LANCZOS))[None, :] for i in image]
            image = np.concatenate(image, axis=0)
            image = np.array(image).astype(np.float32) / 255.0
            image = image.transpose(0, 3, 1, 2)
            image = 2.0 * image - 1.0
            image = torch.from_numpy(image)
        elif isinstance(image[0], torch.Tensor):
            image = torch.cat(image, dim=0)
        
        return image.to(self.device)
    
    def encode(self, image, latent_timestep=None, generator=None):
        """Encode image to latent space and add noise"""
        # Convert image to latents
        image = self.preprocess_image(image)
        image = image.to(device=self.device, dtype=self.vae.dtype)
        
        # Encode with VAE
        with torch.no_grad():
            init_latents = self.vae.encode(image).latent_dist.sample(generator=generator)
            init_latents = self.vae.config.scaling_factor * init_latents

        # Handle batch size
        batch_size = init_latents.shape[0]
        if batch_size > 1:
            init_latents = torch.cat([init_latents], dim=0)

        # Add noise
        shape = init_latents.shape
        noise = torch.randn(shape, generator=generator, device=self.device, dtype=init_latents.dtype)
        
        noisy_latents = self.scheduler.add_noise(init_latents, noise, latent_timestep)

        return noisy_latents


    def decode(self, latents):
        """Decode latents to image"""
        latents = latents.to(self.device)

        with torch.no_grad():
            # Scale and decode the image latents with vae
            latents = 1 / self.vae.config.scaling_factor * latents
            image = self.vae.decode(latents).sample

            # Convert to PIL images
            image = (image / 2 + 0.5).clamp(0, 1)
            image = image.cpu().permute(0, 2, 3, 1).float().numpy()
            image = self.image_processor.numpy_to_pil(image)

        return image[0] if len(image) == 1 else image 