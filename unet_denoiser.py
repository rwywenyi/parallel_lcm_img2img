import torch
from diffusers.pipelines.pipeline_utils import DiffusionPipeline
from diffusers import UNet2DConditionModel

class UNetDenoiser(DiffusionPipeline):
    def __init__(self, unet: UNet2DConditionModel, scheduler, device="cuda"):
        super().__init__()
        self.unet = unet.to(device)
        self.scheduler = scheduler
        self.devices = device
        self.num_inference_steps = 4  # Default value, can be changed

    def get_w_embedding(self, w, embedding_dim=512):
        """Get w-embedding for guidance scale"""
        w = w * 1000.0
        half_dim = embedding_dim // 2
        emb = torch.log(torch.tensor(10000.0)) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, dtype=torch.float32, device=self.devices) * -emb)
        emb = w.to(self.devices)[:, None] * emb[None, :]
        emb = torch.cat([torch.sin(emb), torch.cos(emb)], dim=1)
        if embedding_dim % 2 == 1:
            emb = torch.nn.functional.pad(emb, (0, 1))
        return emb

    def denoise(self, latents, prompt_embeds, num_inference_steps, guidance_scale=8.0, timesteps=None, cross_attention_kwargs=None):
        """Denoise latents using UNet"""
        batch_size = latents.shape[0]
        w = torch.tensor(guidance_scale).repeat(batch_size)
        w_embedding = self.get_w_embedding(w, embedding_dim=256).to(self.devices)

        # Denoising loop
        with self.progress_bar(total=num_inference_steps) as progress_bar:
            for i, t in enumerate(timesteps):
                ts = torch.full((batch_size,), t, device=self.devices, dtype=torch.long)
            
            # UNet prediction
            with torch.no_grad():
                noise_pred = self.unet(
                    latents,
                    ts,
                    timestep_cond=w_embedding,
                    encoder_hidden_states=prompt_embeds,
                    cross_attention_kwargs=cross_attention_kwargs,
                    return_dict=False,
                )[0]

                # Scheduler step
                latents, denoised = self.scheduler.step(noise_pred, i, t, latents, return_dict=False)

                progress_bar.update()

        return denoised 