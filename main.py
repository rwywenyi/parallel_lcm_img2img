import os
import torch
from PIL import Image
from safetensors.torch import load_file
from tqdm import tqdm

from diffusers import AutoencoderKL, UNet2DConditionModel
from diffusers.pipelines.stable_diffusion.safety_checker import StableDiffusionSafetyChecker
from transformers import CLIPTokenizer, CLIPTextModel, CLIPImageProcessor
from diffusers.image_processor import VaeImageProcessor

from vae_processor import VAEProcessor
from unet_denoiser import UNetDenoiser
from lcm_scheduler import LCMScheduler

def load_models(model_id="/root/autodl-tmp/DreamShaper_7"):
    """Load and initialize all required models"""
    print("Loading models...")
    
    # Load basic models from local path
    vae = AutoencoderKL.from_pretrained(model_id, subfolder="vae")
    text_encoder = CLIPTextModel.from_pretrained(model_id, subfolder="text_encoder")
    tokenizer = CLIPTokenizer.from_pretrained(model_id, subfolder="tokenizer")
    
    # Load UNet with special configurations
    unet = UNet2DConditionModel.from_pretrained(
        model_id,
        subfolder="unet",
        device_map=None,
        low_cpu_mem_usage=False,
        local_files_only=True
    )
    
    # Modify UNet time conditioning
    unet.config.time_cond_proj_dim = 256
    unet.add_embedding_to_time = True
    unet.time_embedding.cond_proj = torch.nn.Linear(256, 320, bias=False)
    
    # Load safety checker and feature extractor
    safety_checker = StableDiffusionSafetyChecker.from_pretrained(
        model_id, 
        subfolder="safety_checker"
    )
    feature_extractor = CLIPImageProcessor.from_pretrained(
        model_id, 
        subfolder="feature_extractor"
    )
    
    # Load LCM weights
    print("Loading LCM weights...")
    lcm_path = "./LCM_Dreamshaper_v7_4k.safetensors"
    if os.path.exists(lcm_path):
        ckpt = load_file(lcm_path)
        m, u = unet.load_state_dict(ckpt, strict=False)
        if len(m) > 0:
            print("Missing keys:", m)
        if len(u) > 0:
            print("Unexpected keys:", u)
    else:
        print(f"Warning: LCM weights not found at {lcm_path}")
    
    # Initialize Scheduler:
    scheduler = LCMScheduler(beta_start=0.00085, beta_end=0.0120, beta_schedule="scaled_linear", prediction_type="epsilon")

    return vae, unet, tokenizer, text_encoder, safety_checker, feature_extractor, scheduler

def encode_prompt(prompt, tokenizer, text_encoder, device):
    """Encodes the prompt into text encoder hidden states."""
    if isinstance(prompt, str):
        batch_size = 1
    elif isinstance(prompt, list):
        batch_size = len(prompt)
    else:
        raise ValueError(f"prompt must be string or list, got {type(prompt)}")

    # Tokenize text
    text_inputs = tokenizer(
        prompt,
        padding="max_length", 
        max_length=tokenizer.model_max_length,
        truncation=True,
        return_tensors="pt",
    )
    text_input_ids = text_inputs.input_ids
    untruncated_ids = tokenizer(prompt, padding="longest", return_tensors="pt").input_ids

    if untruncated_ids.shape[-1] >= text_input_ids.shape[-1] and not torch.equal(
        text_input_ids, untruncated_ids
    ):
        removed_text = tokenizer.batch_decode(
            untruncated_ids[:, tokenizer.model_max_length - 1 : -1]
        )
        print(
            "The following part of your input was truncated because CLIP can only handle sequences up to"
            f" {tokenizer.model_max_length} tokens: {removed_text}"
        )

    if hasattr(text_encoder.config, "use_attention_mask") and text_encoder.config.use_attention_mask:
        attention_mask = text_inputs.attention_mask.to(device)
    else:
        attention_mask = None

    prompt_embeds = text_encoder(
        text_input_ids.to(device),
        attention_mask=attention_mask,
    )
    prompt_embeds = prompt_embeds[0]

    # Convert to UNet's expected dtype
    prompt_embeds = prompt_embeds.to(dtype=text_encoder.dtype, device=device)

    return prompt_embeds

def get_timesteps(num_inference_steps, strength, scheduler, device):
    # get the original timestep using init_timestep
    init_timestep = min(int(num_inference_steps * strength), num_inference_steps)
    t_start = max(num_inference_steps - init_timestep, 0)
    timesteps = scheduler.timesteps[t_start:]
    return timesteps, num_inference_steps - t_start

def process_image(image_path):
    """Process input image"""
    image = Image.open(image_path)
    if image.mode != "RGB":
        image = image.convert("RGB")
    return image

def main():
    # Setup device and paths
    device = "cuda" if torch.cuda.is_available() else "cpu"
    save_path = "./lcm_img2img_output"
    os.makedirs(save_path, exist_ok=True)
    
    # Load models
    vae, unet, tokenizer, text_encoder, safety_checker, feature_extractor, scheduler = load_models()
    
    # Move models to device
    vae = vae.to(device)
    unet = unet.to(device)
    text_encoder = text_encoder.to(device)
    
    # Initialize processors
    vae_processor = VAEProcessor(vae, scheduler, device=device)
    unet_denoiser = UNetDenoiser(unet, scheduler, device=device)
    
    # Setup timesteps and parameters
    num_inference_steps = 2  # Default LCM steps
    lcm_origin_steps = 50   # Original LDM steps
    strength = 0.5          # Denoising strength
    guidance_scale = 8.0    # Classifier-free guidance scale
    num_images_per_prompt = 1

    # Configure scheduler timesteps
    scheduler.set_timesteps(num_inference_steps, lcm_origin_steps, device=device)
    timesteps, num_inference_steps = get_timesteps(num_inference_steps, strength, scheduler, device)
    latent_timestep = timesteps[:1].repeat(num_images_per_prompt)

    # Load and process input image
    input_image_path = "/root/autodl-tmp/lcm_images/0.png"  # Replace with your image path
    init_image = process_image(input_image_path)
    
    # Setup prompt
    prompt = "A beautiful cyborg with blue hair, highly detailed, digital painting, artstation, concept art, smooth, sharp focus, illustration"
    prompt_embeds = encode_prompt(prompt, tokenizer, text_encoder, device)
    
    # Generate image
    print("Generating image...")
    with torch.no_grad():
        # 1. VAE Encode
        noisy_latents = vae_processor.encode(
            init_image,
            latent_timestep=latent_timestep,
            generator=torch.Generator(device=device).manual_seed(42)
        )
        
        # 2. UNet Denoise
        denoised_latents = unet_denoiser.denoise(
            noisy_latents,
            prompt_embeds,
            num_inference_steps=num_inference_steps,
            guidance_scale=guidance_scale,
            timesteps=timesteps
        )
        
        # 3. VAE Decode
        output_image = vae_processor.decode(denoised_latents)
    
    # Save output
    output_path = os.path.join(save_path, "img2img_output_0.png")
    output_image.save(output_path)
    print(f"Generated image saved to {output_path}")

if __name__ == "__main__":
    main() 