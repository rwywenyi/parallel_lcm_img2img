import pika
import redis
import torch
import json
import numpy as np
from pipeline_config import *
from main import load_models, UNetDenoiser, encode_prompt, get_timesteps

def process_latents(ch, method, properties, body):
    message = json.loads(body.decode())
    image_id = message['image_id']
    
    # Get latents from Redis
    noisy_latents_bytes = redis_client.get(f"latents:{image_id}")
    noisy_latents = torch.from_numpy(
        np.frombuffer(noisy_latents_bytes, dtype=np.float32)
    ).reshape(1, 4, 96, 96).to(device)
    # noisy_latents = redis_client.get(f"latents:{image_id}")

    # Denoise
    denoised_latents = unet_denoiser.denoise(
        noisy_latents,
        prompt_embeds,
        num_inference_steps=num_inference_steps,
        guidance_scale=8.0,
        timesteps=timesteps
    )
    
    # Store result in Redis
    redis_client.set(f"denoised:{image_id}", denoised_latents.cpu().numpy().tobytes())
    
    # Send message to next stage
    channel.basic_publish(
        exchange='',
        routing_key=QUEUE_NAMES['denoised_latents'],
        body=json.dumps({'image_id': image_id})
    )
    
    ch.basic_ack(delivery_tag=method.delivery_tag)

if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    # Load models
    _, unet, tokenizer, text_encoder, _, _, scheduler = load_models(MODEL_PATH)
    unet = unet.to(device)
    text_encoder = text_encoder.to(device)
    
    # Setup UNet denoiser
    unet_denoiser = UNetDenoiser(unet, scheduler, device=device)
    
    # Encode prompt
    prompt_embeds = encode_prompt(PROMPT, tokenizer, text_encoder, device)
    
    # Setup scheduler
    scheduler.set_timesteps(2, 50, device=device)
    timesteps, num_inference_steps = get_timesteps(2, 0.5, scheduler, device)
    
    # Connect to RabbitMQ
    connection = pika.BlockingConnection(pika.ConnectionParameters(**RABBITMQ_CONFIG))
    channel = connection.channel()
    channel.queue_declare(queue=QUEUE_NAMES['encoded_latents'])
    channel.queue_declare(queue=QUEUE_NAMES['denoised_latents'])
    
    # Connect to Redis
    redis_client = redis.Redis(**REDIS_CONFIG)
    
    # Start consuming
    channel.basic_consume(
        queue=QUEUE_NAMES['encoded_latents'],
        on_message_callback=process_latents
    )
    
    print("Stage 2 (UNet Denoise) waiting for messages...")
    channel.start_consuming() 