import pika
import redis
import torch
import json
from PIL import Image
import io
import base64
from pipeline_config import *
from main import load_models, VAEProcessor, get_timesteps

def process_image(ch, method, properties, body):
    # Decode message
    message = json.loads(body.decode())
    image_data = base64.b64decode(message['image'])
    image_id = message['image_id']
    
    # Process image
    init_image = Image.open(io.BytesIO(image_data))
    if init_image.mode != "RGB":
        init_image = init_image.convert("RGB")
    
    # Generate noisy latents
    latent_timestep = timesteps[:1].repeat(1)
    noisy_latents = vae_processor.encode(
        init_image,
        latent_timestep=latent_timestep,
        generator=torch.Generator(device=device).manual_seed(42)
    )
    
    # Store in Redis
    redis_client.set(f"latents:{image_id}", noisy_latents.cpu().numpy().tobytes())
    # redis_client.set(f"latents:{image_id}", noisy_latents)
    
    # Send message to next stage
    channel.basic_publish(
        exchange='',
        routing_key=QUEUE_NAMES['encoded_latents'],
        body=json.dumps({'image_id': image_id})
    )
    
    ch.basic_ack(delivery_tag=method.delivery_tag)

if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    # Load models
    vae, _, _, _, _, _, scheduler = load_models(MODEL_PATH)
    vae = vae.to(device)
    vae_processor = VAEProcessor(vae, scheduler, device=device)
    
    # Setup scheduler
    scheduler.set_timesteps(2, 50, device=device)
    timesteps, _ = get_timesteps(2, 0.5, scheduler, device)
    
    # Connect to RabbitMQ
    connection = pika.BlockingConnection(pika.ConnectionParameters(**RABBITMQ_CONFIG))
    channel = connection.channel()
    channel.queue_declare(queue=QUEUE_NAMES['input_images'])
    channel.queue_declare(queue=QUEUE_NAMES['encoded_latents'])
    
    # Connect to Redis
    redis_client = redis.Redis(**REDIS_CONFIG)
    
    # Start consuming
    channel.basic_consume(
        queue=QUEUE_NAMES['input_images'],
        on_message_callback=process_image
    )
    
    print("Stage 1 (VAE Encode) waiting for messages...")
    channel.start_consuming() 