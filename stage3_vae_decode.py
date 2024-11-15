import base64
import io
import pika
import redis
import torch
import json
import numpy as np
from pipeline_config import *
from main import load_models, VAEProcessor

def process_denoised(ch, method, properties, body):
    message = json.loads(body.decode())
    image_id = message['image_id']
    
    # Get denoised latents from Redis
    denoised_latents_bytes = redis_client.get(f"denoised:{image_id}")
    denoised_latents = torch.from_numpy(
        np.frombuffer(denoised_latents_bytes, dtype=np.float32)
    ).reshape(1, 4, 96, 96).to(device)
    
    # Decode
    output_image = vae_processor.decode(denoised_latents)
    
    # Convert to bytes
    img_byte_arr = io.BytesIO()
    output_image.save(img_byte_arr, format='PNG')
    img_byte_arr = img_byte_arr.getvalue()
    
    # Send to output queue
    channel.basic_publish(
        exchange='',
        routing_key=QUEUE_NAMES['output_images'],
        body=json.dumps({
            'image_id': image_id,
            'image': base64.b64encode(img_byte_arr).decode()
        })
    )
    
    # Cleanup Redis
    redis_client.delete(f"latents:{image_id}")
    redis_client.delete(f"denoised:{image_id}")
    
    ch.basic_ack(delivery_tag=method.delivery_tag)

if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    # Load models
    vae, _, _, _, _, _, scheduler = load_models(MODEL_PATH)
    vae = vae.to(device)
    vae_processor = VAEProcessor(vae, scheduler, device=device)
    
    # Connect to RabbitMQ
    connection = pika.BlockingConnection(pika.ConnectionParameters(**RABBITMQ_CONFIG))
    channel = connection.channel()
    channel.queue_declare(queue=QUEUE_NAMES['denoised_latents'])
    channel.queue_declare(queue=QUEUE_NAMES['output_images'])
    
    # Connect to Redis
    redis_client = redis.Redis(**REDIS_CONFIG)
    
    # Start consuming
    channel.basic_consume(
        queue=QUEUE_NAMES['denoised_latents'],
        on_message_callback=process_denoised
    )
    
    print("Stage 3 (VAE Decode) waiting for messages...")
    channel.start_consuming() 