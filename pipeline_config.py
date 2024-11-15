import pika

# Shared configuration
RABBITMQ_CONFIG = {
    'host': 'localhost',
    'port': 5672,
    'credentials': pika.PlainCredentials('guest', 'guest')
}

REDIS_CONFIG = {
    'host': 'localhost',
    'port': 6379,
    'db': 0
}

QUEUE_NAMES = {
    'input_images': 'input_images_queue',
    'encoded_latents': 'encoded_latents_queue',
    'denoised_latents': 'denoised_latents_queue',
    'output_images': 'output_images_queue'
}

MODEL_PATH = "/root/autodl-tmp/DreamShaper_7"
PROMPT = "A beautiful cyborg with blue hair, highly detailed, digital painting, artstation, concept art, smooth, sharp focus, illustration" 