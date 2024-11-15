import pika
import base64
import json
import io
from PIL import Image
from pipeline_config import RABBITMQ_CONFIG, QUEUE_NAMES

def send_single_image(image_path):
    # 连接到RabbitMQ
    connection = pika.BlockingConnection(pika.ConnectionParameters(**RABBITMQ_CONFIG))
    channel = connection.channel()
    
    # 确保队列存在
    channel.queue_declare(queue=QUEUE_NAMES['input_images'])
    
    # 读取并处理图片
    with Image.open(image_path) as img:
        if img.mode != "RGB":
            img = img.convert("RGB")
        img_byte_arr = io.BytesIO()
        img.save(img_byte_arr, format='PNG')
        img_byte_arr = img_byte_arr.getvalue()
    
    # 创建消息
    message = {
        'image_id': 'test_image_1',
        'image': base64.b64encode(img_byte_arr).decode('utf-8')
    }
    
    # 发送消息
    channel.basic_publish(
        exchange='',
        routing_key=QUEUE_NAMES['input_images'],
        body=json.dumps(message)
    )
    
    print(f"Sent image: {image_path}")
    connection.close()

if __name__ == "__main__":
    # 替换为你的图片路径
    image_path = "/root/autodl-tmp/lcm_images/3.png"
    send_single_image(image_path) 