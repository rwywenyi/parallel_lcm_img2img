import pika
import json
import base64
import os
from PIL import Image
import io
from pipeline_config import RABBITMQ_CONFIG, QUEUE_NAMES

def save_image(image_data, image_id):
    # 创建输出目录
    output_dir = "output_images"
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    # 解码base64图片数据
    image_bytes = base64.b64decode(image_data)
    image = Image.open(io.BytesIO(image_bytes))
    
    # 保存图片
    output_path = os.path.join(output_dir, f"{image_id}_processed.png")
    image.save(output_path)
    print(f"Saved processed image to: {output_path}")

def process_output(ch, method, properties, body):
    # 解析消息
    message = json.loads(body.decode())
    image_id = message['image_id']
    image_data = message['image']
    
    print(f"Received processed image for {image_id}")
    save_image(image_data, image_id)
    
    # 确认消息
    ch.basic_ack(delivery_tag=method.delivery_tag)

def main():
    # 连接到RabbitMQ
    connection = pika.BlockingConnection(pika.ConnectionParameters(**RABBITMQ_CONFIG))
    channel = connection.channel()
    
    # 确保队列存在
    channel.queue_declare(queue=QUEUE_NAMES['output_images'])
    
    # 设置消费者
    channel.basic_consume(
        queue=QUEUE_NAMES['output_images'],
        on_message_callback=process_output
    )
    
    print("Waiting for processed images. To exit press CTRL+C")
    channel.start_consuming()

if __name__ == "__main__":
    main() 