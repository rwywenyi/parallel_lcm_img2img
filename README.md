# 基于rabbitmq和redis实现三级并行图生图

## 程序说明

**main.py**

main.py实现了一个串行的LCM图生图功能，并且分为三个模块：vae_encode, unet_denoise, vae_decode。

**stage1_vae_encode.py**

1. vae_encode模块实现了vae编码功能，将输入图片编码为latent，并根据强度添加噪声。
2. 作为第一级队列的生产者，将编码后的latent发送到第二级队列。

**stage2_unet_denoise.py**

1. unet_denoise模块实现了unet去噪功能，将latent去噪为最终的latent。
2. 作为第二级队列的消费者，从第一级队列中取出latent进行去噪，并将去噪后的latent发送到第三级队列。

**stage3_vae_decode.py**

1. 从第二级队列消费去噪后的latent数据。
2. vae_decode模块实现了vae解码功能，将最终的latent解码为图片。 

## 架构说明

这种架构的优势：
解耦合: 三个阶段完全独立，可以分别部署和扩展
并行处理: 多个任务可以同时在不同阶段进行处理
负载均衡: 可以根据每个阶段的处理能力配置不同数量的worker
容错性: 如果某个阶段出现问题，不会影响其他阶段的正常运行
工作流程示意：

 Input Image → [Queue1] → VAE Encode → [Queue2] → UNet Denoise → [Queue3] → VAE Decode → Output Image

## send和receive

send.py和receive.py实现了消息的发送和接收功能，分别用于将数据发送到rabbitmq和从rabbitmq中接收数据。

## notes

此代码已经在Linux安装rabbitmq和redis，通过验证，可以下载DreamShaper_7模型和LCM_Dreamshaper_v7_4k.safetensors模型运行。