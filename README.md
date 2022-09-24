# MAE_ViT

## MAE描述

Masked Autoencoders: A MindSpore Implementation，由何凯明团队提出MAE模型，将NLP领域大获成功的自监督预训练模式用在了计算机视觉任务上，效果拔群，在NLP和CV两大领域间架起了一座更简便的桥梁。MAE 是一种简单的自编码方法，可以在给定部分观察的情况下重建原始信号。由编码器将观察到的信号映射到潜在表示，再由解码器从潜在表示重建原始信号。

This is a MindSpore/NPU re-implementation of the paper [Masked Autoencoders Are Scalable Vision Learners](https://arxiv.org/abs/2111.06377):

## 模型架构

<p align="center">
  <img src="https://user-images.githubusercontent.com/11435359/146857310-f258c86c-fde6-48e8-9cee-badd2b21bd2c.png" width="480">
</p>

在预训练期间，大比例的随机的图像块子集（如 75%）被屏蔽掉。编码器用于可见patch的小子集。在编码器之后引入掩码标记，并且完整的编码块和掩码标记集由一个小型解码器处理，该解码器以像素为单位重建原始图像。预训练后，解码器被丢弃，编码器应用于未损坏的图像以生成识别任务的表示。

## 训练过程

### 训练

```shell
  export CUDA_VISIBLE_DEVICES=0
  python train.py --model mae_vit --data_url ./dataset/imagenet > train.log 2>&1 &
```

## 评估过程

### 评估

```shell
python validate.py --model mae_vit --data_url ./dataset/imagenet --checkpoint_path=[CHECKPOINT_PATH]
```

```text 
# grep "accuracy=" eval0/log
accuracy=0.81
```
