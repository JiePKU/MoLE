### An official complement of "MoLE: Human-centric Text-to-image Diffusion with Mixture of Low-rank Experts" 

[Project homepage](https://sites.google.com/view/mole4diffuser/) || [Paper](https://sites.google.com/view/mole4diffuser/) ||
[Hand Close-up](https://sites.google.com/view/response-close-up-of-hand/homepage) || [Human-centric Dataset](https://pan.baidu.com/s/1QL_IImARcBBLwleXEI1wsg) [code: asd4]

## Introduction
This is an official implementation of MoLE, which is a human-centric text-to-image diffusion model. 

## Requirements
Pleae see requirements.txt. We provide the xformers wheel file used in our environment in [here](https://drive.google.com/file/d/1XLvP0T_xoxUyuqA7nCJLjc3Cn1p9KaYG/view?usp=sharing)

## Data Preparation
Download the [Human-centric Dataset](https://pan.baidu.com/s/1QL_IImARcBBLwleXEI1wsg) [code: asd4].

This dataset involves three subsets:human-in-the-scene images, close-up of face images, and close-up of hand images, totally one million images. Moreover these images possess superior quality and boasts high aesthetic scores.

#### We also provide the scripts of downloading raw images from corresponding websites. See directory ./climb_scripts

## Training
MoLE contains three stages.

### Stage 1: Fine-tuning on Human-centric Dataset 

```shell
bash ./command/fine_sd_xformer_lion_300k_ac4_use_all.sh
```

You can also train with log to watch this process by:

```shell
nohup bash ./command/fine_sd_xformer_lion_300k_ac4_use_all.sh &> logs/fine_sd_xformer_lion_300k_ac4_use_all.txt &
```

### Stage 2: Low-rank Expert Generation

Train face expert
```shell
bash ./command/lora_unet_adamw_face-bsd15.sh
```

Train hand expert
```shell
bash ./command/lora_unet_adamw_hand_test_alpha128_bsd15.sh
```

### Stage 3: Soft Mixture Assignment

```shell
bash ./command/moe_unet_adamw_2ex_test_moe_50k_bsd15.sh
```

## Testing
```shell
bash ./gen_command/gen_mole.sh
```

## Ackowledgement
We thank the authors of [XFormer](https://github.com/lucidrains/xformers) for providing us with a great library. Our code is based on [sd-scripts](https://github.com/kohya-ss/sd-scripts). Thank the authors. We also thank Stability.ai for
its open source.




