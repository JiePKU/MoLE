## Introduction
This is an official implementation of MoLE (SD v1.5), which is a human-centric text-to-image diffusion model. 

## Requirements
Pleae see requirements.txt. We provide the xformers wheel file used in our environment in [here](https://drive.google.com/file/d/1XLvP0T_xoxUyuqA7nCJLjc3Cn1p9KaYG/view?usp=sharing)

## Data Preparation
Download the [Human-centric Dataset](https://pan.baidu.com/s/1QL_IImARcBBLwleXEI1wsg) [code: asd4].

This dataset involves three subsets:human-in-the-scene images, close-up of face images, and close-up of hand images, totally one million images. Moreover these images possess superior quality and boasts high aesthetic scores.

#### We also provide the scripts of downloading raw images from corresponding websites. See directory ./climb_scripts

## Training

MoLE contains three stages. We highly recommend you to train with log and using clip-filtered (20 or 25) image-text pairs to improve model performance.

### Stage 1: Fine-tuning on Human-centric Dataset  (Optional)

```shell
bash ./command/fine_sd_xformer_lion_300k_use_all.sh
```

You can also train with log to watch this process by:

```shell
nohup bash ./command/fine_sd_xformer_lion_300k_ac4_use_all.sh &> logs/fine_sd_xformer_lion_300k_use_all.txt &
```

### Stage 2: Low-rank Expert Generation

Train face expert
```shell
bash ./command/lora_unet_adamw_face-bsd15.sh
```

Train hand expert
```shell
bash ./command/lora_unet_adamw_hand-bsd15.sh
```

### Stage 3: Soft Mixture Assignment

```shell
bash ./command/moe_unet_adamw_2ex_gate_50k_bsd15.sh
```

## Testing
```shell
bash ./gen_command/gen_mole.sh
```




