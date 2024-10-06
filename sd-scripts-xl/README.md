## Introduction
This is an official implementation of MoLE (SDXL), which is a human-centric text-to-image diffusion model. 

## Requirements
Pleae see requirements.txt. We provide the xformers wheel file used in our environment in [here](https://drive.google.com/file/d/1XLvP0T_xoxUyuqA7nCJLjc3Cn1p9KaYG/view?usp=sharing)

## Data Preparation
Download the [Human-centric Dataset](https://pan.baidu.com/s/1QL_IImARcBBLwleXEI1wsg) [code: asd4].

This dataset involves three subsets:human-in-the-scene images, close-up of face images, and close-up of hand images, totally one million images. Moreover these images possess superior quality and boasts high aesthetic scores. 


#### Note that the image size currently is 512x512, it is recommended to resize them into 1024x1024 for SDXL. We also provide the scripts of downloading raw images from corresponding websites. See directory ./climb_scripts. And you can run them to obtain raw images and preprocess them into 1024x1024 images by yourself to further improve the quality of images.

## Training
For SDXL, we use two stages in MoLE. We highly recommend you to train with log and using clip-filtered (20 or 25) image-text pairs to improve model performance.


### Preprocess: Cache Image-Text Pairs (Optional)

For faster training and saving memory, we cache image-text pairs with the following commands:

```shell
bash ./command/cache_text_encoder_outputs_X.sh
```

```shell 
bash ./command/catch_image_latent_X.sh
```

### Stage 1: Low-rank Expert Generation

Train face expert
```shell
bash ./command/lora_sdxl_face_llavacap_30k-lr2e-5-d256.sh
```

Train hand expert
```shell
bash ./command/lora_sdxl_hand_llavacap_60k-lr1e-5-d256.sh
```

### Stage 2: Soft Mixture Assignment

```shell
bash ./command/moe_sdxl_clip25_50K-highquality-llavacap.sh
```

## Testing
```shell
bash ./gen_command/moe_sdxl_gen-id0-llava.sh
```




