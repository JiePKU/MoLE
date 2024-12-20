<p align="center">
  <img src="img/title.png"  height=120>
</p>

### <div align="center"> MoLE: Enhancing Human-centric Text-to-image Diffusion via Mixture of Low-rank Experts<div> 
### <div align="center"> NeurIPS 2024 <div> 

<div align="center">

[Project homepage](https://sites.google.com/view/mole4diffuser/) || [Paper](https://arxiv.org/abs/2410.23332) ||  [Model](https://huggingface.co/jiezhueval/MoLE-SDXL/tree/main) || [Human-centric Dataset](https://pan.baidu.com/s/1QL_IImARcBBLwleXEI1wsg) [code: asd4]

</div>

<div align="center">

>[Jie Zhu](https://scholar.google.com/citations?user=ZL506kEAAAAJ&hl=zh-CN), [Yixiong Chen](https://schuture.github.io/), 
[Mingyu Ding](https://dingmyu.github.io/), [Ping Luo](http://luoping.me/),
[Leye Wang](https://wangleye.github.io/)&#8224;,
[Jingdong Wang](https://jingdongwang2017.github.io/)&#8224;
<br>Peking University, Johns Hopkins University, UC Berkeley, The University of Hong Kong, Baidu<br>

</div>

<p align="center">
  <img src="img/case.png"  height=400>
</p>


## Introduction
This is an official implementation of MoLE, which is a human-centric text-to-image diffusion model. We provide the code for SD v1.5 and SDXL, respectively.  

## Requirements
Pleae see requirements.txt. We provide the xformers file used in our environment in [here](https://drive.google.com/drive/folders/1h390KY7VVXhXqXd1r1-np4E6vdEXxUUU?usp=sharing)

## Data Preparation
Download the Human-centric Dataset from [Baidu Pan](https://pan.baidu.com/s/1QL_IImARcBBLwleXEI1wsg) (code: asd4) or [Google Drive](https://drive.google.com/drive/folders/1jWOouxul6LZWHsU1NsH2RvZemzfgR3PN?usp=sharing).

This dataset involves three subsets:human-in-the-scene images, close-up of face images, and close-up of hand images, totally one million images. Moreover these images possess superior quality and boasts high aesthetic scores.

We also provide the scripts of downloading raw images from corresponding websites. See directory ./climb_scripts

##### NOTE: Our dataset is allowed for academic purposes only. When using it, the users are requested to ensure compliance with legal regulations. See LICENSE.txt for details.

#### If it is helpful, please give us a star and cite our paper. Thanks!

## Ackowledgement
We thank the authors of [XFormer](https://github.com/lucidrains/xformers) for providing us with a great library. Our code is based on [sd-scripts](https://github.com/kohya-ss/sd-scripts). Thank the authors. We also thank Stability.ai for
its open source.




