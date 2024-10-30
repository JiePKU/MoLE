
tmp_my_name=${0##*/}
my_name=${tmp_my_name%.*}
echo $my_name

output_dir=$my_name

export MODEL="/home/cxk/xknfs/zhujie/stable-diffusion-v1-5" ## init model weight 
export MODEL_NAME="/home/cxk/xknfs/zhujie/stable-diffusion-v1-5"
export DATASET_NAME="./data_config/hdimage.toml"  ## image data we use 
export OUTPUT_DIR="./sd_output/$my_name" 

PYTHON=/root/paddlejob/workspace/env_run/qjy_dataset_env/diffusion/bin/python

OMP_NUM_THREADS=1 $PYTHON -m torch.distributed.launch --nproc_per_node 4 \
    --use_env fine_tune.py \
    --pretrained_model_name_or_path=$MODEL_NAME \
    --load_model=$MODEL \
    --output_dir=$OUTPUT_DIR \
    --output_name=$my_name \
    --dataset_config=$DATASET_NAME \
    --learning_rate=2e-6 --max_train_steps=300000 \
    --xformers \
    --min_snr_gamma=5 \
    --mixed_precision=bf16 \
    --save_every_n_steps=1000 \
    --gradient_accumulation_steps=4 \
    --save_state \
    --clip_skip=1 \
    --sample_every_n_steps=500 \
    --diffusers_xformers \
    --use_lion_optimizer \
    --sample_prompts='./prompts_example/prompts.txt' \
    # --train_text_encoder \ ## to save memory and train unet only
    # --global_step=212500 \ ## for resume training



