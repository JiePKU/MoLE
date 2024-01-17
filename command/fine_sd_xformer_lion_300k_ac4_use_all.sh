
tmp_my_name=${0##*/}
my_name=${tmp_my_name%.*}
echo $my_name

output_dir=$my_name

export MODEL="./stable-diffusion-v1-5"
export MODEL_NAME="./stable-diffusion-v1-5"
export DATASET_NAME="./data-config/hqdata.toml"
export OUTPUT_DIR="./sd_output/$my_name"

PYTHON=/env_run/diffusion/bin/python

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
    --global_step=212500 \
    --mixed_precision=bf16 \
    --save_every_n_steps=500 \
    --gradient_accumulation_steps=4 \
    --save_state \
    --clip_skip=1 \
    --sample_every_n_steps=500 \
    --diffusers_xformers \
    --train_text_encoder \
    --use_lion_optimizer \
    --sample_prompts='./prompts_example/prompts.txt' \



