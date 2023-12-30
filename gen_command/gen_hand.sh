
tmp_my_name=${0##*/}
my_name=${tmp_my_name%.*}
echo $my_name

output_dir=$my_name

export CUDA_VISIBLE_DEVICES="1"

export MODEL_NAME="./sd_output/fine_sd_xformer_lion_300k_ac4_use_all"
export OUTPUT_DIR="./gen_output/$my_name"

PYTHON=/root/paddlejob/workspace/env_run/diffusion/bin/python

$PYTHON gen_img_diffusers.py  \
    --ckpt $MODEL_NAME \
    --outdir $OUTPUT_DIR \
    --xformers \
    --fp16  \
    --W 512 \
    --H 512 --scale 7.5 \
    --sampler ddim \
    --steps 30 \
    --batch_size 10 \
    --images_per_prompt 10 \
    --from_file './prompts_example/hand.txt' \
    --network_module networks.lora \
    --network_weights "./sd_output/lora_unet_adamw_hand_test_alpha256_2356/lora_unet_adamw_hand_test_alpha256_2356.safetensors" \

