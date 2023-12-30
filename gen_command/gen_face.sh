
tmp_my_name=${0##*/}
my_name=${tmp_my_name%.*}
echo $my_name

output_dir=$my_name

export CUDA_VISIBLE_DEVICES="0"

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
    --images_per_prompt 30 \
    --from_file './prompts_example/test.txt' \
    --network_module networks.lora \
    --network_weights "./sd_output/lora_unet_adamw_face/lora_unet_adamw_face.safetensors" \