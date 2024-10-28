tmp_my_name=${0##*/}
my_name=${tmp_my_name%.*}
echo $my_name

output_dir=$my_name

export CUDA_VISIBLE_DEVICES="0"

export MODEL_NAME="/root/paddlejob/workspace/env_run/output/sd_xl_base_1.0.safetensors"
export OUTPUT_DIR="./gen_output/$my_name"

PYTHON=/root/paddlejob/workspace/env_run/qjy_dataset_env/diffusion/bin/python

$PYTHON sdxl_gen_img.py  \
    --ckpt $MODEL_NAME \
    --outdir $OUTPUT_DIR \
    --batch_size 2 \
    --sampler ddim \
    --steps 30 \
    --bf16  \
    --no_half_vae \
    --images_per_prompt 5 \
    --from_file './prompt_example/prompt_sdxl.txt' \
    # --network_module networks.lora \
    # --network_weights "./sd_output/lora_sdxl_face_llavacap_30k-lr2e-5-d256/lora_sdxl_face_llavacap_30k-lr2e-5-d256-step00030000.safetensors" \