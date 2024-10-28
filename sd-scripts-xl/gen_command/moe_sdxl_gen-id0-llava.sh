tmp_my_name=${0##*/}
my_name=${tmp_my_name%.*}
echo $my_name

output_dir=$my_name

export CUDA_VISIBLE_DEVICES="6"

export MODEL_NAME="/root/paddlejob/workspace/env_run/output/sd_xl_base_1.0.safetensors"
export OUTPUT_DIR="./gen_output/$my_name"

PYTHON=/root/paddlejob/workspace/env_run/qjy_dataset_env/diffusion/bin/python

$PYTHON moe_sdxl_gen_img.py  \
    --ckpt $MODEL_NAME \
    --outdir $OUTPUT_DIR \
    --batch_size 2 \
    --sampler ddim \
    --steps 50 \
    --bf16  \
    --scale 7.5 \
    --n_iter 1 \
    --no_half_vae \
    --images_per_prompt 2 \
    --from_file './prompt_example/captions_val2014_person_v1.txt' \
    --network_module networks.lora networks.lora \
    --network_weights "./sd_output/lora_sdxl_face_llavacap_30k-lr2e-5-d256/lora_sdxl_face_llavacap_30k-lr2e-5-d256-step00030000.safetensors" "./sd_output/lora_sdxl_hand_llavacap_60k-lr1e-5-d256/lora_sdxl_hand_llavacap_60k-lr1e-5-d256-step00060000.safetensors" \
    --network_module_moe networks.moe \
    --network_weights_moe "./sd_output/moe_sdxl_clip25_50K-highquality-llavacap/moe_sdxl_clip25_50K-highquality-llavacap-step00030000.safetensors" \