
tmp_my_name=${0##*/}
my_name=${tmp_my_name%.*}
echo $my_name

output_dir=$my_name

export CUDA_VISIBLE_DEVICES="0"

export MODEL_NAME="./sd_output/fine_sd_xformer_lion_300k_ac4_use_all"
export OUTPUT_DIR="./gen_output/$my_name"

PYTHON=/root/paddlejob/workspace/env_run/diffusion/bin/python

$PYTHON gen_img_diffusers_moe.py  \
    --ckpt $MODEL_NAME \
    --outdir $OUTPUT_DIR \
    --xformers \
    --fp16  \
    --W 512 \
    --H 512 --scale 7.5 \
    --sampler ddim \
    --steps 50 \
    --batch_size 1 \
    --images_per_prompt 1 \
    --from_file './prompts_example/test.txt' \
    --network_module networks.lora networks.lora \
    --network_weights "./sd_output/lora_unet_adamw_hand_test_alpha128_bsd15/lora_unet_adamw_hand_test_alpha128_bsd15.safetensors" "../sd_output/lora_unet_adamw_face-bsd15/lora_unet_adamw_face-bsd15.safetensors" \
    --network_module_moe networks.moe \
    --network_weights_moe "./sd_output/moe_unet_adamw_2ex_test_gate_50k_bsd15/moe_unet_adamw_2ex_test_gate_50k_bsd15.safetensors" \


