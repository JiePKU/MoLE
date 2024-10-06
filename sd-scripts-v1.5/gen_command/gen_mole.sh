
tmp_my_name=${0##*/}
my_name=${tmp_my_name%.*}
echo $my_name

output_dir=$my_name

export CUDA_VISIBLE_DEVICES="0"

export MODEL_NAME="/home/cxk/xknfs/zhujie/stable-diffusion-v1-5"  ## Path to the fine-tuned basemodel. you can also use sd v1.5 if you do not want to fine-tune.
export OUTPUT_DIR="./gen_output/$my_name"

PYTHON=/root/paddlejob/workspace/env_run/qjy_dataset_env/diffusion/bin/python

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
    --network_weights "/path/to/hand-lora/weight" "/path/to/face-lora/weight" \
    --network_module_moe networks.moe \
    --network_weights_moe "/path/to/moe/weight" \


