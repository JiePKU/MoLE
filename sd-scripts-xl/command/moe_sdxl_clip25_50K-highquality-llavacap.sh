
tmp_my_name=${0##*/}
my_name=${tmp_my_name%.*}
echo $my_name

output_dir=$my_name

# export MODEL="/root/paddlejob/workspace/env_run/output/sd_xl_base_1.0.safetensors"
export MODEL_NAME="/root/paddlejob/workspace/env_run/output/sd_xl_base_1.0.safetensors"
export DATASET_NAME="./data_config/high_quality_1024.toml"
export OUTPUT_DIR="./sd_output/$my_name"

export CUDA_VISIBLE_DEVICES="6,7"                                                                       
ADDR=10.95.147.13

NNODES=1
#RANK=$PADDLE_TRAINER_ID                                                                                  
RANK=0                                        
PORT=8977

PYTHON=/root/paddlejob/workspace/env_run/qjy_dataset_env/diffusion/bin/python

OMP_NUM_THREADS=1 $PYTHON -m torch.distributed.launch --nproc_per_node 2 \
    --nnodes=$NNODES \
    --node_rank=$RANK \
    --master_addr=$ADDR \
    --master_port=$PORT \
    --use_env sdxl_train_moe.py \
    --pretrained_model_name_or_path=$MODEL_NAME \
    --dataset_config=$DATASET_NAME \
    --output_dir=$OUTPUT_DIR \
    --output_name=$my_name \
    --prior_loss_weight=1.0 \
    --max_train_steps=50000 \
    --learning_rate=1e-5 \
    --network_dim=256 \
    --full_bf16 \
    --mixed_precision="bf16" \
    --save_precision="bf16" \
    --face_network_weights="/root/paddlejob/workspace/env_run/output/sd-scripts-xl/sd_output/lora_sdxl_face_llavacap_30k-lr2e-5-d256/lora_sdxl_face_llavacap_30k-lr2e-5-d256-step00030000.safetensors" \
    --hand_network_weights="/root/paddlejob/workspace/env_run/output/sd-scripts-xl/sd_output/lora_sdxl_hand_llavacap_60k-lr1e-5-d256/lora_sdxl_hand_llavacap_60k-lr1e-5-d256-step00060000.safetensors" \
    --optimizer_type="AdamW" \
    --network_train_unet_only \
    --sample_every_n_steps=1000 \
    --save_every_n_steps=500 \
    --network_module_lora=networks.lora \
    --network_module_moe=networks.moe \
    --sample_prompts='./prompts_example/prompts.txt' \
    --gradient_accumulation_steps=12 \
    --no_half_vae \
    --mem_eff_attn \
    --cache_text_encoder_outputs \
    --cache_text_encoder_outputs_to_disk \
    --cache_latents \
    --cache_latents_to_disk \
    --dim_from_weights \
    # --min_snr_gamma=5 \
    # --noise_offset 0.07 \
    # --xformers \
    # --noise_offset_random_strength \
    
