
tmp_my_name=${0##*/}
my_name=${tmp_my_name%.*}
echo $my_name

output_dir=$my_name


# export MODEL="/root/paddlejob/workspace/env_run/output/sd_xl_base_1.0.safetensors"
export MODEL_NAME="/root/paddlejob/workspace/env_run/output/sd_xl_base_1.0.safetensors"
export DATASET_NAME="./data_config/test_0.toml"
export OUTPUT_DIR="./sd_output/$my_name"

export CUDA_VISIBLE_DEVICES="0,1"                                                                       
# ADDR=`echo $PADDLE_TRAINERS | awk -F "," '{print $1}'`                                                                                           
# NNODES=$PADDLE_TRAINERS_NUM
# RANK=$PADDLE_TRAINER_ID                                                                                                                          
PORT=8940

ADDR=10.95.146.143

NNODES=1
#RANK=$PADDLE_TRAINER_ID                                                                                  
RANK=0                                        
# PORT=8977


PYTHON=/root/paddlejob/workspace/env_run/qjy_dataset_env/diffusion/bin/python

OMP_NUM_THREADS=1 $PYTHON -m torch.distributed.launch --nproc_per_node 2 \
    --nnodes=$NNODES \
    --node_rank=$RANK \
    --master_addr=$ADDR \
    --master_port=$PORT \
    --use_env sdxl_train_network.py \
    --pretrained_model_name_or_path=$MODEL_NAME \
    --dataset_config=$DATASET_NAME \
    --output_dir=$OUTPUT_DIR \
    --output_name=$my_name \
    --max_train_steps=30000 \
    --gradient_accumulation_steps=16 \
    --learning_rate=2e-5 \
    --network_dim=256 \
    --full_bf16 \
    --mixed_precision="bf16" \
    --save_precision="bf16" \
    --network_alpha=64 \
    --optimizer_type="AdamW" \
    --network_train_unet_only \
    --save_every_n_steps=500 \
    --network_module=networks.lora \
    --sample_prompts='./prompts_example/face_prompts.txt' \
    --no_half_vae \
    --cache_text_encoder_outputs \
    --cache_text_encoder_outputs_to_disk \
    --cache_latents \
    --cache_latents_to_disk \
    --prior_loss_weight=1.0 \
    --full_bf16 \
    # --mem_eff_attn \
    # --xformers \
    # --noise_offset 0.07 \
    # --noise_offset_random_strength \
    