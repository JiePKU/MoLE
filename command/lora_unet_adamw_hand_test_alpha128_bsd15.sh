
tmp_my_name=${0##*/}
my_name=${tmp_my_name%.*}
echo $my_name

output_dir=$my_name


export MODEL="./sd_output/fine_sd_xformer_lion_300k_ac4_use_all/fine_sd_xformer_lion_300k_ac4_use_all"
export MODEL_NAME="./sd_output/fine_sd_xformer_lion_300k_ac4_use_all/fine_sd_xformer_lion_300k_ac4_use_all"
export DATASET_NAME="./data-config/hand.toml"
export OUTPUT_DIR="./sd_output/$my_name"

export CUDA_VISIBLE_DEVICES="0,1,2,3"                                                                       
ADDR=`echo $PADDLE_TRAINERS | awk -F "," '{print $1}'`                                                                                           
NNODES=$PADDLE_TRAINERS_NUM
RANK=$PADDLE_TRAINER_ID                                                                                                                          
PORT=8932


PYTHON=/root/paddlejob/workspace/env_run/diffusion/bin/python

OMP_NUM_THREADS=1 $PYTHON -m torch.distributed.launch --nproc_per_node 4 \
    --nnodes=$NNODES \
    --node_rank=$RANK \
    --master_addr=$ADDR \
    --master_port=$PORT \
    --use_env train_network.py \
    --pretrained_model_name_or_path=$MODEL_NAME \
    --load_model=$MODEL \
    --dataset_config=$DATASET_NAME \
    --output_dir=$OUTPUT_DIR \
    --output_name=$my_name \
    --prior_loss_weight=1.0 \
    --max_train_steps=60000 \
    --learning_rate=1e-5 \
    --network_dim=256 \
    --gradient_accumulation_steps=4 \
    --xformer \
    --network_alpha=128 \
    --save_state \
    --optimizer_type="AdamW" \
    --mixed_precision="bf16" \
    --network_train_unet_only \
    --save_every_n_steps=500 \
    --sample_every_n_steps=500 \
    --network_module=networks.lora \
    --sample_prompts='./prompts_example/hand_prompts.txt' \