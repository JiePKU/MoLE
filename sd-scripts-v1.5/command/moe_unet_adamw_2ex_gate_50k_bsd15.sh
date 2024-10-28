
tmp_my_name=${0##*/}
my_name=${tmp_my_name%.*}
echo $my_name

output_dir=$my_name


export MODEL="/home/cxk/xknfs/zhujie/stable-diffusion-v1-5"  ## Path to the fine-tuned basemodel. you can also use sd v1.5 if you do not want to fine-tune.
export MODEL_NAME="/home/cxk/xknfs/zhujie/stable-diffusion-v1-5"
export DATASET_NAME="./data_config/hqdata.toml"
export OUTPUT_DIR="./sd_output/$my_name"

export CUDA_VISIBLE_DEVICES="0,1,2,3"                                                                       
ADDR=`echo $PADDLE_TRAINERS | awk -F "," '{print $1}'`                                                                                           
NNODES=$PADDLE_TRAINERS_NUM
RANK=$PADDLE_TRAINER_ID                                                                                                                          
PORT=8932


PYTHON=/root/paddlejob/workspace/env_run/qjy_dataset_env/diffusion/bin/python

OMP_NUM_THREADS=1 $PYTHON -m torch.distributed.launch --nproc_per_node 4 \
    --nnodes=$NNODES \
    --node_rank=$RANK \
    --master_addr=$ADDR \
    --master_port=$PORT \
    --use_env train_moe.py \
    --pretrained_model_name_or_path=$MODEL_NAME \
    --load_model=$MODEL \
    --dataset_config=$DATASET_NAME \
    --output_dir=$OUTPUT_DIR \
    --output_name=$my_name \
    --prior_loss_weight=1.0 \
    --max_train_steps=50000 \
    --learning_rate=1e-5 \
    --network_dim_face=256 \
    --network_dim_hand=256 \
    --network_alpha_face=64 \
    --network_alpha_hand=128 \
    --face_network_weights="/path/to/facelora/weight" \
    --hand_network_weights="/path/to/hanlora/weight" \
    --xformers \
    --save_state \
    --optimizer_type="AdamW" \
    --mixed_precision="bf16" \
    --network_train_unet_only \
    --sample_every_n_steps=1000 \
    --network_module_lora=networks.lora \
    --network_module_moe=networks.moe \
    --sample_prompts='./prompts_example/human_prompts.txt' \
    --gradient_accumulation_steps=6 \
    --save_every_n_steps=1000 \