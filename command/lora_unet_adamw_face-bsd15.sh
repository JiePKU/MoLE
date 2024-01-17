
tmp_my_name=${0##*/}
my_name=${tmp_my_name%.*}
echo $my_name

output_dir=$my_name


export MODEL="./sd_output/fine_sd_xformer_lion_300k_ac4_use_all/fine_sd_xformer_lion_300k_ac4_use_all"
export MODEL_NAME="./sd_output/fine_sd_xformer_lion_300k_ac4_use_all/fine_sd_xformer_lion_300k_ac4_use_all"
export DATASET_NAME="./data-config/face.toml"
export OUTPUT_DIR="./sd_output/$my_name"

export CUDA_VISIBLE_DEVICES="0,1,2,3"                                                                       
ADDR=`echo $TRAINERS | awk -F "," '{print $1}'`                                                                                           
NNODES=$TRAINERS_NUM
RANK=$TRAINER_ID                                                                                                                          
PORT=8940


PYTHON=/env_run/diffusion/bin/python

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
    --max_train_steps=30000 \
    --gradient_accumulation_steps=4 \
    --learning_rate=2e-5 \
    --network_dim=256 \
    --network_alpha=64 \
    --xformer \
    --save_state \
    --optimizer_type="AdamW" \
    --mixed_precision="bf16" \
    --network_train_unet_only \
    --save_every_n_steps=500 \
    --sample_every_n_steps=500 \
    --network_module=networks.lora \
    --sample_prompts='./prompts_example/face_prompts.txt' \