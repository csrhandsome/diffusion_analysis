export NCCL_IB_HCA=mlx5_0:1,mlx5_1:1,mlx5_2:1,mlx5_3:1,mlx5_4:1,mlx5_7:1,mlx5_8:1,mlx5_9:1
export NCCL_IB_DISABLE=0
# 添加过NCCL的通信端口
export NCCL_SOCKET_IFNAME=enp3s0
export NCCL_DEBUG=INFO
export NCCL_NVLS_ENABLE=0

export TEXT_ENCODER_NAME="google/t5-v1_1-base"
export VISION_ENCODER_NAME="google/siglip-base-patch16-224"
export OUTPUT_DIR="./checkpoints/rdt-pretrain-1b-nodeepspeed"
export CFLAGS="-I/usr/include"
export LDFLAGS="-L/usr/lib/x86_64-linux-gnu"
export CUTLASS_PATH="${HOME}/cutlass"

export WANDB_PROJECT="rdt_train_nodeepseed"

if [ ! -d "$OUTPUT_DIR" ]; then
    mkdir "$OUTPUT_DIR"
    echo "Folder '$OUTPUT_DIR' created"
else
    echo "Folder '$OUTPUT_DIR' already exists"
fi

# For run in a mutiple node/machine
# deepspeed --hostfile=hostfile.txt main.py \
#     --deepspeed="./configs/zero2.json" \

#   这个是single node
# - 仍然使用了 DeepSpeed 的优化功能
# - 包括 ZeRO 优化器、混合精度训练等
# - 但不是用于多机分布式训练
# - 主要用于优化单机训练的性能和内存使用
#   修改过batch size,max_train_steps
accelerate launch main.py \
    --deepspeed="./configs/zero2.json" \
    --pretrained_text_encoder_name_or_path=$TEXT_ENCODER_NAME \
    --pretrained_vision_encoder_name_or_path=$VISION_ENCODER_NAME \
    --output_dir=$OUTPUT_DIR \
    --train_batch_size=4 \
    --sample_batch_size=8 \
    --max_train_steps=10000 \
    --checkpointing_period=1000 \
    --sample_period=500 \
    --checkpoints_total_limit=40 \
    --lr_scheduler="constant" \
    --learning_rate=1e-4 \
    --mixed_precision="bf16" \
    --dataloader_num_workers=8 \
    --dataset_type="pretrain" \
    --report_to=wandb \
    --load_from_hdf5
    # --load_from_hdf5 被定义为一个布尔类型的标志参数（action="store_true"），这意味着它的值是由是否存在该参数来决定的，而不是通过 =True 或 =False 来显式指定的
    # Use this to resume training from some previous checkpoint
    # --resume_from_checkpoint="checkpoint-1000" \
