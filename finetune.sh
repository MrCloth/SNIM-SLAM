#!/bin/bash

# 设置变量
REPLICA_ROOT="/root/autodl-tmp/sni-slam/replica"  # Replica数据集路径
OUTPUT_DIR="/root/autodl-tmp/sni-slam/replica_processed"  # 处理后数据路径
CHECKPOINT_PATH="/root/autodl-tmp/sni-slam/seg/sam2/checkpoints/sam2.1_hiera_tiny.pt"  # SAM2预训练权重
FINETUNE_DIR="/root/autodl-tmp/sni-slam/seg/sam2/finetune"  # 微调模型保存路径

# 第1步：准备数据
echo "=== 准备Replica数据集 ==="
python prepare_replica_data.py \
    --replica_root $REPLICA_ROOT \
    --output_dir $OUTPUT_DIR \
    --views_per_scene 500 \
    --img_height 680 \
    --img_width 1200

# 第2步：执行微调
echo "=== 开始微调SAM2SEG模型 ==="
python finetune_sam2seg_replica.py \
    --data_root $OUTPUT_DIR \
    --checkpoint_path $CHECKPOINT_PATH \
    --output_dir $FINETUNE_DIR \
    --batch_size 8 \
    --num_epochs 30 \
    --lr 5e-4 \
    --finetune_strategy last_layers \
    --img_h 680 \
    --img_w 1200 \
    --edge 10 \
    --views_per_scene 500

echo "微调完成，模型保存在 $FINETUNE_DIR"

