#!/bin/bash

# 设置路径变量
PRETRAIN_PATH="/root/autodl-tmp/sni-slam/seg/sam2/checkpoints/sam2.1_hiera_tiny.pt"  # SAM2预训练权重路径，请替换为实际路径
CONFIG_PATH="/root/autodl-tmp/sni-slam/configs/SNI-SLAM.yaml"           # 配置文件路径
SAVE_DIR="/root/autodl-tmp/sni-slam/sam2Train/checkpoints/sam2seg"                # 模型保存目录
LOG_DIR="/root/autodl-tmp/sni-slam/sam2Train/logs/sam2seg"                        # 日志保存目录

# 创建必要的目录
mkdir -p $SAVE_DIR
mkdir -p $LOG_DIR

# 示例5: 从之前的检查点恢复训练
python train_sam2seg.py \
  --config $CONFIG_PATH \
  --pretrain_path $PRETRAIN_PATH \
  --mode finetuning \
  --batch_size 8 \
  --epochs 30 \
  --finetune_strategy last_layers \
  --lr 1e-5 \
  --save_dir "${SAVE_DIR}/last_layers_continued" \
  --log_dir "${LOG_DIR}/last_layers_continued" \
  --save_freq 5 \
  --resume "${SAVE_DIR}/last_layers/sam2seg_epoch_20.pth" 