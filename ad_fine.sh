#!/bin/bash

# 设置路径变量
PRETRAIN_PATH="/path/to/sam2_pretrained.pth"  # SAM2预训练权重路径，请替换为实际路径
CONFIG_PATH="configs/SNI-SLAM.yaml"           # 配置文件路径
SAVE_DIR="checkpoints/sam2seg"                # 模型保存目录
LOG_DIR="logs/sam2seg"                        # 日志保存目录

# 创建必要的目录
mkdir -p $SAVE_DIR
mkdir -p $LOG_DIR

# 示例1: 只训练头部（默认策略）
python train_sam2seg.py \
  --config $CONFIG_PATH \
  --pretrain_path $PRETRAIN_PATH \
  --finetune_strategy head_only \
  --mode finetuning \
  --batch_size 8 \
  --epochs 20 \
  --lr 1e-4 \
  --save_dir "${SAVE_DIR}/head_only" \
  --log_dir "${LOG_DIR}/head_only" \
  --save_freq 5

# 示例2: 训练最后几层
python train_sam2seg.py \
  --config $CONFIG_PATH \
  --pretrain_path $PRETRAIN_PATH \
  --finetune_strategy last_layers \
  --mode finetuning \
  --batch_size 8 \
  --epochs 20 \
  --lr 5e-5 \
  --save_dir "${SAVE_DIR}/last_layers" \
  --log_dir "${LOG_DIR}/last_layers" \
  --save_freq 5

# 示例3: 只训练Neck部分
python train_sam2seg.py \
  --config $CONFIG_PATH \
  --pretrain_path $PRETRAIN_PATH \
  --finetune_strategy neck_only \
  --mode finetuning \
  --batch_size 8 \
  --epochs 20 \
  --lr 1e-4 \
  --save_dir "${SAVE_DIR}/neck_only" \
  --log_dir "${LOG_DIR}/neck_only" \
  --save_freq 5

# 示例4: 完全微调（所有参数）
python train_sam2seg.py \
  --config $CONFIG_PATH \
  --pretrain_path $PRETRAIN_PATH \
  --finetune_strategy full_finetune \
  --mode finetuning \
  --batch_size 4 \  # 由于训练更多参数，降低批次大小以节省显存
  --epochs 15 \
  --lr 1e-5 \  # 较低的学习率避免破坏预训练特征
  --save_dir "${SAVE_DIR}/full_finetune" \
  --log_dir "${LOG_DIR}/full_finetune" \
  --save_freq 3

# 示例5: 从之前的检查点恢复训练
python train_sam2seg.py \
  --config $CONFIG_PATH \
  --pretrain_path $PRETRAIN_PATH \
  --resume "${SAVE_DIR}/last_layers/sam2seg_epoch_10.pth" \ # 从第10轮保存的检查点恢复
  --finetune_strategy last_layers \
  --mode finetuning \
  --batch_size 8 \
  --epochs 30 \  # 继续训练到30轮
  --lr 1e-5 \    # 可以降低学习率
  --save_dir "${SAVE_DIR}/last_layers_continued" \
  --log_dir "${LOG_DIR}/last_layers_continued" \
  --save_freq 5

# 示例6: 以映射模式训练
python train_sam2seg.py \
  --config $CONFIG_PATH \
  --pretrain_path $PRETRAIN_PATH \
  --finetune_strategy last_layers \
  --mode mapping \     # 使用mapping模式
  --batch_size 8 \
  --epochs 20 \
  --lr 1e-4 \
  --save_dir "${SAVE_DIR}/mapping_mode" \
  --log_dir "${LOG_DIR}/mapping_mode" \
  --save_freq 5