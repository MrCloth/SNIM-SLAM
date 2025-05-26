import torch
import os
import sys
import argparse
import tqdm
from tqdm import tqdm
import yaml
from datetime import datetime
import torch.optim as optim
import torch.nn as nn
from torch.utils.data import DataLoader
import logging
sys.path.append('/root/autodl-tmp/sni-slam')  # 确保路径正确

# 设置日志
def setup_logger(log_dir):
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)
    
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    log_file = os.path.join(log_dir, f'training_{timestamp}.log')
    
    # 配置日志
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler()
        ]
    )
    return logging.getLogger()

def load_pretrained_weights(model, pretrained_path, device, finetune_strategy='head_only'):
    """
    加载预训练权重到模型
    
    参数:
        model: SAM2SEG模型实例
        pretrained_path: 预训练权重文件路径
        device: 设备 (cpu/cuda)
        finetune_strategy: 微调策略
    
    返回:
        model: 加载权重后的模型
    """
    logger.info(f"加载SAM2预训练权重: {pretrained_path}")
    checkpoint = torch.load(pretrained_path, map_location=device)
    
    # 提取"model"键下的权重字典
    if "model" in checkpoint:
        state_dict = checkpoint["model"]
    else:
        state_dict = checkpoint
    
    # 创建新的权重字典，只保留和映射需要的部分
    new_state_dict = {}
    
    # 记录处理的键名数量
    trunk_keys = 0
    neck_keys = 0
    other_keys = 0
    
    # 执行有针对性的键名映射 - 只提取backbone所需部分
    for key, value in state_dict.items():
        if key.startswith('image_encoder.trunk'):
            new_key = key.replace('image_encoder.trunk', 'backbone.trunk')
            new_state_dict[new_key] = value
            trunk_keys += 1
        elif key.startswith('image_encoder.neck'):
            new_key = key.replace('image_encoder.neck', 'backbone.neck')
            new_state_dict[new_key] = value
            neck_keys += 1
        else:
            # 忽略其他不相关的键
            other_keys += 1
    
    # 使用非严格模式加载权重
    missing_keys, unexpected_keys = model.load_state_dict(new_state_dict, strict=False)
    
    logger.info(f"预训练模型处理情况:")
    logger.info(f"- trunk部分键: {trunk_keys}个")
    logger.info(f"- neck部分键: {neck_keys}个")
    logger.info(f"- 忽略的键: {other_keys}个")
    logger.info(f"- 模型中缺失的键: {len(missing_keys)}个")
    logger.info(f"- 预训练中多余的键: {len(unexpected_keys)}个")
    
    # 手动初始化缺失的参数
    logger.info("正在特定初始化缺失的参数...")
    
    # 获取模型当前状态字典
    model_state_dict = model.state_dict()
    
    # 初始化segmentation_head参数
    # segmentation_head.1 - 第一个卷积层，后面跟BN和ReLU
    w_key = 'segmentation_head.1.weight'
    b_key = 'segmentation_head.1.bias'
    if w_key in missing_keys:
        # 使用He初始化，适合ReLU激活函数
        nn.init.kaiming_normal_(model_state_dict[w_key], mode='fan_out', nonlinearity='relu')
        logger.info(f"使用He初始化 {w_key}")
    if b_key in missing_keys:
        nn.init.zeros_(model_state_dict[b_key])
        logger.info(f"初始化 {b_key} 为零")
    
    # segmentation_head.2 - BN层
    w_key = 'segmentation_head.2.weight'
    b_key = 'segmentation_head.2.bias'
    mean_key = 'segmentation_head.2.running_mean'
    var_key = 'segmentation_head.2.running_var'
    
    if w_key in missing_keys:
        # BN权重初始化为1
        nn.init.ones_(model_state_dict[w_key])
        logger.info(f"初始化 {w_key} 为1")
    if b_key in missing_keys:
        nn.init.zeros_(model_state_dict[b_key])
        logger.info(f"初始化 {b_key} 为零")
    if mean_key in missing_keys:
        nn.init.zeros_(model_state_dict[mean_key])
        logger.info(f"初始化 {mean_key} 为零")
    if var_key in missing_keys:
        nn.init.ones_(model_state_dict[var_key])
        logger.info(f"初始化 {var_key} 为1")
    
    # segmentation_head.5 - 最终分类卷积层
    w_key = 'segmentation_head.5.weight'
    b_key = 'segmentation_head.5.bias'
    if w_key in missing_keys:
        # 最后一层使用Xavier初始化，适合输出层
        nn.init.xavier_uniform_(model_state_dict[w_key])
        logger.info(f"使用Xavier初始化 {w_key}")
    if b_key in missing_keys:
        # 偏置初始化为0
        nn.init.zeros_(model_state_dict[b_key])
        logger.info(f"初始化 {b_key} 为零")
    
    # 将初始化后的参数加载回模型
    model.load_state_dict(model_state_dict)
    
    logger.info("缺失参数特定初始化完成")
    
    # 应用指定的微调策略
    model._apply_finetune_strategy()
    
    return model

def load_finetuned_model(model, checkpoint_path, device):
    """加载已微调的模型权重"""
    logger.info(f"加载微调后的SAM2权重: {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location=device)
    
    # 处理checkpoint可能是字典的情况
    if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'])
    else:
        model.load_state_dict(checkpoint)
        
    logger.info("微调模型加载成功")
    return model

def save_checkpoint(model, optimizer, epoch, loss, save_dir, filename=None):
    """保存模型检查点"""
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    
    if filename is None:
        filename = f'sam2seg_epoch_{epoch+1}.pth'
    
    filepath = os.path.join(save_dir, filename)
    
    torch.save({
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'loss': loss,
    }, filepath)
    
    logger.info(f"模型检查点已保存至: {filepath}")


def main():
    parser = argparse.ArgumentParser(description='SAM2SEG模型训练')
    parser.add_argument('--config', type=str, default='configs/SNI-SLAM.yaml', help='配置文件路径')
    parser.add_argument('--pretrain_path', type=str, required=True, help='SAM2预训练权重路径')
    parser.add_argument('--resume', type=str, default=None, help='恢复训练的检查点路径')
    parser.add_argument('--finetune_strategy', type=str, default='head_only', 
                        choices=['head_only', 'last_layers', 'neck_only', 'full_finetune'],
                        help='微调策略')
    parser.add_argument('--mode', type=str, default='finetuning', 
                       choices=['mapping', 'finetuning', 'inference'],
                       help='模型运行模式')
    parser.add_argument('--batch_size', type=int, default=8, help='批次大小')
    parser.add_argument('--epochs', type=int, default=10, help='训练轮数')
    parser.add_argument('--lr', type=float, default=1e-4, help='学习率')
    parser.add_argument('--save_dir', type=str, default='checkpoints', help='模型保存目录')
    parser.add_argument('--log_dir', type=str, default='logs', help='日志保存目录')
    parser.add_argument('--save_freq', type=int, default=5, help='模型保存频率（轮数）')
    
    args = parser.parse_args()
    
    # 设置日志
    global logger
    logger = setup_logger(args.log_dir)
    
    # 记录训练参数
    logger.info("========== 训练参数 ==========")
    for arg, value in vars(args).items():
        logger.info(f"{arg}: {value}")
    logger.info("=============================")

    # 加载配置文件
    from src import config
    cfg = config.load_config(args.config)
    
    # 导入模型和数据集
    from src.networks.sam2_seg2 import SAM2SEG
    from src.datasets.dataset_load import SAM2SEGDataset
    
    # 初始化数据集
    dataset = SAM2SEGDataset(cfg)
    logger.info(f"数据集初始化完成，样本数: {len(dataset)}")
    
    # 可视化标签映射关系和样本
    dataset.visualize_label_mapping('label_mapping.png')
    dataset.visualize_sample_labels(num_samples=3, save_dir='./label_samples')
    logger.info("标签映射和样本可视化已完成")
    
    # 创建数据加载器
    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True, num_workers=4)
    logger.info(f"数据加载器创建完成，批次大小: {args.batch_size}")
    
    # 设备设置
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"使用设备: {device}")
    
    # 初始化模型
    model = SAM2SEG(
        img_h=cfg['cam']['H'],
        img_w=cfg['cam']['W'],
        num_cls=cfg['model']['cnn']['n_classes'],  # 使用配置文件中的类别数
        edge=cfg['cam']['crop_edge'],  # 使用配置中的crop_edge
        dim=cfg['model']['c_dim'],  # 使用配置中的维度信息
        backbone='sam2.1_t',
        mode=args.mode,  # 使用传入的模式
        finetune_strategy=args.finetune_strategy  # 使用传入的微调策略
    )
    
    # 加载模型权重
    if args.resume:
        # 恢复之前的训练
        model = load_finetuned_model(model, args.resume, device)
        # 应用微调策略（确保在加载后重新应用）
        model._apply_finetune_strategy() 
    else:
        # 从预训练权重开始
        model = load_pretrained_weights(model, args.pretrain_path, device, args.finetune_strategy)
    
    # 将模型放到指定设备
    model = model.to(device)
    
    # # 打印模型架构
    # logger.info(f"模型架构:\n{model}")
    
    # 优化器和损失函数
    optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=args.lr)
    criterion = nn.CrossEntropyLoss(ignore_index=255)
    
    # 如果恢复训练，加载优化器状态
    start_epoch = 0
    if args.resume and os.path.isfile(args.resume):
        checkpoint = torch.load(args.resume, map_location=device)
        if 'optimizer_state_dict' in checkpoint:
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        if 'epoch' in checkpoint:
            start_epoch = checkpoint['epoch'] + 1
            logger.info(f"从epoch {start_epoch}恢复训练")
    
    # 训练过程
    best_loss = float('inf')
    
    logger.info("开始训练...")
    for epoch in range(start_epoch, args.epochs):
        model.train()
        running_loss = 0.0
        
        # 使用tqdm包装dataloader
        pbar = tqdm(dataloader, desc=f"Epoch {epoch+1}/{args.epochs}")
        for batch_idx, (color_data, depth_data, poses, semantic_data) in enumerate(pbar):
            color_data = color_data.to(device)
            semantic_data = semantic_data.to(device)
            
            optimizer.zero_grad()  # 清空梯度
            
            # 前向传播
            outputs = model(color_data)
            
            # 计算损失
            loss = criterion(outputs, semantic_data)
            
            # 反向传播
            loss.backward()
            
            # 更新参数
            optimizer.step()
            
            running_loss += loss.item()
            
            # 更新进度条显示的损失值
            pbar.set_postfix({"loss": f"{loss.item():.4f}"})
            
            # 每100个批次记录一次详细信息
            if batch_idx % 100 == 0:
                logger.info(f"Epoch [{epoch+1}/{args.epochs}], Batch [{batch_idx}/{len(dataloader)}], Loss: {loss.item():.4f}")
        
        # 计算平均损失
        avg_loss = running_loss / len(dataloader)
        logger.info(f"Epoch [{epoch+1}/{args.epochs}], Average Loss: {avg_loss:.4f}")
        
        # 每隔指定轮数保存模型
        if (epoch + 1) % args.save_freq == 0:
            save_checkpoint(model, optimizer, epoch, avg_loss, args.save_dir)
        
        # 保存最佳模型
        if avg_loss < best_loss:
            best_loss = avg_loss
            save_checkpoint(model, optimizer, epoch, avg_loss, args.save_dir, "sam2seg_best.pth")
            logger.info(f"保存最佳模型，损失: {best_loss:.4f}")
    
    # 保存最终模型
    save_checkpoint(model, optimizer, args.epochs - 1, avg_loss, args.save_dir, "sam2seg_final.pth")
    logger.info("训练完成！")


if __name__ == "__main__":
    main()