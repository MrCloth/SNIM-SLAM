import torch
import os
import sys
import argparse
import yaml
import numpy as np
from tqdm import tqdm
import logging
from datetime import datetime
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
import seaborn as sns
sys.path.append('/root/autodl-tmp/sni-slam')  # 确保路径正确

def setup_logger(log_dir):
    """设置日志记录器"""
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)
    
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    log_file = os.path.join(log_dir, f'evaluation_{timestamp}.log')
    
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

def compute_miou(pred, target, num_classes, ignore_index=255):
    """
    计算平均交并比(mIoU)
    
    参数:
        pred: 预测结果 (N, H, W)
        target: 真实标签 (N, H, W)
        num_classes: 类别数量
        ignore_index: 忽略的标签索引
        
    返回:
        miou: 平均交并比
        class_iou: 每个类别的IoU
    """
    # 扁平化张量
    mask = (target != ignore_index)
    pred_flat = pred[mask].cpu().numpy()
    target_flat = target[mask].cpu().numpy()
    
    # 计算混淆矩阵
    conf_matrix = confusion_matrix(
        target_flat, 
        pred_flat, 
        labels=list(range(num_classes))
    )
    
    # 计算每个类别的IoU
    class_iou = np.zeros(num_classes)
    for class_idx in range(num_classes):
        # IoU = TP / (TP + FP + FN)
        # TP = 对角元素
        tp = conf_matrix[class_idx, class_idx]
        # TP + FP = 列和 (预测为该类的元素总数)
        tp_fp = np.sum(conf_matrix[:, class_idx])
        # TP + FN = 行和 (真实为该类的元素总数)
        tp_fn = np.sum(conf_matrix[class_idx, :])
        
        # 计算IoU，避免除零错误
        if (tp_fp + tp_fn - tp) == 0:
            class_iou[class_idx] = 0
        else:
            class_iou[class_idx] = tp / (tp_fp + tp_fn - tp)
    
    # 计算有效类别的平均IoU (跳过计算值为零的类别，这些可能是数据集中不存在的类)
    valid_classes = np.where(np.sum(conf_matrix, axis=1) > 0)[0]
    if len(valid_classes) == 0:
        miou = 0
    else:
        miou = np.mean(class_iou[valid_classes])
    
    # 返回mIoU和每个类别的IoU
    return miou, class_iou, conf_matrix

def visualize_confusion_matrix(conf_matrix, class_names, save_path):
    """可视化混淆矩阵"""
    plt.figure(figsize=(12, 10))
    sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', 
                xticklabels=class_names, yticklabels=class_names)
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title('Confusion Matrix')
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()
    logger.info(f"混淆矩阵已保存至: {save_path}")

def visualize_iou_per_class(class_iou, class_names, save_path):
    """可视化每个类别的IoU"""
    plt.figure(figsize=(12, 8))
    bars = plt.bar(class_names, class_iou)
    
    # 在柱状图上添加数值标签
    for bar, iou in zip(bars, class_iou):
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height,
                 f'{iou:.3f}', ha='center', va='bottom', rotation=0)
    
    plt.xlabel('Classes')
    plt.ylabel('IoU')
    plt.title('IoU per Class')
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()
    logger.info(f"类别IoU图表已保存至: {save_path}")

def save_evaluation_results(miou, class_iou, class_names, save_path):
    """保存评估结果到文本文件"""
    with open(save_path, 'w') as f:
        f.write(f"平均交并比 (mIoU): {miou:.4f}\n\n")
        f.write("每个类别的IoU:\n")
        for i, (name, iou) in enumerate(zip(class_names, class_iou)):
            f.write(f"{i}: {name} - {iou:.4f}\n")
    
    logger.info(f"评估结果已保存至: {save_path}")

def evaluate_model(model, dataloader, device, num_classes, output_dir, class_names):
    """评估模型并计算mIoU"""
    model.eval()
    
    all_preds = []
    all_targets = []
    
    logger.info("开始评估模型...")
    
    with torch.no_grad():
        for batch_idx, (color_data, depth_data, poses, semantic_data) in enumerate(tqdm(dataloader, desc="Evaluating")):
            color_data = color_data.to(device)
            semantic_data = semantic_data.to(device)
            
            # 前向传播
            outputs = model(color_data)
            
            # 获取预测结果 (B, C, H, W) -> (B, H, W)
            preds = torch.argmax(outputs, dim=1)
            
            # 收集预测和目标
            all_preds.append(preds)
            all_targets.append(semantic_data)
    
    # 拼接所有批次的数据
    all_preds = torch.cat(all_preds, dim=0)
    all_targets = torch.cat(all_targets, dim=0)
    
    # 计算mIoU
    miou, class_iou, conf_matrix = compute_miou(all_preds, all_targets, num_classes)
    
    # 确保输出目录存在
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    # 可视化每个类别的IoU
    visualize_iou_per_class(
        class_iou, 
        class_names, 
        os.path.join(output_dir, 'iou_per_class.png')
    )
    
    # 可视化混淆矩阵
    visualize_confusion_matrix(
        conf_matrix, 
        class_names, 
        os.path.join(output_dir, 'confusion_matrix.png')
    )
    
    # 保存评估结果
    save_evaluation_results(
        miou, 
        class_iou, 
        class_names, 
        os.path.join(output_dir, 'evaluation_results.txt')
    )
    
    logger.info(f"模型评估完成，mIoU: {miou:.4f}")
    
    # 打印每个类别的IoU
    logger.info("每个类别的IoU:")
    for i, (name, iou) in enumerate(zip(class_names, class_iou)):
        logger.info(f"  {i}: {name} - {iou:.4f}")
    
    return miou, class_iou

def main():
    parser = argparse.ArgumentParser(description='SAM2SEG模型评估 - mIoU计算')
    parser.add_argument('--config', type=str, default='/root/autodl-tmp/sni-slam/configs/SNI-SLAM.yaml', help='配置文件路径')
    parser.add_argument('--checkpoint', type=str, required=True, help='模型检查点路径')
    parser.add_argument('--batch_size', type=int, default=8, help='批次大小')
    parser.add_argument('--output_dir', type=str, default='evaluation_results', help='评估结果保存目录')
    parser.add_argument('--log_dir', type=str, default='logs', help='日志保存目录')
    parser.add_argument('--split', type=str, default='val', choices=['train', 'val', 'test'], help='评估数据集分割')
    
    args = parser.parse_args()
    
    # 设置日志
    global logger
    logger = setup_logger(args.log_dir)
    
    # 记录评估参数
    logger.info("========== 评估参数 ==========")
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
    # 设置评估模式，以便数据集可以相应地加载验证/测试数据
    dataset = SAM2SEGDataset(cfg, split=args.split)
    logger.info(f"{args.split}数据集初始化完成，样本数: {len(dataset)}")
    
    # 获取类别名称
    class_names = dataset.get_class_names()
    num_classes = len(class_names)
    logger.info(f"类别数量: {num_classes}")
    logger.info(f"类别名称: {class_names}")
    
    # 创建数据加载器
    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, num_workers=4)
    logger.info(f"数据加载器创建完成，批次大小: {args.batch_size}")
    
    # 设备设置
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"使用设备: {device}")
    
    # 初始化模型
    model = SAM2SEG(
        img_h=cfg['cam']['H'],
        img_w=cfg['cam']['W'],
        num_cls=cfg['model']['cnn']['n_classes'],
        edge=cfg['cam']['crop_edge'],
        dim=cfg['model']['c_dim'],
        backbone='sam2.1_t',
        mode='inference'  # 评估时使用推理模式
    )
    
    # 加载模型权重
    logger.info(f"加载模型检查点: {args.checkpoint}")
    checkpoint = torch.load(args.checkpoint, map_location=device)
    
    # 处理checkpoint可能是字典的情况
    if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'])
    else:
        model.load_state_dict(checkpoint)
    
    # 将模型放到指定设备
    model = model.to(device)
    model.eval()
    
    # 评估模型
    miou, class_iou = evaluate_model(
        model, 
        dataloader, 
        device, 
        num_classes, 
        args.output_dir,
        class_names
    )
    
    logger.info(f"评估完成! 最终mIoU: {miou:.4f}")
    logger.info(f"详细评估结果已保存至: {args.output_dir}")


if __name__ == "__main__":
    main()