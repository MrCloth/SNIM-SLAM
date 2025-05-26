import torch
import os
import sys
import argparse
import tqdm
from tqdm import tqdm
sys.path.append('/root/autodl-tmp/sni-slam')  # 确保路径正确
import torch.optim as optim
import torch.nn as nn
from src import config
from torch.utils.data import DataLoader
from src.datasets.dataset_load import SAM2SEGDataset  # 导入数据集类

# 加载配置文件
import yaml
parser = argparse.ArgumentParser(description='Arguments for running SNI_SLAM.')
args = parser.parse_args()

# 加载配置文件
cfg = config.load_config('configs/SNI-SLAM.yaml')

# 导入模型定义
from src.networks.sam2_seg2 import SAM2SEG

# 初始化数据集
dataset = SAM2SEGDataset(cfg)

# 可视化标签映射关系和样本
dataset.visualize_label_mapping('label_mapping.png')
dataset.visualize_sample_labels(num_samples=3, save_dir='./label_samples')

dataloader = DataLoader(dataset, batch_size=8, shuffle=True)
# 创建数据加载器

# 使用适当的损失函数，忽略255值（无效标签）

# 初始化模型
model = SAM2SEG(
    img_h=cfg['cam']['H'],
    img_w=cfg['cam']['W'],
    num_cls=cfg['model']['cnn']['n_classes'],  # 使用配置文件中的类别数
    edge=cfg['cam']['crop_edge'],  # 使用配置中的crop_edge
    dim=cfg['model']['c_dim'],  # 使用配置中的维度信息
    backbone='sam2.1_t',
    mode='finetuning',  # 设置训练模式
    finetune_strategy='last_layers'  # 微调最后一层
)

# 优化器和损失函数
optimizer = optim.Adam(model.parameters(), lr=1e-4)
criterion = nn.CrossEntropyLoss(ignore_index=255)

# 设备设置
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)
# 训练过程

num_epochs = 10
for epoch in range(num_epochs):
    model.train()
    running_loss = 0.0
    # 使用tqdm包装dataloader
    pbar = tqdm(dataloader, desc=f"Epoch {epoch+1}/{num_epochs}")
    for color_data, depth_data, poses, semantic_data in pbar:
        color_data, depth_data, semantic_data = color_data.to(device), depth_data.to(device), semantic_data.to(device)
        
        optimizer.zero_grad()  # 清空梯度
        
        # 前向传播
        outputs = model(color_data)  # 假设模型输出为类别的预测    
        # 计算损失
        loss = criterion(outputs, semantic_data)  # 计算交叉熵损失
        
        # 反向传播
        loss.backward()
        
        # 更新参数
        optimizer.step()
        
        running_loss += loss.item()
        # 更新进度条显示的损失值
        pbar.set_postfix({"loss": f"{loss.item():.4f}"})
    
    # 输出每个epoch的平均损失
    avg_loss = running_loss / len(dataloader)
    print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {avg_loss:.4f}")
    # 每隔一定时期保存模型
    if (epoch + 1) % 5 == 0:
        torch.save(model.state_dict(), f"sam2seg_epoch_{epoch+1}.pth")
