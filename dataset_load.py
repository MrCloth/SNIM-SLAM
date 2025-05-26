import glob
import cv2
import torch
import numpy as np
import sys
import pickle
import os
sys.path.append('/root/autodl-tmp/sni-slam')
from torch.utils.data import Dataset
import torch.nn.functional as F
import matplotlib.pyplot as plt
from collections import Counter
import pandas as pd
from tqdm import tqdm

class SAM2SEGDataset(Dataset):
    def __init__(self, cfg, device='cuda:0'):
        super(SAM2SEGDataset, self).__init__()
        self.device = device
        self.input_folder = cfg['data']['input_folder']
        
        # 文件路径
        self.color_paths = sorted(glob.glob(f'{self.input_folder}/rgb/rgb_*.png'))
        self.depth_paths = sorted(glob.glob(f'{self.input_folder}/depth/depth_*.png'))
        self.semantic_paths = sorted(glob.glob(f'{self.input_folder}/semantic_class/semantic_class_*.png'))
        
        self.n_img = len(self.color_paths)
        
        # 相机参数等
        self.H, self.W = cfg['cam']['H'], cfg['cam']['W']
        self.fx, self.fy, self.cx, self.cy = cfg['cam']['fx'], cfg['cam']['fy'], cfg['cam']['cx'], cfg['cam']['cy']
        
        self.crop_edge = cfg['cam']['crop_edge']
        self.crop_size = cfg['cam'].get('crop_size', None)  # 如果没有提供，则为None
        self.load_poses(f'{self.input_folder}/traj.txt')
        
        # 加载语义类别映射
        self.ignore_label = 255  # 用于表示忽略的标签值
        self.load_semantic_classes()
        
        # 打印标签映射信息
        print(f"已加载语义标签映射: 将 {len(self.label_map)} 个原始标签映射到 0-{len(self.label_map)-1}")

    def __len__(self):
        return self.n_img

    def __getitem__(self, idx):
        color_path = self.color_paths[idx]
        depth_path = self.depth_paths[idx]
        
        # 加载数据
        color_data = cv2.imread(color_path)
        color_data = cv2.cvtColor(color_data, cv2.COLOR_BGR2RGB) / 255.0  # RGB归一化到[0, 1]
        
        depth_data = cv2.imread(depth_path, cv2.IMREAD_UNCHANGED) / 1000.0  # 将深度图进行归一化
        
        # 加载语义标签数据
        semantic_data = cv2.imread(self.semantic_paths[idx], cv2.IMREAD_UNCHANGED)
        
        # 根据配置进行图像裁剪、缩放等操作
        if self.crop_size:
            color_data = cv2.resize(color_data, (self.crop_size, self.crop_size))
            depth_data = cv2.resize(depth_data, (self.crop_size, self.crop_size))
            semantic_data = cv2.resize(semantic_data, (self.crop_size, self.crop_size), interpolation=cv2.INTER_NEAREST)
        
        # 将原始标签映射到连续索引
        semantic_data = self.map_semantic_labels(semantic_data)
        
        color_data = torch.from_numpy(color_data).float().permute(2, 0, 1)  # 转为Tensor并调整通道顺序
        depth_data = torch.from_numpy(depth_data).float()  # 深度图为单通道
        semantic_data = torch.from_numpy(semantic_data).long()  # 标签转为long类型
        
        # 裁剪边界
        if self.crop_edge > 0:
            color_data = color_data[:, self.crop_edge:-self.crop_edge, self.crop_edge:-self.crop_edge]
            depth_data = depth_data[self.crop_edge:-self.crop_edge, self.crop_edge:-self.crop_edge]
            semantic_data = semantic_data[self.crop_edge:-self.crop_edge, self.crop_edge:-self.crop_edge]
        
        pose = self.poses[idx]  # 位姿
        
        return color_data, depth_data, pose, semantic_data
    
    def load_poses(self, path):
        self.poses = []
        with open(path, 'r') as f:
            lines = f.readlines()
        
        for line in lines:
            c2w = np.array(list(map(float, line.split()))).reshape(4, 4)
            c2w = torch.from_numpy(c2w).float()
            self.poses.append(c2w)
    
    def load_semantic_classes(self):
        """加载语义类别映射"""
        try:
            # 尝试加载semantic_classes.pkl
            semantic_classes_path = '/root/autodl-tmp/sni-slam/seg/semantic_classes.pkl'
            
            if not os.path.exists(semantic_classes_path):
                # 尝试其他可能的位置
                alt_paths = [
                    os.path.join(self.input_folder, 'semantic_classes.pkl'),
                    os.path.join(os.path.dirname(self.input_folder), 'semantic_classes.pkl'),
                    os.path.join(os.path.dirname(os.path.dirname(self.input_folder)), 'semantic_classes.pkl')
                ]
                
                for path in alt_paths:
                    if os.path.exists(path):
                        semantic_classes_path = path
                        break
            
            if not os.path.exists(semantic_classes_path):
                print(f"警告: 找不到语义类别文件 {semantic_classes_path}")
                self.label_map = None
                return
            
            # 加载语义类别
            with open(semantic_classes_path, 'rb') as f:
                semantic_classes = pickle.load(f)
                
            # 创建映射字典: 原始标签ID -> 连续索引
            self.label_map = {int(label): idx for idx, label in enumerate(semantic_classes)}
            
            # 输出映射信息
            print(f"从 {semantic_classes_path} 加载了语义类别映射")
            print(f"原始标签 -> 映射索引:")
            for orig, new in list(self.label_map.items())[:5]:
                print(f"  {orig} -> {new}")
            print("  ...")
            for orig, new in list(self.label_map.items())[-5:]:
                print(f"  {orig} -> {new}")
                
        except Exception as e:
            print(f"加载语义类别映射时出错: {e}")
            self.label_map = None
    
    def map_semantic_labels(self, semantic_data):
        """将原始标签映射到连续索引"""
        if self.label_map is None:
            return semantic_data
        
        # 创建输出数组
        mapped_labels = np.ones_like(semantic_data) * self.ignore_label
        
        # 遍历映射字典，将所有匹配的原始标签替换为新标签
        for orig_label, new_label in self.label_map.items():
            mask = (semantic_data == orig_label)
            mapped_labels[mask] = new_label
        
        return mapped_labels
    
    def visualize_label_mapping(self, save_path='label_mapping.png'):
        """可视化标签映射关系"""
        if self.label_map is None:
            print("警告: 未加载语义类别映射，无法可视化")
            return
        
        # 提取映射数据
        orig_labels = list(self.label_map.keys())
        mapped_labels = list(self.label_map.values())
        
        # 创建散点图
        plt.figure(figsize=(10, 6))
        plt.scatter(orig_labels, mapped_labels, alpha=0.7)
        plt.xlabel('original ID')
        plt.ylabel('Mapped ID(0-51)')
        plt.title('Semantic Label Mapping')
        plt.grid(True, linestyle='--', alpha=0.7)
        
        # 添加最大值线
        plt.axhline(y=max(mapped_labels), color='r', linestyle='--', alpha=0.5, label=f'max mapped labels: {max(mapped_labels)}')
        
        plt.legend()
        plt.tight_layout()
        plt.savefig(save_path)
        plt.close()
        print(f"标签映射可视化已保存到: {save_path}")
    
    def visualize_sample_labels(self, indices=None, num_samples=4, save_dir='./label_samples'):
        """可视化几个样本的原始标签和映射后的标签"""
        os.makedirs(save_dir, exist_ok=True)
        
        if indices is None:
            # 随机选择样本
            indices = np.random.choice(self.n_img, num_samples, replace=False)
        
        # 确保索引在范围内且数量正确
        indices = [i for i in indices if 0 <= i < self.n_img][:num_samples]
        actual_num_samples = len(indices)
        
        # 创建一个2行、样本数列的图表
        fig, axes = plt.subplots(actual_num_samples, 3, figsize=(15, 5*actual_num_samples))
        if actual_num_samples == 1:
            axes = [axes]
            
        for i, idx in enumerate(indices):
            # 加载原始图像和标签
            color_path = self.color_paths[idx]
            semantic_path = self.semantic_paths[idx]
            
            color_data = cv2.imread(color_path)
            color_data = cv2.cvtColor(color_data, cv2.COLOR_BGR2RGB) / 255.0
            orig_semantic = cv2.imread(semantic_path, cv2.IMREAD_UNCHANGED)
            
            # 应用与__getitem__相同的预处理
            if self.crop_size:
                color_data = cv2.resize(color_data, (self.crop_size, self.crop_size))
                orig_semantic = cv2.resize(orig_semantic, (self.crop_size, self.crop_size), 
                                           interpolation=cv2.INTER_NEAREST)
                
            # 获取映射后的标签
            mapped_semantic = self.map_semantic_labels(orig_semantic)
            
            # 获取标签统计信息
            orig_unique = np.unique(orig_semantic)
            mapped_unique = np.unique(mapped_semantic)
            orig_min, orig_max = orig_semantic.min(), orig_semantic.max()
            mapped_min, mapped_max = mapped_semantic.min(), mapped_semantic.max()
            
            # 显示RGB图像
            axes[i][0].imshow(color_data)
            axes[i][0].set_title(f"sample {idx} - RGB image")
            axes[i][0].axis('off')
            
            # 显示原始标签
            im1 = axes[i][1].imshow(orig_semantic, cmap='tab20', interpolation='nearest')
            axes[i][1].set_title(f"original label (range: {orig_min}-{orig_max})")
            axes[i][1].axis('off')
            fig.colorbar(im1, ax=axes[i][1])
            
            # 显示映射后的标签
            im2 = axes[i][2].imshow(mapped_semantic, cmap='tab20', interpolation='nearest')
            axes[i][2].set_title(f"mapped label (range: {mapped_min}-{mapped_max})")
            axes[i][2].axis('off')
            fig.colorbar(im2, ax=axes[i][2])
            
            # 打印标签信息
            print(f"样本 {idx} 标签统计:")
            print(f"  原始标签范围: {orig_min}-{orig_max}, 唯一值: {len(orig_unique)}")
            print(f"  映射标签范围: {mapped_min}-{mapped_max}, 唯一值: {len(mapped_unique)}")
            if self.ignore_label in mapped_unique:
                ignore_percent = np.sum(mapped_semantic == self.ignore_label) / mapped_semantic.size * 100
                print(f"  忽略标签占比: {ignore_percent:.2f}%")
            
        plt.tight_layout()
        plt.savefig(f'{save_dir}/sample_labels_comparison.png')
        plt.close()
        print(f"样本标签对比可视化已保存到: {save_dir}/sample_labels_comparison.png")
        
        return indices
