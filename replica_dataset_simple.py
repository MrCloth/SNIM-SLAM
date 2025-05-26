import os
import json
import glob
import torch
import numpy as np
from torch.utils.data import Dataset
from PIL import Image
import torchvision.transforms as transforms
import random

class ReplicaDatasetSimple(Dataset):
    def __init__(self, root, split='train', img_height=680, img_width=1200, transform=None):
        """
        简化版Replica数据集加载器，假设已经有渲染好的图像和分割标签
        
        参数:
            root: 数据目录，包含渲染好的图像和分割标签
            split: 'train'或'val'分割
            img_height: 输出图像高度
            img_width: 输出图像宽度
            transform: 图像转换
        """
        self.root = root
        self.split = split
        self.img_height = img_height
        self.img_width = img_width
        
        # 假设已经有预先渲染好的数据
        # 目录结构: root/images 和 root/labels
        self.images_dir = os.path.join(root, "images")
        self.labels_dir = os.path.join(root, "labels")
        
        assert os.path.exists(self.images_dir), f"图像目录不存在: {self.images_dir}"
        assert os.path.exists(self.labels_dir), f"标签目录不存在: {self.labels_dir}"
        
        # 获取所有图像文件
        self.image_files = sorted(glob.glob(os.path.join(self.images_dir, "*.jpg")) + 
                                 glob.glob(os.path.join(self.images_dir, "*.png")))
        
        # 将数据集分为训练集和验证集
        random.seed(42)  # 设置随机种子以确保可重现性
        random.shuffle(self.image_files)
        
        # 使用80%的数据作为训练集，20%作为验证集
        if split == 'train':
            self.image_files = self.image_files[:int(len(self.image_files) * 0.8)]
        else:  # 'val'
            self.image_files = self.image_files[int(len(self.image_files) * 0.8):]
        
        print(f"找到 {len(self.image_files)} 个图像用于 {split}")
        
        # 默认转换
        if transform is None:
            self.transform = transforms.Compose([
                transforms.Resize((img_height, img_width)),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            ])
        else:
            self.transform = transform
            
        # 标签转换
        self.label_transform = transforms.Compose([
            transforms.Resize((img_height, img_width), interpolation=transforms.InterpolationMode.NEAREST),
            transforms.PILToTensor()
        ])
        
        # 加载类别映射（如果有）
        self.class_mapping = self._load_class_mapping()
    
    def _load_class_mapping(self):
        """加载类别映射（如果有）"""
        mapping_file = os.path.join(self.root, "class_mapping.json")
        if os.path.exists(mapping_file):
            with open(mapping_file, 'r') as f:
                return json.load(f)
        return None
    
    def __len__(self):
        return len(self.image_files)
    
    def __getitem__(self, idx):
        # 获取图像路径
        img_path = self.image_files[idx]
        
        # 获取对应的标签路径
        # 假设标签文件与图像文件同名，但在labels目录下
        img_name = os.path.basename(img_path)
        label_name = os.path.splitext(img_name)[0] + '.png'  # 假设标签是PNG格式
        label_path = os.path.join(self.labels_dir, label_name)
        
        # 确保标签文件存在
        if not os.path.exists(label_path):
            # 尝试其他可能的扩展名
            for ext in ['.jpg', '.tif', '.tiff']:
                alt_label_path = os.path.join(self.labels_dir, os.path.splitext(img_name)[0] + ext)
                if os.path.exists(alt_label_path):
                    label_path = alt_label_path
                    break
            
            # 如果仍然找不到，则抛出错误
            assert os.path.exists(label_path), f"找不到标签文件: {label_path}"
        
        # 加载图像和标签
        image = Image.open(img_path).convert('RGB')
        label = Image.open(label_path)
        
        # 应用转换
        image = self.transform(image)
        label = self.label_transform(label).squeeze().long()
        
        # 如果需要，应用类别映射
        if self.class_mapping:
            # 将标签映射到模型所需的类别ID
            mapped_label = torch.zeros_like(label)
            for orig_id_str, new_id in self.class_mapping.items():
                orig_id = int(orig_id_str)
                mapped_label[label == orig_id] = new_id
            label = mapped_label
        
        return image, label
