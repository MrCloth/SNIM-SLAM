import os
import json
import glob
import torch
import numpy as np
from torch.utils.data import Dataset
from PIL import Image
import torchvision.transforms as transforms
import habitat_sim
from habitat_sim.utils.data import ImageExtractor

class ReplicaDataset(Dataset):
    def __init__(self, root, split='train', img_height=680, img_width=1200, num_views_per_scene=500, transform=None):
        """
        Replica数据集加载器，用于从Replica场景生成和加载RGB图像和语义分割标签
        
        参数:
            root: Replica数据集根目录
            split: 'train'或'val'分割
            img_height: 输出图像高度
            img_width: 输出图像宽度
            num_views_per_scene: 每个场景生成的视图数量
            transform: 数据增强转换
        """
        self.root = root
        self.split = split
        self.img_height = img_height
        self.img_width = img_width
        self.num_views_per_scene = num_views_per_scene
        
        # 获取所有场景路径
        self.scene_paths = glob.glob(os.path.join(root, '*'))
        self.scene_paths = [p for p in self.scene_paths if os.path.isdir(p)]
        
        # 划分训练和验证场景
        # 使用80%的场景作为训练集，20%作为验证集
        # 或者按照特定场景名称划分
        if split == 'train':
            self.scene_paths = self.scene_paths[:int(len(self.scene_paths) * 0.8)]
        else:  # 'val'
            self.scene_paths = self.scene_paths[int(len(self.scene_paths) * 0.8):]
        
        print(f"Using {len(self.scene_paths)} scenes for {split}")
        
        # 创建数据集缓存目录
        self.cache_dir = os.path.join(root, f"cache_{split}")
        os.makedirs(self.cache_dir, exist_ok=True)
        
        # 检查是否已有生成的数据
        self.image_files = glob.glob(os.path.join(self.cache_dir, "*_rgb.jpg"))
        
        # 如果没有缓存数据，则生成
        if len(self.image_files) == 0:
            self._generate_dataset()
            # 重新获取生成的文件列表
            self.image_files = glob.glob(os.path.join(self.cache_dir, "*_rgb.jpg"))
        
        print(f"Found {len(self.image_files)} images for {split}")
        
        # 排序以确保rgb和semantic对应
        self.image_files.sort()
        
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
        
        # 加载语义映射信息（类别ID到类别名称的映射）
        self.semantic_id_to_class = self._load_semantic_mapping()
    
    def _load_semantic_mapping(self):
        """加载语义ID到类别的映射"""
        semantic_mapping = {}
        for scene_path in self.scene_paths:
            info_path = os.path.join(scene_path, "habitat", "info_semantic.json")
            if os.path.exists(info_path):
                with open(info_path, 'r') as f:
                    info = json.load(f)
                
                # 提取ID到类别的映射
                if "id_to_label" in info:
                    for id_str, label in info["id_to_label"].items():
                        id_int = int(id_str)
                        if "raw_category" in label:
                            category = label["raw_category"]
                            semantic_mapping[id_int] = category
        
        # 如果没有找到映射，使用默认的52个类别
        if not semantic_mapping:
            # 使用默认的类别映射（根据实际的Replica数据集调整）
            semantic_mapping = {i: f"class_{i}" for i in range(52)}
        
        return semantic_mapping
    
    def _generate_dataset(self):
        """生成和保存数据集图像和标签"""
        print("正在生成Replica数据集图像和标签...")
        
        for scene_path in self.scene_paths:
            scene_name = os.path.basename(scene_path)
            print(f"处理场景: {scene_name}")
            
            # 配置habitat场景
            mesh_semantic_path = os.path.join(scene_path, "habitat", "mesh_semantic.ply")
            
            # 检查必要文件是否存在
            if not os.path.exists(mesh_semantic_path):
                print(f"警告: 场景 {scene_name} 缺少必要文件，跳过")
                continue
            
            # 使用Habitat-Sim提取图像和语义标签
            try:
                # 初始化图像提取器
                extractor = ImageExtractor(
                    mesh_semantic_path,
                    img_size=(self.img_width, self.img_height),
                    output=["rgba", "semantic"],
                    semantics_mapping_file=os.path.join(scene_path, "habitat", "info_semantic.json")
                )
                
                # 随机生成视点
                for i in range(self.num_views_per_scene):
                    # 随机获取视点
                    obs = extractor.get_random_view()
                    
                    # 提取图像和语义
                    rgba = obs["rgba"]
                    semantic = obs["semantic"]
                    
                    # 将RGBA转换为RGB
                    rgb = rgba[:, :, :3]
                    
                    # 保存图像和语义标签
                    rgb_path = os.path.join(self.cache_dir, f"{scene_name}_{i:04d}_rgb.jpg")
                    semantic_path = os.path.join(self.cache_dir, f"{scene_name}_{i:04d}_semantic.png")
                    
                    # 保存RGB图像
                    Image.fromarray(rgb).save(rgb_path)
                    
                    # 保存语义分割图
                    Image.fromarray(semantic.astype(np.uint8)).save(semantic_path)
                    
                print(f"为场景 {scene_name} 生成了 {self.num_views_per_scene} 个视图")
                
            except Exception as e:
                print(f"处理场景 {scene_name} 时出错: {e}")
                
                # 尝试备选方法：直接使用预渲染的样本（如果有）
                # 这里可以添加备选的数据生成方法
        
        print("数据集生成完成")
    
    def __len__(self):
        return len(self.image_files)
    
    def __getitem__(self, idx):
        # 获取图像路径
        img_path = self.image_files[idx]
        # 根据命名规则获取对应的语义标签路径
        label_path = img_path.replace("_rgb.jpg", "_semantic.png")
        
        # 确保文件存在
        assert os.path.exists(img_path), f"图像文件 {img_path} 不存在"
        assert os.path.exists(label_path), f"标签文件 {label_path} 不存在"
        
        # 加载图像和标签
        image = Image.open(img_path).convert('RGB')
        label = Image.open(label_path)
        
        # 应用转换
        image = self.transform(image)
        label = self.label_transform(label).squeeze().long()
        
        # 确保标签在有效范围内
        # 可能需要重映射语义ID
        label = self._remap_semantic_ids(label)
        
        return image, label
    
    def _remap_semantic_ids(self, label_tensor):
        """将原始语义ID重映射到连续的类别ID范围内"""
        # 创建重映射张量
        unique_ids = torch.unique(label_tensor)
        id_map = {}
        
        for i, id in enumerate(unique_ids):
            if id.item() != 255:  # 保留忽略标签
                id_map[id.item()] = i
        
        # 应用重映射
        result = label_tensor.clone()
        for old_id, new_id in id_map.items():
            result[label_tensor == old_id] = new_id
        
        return result
