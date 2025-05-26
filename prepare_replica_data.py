import os
import json
import numpy as np
import argparse
from PIL import Image
import random
import shutil
from tqdm import tqdm

try:
    import habitat_sim
    from habitat_sim.utils.data import ImageExtractor
    HAS_HABITAT = True
except ImportError:
    HAS_HABITAT = False
    print("habitat-sim未安装，将使用备选数据处理方法")

def parse_args():
    parser = argparse.ArgumentParser(description='准备Replica数据集用于SAM2SEG微调')
    parser.add_argument('--replica_root', type=str, required=True, help='Replica数据集根目录')
    parser.add_argument('--output_dir', type=str, default='replica_processed', help='输出目录')
    parser.add_argument('--views_per_scene', type=int, default=500, help='每个场景渲染的视图数量')
    parser.add_argument('--img_height', type=int, default=680, help='渲染图像高度')
    parser.add_argument('--img_width', type=int, default=1200, help='渲染图像宽度')
    parser.add_argument('--seed', type=int, default=42, help='随机种子')
    return parser.parse_args()

def prepare_with_habitat(args):
    """使用habitat-sim渲染和提取数据"""
    print("使用Habitat-Sim处理Replica数据")
    
    # 获取所有场景路径
    scene_dirs = [d for d in os.listdir(args.replica_root) if os.path.isdir(os.path.join(args.replica_root, d))]
    
    # 创建输出目录
    os.makedirs(os.path.join(args.output_dir, 'images'), exist_ok=True)
    os.makedirs(os.path.join(args.output_dir, 'labels'), exist_ok=True)
    
    # 收集所有类别
    all_categories = set()
    category_to_id = {}
    
    # 处理每个场景
    for scene_dir in tqdm(scene_dirs, desc="处理场景"):
        scene_path = os.path.join(args.replica_root, scene_dir)
        
        # 检查必要文件是否存在
        mesh_semantic_path = os.path.join(scene_path, "habitat", "mesh_semantic.ply")
        info_semantic_path = os.path.join(scene_path, "habitat", "info_semantic.json")
        
        if not os.path.exists(mesh_semantic_path) or not os.path.exists(info_semantic_path):
            print(f"警告: 场景 {scene_dir} 缺少必要文件，跳过")
            continue
        
        # 读取语义信息
        with open(info_semantic_path, 'r') as f:
            semantic_info = json.load(f)
        
        # 提取类别信息
        for id_str, label_info in semantic_info.get("id_to_label", {}).items():
            if "raw_category" in label_info:
                category = label_info["raw_category"]
                all_categories.add(category)
        
        # 初始化图像提取器
        try:
            extractor = ImageExtractor(
                mesh_semantic_path,
                img_size=(args.img_width, args.img_height),
                output=["rgba", "semantic"],
                semantics_mapping_file=info_semantic_path
            )
            
            # 为每个场景渲染多个视图
            for i in range(args.views_per_scene):
                # 随机获取视点
                obs = extractor.get_random_view()
                
                # 提取图像和语义
                rgba = obs["rgba"]
                semantic = obs["semantic"]
                
                # 将RGBA转换为RGB
                rgb = rgba[:, :, :3]
                
                # 保存图像和语义标签
                rgb_path = os.path.join(args.output_dir, 'images', f"{scene_dir}_{i:04d}.jpg")
                semantic_path = os.path.join(args.output_dir, 'labels', f"{scene_dir}_{i:04d}.png")
                
                # 保存RGB图像
                Image.fromarray(rgb).save(rgb_path)
                
                # 保存语义分割图
                Image.fromarray(semantic.astype(np.uint8)).save(semantic_path)
            
            print(f"为场景 {scene_dir} 生成了 {args.views_per_scene} 个视图")
            
        except Exception as e:
            print(f"处理场景 {scene_dir} 时出错: {e}")
    
    # 创建类别映射
    for i, category in enumerate(sorted(all_categories)):
        category_to_id[category] = i
    
    # 保存类别映射
    with open(os.path.join(args.output_dir, 'class_mapping.json'), 'w') as f:
        json.dump(category_to_id, f, indent=4)
    
    print(f"共处理 {len(scene_dirs)} 个场景，发现 {len(all_categories)} 个类别")
    print(f"类别映射已保存到 {os.path.join(args.output_dir, 'class_mapping.json')}")
    
    # 创建训练/验证划分
    create_train_val_split(args.output_dir)

def prepare_alternative(args):
    """不使用habitat-sim的备选方法 - 直接使用mesh_semantic.ply文件和可视化工具"""
    print("使用备选方法处理Replica数据")
    print("注意: 这种方法需要您使用第三方工具（如MeshLab或Blender）手动渲染图像和分割标签")
    
    # 获取所有场景路径
    scene_dirs = [d for d in os.listdir(args.replica_root) if os.path.isdir(os.path.join(args.replica_root, d))]
    
    # 创建输出目录
    os.makedirs(os.path.join(args.output_dir, 'scenes'), exist_ok=True)
    
    # 为每个场景创建符号链接或复制mesh_semantic.ply和info_semantic.json
    for scene_dir in tqdm(scene_dirs, desc="处理场景"):
        scene_path = os.path.join(args.replica_root, scene_dir)
        
        # 检查必要文件是否存在
        mesh_semantic_path = os.path.join(scene_path, "habitat", "mesh_semantic.ply")
        info_semantic_path = os.path.join(scene_path, "habitat", "info_semantic.json")
        
        if not os.path.exists(mesh_semantic_path) or not os.path.exists(info_semantic_path):
            print(f"警告: 场景 {scene_dir} 缺少必要文件，跳过")
            continue
        
        # 创建场景输出目录
        scene_output_dir = os.path.join(args.output_dir, 'scenes', scene_dir)
        os.makedirs(scene_output_dir, exist_ok=True)
        
        # 复制文件
        shutil.copy2(mesh_semantic_path, os.path.join(scene_output_dir, 'mesh_semantic.ply'))
        shutil.copy2(info_semantic_path, os.path.join(scene_output_dir, 'info_semantic.json'))
        
        print(f"已复制场景 {scene_dir} 的文件，请使用3D工具手动渲染")
    
    print(f"已处理 {len(scene_dirs)} 个场景的文件")
    print(f"请使用3D渲染工具（如MeshLab或Blender）渲染每个场景的图像和分割标签")
    print(f"渲染后，请将图像放入 {os.path.join(args.output_dir, 'images')} 目录")
    print(f"将标签放入 {os.path.join(args.output_dir, 'labels')} 目录")

def create_train_val_split(data_dir, train_ratio=0.8):
    """创建训练集和验证集划分"""
    images_dir = os.path.join(data_dir, 'images')
    
    # 获取所有图像文件
    image_files = [f for f in os.listdir(images_dir) if f.endswith(('.jpg', '.png'))]
    
    # 按场景名称分组
    scene_images = {}
    for img_file in image_files:
        scene_name = img_file.split('_')[0]  # 假设文件名格式为scene_xxxx.jpg
        if scene_name not in scene_images:
            scene_images[scene_name] = []
        scene_images[scene_name].append(img_file)
    
    # 随机划分场景
    random.seed(42)
    scenes = list(scene_images.keys())
    random.shuffle(scenes)
    
    train_scenes = scenes[:int(len(scenes) * train_ratio)]
    val_scenes = scenes[int(len(scenes) * train_ratio):]
    
    # 创建训练集和验证集文件列表
    train_files = []
    for scene in train_scenes:
        train_files.extend(scene_images[scene])
    
    val_files = []
    for scene in val_scenes:
        val_files.extend(scene_images[scene])
    
    # 保存划分结果
    with open(os.path.join(data_dir, 'train.txt'), 'w') as f:
        for img_file in train_files:
            f.write(os.path.splitext(img_file)[0] + '\n')
    
    with open(os.path.join(data_dir, 'val.txt'), 'w') as f:
        for img_file in val_files:
            f.write(os.path.splitext(img_file)[0] + '\n')
    
    print(f"训练集: {len(train_files)} 图像, 来自 {len(train_scenes)} 个场景")
    print(f"验证集: {len(val_files)} 图像, 来自 {len(val_scenes)} 个场景")

if __name__ == "__main__":
    args = parse_args()
    
    # 创建输出目录
    os.makedirs(args.output_dir, exist_ok=True)
    
    # 根据是否安装了habitat-sim选择数据处理方法
    if HAS_HABITAT:
        prepare_with_habitat(args)
    else:
        prepare_alternative(args)
