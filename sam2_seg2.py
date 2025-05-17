import os
import torch
import torch.nn as nn
import sys
import torch.nn.functional as F

parent_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
module_path = os.path.join(parent_dir, 'seg', 'sam2')
sys.path.append(module_path)


sam2_backbones = {
    'sam2.1_t': {
        'scalp': 1,
        'trunk': {
            '_target_': 'sam2.modeling.backbones.hieradet.Hiera',
            'embed_dim': 96,
            'num_heads': 1,
            'stages': [1, 2, 7, 2],
            'global_att_blocks': [5, 7, 9],
            'window_pos_embed_bkg_spatial_size': [7, 7]
        },
        'neck': {
            '_target_': 'sam2.modeling.backbones.image_encoder.FpnNeck',
            'position_encoding': {
                '_target_': 'sam2.modeling.position_encoding.PositionEmbeddingSine',
                'num_pos_feats': 256,
                'normalize': True,
                'scale': None,
                'temperature': 10000
            },
            'd_model': 256,
            'backbone_channel_list': [768, 384, 192, 96],
            'fpn_top_down_levels': [2, 3],  # output level 0 and 1 directly use the backbone features
            'fpn_interp_model': 'nearest'
        }
    }
}

def make_sam2_model(
        *,
        arch_name: str = "sam2.1_t",
        scalp: int = 1,
        trunk_embed_dim: int = 96,
        trunk_num_heads: int = 1,
        trunk_stages: list = [1, 2, 7, 2],
        trunk_global_att_blocks: list = [5, 7, 9],
        trunk_window_pos_embed_bkg_spatial_size: list = [7, 7],
        neck_d_model: int = 256,
        neck_backbone_channel_list: list = [768, 384, 192, 96],
        neck_fpn_top_down_levels: list = [2, 3],
        neck_fpn_interp_model: str = "nearest",
        neck_pos_encoding_num_pos_feats: int = 256,
        neck_pos_encoding_normalize: bool = True,
        neck_pos_encoding_scale = None,
        neck_pos_encoding_temperature: int = 10000,
):
    from seg.sam2.sam2.modeling.backbones import image_encoder
    from seg.sam2.sam2.modeling.backbones.hieradet import Hiera
    from seg.sam2.sam2.modeling.position_encoding import PositionEmbeddingSine
    
    # 配置position encoding
    position_encoding = PositionEmbeddingSine(
        num_pos_feats=neck_pos_encoding_num_pos_feats,
        normalize=neck_pos_encoding_normalize,
        scale=neck_pos_encoding_scale,
        temperature=neck_pos_encoding_temperature
    )
    
    # 配置trunk
    trunk = Hiera(
        embed_dim=trunk_embed_dim,
        num_heads=trunk_num_heads,
        stages=trunk_stages,
        global_att_blocks=trunk_global_att_blocks,
        window_pos_embed_bkg_spatial_size=trunk_window_pos_embed_bkg_spatial_size
    )
    
    # 配置neck
    neck = image_encoder.FpnNeck(
        position_encoding=position_encoding,
        d_model=neck_d_model,
        backbone_channel_list=neck_backbone_channel_list,
        fpn_top_down_levels=neck_fpn_top_down_levels,
        fpn_interp_model=neck_fpn_interp_model
    )
    
    # 创建ImageEncoder
    model = image_encoder.ImageEncoder(
        scalp=scalp,
        trunk=trunk,
        neck=neck
    )
    
    return model
class SAM2SEG(nn.Module):
    def __init__(self, img_h, img_w, num_cls, backbone='sam2.1_t', mode='mapping', edge=10, dim=16, d_model=256, finetune_strategy='head_only'):
        super(SAM2SEG, self).__init__()
        
        # 基本参数
        self.num_class = num_cls
        self.img_h = img_h
        self.img_w = img_w
        self.edge = edge
        self.d_model = d_model
        self.mode = mode
        # 图像特征提取器
        self.backbones = sam2_backbones
        self.embed_dim = self.backbones[backbone]['trunk']['embed_dim']
        self.d_model = self.backbones[backbone]['neck']['d_model']
        self.backbone = make_sam2_model(arch_name='sam2.1_t')
        
        # 微调策略，控制哪些部分可训练
        self.finetune_strategy = finetune_strategy
        
        # 尺寸调整的上采样层
        h = ((self.img_h - 2 * edge) // 32) * 32  # 调整为32倍 (Hiera的最大下采样率)
        w = ((self.img_w - 2 * edge) // 32) * 32
        self.upsample = nn.Upsample((h, w))
        
        # 特征融合后的处理模块
        self.fusion_convs = nn.ModuleList([
            nn.Conv2d(d_model, d_model, kernel_size=3, padding=1),
            nn.Conv2d(d_model, d_model, kernel_size=3, padding=1),
            nn.Conv2d(d_model, d_model, kernel_size=3, padding=1),
        ])
        
        # 最终分割头
        self.segmentation_head = nn.Sequential(
            nn.Upsample(scale_factor=4),
            nn.Conv2d(d_model, dim, kernel_size=3, padding=1),
            nn.BatchNorm2d(dim),
            nn.ReLU(inplace=True),
            nn.Upsample(size=(self.img_h - 2 * edge, self.img_w - 2 * edge)),
            nn.Conv2d(dim, self.num_class, (3, 3), padding=(1, 1))
        )
        
        # 应用微调策略
        self._apply_finetune_strategy()

    def _apply_finetune_strategy(self):
        """应用不同的微调策略"""
        # 默认：先冻结所有backbone参数
        for name, param in self.backbone.named_parameters():
            param.requires_grad = False
            
        if self.finetune_strategy == 'head_only':
            # 仅训练融合卷积和分割头，backbone完全冻结
            pass  # 已经冻结backbone
            
        elif self.finetune_strategy == 'last_layers':
            # 训练backbone的最后一些层级 + 解码头
            # 解冻trunk的最后一个阶段
            for name, param in self.backbone.trunk.named_parameters():
                if 'stages.3' in name:  # 解冻最后一个stage
                    param.requires_grad = True
                    
        elif self.finetune_strategy == 'neck_only':
            # 仅训练FPN Neck部分 + 解码头
            for name, param in self.backbone.neck.named_parameters():
                param.requires_grad = True
                
        elif self.finetune_strategy == 'full_finetune':
            # 完全微调所有参数
            for name, param in self.backbone.named_parameters():
                param.requires_grad = True
        
        # 确保融合卷积和分割头是可训练的
        for name, param in self.fusion_convs.named_parameters():
            param.requires_grad = True
        
        for name, param in self.segmentation_head.named_parameters():
            param.requires_grad = True
        
        # 计算并显示冻结/未冻结参数数量
        backbone_params = sum(p.numel() for p in self.backbone.parameters())
        frozen_params = sum(p.numel() for p in self.parameters() if not p.requires_grad)
        total_params = sum(p.numel() for p in self.parameters())
        trainable_params = total_params - frozen_params
        
        print(f"=== 参数冻结统计 ({self.finetune_strategy}) ===")
        print(f"backbone参数: {backbone_params:,}")
        print(f"冻结参数总数: {frozen_params:,}")
        print(f"可训练参数总数: {trainable_params:,}")
        print(f"冻结比例: {frozen_params/total_params:.2%}")
        print(f"========================")

    def _upsample_add(self, x, y):
        """上采样x并与y相加"""
        return F.interpolate(x, size=y.shape[2:], mode='bilinear', align_corners=False) + y

    def forward(self, x):
        # 调整输入大小
        # print(f"输入大小: {x.shape}")
        x = self.upsample(x)
        # print(f"上采样后大小: {x.shape}")
        bs = x.shape[0]
        # 提取多尺度特征
        encoder_output = self.backbone(x.float())
        features = encoder_output["backbone_fpn"]
        ##############！！！！！！！！！！！！！！！！！！！！！！
        # 特征金字塔网络处理 (FPN风格，自顶向下路径,不需要)
        laterals = []
        for i, feat in enumerate(features):
            laterals.append(feat)
        
        # 自顶向下融合路径
        fpn_features = [laterals[0]]  # 从最高级别特征开始
        for i in range(len(laterals) - 1):
            # 上采样当前特征并与下一级特征相加
            fused = self._upsample_add(fpn_features[-1], laterals[i + 1])
            # 应用卷积进一步处理融合特征
            fused = self.fusion_convs[i](fused)
            fpn_features.append(fused)
        
        # 使用最高分辨率的融合特征进行最终预测
        out = fpn_features[-1]
        # print(f"特征融合后,传入分割头之前的大小: {out.shape}")
        # 分割头输出
        if self.mode == 'mapping':
            for i in range(5):
                out = self.segmentation_head[i](out)
            # print(f"mapping状态输出大小: {out.shape}")
        elif self.mode == 'finetuning':
            out = self.segmentation_head(out)
        else:
            outputs = self.segmentation_head(out)
            out = torch.max(outputs, 1).indices.squeeze()
            # print(f"fei输出大小: {out.shape}")
        
        return out
