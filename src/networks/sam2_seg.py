# This file is a part of SNI-SLAM.

import os
import torch
import torch.nn as nn
import sys

parent_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
module_path = os.path.join(parent_dir, 'seg', 'facebookresearch_sam2_main')
sys.path.append(module_path)

sam2_backbones = {
    'sam2_b':{
        'name':'sam2_vitb14',
        'embedding_size':768,
        'patch_size':14
    },
    'sam2_l':{
        'name':'sam2_vitl14',
        'embedding_size':1024,
        'patch_size':14
    },
}

def make_sam2_model(
    *,
    arch_name: str = "vit_large",
    img_size: int = 518,
    patch_size: int = 14,
    init_values: float = 1.0,
    ffn_layer: str = "mlp",
    block_chunks: int = 0,
    pretrained: bool = False,
    **kwargs,
):
    from sam2.modeling.sam2_base import SAM2Base

    vit_kwargs = dict(
        img_size=img_size,
        patch_size=patch_size,
        init_values=init_values,
        ffn_layer=ffn_layer,
        block_chunks=block_chunks,
    )
    vit_kwargs.update(**kwargs)
    model = SAM2Base(**vit_kwargs)

    return model


class SAM2SEG(nn.Module):
    def __init__(self, img_h, img_w, num_cls, backbone='sam2_b', mode='mapping', edge=10, dim=16):
        super(SAM2SEG, self).__init__()
        self.backbones = sam2_backbones
        self.embedding_size = self.backbones[backbone]['embedding_size']
        self.patch_size = self.backbones[backbone]['patch_size']

        # Create the model
        if backbone == 'sam2_b':
            self.backbone = make_sam2_model(arch_name="vit_base")
        elif backbone == 'sam2_l':
            self.backbone = make_sam2_model(arch_name="vit_large")

        self.num_class = num_cls
        self.mode = mode
        self.img_h = img_h
        self.img_w = img_w

        self.segmentation_conv = nn.Sequential(
            nn.Upsample(scale_factor=4),
            nn.Conv2d(self.embedding_size, dim, (3, 3), padding=(1, 1)),
            nn.Upsample((self.img_h - 2 * edge, self.img_w - 2 * edge)),
            nn.Conv2d(dim, self.num_class, (3, 3), padding=(1, 1))
        )
        h = ((self.img_h - 2 * edge) // 14) * 14
        w = ((self.img_w - 2 * edge) // 14) * 14
        self.upsample = nn.Upsample((h, w))

    def forward(self, x):
        x = self.upsample(x)
        bs = x.shape[0]
        mask_dim = (x.shape[2] / self.patch_size, x.shape[3] / self.patch_size)

        # Use SAM2Base to get the features
        features = self.backbone.extract_features(x)

        out = features.permute(0, 2, 3, 1).reshape(bs, self.embedding_size, int(mask_dim[0]), int(mask_dim[1]))  # [1,768,40,72]

        if self.mode == 'mapping':
            for i in range(3):
                out = self.segmentation_conv[i](out)
        else:
            outputs = self.segmentation_conv(out)
            out = torch.max(outputs, 1).indices.squeeze()

        return out