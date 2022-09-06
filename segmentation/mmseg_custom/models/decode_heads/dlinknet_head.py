from functools import partial

import torch
import torch.nn as nn
import torch.nn.functional as F

from mmseg.models.builder import HEADS, build_loss
from mmseg.models.decode_heads.decode_head import BaseDecodeHead

from .vit_backbone import ViTBackbone, Norm2d


nonlinearity = partial(F.relu, inplace=True)


# class Conv3x3LN(nn.Module):
#     """
#     3x3 Conv + LN: 减少上采样的 aliasing effect
#     暂时不使用
#     ViTDet Page 15 - A.2 Implementation Details
#     "Then for each pyramid level, we apply a 1x1 convolution with LN to reduce dimension
#      to 256 and then a 3x3 convolution also with LN, similar to the per-level processing of FPN"
#     """
#     def __init__(self, in_channels, out_channels):
#         super().__init__()
#         self.module = nn.Sequential(
#             nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
#             Norm2d(out_channels)
#         )
    
#     def forward(self, x):
#         return self.module(x)

@HEADS.register_module()
class ViT_DLinkNet(BaseDecodeHead):
    """
    原 D-LinkNet 34 中 encoder 的 4 个输出的 shape (假设 batch size 为 4):
        torch.Size([4, 64, 256, 256])
        torch.Size([4, 128, 128, 128])
        torch.Size([4, 256, 64, 64])
        torch.Size([4, 512, 32, 32])
    """

    # ViT 的变体:
    params = {
        'tiny': {
            'embed_dim': 192,
            'depth': 12,
            'num_heads': 3
        },
        'small': {
            'embed_dim': 384,
            'depth': 12,
            'num_heads': 6
        },
        'base': {
            'embed_dim': 768,
            'depth': 12,
            'num_heads': 12
        },
        'large': {
            'embed_dim': 1024,
            'depth': 24,
            'num_heads': 16
        }
    }

    def __init__(self, patch_size=16, backbone_type='tiny', pretrained_backbone=None):
        super(ViT_DLinkNet, self).__init__()

        self.vit_backbone = ViTBackbone(
            img_size=1024,
            patch_size=patch_size,
            num_classes=2,
            embed_dim=self.params[backbone_type]['embed_dim'],
            depth=self.params[backbone_type]['depth'],
            num_heads=self.params[backbone_type]['num_heads'],
            mlp_ratio=4,
            qkv_bias=True,
            drop_rate=0.,
            attn_drop_rate=0.,
            drop_path_rate=0.1,
            use_abs_pos_emb=True,
            pretrained=pretrained_backbone
        )

        self.dblock = Dblock(512)

        filters = [64, 128, 256, 512]
        self.decoder4 = DecoderBlock(filters[3], filters[2])
        self.decoder3 = DecoderBlock(filters[2], filters[1])
        self.decoder2 = DecoderBlock(filters[1], filters[0])
        self.decoder1 = DecoderBlock(filters[0], filters[0])

        self.finaldeconv1 = nn.ConvTranspose2d(filters[0], 32, 4, 2, 1)
        self.finalrelu1 = nonlinearity
        self.finalconv2 = nn.Conv2d(32, 32, 3, padding=1)
        self.finalrelu2 = nonlinearity
        self.finalconv3 = nn.Conv2d(32, 1, 3, padding=1)

    def init_weights(self):
        pass

    @torch.jit.ignore
    def no_weight_decay(self):
        return {'pos_embed', 'cls_token'}

    def get_num_layers(self):
        return len(self.vit_backbone.blocks)

    def _forward(self, x):
        # Encoder
        e1, e2, e3, e4 = self.vit_backbone(x)

        # Center
        e4 = self.dblock(e4)

        # Decoder
        d4 = self.decoder4(e4) + e3
        d3 = self.decoder3(d4) + e2
        d2 = self.decoder2(d3) + e1
        d1 = self.decoder1(d2)

        out = self.finaldeconv1(d1)
        out = self.finalrelu1(out)
        out = self.finalconv2(out)
        out = self.finalrelu2(out)
        out = self.finalconv3(out)

        return F.sigmoid(out)

    def load_mae_pretrained(self, pretrained_weight_path):
        print('Loading MAE pretrained encoder and decoder from', pretrained_weight_path)
        pretrained_model = torch.load(pretrained_weight_path, map_location='cpu')['model']
        state_dict = {}

        for key, value in pretrained_model.items():
            # Position Embedding
            if key == 'pos_embed':
                # num_patches + 1 -> num_patches
                state_dict['vit_backbone.pos_embed'] = value[:, :value.shape[1]-1, :]
                
            if key.startswith('patch_embed') or key.startswith('norm') or key.startswith('fpn'):
                state_dict['vit_backbone.' + key] = value
            
            # Transformer blocks in ViT backbone (except for propogation blocks)
            elif key.startswith('blocks.'):
                block_interval = 12 // 4    # depth=12
                conv_prop_block_idx = [block_interval * (i+1) for i in range(4)]
                block_idx = int(key.split('.')[1])
                if block_idx not in conv_prop_block_idx:
                    state_dict['vit_backbone.' + key] = value
                else:
                    print('Droppig: ', key)
            
            elif key.startswith('dblock') or key.startswith('decoder4') or \
                key.startswith('decoder3') or key.startswith('decoder2') or \
                key.startswith('decoder1') or key.startswith('finaldeconv1') or \
                key.startswith('finalconv2'):
                state_dict[key] = value
            
            else:
                print('Droppig: ', key)

        # The final convolution layer does not need to load
        # state_dict['finalconv3.weight'] = self.finalconv3.weight
        # state_dict['finalconv3.bias'] = self.finalconv3.bias

        self.load_state_dict(state_dict, strict=False)


class ViT_DLinkNet_wo_DBlock(nn.Module):
    # ViT 的变体:
    params = {
        'tiny': {
            'embed_dim': 192,
            'depth': 12,
            'num_heads': 3
        },
        'small': {
            'embed_dim': 384,
            'depth': 12,
            'num_heads': 6
        },
        'base': {
            'embed_dim': 768,
            'depth': 12,
            'num_heads': 12
        },
        'large': {
            'embed_dim': 1024,
            'depth': 24,
            'num_heads': 16
        }
    }

    def __init__(self, patch_size=16, backbone_type='tiny', pretrained_backbone=None):
        super(ViT_DLinkNet_wo_DBlock, self).__init__()

        self.vit_backbone = ViTBackbone(
            img_size=1024,
            patch_size=patch_size,
            num_classes=2,
            embed_dim=self.params[backbone_type]['embed_dim'],
            depth=self.params[backbone_type]['depth'],
            num_heads=self.params[backbone_type]['num_heads'],
            mlp_ratio=4,
            qkv_bias=True,
            drop_rate=0.,
            attn_drop_rate=0.,
            drop_path_rate=0.1,
            use_abs_pos_emb=True,
            pretrained=pretrained_backbone
        )

        filters = [64, 128, 256, 512]
        self.decoder4 = DecoderBlock(filters[3], filters[2])
        self.decoder3 = DecoderBlock(filters[2], filters[1])
        self.decoder2 = DecoderBlock(filters[1], filters[0])
        self.decoder1 = DecoderBlock(filters[0], filters[0])

        self.finaldeconv1 = nn.ConvTranspose2d(filters[0], 32, 4, 2, 1)
        self.finalrelu1 = nonlinearity
        self.finalconv2 = nn.Conv2d(32, 32, 3, padding=1)
        self.finalrelu2 = nonlinearity
        self.finalconv3 = nn.Conv2d(32, 1, 3, padding=1)

    @torch.jit.ignore
    def no_weight_decay(self):
        return {'pos_embed', 'cls_token'}

    def get_num_layers(self):
        return len(self.vit_backbone.blocks)

    def forward(self, x):
        # Encoder
        e1, e2, e3, e4 = self.vit_backbone(x)
        
        # Decoder
        d4 = self.decoder4(e4) + e3
        d3 = self.decoder3(d4) + e2
        d2 = self.decoder2(d3) + e1
        d1 = self.decoder1(d2)

        out = self.finaldeconv1(d1)
        out = self.finalrelu1(out)
        out = self.finalconv2(out)
        out = self.finalrelu2(out)
        out = self.finalconv3(out)

        return F.sigmoid(out)

    def load_mae_pretrained(self, pretrained_weight_path):
        print('Loading MAE pretrained encoder and decoder from', pretrained_weight_path)
        pretrained_model = torch.load(pretrained_weight_path, map_location='cpu')['model']
        state_dict = {}

        for key, value in pretrained_model.items():
            # Position Embedding
            if key == 'pos_embed':
                # num_patches + 1 -> num_patches
                state_dict['vit_backbone.pos_embed'] = value[:, :value.shape[1]-1, :]
                
            if key.startswith('patch_embed') or key.startswith('norm') or key.startswith('fpn'):
                state_dict['vit_backbone.' + key] = value
            
            # Transformer blocks in ViT backbone (except for propogation blocks)
            elif key.startswith('blocks.'):
                block_interval = 12 // 4    # depth=12
                conv_prop_block_idx = [block_interval * (i+1) for i in range(4)]
                block_idx = int(key.split('.')[1])
                if block_idx not in conv_prop_block_idx:
                    state_dict['vit_backbone.' + key] = value
                else:
                    print('Droppig: ', key)
            
            elif key.startswith('decoder4') or \
                key.startswith('decoder3') or key.startswith('decoder2') or \
                key.startswith('decoder1') or key.startswith('finaldeconv1') or \
                key.startswith('finalconv2'):
                state_dict[key] = value
            
            else:
                print('Droppig: ', key)

        # The final convolution layer does not need to load
        # state_dict['finalconv3.weight'] = self.finalconv3.weight
        # state_dict['finalconv3.bias'] = self.finalconv3.bias

        self.load_state_dict(state_dict, strict=False)

class Dblock(nn.Module):
    def __init__(self,channel):
        super(Dblock, self).__init__()
        self.dilate1 = nn.Conv2d(channel, channel, kernel_size=3, dilation=1, padding=1)
        self.dilate2 = nn.Conv2d(channel, channel, kernel_size=3, dilation=2, padding=2)
        self.dilate3 = nn.Conv2d(channel, channel, kernel_size=3, dilation=4, padding=4)
        self.dilate4 = nn.Conv2d(channel, channel, kernel_size=3, dilation=8, padding=8)
        #self.dilate5 = nn.Conv2d(channel, channel, kernel_size=3, dilation=16, padding=16)
        for m in self.modules():
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d):
                if m.bias is not None:
                    m.bias.data.zero_()
                    
    def forward(self, x):
        dilate1_out = nonlinearity(self.dilate1(x))
        dilate2_out = nonlinearity(self.dilate2(dilate1_out))
        dilate3_out = nonlinearity(self.dilate3(dilate2_out))
        dilate4_out = nonlinearity(self.dilate4(dilate3_out))
        #dilate5_out = nonlinearity(self.dilate5(dilate4_out))
        out = x + dilate1_out + dilate2_out + dilate3_out + dilate4_out# + dilate5_out
        return out

class DecoderBlock(nn.Module):
    def __init__(self, in_channels, n_filters):
        super(DecoderBlock,self).__init__()

        self.conv1 = nn.Conv2d(in_channels, in_channels // 4, 1)
        self.norm1 = nn.BatchNorm2d(in_channels // 4)
        self.relu1 = nonlinearity

        self.deconv2 = nn.ConvTranspose2d(in_channels // 4, in_channels // 4, 3, stride=2, padding=1, output_padding=1)
        self.norm2 = nn.BatchNorm2d(in_channels // 4)
        self.relu2 = nonlinearity

        self.conv3 = nn.Conv2d(in_channels // 4, n_filters, 1)
        self.norm3 = nn.BatchNorm2d(n_filters)
        self.relu3 = nonlinearity

    def forward(self, x):
        x = self.conv1(x)
        x = self.norm1(x)
        x = self.relu1(x)
        x = self.deconv2(x)
        x = self.norm2(x)
        x = self.relu2(x)
        x = self.conv3(x)
        x = self.norm3(x)
        x = self.relu3(x)
        return x