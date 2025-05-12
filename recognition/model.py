# D-ViT model 구현
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.cuda.amp import autocast, GradScaler
from torch.utils.tensorboard import SummaryWriter

class TransformerBlock(nn.Module):
    def __init__(self, dim, heads, mlp_dim):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = nn.MultiheadAttention(dim, heads, batch_first=True)
        self.norm2 = nn.LayerNorm(dim)
        self.mlp = nn.Sequential(
            nn.Linear(dim, mlp_dim),
            nn.GELU(),
            nn.Linear(mlp_dim, dim)
        )

    def forward(self, x):
        x = x + self.attn(self.norm1(x), self.norm1(x), self.norm1(x))[0]
        x = x + self.mlp(self.norm2(x))
        return x

class ChannelSplitViT(nn.Module):
    def __init__(self, in_channels=64, patch_size=8, dim=64, depth=2, heads=4, mlp_dim=128):
        super().__init__()
        self.patch_embed = nn.Conv2d(in_channels, dim, kernel_size=patch_size, stride=patch_size)
        self.transformer = nn.Sequential(
            *[TransformerBlock(dim, heads, mlp_dim) for _ in range(depth)]
        )
        self.upsample = nn.Sequential(
            nn.ConvTranspose2d(dim, dim, kernel_size=4, stride=4),
            nn.BatchNorm2d(dim),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        B, C, H, W = x.shape
        x = self.patch_embed(x)  # (B, dim, H/8, W/8)
        x = x.flatten(2).transpose(1, 2)  # (B, N, dim)
        x = self.transformer(x)
        x = x.transpose(1, 2).reshape(B, -1, H // 8, W // 8)
        x = self.upsample(x)
        return x

class SpatialSplitViT(nn.Module):
    def __init__(self, in_channels=64, patch_size=8, dim=64, depth=2, heads=4, mlp_dim=128):
        super().__init__()
        self.patch_embed = nn.Conv2d(in_channels, dim, kernel_size=patch_size, stride=patch_size)
        self.transformer = nn.Sequential(
            *[TransformerBlock(dim, heads, mlp_dim) for _ in range(depth)]
        )
        self.upsample = nn.Sequential(
            nn.ConvTranspose2d(dim, dim, kernel_size=4, stride=4),
            nn.BatchNorm2d(dim),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        B, C, H, W = x.shape
        x = self.patch_embed(x)  # (B, dim, H/8, W/8)
        x = x.flatten(2).transpose(1, 2)  # (B, N, dim)
        x = self.transformer(x)
        x = x.transpose(1, 2).reshape(B, -1, H // 8, W // 8)
        x = self.upsample(x)
        return x

class PredictionBlock(nn.Module):
    def __init__(self):
        super().__init__()
        self.spat = SpatialSplitViT()
        self.chan = ChannelSplitViT()
        self.final = nn.Conv2d(64, 68, 1)
        
        self.skip_conv = nn.Conv2d(64, 64, 1)
        self.fusion = nn.Sequential(
            nn.Conv2d(128, 64, 1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True)
        )

    def forward(self, x, low_feat):
        skip_feat = self.skip_conv(low_feat)
        
        spat_feat = self.spat(x)
        chan_feat = self.chan(x)
        
        feat = torch.cat([spat_feat, chan_feat], dim=1)
        feat = self.fusion(feat)
        
        skip_feat = F.interpolate(skip_feat, size=feat.shape[2:], mode="bilinear", align_corners=False)
        feat = feat + skip_feat
        
        out = self.final(feat)
        out = F.interpolate(out, size=(256,256), mode='bilinear', align_corners=False)
        return out

class DViTModel(nn.Module):
    def __init__(self, nstack=8):
        super().__init__()
        self.backbone = nn.Sequential(
            nn.Conv2d(3, 64, 7, stride=2, padding=3),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2)
        )
        self.blocks = nn.ModuleList([PredictionBlock() for _ in range(nstack)])

    def forward(self, x):
        low_feat = self.backbone(x)
        outputs = []
        for block in self.blocks:
            out = block(low_feat, low_feat)  # Long Skip Connection
            outputs.append(out)
        return outputs

# AMP + TensorBoard logging integration in training loop
# scaler = GradScaler()
# writer = SummaryWriter(log_dir="runs/dvit_300w")
# with autocast():
#     output = model(input)
#     loss = criterion(output, target)
# scaler.scale(loss).backward()
# scaler.step(optimizer)
# scaler.update()
# writer.add_scalar("Loss/train", loss.item(), global_step)
# writer.close()
