"""
Enhanced YOLOv8 Modules for Performance Optimization
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import List, Optional, Tuple

from ultralytics.nn.modules.conv import Conv, DWConv, autopad
from ultralytics.nn.modules.block import C2f, Bottleneck


class DyConv(nn.Module):
    """Dynamic Convolution with adaptive kernel weights"""
    def __init__(self, c1, c2, k=3, s=1, p=None, g=1, d=1, act=True, num_experts=4):
        super().__init__()
        self.c1 = c1
        self.c2 = c2
        self.k = k
        self.s = s
        self.p = autopad(k, p, d) if p is None else p
        self.g = g
        self.d = d
        self.num_experts = num_experts
        
        # Multiple expert convolutions
        self.experts = nn.ModuleList([
            nn.Conv2d(c1, c2, k, s, self.p, d, g, bias=False) 
            for _ in range(num_experts)
        ])
        
        # Attention mechanism for expert selection
        self.attention = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(c1, max(c1 // 16, 1), 1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(max(c1 // 16, 1), num_experts, 1, bias=False),
            nn.Softmax(dim=1)
        )
        
        self.bn = nn.BatchNorm2d(c2)
        self.act = nn.SiLU() if act else nn.Identity()
        
    def forward(self, x):
        # Get attention weights for expert selection
        attn_weights = self.attention(x)  # (B, num_experts, 1, 1)
        
        # Apply experts and combine
        expert_outputs = []
        for expert in self.experts:
            expert_outputs.append(expert(x))
        
        # Weighted combination of expert outputs
        output = torch.zeros_like(expert_outputs[0])
        for i, expert_out in enumerate(expert_outputs):
            output += attn_weights[:, i:i+1] * expert_out
            
        return self.act(self.bn(output))


class ECAAttention(nn.Module):
    """Efficient Channel Attention mechanism"""
    def __init__(self, c1, gamma=2, b=1):
        super().__init__()
        self.gamma = gamma
        self.b = b
        
        # Adaptive kernel size calculation
        t = int(abs((math.log(c1, 2) + self.b) / self.gamma))
        k = t if t % 2 else t + 1
        k = max(k, 3)  # Minimum kernel size
        
        self.conv = nn.Conv1d(1, 1, kernel_size=k, padding=(k - 1) // 2, bias=False)
        self.sigmoid = nn.Sigmoid()
        
    def forward(self, x):
        # Global average pooling
        y = F.adaptive_avg_pool2d(x, 1).squeeze(-1).transpose(-1, -2)
        
        # 1D convolution for channel attention
        y = self.conv(y).transpose(-1, -2).unsqueeze(-1)
        
        # Apply attention weights
        return x * self.sigmoid(y).expand_as(x)


class C2f_ECA(nn.Module):
    """Enhanced C2f block with ECA attention mechanism"""
    def __init__(self, c1, c2, n=1, shortcut=False, g=1, e=0.5):
        super().__init__()
        self.c = int(c2 * e)
        self.cv1 = Conv(c1, 2 * self.c, 1, 1)
        self.cv2 = Conv((2 + n) * self.c, c2, 1)
        self.m = nn.ModuleList(Bottleneck(self.c, self.c, shortcut, g, k=((3, 3), (3, 3)), e=1.0) for _ in range(n))
        self.eca = ECAAttention(c2)
        
    def forward(self, x):
        y = list(self.cv1(x).chunk(2, 1))
        y.extend(m(y[-1]) for m in self.m)
        output = self.cv2(torch.cat(y, 1))
        return self.eca(output)


class FeatureFusion(nn.Module):
    """Advanced Feature Fusion module"""
    def __init__(self, c1, c2):
        super().__init__()
        self.cv1 = Conv(c1, c2, 1, 1)
        self.cv2 = Conv(c1, c2, 1, 1)
        
        # Spatial attention for feature fusion
        self.spatial_attn = nn.Sequential(
            nn.Conv2d(2, 1, 7, padding=3, bias=False),
            nn.Sigmoid()
        )
        
        # Channel attention for feature fusion
        self.channel_attn = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(c2, max(c2 // 16, 1), 1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(max(c2 // 16, 1), c2, 1, bias=False),
            nn.Sigmoid()
        )
        
    def forward(self, x):
        # x puede ser una lista de tensores o un solo tensor
        if isinstance(x, list):
            x1, x2 = x[0], x[1]
        else:
            # Si es un solo tensor, lo dividimos por la mitad en canales
            x1, x2 = x.chunk(2, 1)
        
        # Ensure same spatial dimensions
        if x1.shape[2:] != x2.shape[2:]:
            x2 = F.interpolate(x2, size=x1.shape[2:], mode='bilinear', align_corners=False)
        
        # Process features
        f1 = self.cv1(x1)
        f2 = self.cv2(x2)
        
        # Spatial attention
        avg_pool = torch.mean(f1 + f2, dim=1, keepdim=True)
        max_pool, _ = torch.max(f1 + f2, dim=1, keepdim=True)
        spatial_attn = self.spatial_attn(torch.cat([avg_pool, max_pool], dim=1))
        
        # Channel attention
        channel_attn = self.channel_attn(f1 + f2)
        
        # Fused features with attention
        fused = (f1 + f2) * spatial_attn * channel_attn
        
        return fused


class SPPF_Enhanced(nn.Module):
    """Enhanced Spatial Pyramid Pooling Fast"""
    def __init__(self, c1, c2, k=5):
        super().__init__()
        c_ = c1 // 2
        self.cv1 = Conv(c1, c_, 1, 1)
        self.cv2 = Conv(c_ * 4, c2, 1, 1)
        
        # Multiple pooling scales
        self.m1 = nn.MaxPool2d(kernel_size=k, stride=1, padding=k // 2)
        self.m2 = nn.MaxPool2d(kernel_size=k+2, stride=1, padding=(k+2) // 2)
        self.m3 = nn.MaxPool2d(kernel_size=k+4, stride=1, padding=(k+4) // 2)
        
    def forward(self, x):
        x = self.cv1(x)
        y1 = self.m1(x)
        y2 = self.m2(x)  
        y3 = self.m3(x)
        return self.cv2(torch.cat([x, y1, y2, y3], 1))

from ultralytics.nn.modules.head import Detect as DetectEnhanced

def create_enhanced_module(module_name, *args, **kwargs):
    """
    Factory function to create enhanced modules
    """
    modules = {
        'DyConv': DyConv,
        'C2f_ECA': C2f_ECA,
        'FeatureFusion': FeatureFusion,
        'SPPF_Enhanced': SPPF_Enhanced,
        'DetectEnhanced': DetectEnhanced,
        'ECAAttention': ECAAttention
    }
    
    if module_name in modules:
        return modules[module_name](*args, **kwargs)
    else:
        raise ValueError(f"Unknown module: {module_name}")


# __all__ exports
__all__ = [
    'DyConv', 
    'C2f_ECA', 
    'FeatureFusion', 
    'SPPF_Enhanced', 
    'DetectEnhanced', 
    'ECAAttention',
    'create_enhanced_module'
]