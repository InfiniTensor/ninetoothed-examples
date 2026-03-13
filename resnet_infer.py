#!/usr/bin/env python3
"""
run_resnet50.py
一键跑通 ResNet-50 预训练模型推理
"""

import torch
import torch.nn as nn
import torchvision.transforms as T
from torchvision.models import resnet50, ResNet50_Weights
from PIL import Image
import os

import ops.ninetoothed.torch
import json
import sys
import argparse
from utils import replace_module
from relu import ReLU,relu_backend
from softmax import Softmax, softmax_backend
from max_pool2d import MaxPool2d, max_pool2d_backend
from conv2d import Conv2d, conv2d_backend
from avg_pool2d import AvgPool2d, avg_pool2d_backend

torch.manual_seed(42)

# BottleNeck 模块（ResNet50 使用）
class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, in_channels, out_channels, stride=1, downsample=None, groups=1, base_width=64):
        super().__init__()
        width = int(out_channels * (base_width / 64.0)) * groups

        self.conv1 = nn.Conv2d(in_channels, width, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(width)
        self.conv2 = nn.Conv2d(width, width, kernel_size=3, stride=stride, padding=1, groups=groups, bias=False)
        self.bn2 = nn.BatchNorm2d(width)
        self.conv3 = nn.Conv2d(width, out_channels * self.expansion, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(out_channels * self.expansion)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)
        return out

# 自定义 ResNet50 主干
class CustomResNet50(nn.Module):
    def __init__(self, num_classes=1000, zero_init_residual=False):
        super().__init__()
        self.in_channels = 64

        # Stem
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        # Residual layers
        self.layer1 = self._make_layer(Bottleneck, 64, 3)
        self.layer2 = self._make_layer(Bottleneck, 128, 4, stride=2)
        self.layer3 = self._make_layer(Bottleneck, 256, 6, stride=2)
        self.layer4 = self._make_layer(Bottleneck, 512, 3, stride=2)

        # self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.avgpool = nn.AvgPool2d(kernel_size=7, stride=7)
        self.fc = nn.Linear(512 * Bottleneck.expansion, num_classes)

        # 初始化权重（与 torchvision 一致）
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

        # Zero-initialize the last BN in each residual branch
        if zero_init_residual:
            for m in self.modules():
                if isinstance(m, Bottleneck):
                    nn.init.constant_(m.bn3.weight, 0)

    def _make_layer(self, block, out_channels, blocks, stride=1):
        downsample = None
        if stride != 1 or self.in_channels != out_channels * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.in_channels, out_channels * block.expansion, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels * block.expansion),
            )
        layers = []
        layers.append(block(self.in_channels, out_channels, stride, downsample))
        self.in_channels = out_channels * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.in_channels, out_channels))
        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)
        return x

def download_resnet50_weights(save_path="weights/resnet50.pth"):
    if os.path.exists(save_path):
        print(f"✅ Found weights at {save_path}")
        return save_path

    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    print(f"⬇️ Downloading ResNet50 weights from torchvision...")
    weights = ResNet50_Weights.DEFAULT
    state_dict = weights.get_state_dict(progress=True)
    torch.save(state_dict, save_path)
    print(f"💾 Saved weights to {save_path}")
    return save_path

def load_weights_to_custom_model(model, weight_path):
    # 加载官方 state_dict
    state_dict = torch.load(weight_path, map_location="cpu", weights_only=True)

    # 严格加载（CustomResNet50 结构与官方一致，应完全匹配）
    model.load_state_dict(state_dict, strict=True)
    print("✅ Successfully loaded pretrained weights into custom ResNet50.")

parser = argparse.ArgumentParser(
        description="Generate text using a causal language model."
    )

parser.add_argument(
    "--model",
    type=str,
    required=False,
    help="Path to the model or model identifier from Hugging Face.",
)
parser.add_argument(
    "--prompts",
    type=str,
    nargs="+",
    required=False,
    help="List of prompts for text generation.",
)
parser.add_argument(
    "--device",
    type=str,
    default="cpu",
    help='Device to use for inference (e.g., "cuda", "cpu").',
)
parser.add_argument(
    "--backend",
    type=str,
    default="ninetoothed",
    help='Backend to use for inference (e.g., "ninetoothed", "triton", "torch").',
)
args = parser.parse_args()
backend = args.backend
print("Using backend:", backend)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

weights = ResNet50_Weights.DEFAULT
# model = resnet50(weights=weights).to(device)
preprocess = weights.transforms()


# 1. 下载权重（如果不存在）
weight_file = "/data-aisoft/qyq_models/resnet50/resnet50.pth"
download_resnet50_weights(weight_file)

# 2. 创建自定义模型
model = CustomResNet50(num_classes=1000)

# 3. 加载预训练权重
load_weights_to_custom_model(model, weight_file)

model = model.to(device).eval()
        
if backend != "torch":
    replace_module(model, ReLU)
    replace_module(model, MaxPool2d)
    replace_module(model, Conv2d)
    replace_module(model, AvgPool2d)

# if len(sys.argv) < 2:
#     print("Usage: python run_resnet50.py <image_path>")
#     sys.exit(1)

# img_path = sys.argv[1]
# img = Image.open(img_path).convert("RGB")
img = torch.randn(3, 224, 224, device=device)

batch = preprocess(img).unsqueeze(0).to(device)

with torch.no_grad():
    with (relu_backend(backend),
          max_pool2d_backend(backend),
          conv2d_backend(backend),
          avg_pool2d_backend(backend)
          ):
        logits = model(batch)          # shape: [1, 1000]

if backend == "ninetoothed":
    probs = ops.ninetoothed.torch.softmax(logits[0].view(1, logits[0].shape[0]))
elif backend == "torch":
    probs = torch.nn.functional.softmax(logits[0], dim=0)
    
top5_prob, top5_idx = torch.topk(probs.squeeze(0), 5)

# 加载 ImageNet 类别标签
labels = weights.meta["categories"]

for i in range(5):
    print(f"{labels[top5_idx[i]]:>20s}: {top5_prob[i].item()*100:5.2f}%")
    
    
    
    
    
    
    
    
    
    
    
    
    
    
