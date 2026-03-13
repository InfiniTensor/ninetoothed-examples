#!/usr/bin/env python3
"""
run_resnet50.py
一键跑通 ResNet-50 预训练模型推理 + 性能测试对比
"""

import torch
import torch.nn as nn
import torchvision.transforms as T
from torchvision.models import resnet50, ResNet50_Weights
from PIL import Image
import os
import time
import argparse
from collections import defaultdict
import matplotlib.pyplot as plt
import numpy as np

import ops.ninetoothed.torch
from utils import replace_module
from relu import ReLU, relu_backend
from softmax import Softmax, softmax_backend
from max_pool2d import MaxPool2d, max_pool2d_backend
from conv2d import Conv2d, conv2d_backend
from avg_pool2d import AvgPool2d, avg_pool2d_backend

torch.manual_seed(42)

# ============== 模型定义部分（保持不变）=============
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
        out = self.conv1(x); out = self.bn1(out); out = self.relu(out)
        out = self.conv2(out); out = self.bn2(out); out = self.relu(out)
        out = self.conv3(out); out = self.bn3(out)
        if self.downsample is not None:
            identity = self.downsample(x)
        out += identity
        out = self.relu(out)
        return out

class CustomResNet50(nn.Module):
    def __init__(self, num_classes=1000, zero_init_residual=False):
        super().__init__()
        self.in_channels = 64
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(Bottleneck, 64, 3)
        self.layer2 = self._make_layer(Bottleneck, 128, 4, stride=2)
        self.layer3 = self._make_layer(Bottleneck, 256, 6, stride=2)
        self.layer4 = self._make_layer(Bottleneck, 512, 3, stride=2)
        self.avgpool = nn.AvgPool2d(kernel_size=7, stride=7)
        self.fc = nn.Linear(512 * Bottleneck.expansion, num_classes)
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1); nn.init.constant_(m.bias, 0)
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
        layers = [block(self.in_channels, out_channels, stride, downsample)]
        self.in_channels = out_channels * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.in_channels, out_channels))
        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x); x = self.bn1(x); x = self.relu(x); x = self.maxpool(x)
        x = self.layer1(x); x = self.layer2(x); x = self.layer3(x); x = self.layer4(x)
        x = self.avgpool(x); x = torch.flatten(x, 1); x = self.fc(x)
        return x

# ============== 工具函数 ==============
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
    state_dict = torch.load(weight_path, map_location="cpu", weights_only=True)
    model.load_state_dict(state_dict, strict=True)
    print("✅ Successfully loaded pretrained weights into custom ResNet50.")

def create_model(backend="torch", device="cuda"):
    """创建并初始化模型"""
    model = CustomResNet50(num_classes=1000)
    weight_file = "/data-aisoft/qyq_models/resnet50/resnet50.pth"
    download_resnet50_weights(weight_file)
    load_weights_to_custom_model(model, weight_file)
    model = model.to(device).eval()
    
    if backend != "torch":
        replace_module(model, ReLU)
        replace_module(model, MaxPool2d)
        replace_module(model, Conv2d)
        replace_module(model, AvgPool2d)
    return model

def prepare_input(batch_size, device, weights):
    """准备输入数据"""
    # 使用随机张量避免IO瓶颈，更纯粹测试计算性能
    return torch.randn(batch_size, 3, 224, 224, device=device)

def benchmark_inference(model, input_tensor, backend, device, 
                       num_warmup=10, num_runs=100, use_cuda_event=True):
    """
    基准测试推理性能
    返回: dict containing latency stats (ms)
    """
    latencies = []
    
    # Warmup
    print(f"🔥 Warming up ({num_warmup} iterations)...")
    model.eval()
    with torch.no_grad():
        for _ in range(num_warmup):
            if backend == "ninetoothed":
                with (relu_backend(backend), max_pool2d_backend(backend),
                      conv2d_backend(backend), avg_pool2d_backend(backend)):
                    _ = model(input_tensor)
            else:
                _ = model(input_tensor)
    
    if device.type == "cuda" and use_cuda_event:
        # 使用 CUDA Event 精确计时（排除 CPU-GPU 同步开销）
        start_event = torch.cuda.Event(enable_timing=True)
        end_event = torch.cuda.Event(enable_timing=True)
        
        print(f"⏱️  Running benchmark ({num_runs} iterations, CUDA events)...")
        with torch.no_grad():
            for _ in range(num_runs):
                start_event.record()
                if backend == "ninetoothed":
                    with (relu_backend(backend), max_pool2d_backend(backend),
                          conv2d_backend(backend), avg_pool2d_backend(backend)):
                        _ = model(input_tensor)
                else:
                    _ = model(input_tensor)
                end_event.record()
        torch.cuda.synchronize()
        
        for _ in range(num_runs):
            # 重新运行以获取每个iter的时间（或缓存之前结果）
            pass
        # 更准确：在循环内记录每个时间
        latencies = []
        with torch.no_grad():
            for _ in range(num_runs):
                start_event.record()
                if backend == "ninetoothed":
                    with (relu_backend(backend), max_pool2d_backend(backend),
                          conv2d_backend(backend), avg_pool2d_backend(backend)):
                        _ = model(input_tensor)
                else:
                    _ = model(input_tensor)
                end_event.record()
                torch.cuda.synchronize()
                latencies.append(start_event.elapsed_time(end_event))
    else:
        # CPU 或回退方案：使用 time.time()
        print(f"⏱️  Running benchmark ({num_runs} iterations, time.time())...")
        with torch.no_grad():
            for _ in range(num_runs):
                start = time.perf_counter()
                if backend == "ninetoothed":
                    with (relu_backend(backend), max_pool2d_backend(backend),
                          conv2d_backend(backend), avg_pool2d_backend(backend)):
                        _ = model(input_tensor)
                else:
                    _ = model(input_tensor)
                if device.type == "cuda":
                    torch.cuda.synchronize()
                end = time.perf_counter()
                latencies.append((end - start) * 1000)  # ms
    
    return {
        "mean": np.mean(latencies),
        "median": np.median(latencies),
        "std": np.std(latencies),
        "min": np.min(latencies),
        "max": np.max(latencies),
        "p95": np.percentile(latencies, 95),
        "p99": np.percentile(latencies, 99),
        "raw": latencies
    }

def plot_comparison(results_dict, save_path="benchmark_comparison.png"):
    """绘制性能对比图"""
    backends = list(results_dict.keys())
    metrics = ["mean", "median", "p95", "p99"]
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle("ResNet-50 Inference Performance Comparison", fontsize=16, fontweight='bold')
    
    colors = plt.cm.Set2(np.linspace(0, 1, len(backends)))
    
    for idx, metric in enumerate(metrics):
        ax = axes[idx // 2, idx % 2]
        values = [results_dict[b][metric] for b in backends]
        bars = ax.bar(backends, values, color=colors[:len(backends)], edgecolor='black')
        
        # 添加数值标签
        for bar in bars:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                   f'{height:.2f}', ha='center', va='bottom', fontsize=9)
        
        ax.set_ylabel("Latency (ms)")
        ax.set_title(f"{metric.upper()} Latency")
        ax.grid(axis='y', alpha=0.3, linestyle='--')
        ax.set_axisbelow(True)
    
    # 添加吞吐量对比子图
    ax_thr = plt.subplot2grid((3, 2), (2, 0), colspan=2)
    batch_size = list(results_dict.values())[0].get("batch_size", 1)
    throughputs = [(1000 / results_dict[b]["mean"]) * batch_size for b in backends]  # images/sec
    bars = ax_thr.bar(backends, throughputs, color=colors[:len(backends)], edgecolor='black')
    for bar in bars:
        height = bar.get_height()
        ax_thr.text(bar.get_x() + bar.get_width()/2., height,
                   f'{height:.1f}', ha='center', va='bottom', fontsize=9)
    ax_thr.set_ylabel("Throughput (images/sec)")
    ax_thr.set_title(f"Throughput @ batch_size={batch_size}")
    ax_thr.grid(axis='y', alpha=0.3, linestyle='--')
    ax_thr.set_axisbelow(True)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"📊 Saved comparison plot to {save_path}")
    plt.show()

def print_summary(results_dict, batch_size):
    """打印性能摘要"""
    print("\n" + "="*70)
    print("📈 PERFORMANCE SUMMARY")
    print("="*70)
    print(f"Batch Size: {batch_size} | Input: 3x224x224")
    print("-"*70)
    print(f"{'Backend':<15} {'Mean(ms)':>10} {'Median(ms)':>12} {'P95(ms)':>10} {'Throughput(img/s)':>18}")
    print("-"*70)
    
    for backend, stats in results_dict.items():
        mean, median, p95 = stats["mean"], stats["median"], stats["p95"]
        throughput = (1000 / mean) * batch_size
        print(f"{backend:<15} {mean:>10.2f} {median:>12.2f} {p95:>10.2f} {throughput:>18.1f}")
    
    # 计算加速比
    if "torch" in results_dict and "ninetoothed" in results_dict:
        torch_mean = results_dict["torch"]["mean"]
        ninetoothed_mean = results_dict["ninetoothed"]["mean"]
        speedup = torch_mean / ninetoothed_mean
        print("-"*70)
        if speedup > 1:
            print(f"🚀 ninetoothed is {speedup:.2f}x faster than PyTorch (mean latency)")
        else:
            print(f"🐌 ninetoothed is {1/speedup:.2f}x slower than PyTorch (mean latency)")
    print("="*70 + "\n")

# ============== 主函数 ==============
def main():
    parser = argparse.ArgumentParser(description="ResNet-50 Performance Benchmark")
    parser.add_argument("--backends", type=str, nargs="+", default=["torch", "ninetoothed"],
                       help="Backends to test: torch, ninetoothed")
    parser.add_argument("--batch-sizes", type=int, nargs="+", default=[1],
                       help="Batch sizes to test")
    parser.add_argument("--warmup", type=int, default=10, help="Warmup iterations")
    parser.add_argument("--runs", type=int, default=100, help="Benchmark iterations")
    parser.add_argument("--device", type=str, default=None, help="Device (cuda/cpu)")
    parser.add_argument("--output", type=str, default="benchmark_results.png",
                       help="Output plot path")
    args = parser.parse_args()
    
    device = torch.device(args.device if args.device else 
                         ("cuda" if torch.cuda.is_available() else "cpu"))
    print(f"🔧 Using device: {device}")
    
    weights = ResNet50_Weights.DEFAULT
    results = {}
    
    for batch_size in args.batch_sizes:
        print(f"\n🧪 Testing batch_size={batch_size}")
        for backend in args.backends:
            print(f"\n▶️  Backend: {backend}")
            
            # 为每个配置创建独立模型实例，避免状态污染
            model = create_model(backend=backend, device=device)
            input_tensor = prepare_input(batch_size, device, weights)
            
            stats = benchmark_inference(
                model=model,
                input_tensor=input_tensor,
                backend=backend,
                device=device,
                num_warmup=args.warmup,
                num_runs=args.runs
            )
            stats["batch_size"] = batch_size  # 记录batch size用于绘图
            
            key = f"{backend}_bs{batch_size}" if len(args.batch_sizes) > 1 else backend
            results[key] = stats
            
            print(f"   ✓ Mean: {stats['mean']:.2f}ms | Median: {stats['median']:.2f}ms | P95: {stats['p95']:.2f}ms")
            
            # 清理内存
            del model, input_tensor
            if device.type == "cuda":
                torch.cuda.empty_cache()
    
    # 打印摘要 & 绘图
    print_summary(results, batch_size=args.batch_sizes[0])
    plot_comparison(results, save_path=args.output)
    
    # 可选：保存原始数据
    import json
    with open("benchmark_raw.json", "w") as f:
        # 序列化时排除raw列表避免文件过大
        serializable = {k: {kk: vv for kk, vv in v.items() if kk != "raw"} 
                       for k, v in results.items()}
        json.dump(serializable, f, indent=2)
    print("💾 Raw stats saved to benchmark_raw.json")

if __name__ == "__main__":
    main()