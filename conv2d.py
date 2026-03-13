from contextlib import contextmanager
import torch
import torch.nn.functional as F
import triton
import torch.nn as nn

import ops.ninetoothed.torch
import ops.triton.torch
import ntops.torch

class Conv2d(nn.Module):
    conv2d = None

    def __init__(self, other):
        super().__init__()

        self.__dict__ = other.__dict__

    def forward(self, input):
        def _pair(x):
            return x if isinstance(x, (tuple, list)) else (x, x)
        
        stride = _pair(self.stride)
        padding = _pair(self.padding)
        dilation = _pair(self.dilation)
        
        return type(self).conv2d(
            input,
            self.weight,
            self.bias,
            stride=stride,
            padding=padding,
            dilation=dilation,
            groups=self.groups,
        )
    
@contextmanager
def conv2d_backend(backend_name):
    _prev_impl = Conv2d.conv2d
    if backend_name == "ninetoothed":
        impl = ntops.torch.conv2d
    elif backend_name == "torch":
        impl = F.conv2d
    else:
        raise ValueError(f"unknown backend: `{backend_name}`")

    Conv2d.conv2d = impl

    try:
        yield
    finally:
        Conv2d.conv2d = _prev_impl

if __name__ == "__main__":
    torch.manual_seed(0)

    n, c, h, w = 2, 3, 112, 112
    k, _, r, s = 4, c, 3, 3
    dtype = torch.float16
    device = "cuda"

    input = torch.randn((n, c, h, w), dtype=dtype, device=device)
    weight = torch.randn((k, c, r, s), dtype=dtype, device=device)
    bias = torch.randn((k,), dtype=dtype, device=device)
    stride = 3
    dilation = 1
    
    ninetoothed_output = ops.ninetoothed.torch.conv2d(
        input, weight, bias=bias, stride=stride, dilation=dilation
    )
    reference_output = F.conv2d(
        input, weight, bias=bias, stride=stride, dilation=dilation
    )

    
    print(ninetoothed_output)
    print(reference_output)
    

    if torch.allclose(ninetoothed_output, reference_output, atol=0.01, rtol=0.01):
        print("✅ NineToothed and PyTorch match.")
    else:
        print("❌ NineToothed and PyTorch differ.")
    