import torch

print("=== Stage 1: Tensor Operations & GPU Test ===")

# 检查 GPU
print("CUDA available:", torch.cuda.is_available())
device = "cuda" if torch.cuda.is_available() else "cpu"
print("Using device:", device)

# 张量创建与操作
x_cpu = torch.randn(3, 3)
x_gpu = x_cpu.to(device)

print("CPU tensor:\n", x_cpu)
print("GPU tensor:\n", x_gpu)

# 基本运算
y = x_gpu * 2 + 1
print("Computation result on GPU:\n", y)
