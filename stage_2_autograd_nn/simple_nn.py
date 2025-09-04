import torch
import torch.nn as nn

print("=== Stage 2: Autograd & Simple NN ===")
device = "cuda" if torch.cuda.is_available() else "cpu"

# 自动求导示例
x = torch.tensor([2.0, 3.0], requires_grad=True, device=device)
y = x**2 + 3 * x
y.sum().backward()
print("Gradient:", x.grad)

# 简单神经网络
model = nn.Sequential(nn.Linear(5, 3), nn.ReLU(), nn.Linear(3, 1)).to(device)

sample_input = torch.randn(10, 5, device=device)
output = model(sample_input)
print("NN output:\n", output)
