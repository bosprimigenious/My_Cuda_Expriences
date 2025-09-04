import torch
import torch.nn as nn
import torch.optim as optim

device = "cuda" if torch.cuda.is_available() else "cpu"
print("=== Stage 3: Linear Regression on GPU ===")

# 生成数据
X = torch.randn(100, 1, device=device)
Y = 3 * X + 2 + 0.1 * torch.randn(100, 1, device=device)

# 模型
model = nn.Linear(1, 1).to(device)
criterion = nn.MSELoss()
optimizer = optim.SGD(model.parameters(), lr=0.1)

# 训练
for epoch in range(100):
    optimizer.zero_grad()
    y_pred = model(X)
    loss = criterion(y_pred, Y)
    loss.backward()
    optimizer.step()
    if epoch % 20 == 0:
        print(f"Epoch {epoch}: loss={loss.item():.4f}")

print("Trained weight:", model.weight.item(), "bias:", model.bias.item())
