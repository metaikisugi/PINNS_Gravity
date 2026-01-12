import torch
import torch.nn as nn
import numpy as np

# Reproduce the exact setup
torch.manual_seed(42)

class SimpleModel(nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self, t):
        angle = 2 * np.pi * t
        x = torch.cos(angle)
        y = torch.sin(angle)
        # return torch.cat([x, y], dim=1) 
        # ↑ ここでまとめると微分時に分離が面倒なので、計算時はバラで扱います
        return x, y 

# Test Point
t = torch.tensor([[0.25]], dtype=torch.float32, requires_grad=True)

model = SimpleModel()
x, y = model(t) # バラで受け取る

print(f"t: {t.item()}")
print(f"x: {x.item()} (Expected: 0)")
print(f"y: {y.item()} (Expected: 1)")

# --- First Derivative (修正ポイント) ---
# xとyを個別にtで微分します
# .sum()を使うことでスカラーとして微分し、grad_outputsの明示を避けます（形状不一致の警告を防ぐため）
dx_dt = torch.autograd.grad(x.sum(), t, create_graph=True)[0]
dy_dt = torch.autograd.grad(y.sum(), t, create_graph=True)[0]

print(f"dx/dt: {dx_dt.item():.4f} (Expected: -6.28)")
print(f"dy/dt: {dy_dt.item():.4f} (Expected: 0)")

# --- Second Derivative ---
# 上で求めた1階微分をさらにtで微分します
d2x_dt2 = torch.autograd.grad(dx_dt.sum(), t, create_graph=True)[0]
d2y_dt2 = torch.autograd.grad(dy_dt.sum(), t, create_graph=True)[0]

print(f"d2x/dt2: {d2x_dt2.item():.4f} (Expected: 0)")
print(f"d2y/dt2: {d2y_dt2.item():.4f} (Expected: -39.48)")

# --- Physics Check ---
r = torch.sqrt(x**2 + y**2)
Pi_target = (2 * np.pi)**2 # 約 39.478

# Equation: d2r/dt2 + Pi * r / |r|^3 = 0
# 成分ごとに: d2x + Pi * x / r^3
res_x = d2x_dt2 + Pi_target * x / (r**3)
res_y = d2y_dt2 + Pi_target * y / (r**3)

print("-" * 20)
print(f"Residual X: {res_x.item():.6f}") # ほぼ0になるはず
print(f"Residual Y: {res_y.item():.6f}") # ほぼ0になるはず
print(f"Physics Loss: {(res_x**2 + res_y**2).item():.6f}")
