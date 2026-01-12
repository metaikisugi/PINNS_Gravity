
import torch
import torch.nn as nn
import numpy as np

# Reproduce the exact setup
torch.manual_seed(42)

# Define a simple model that approximates x = cos(2*pi*t)
class SimpleModel(nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self, t):
        # Emulate the perfect learned model
        angle = 2 * np.pi * t
        x = torch.cos(angle)
        y = torch.sin(angle)
        return torch.cat([x, y], dim=1)

# Test Point
t = torch.tensor([[0.25]], dtype=torch.float32, requires_grad=True)

model = SimpleModel()
output = model(t)
x = output[:, 0:1]
y = output[:, 1:2]

print(f"t: {t.item()}")
print(f"x: {x.item()} (Expected: cos(pi/2)=0)")
print(f"y: {y.item()} (Expected: sin(pi/2)=1)")

# First Derivative
grads = torch.autograd.grad(output, t, grad_outputs=torch.ones_like(output), create_graph=True)[0]
dx_dt = grads[:, 0:1]
dy_dt = grads[:, 1:2]

print(f"dx/dt: {dx_dt.item()} (Expected: -2pi sin(pi/2) = -6.28)")
print(f"dy/dt: {dy_dt.item()} (Expected: 2pi cos(pi/2) = 0)")

# Second Derivative
grads2_x = torch.autograd.grad(dx_dt, t, grad_outputs=torch.ones_like(dx_dt), create_graph=True)[0]
d2x_dt2 = grads2_x[:, 0:1]

grads2_y = torch.autograd.grad(dy_dt, t, grad_outputs=torch.ones_like(dy_dt), create_graph=True)[0]
d2y_dt2 = grads2_y[:, 0:1]

print(f"d2x/dt2: {d2x_dt2.item()} (Expected: -4pi^2 cos(pi/2) = 0)")
print(f"d2y/dt2: {d2y_dt2.item()} (Expected: -4pi^2 sin(pi/2) = -39.48)")

# Physics Check
r = torch.sqrt(x**2 + y**2) # Should be 1
Pi_target = 39.478

# Equation: d2x + Pi * x / r^3
res_x = d2x_dt2 + Pi_target * x / (r**3)
res_y = d2y_dt2 + Pi_target * y / (r**3)

print(f"Residual X: {res_x.item()}")
print(f"Residual Y: {res_y.item()}")
print(f"Physics Loss: {res_x.item()**2 + res_y.item()**2}")
