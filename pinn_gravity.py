
import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt

# Set random seed for reproducibility
torch.manual_seed(42)
np.random.seed(42)

# ==========================================
# 1. Physical Constants & Data Generation
# ==========================================
# Real SI Units
G_real = 6.67430e-11 # m^3 kg^-1 s^-2 (Target)
M_sun = 1.989e30     # kg
AU = 1.496e11        # m
Year = 3.15576e7     # s (Julian Year)

# Scaling Factors (Nondimensionalization)
L_scale = AU          # Length scale
T_scale = Year        # Time scale
M_scale = M_sun       # Mass scale (just for reference)

# Normalized True Parameter
# Physics: d2r/dt2 = -GM/r^3 * r
# Normalized: (L/T^2) d2r_hat/dt_hat2 = -G * M / (L^3 r_hat^3) * L r_hat
# d2r_hat/dt_hat2 = - (G * M * T^2 / L^3) * r_hat / r_hat^3
# Let Pi = G * M * T^2 / L^3. This is what the network learns.
Pi_true = G_real * M_sun * (T_scale**2) / (L_scale**3)
print(f"Target Dimensionless Parameter Pi: {Pi_true:.5f} (Should be approx 4pi^2 = {4*np.pi**2:.5f})")

def generate_data(n_points=100):
    # Earth orbit approx: circular, r = 1 AU, period = 1 Year.
    # Angle theta = 2pi * t / Period
    # t_real goes from 0 to Year
    t_real = np.linspace(0, Year, n_points)
    
    # Normalized time t_hat goes from 0 to 1
    t_hat = t_real / T_scale 
    
    # Angle
    theta = 2 * np.pi * t_hat
    
    # Position (Normalized)
    # x_hat = cos(theta), y_hat = sin(theta) -> r_hat = 1
    x_hat = np.cos(theta)
    y_hat = np.sin(theta)
    
    return t_hat, x_hat, y_hat

# Generate Data
t_train, x_train, y_train = generate_data(100)
t_tensor = torch.tensor(t_train, dtype=torch.float32).view(-1, 1).requires_grad_(True)
x_tensor = torch.tensor(x_train, dtype=torch.float32).view(-1, 1)
y_tensor = torch.tensor(y_train, dtype=torch.float32).view(-1, 1)

# ==========================================
# 2. PINN Model
# ==========================================
class PINN(nn.Module):
    def __init__(self):
        super().__init__()
        # Input: Normalized Time (0 to 1) -> Output: Normalized Pos (-1 to 1)
        self.net = nn.Sequential(
            nn.Linear(1, 64),
            nn.Tanh(),
            nn.Linear(64, 64),
            nn.Tanh(),
            nn.Linear(64, 64),
            nn.Tanh(),
            nn.Linear(64, 2)
        )
        
        # Learnable Dimensionless Parameter Pi
        # Initialize near 4pi^2 (~39.4) or random?
        # Let's initialize at 20.0 to show convergence
        self.Pi = nn.Parameter(torch.tensor([20.0], dtype=torch.float32))

    def forward(self, t):
        return self.net(t)

model = PINN()

# ==========================================
# 3. Training Loop (Freeze Strategy)
# ==========================================
# Phase 1: Learn Orbit (Data Only)
# Phase 2: Learn G (Physics Only, Frozen Net)
epochs_phase1 = 10000 # Increase to ensure perfect circle
epochs_phase2 = 10000 

loss_history = []
Pi_history = []

print("--- Phase 1: Learning Orbit Shape (Data Only) ---")
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

for epoch in range(epochs_phase1):
    optimizer.zero_grad()
    output = model(t_tensor)
    loss = torch.mean((output[:, 0:1] - x_tensor)**2 + (output[:, 1:2] - y_tensor)**2)
    loss.backward()
    optimizer.step()
    
    loss_history.append(loss.item())
    Pi_history.append(model.Pi.item())
    
    if epoch % 1000 == 0:
        print(f"Epoch {epoch}: DataLoss={loss.item():.8f}")

print("--- Phase 2: Deriving G (Physics Only, Frozen Network) ---")
# Freeze Network
for param in model.net.parameters():
    param.requires_grad = False

# Optimize Pi only
optimizer_Pi = torch.optim.Adam([model.Pi], lr=1e-2) # Higher LR for single param

for epoch in range(epochs_phase2):
    optimizer_Pi.zero_grad()
    
    # Physics Collocation Points (Normalized time 0 to 1)
    t_phys = torch.rand(2000, 1).requires_grad_(True)
    
    output_phys = model(t_phys) # Network is frozen
    x_phys = output_phys[:, 0:1]
    y_phys = output_phys[:, 1:2]
    
    # Correctly compute derivatives for x and y separately
    # Use .sum() to differentiate as scalar, identical to debug_derivatives.py fix
    dx_dt = torch.autograd.grad(x_phys.sum(), t_phys, create_graph=True)[0]
    dy_dt = torch.autograd.grad(y_phys.sum(), t_phys, create_graph=True)[0]
    
    d2x_dt2 = torch.autograd.grad(dx_dt.sum(), t_phys, create_graph=True)[0]
    d2y_dt2 = torch.autograd.grad(dy_dt.sum(), t_phys, create_graph=True)[0]
    
    r_phys = torch.sqrt(x_phys**2 + y_phys**2)
    
    # Normalized Physics Equation:
    # d2x/dt2 + Pi * x / r^3 = 0
    # Avoid zero div? r should be ~1
    res_x = d2x_dt2 + model.Pi * x_phys / (r_phys**3)
    res_y = d2y_dt2 + model.Pi * y_phys / (r_phys**3)
    
    loss_phys = torch.mean(res_x**2 + res_y**2)
    loss_phys.backward()
    optimizer_Pi.step()
    
    loss_history.append(loss_phys.item())
    Pi_history.append(model.Pi.item())
    
    if epoch % 1000 == 0:
        print(f"Epoch {epoch+epochs_phase1}: PhysLoss={loss_phys.item():.8f}, Pi={model.Pi.item():.5f}")

# ==========================================
# 4. Analysis & Output
# ==========================================
Pi_final = model.Pi.item()
# Recover G
# Pi = G * M * T^2 / L^3  =>  G = Pi * L^3 / (M * T^2)
G_derived = Pi_final * (L_scale**3) / (M_sun * T_scale**2)

print("\n" + "="*30)
print(f"RESULTS (SI Units)")
print(f"True G    : {G_real:.5e}")
print(f"Derived G : {G_derived:.5e}")
error_percent = abs(G_derived - G_real) / G_real * 100
print(f"Error     : {error_percent:.2f}%")
print("="*30)

# Plots
plt.figure(figsize=(10, 5))
plt.subplot(1, 2, 1)
plt.plot(Pi_history)
plt.axhline(Pi_true, color='r', linestyle='--', label='True Pi')
plt.title('Convergence of Dimensionless Parameter Pi')
plt.legend()

plt.subplot(1, 2, 2)
# Reconstruct Orbit
t_plot = torch.linspace(0, 1, 100).view(-1, 1)
out_plot = model(t_plot).detach().numpy()
plt.plot(out_plot[:, 0], out_plot[:, 1], label='Model')
plt.plot(x_train, y_train, 'k.', markersize=2, label='Data')
plt.axis('equal')
plt.title('Orbit (Normalized)')
plt.legend()

plt.savefig('result_real_scale.png')
print("Saved result_real_scale.png")
