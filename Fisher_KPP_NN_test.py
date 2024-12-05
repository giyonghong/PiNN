import numpy as np
import pandas as pd
import torch
import matplotlib.pyplot as plt


np.random.seed(0)
torch.manual_seed(0)

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
file = pd.read_csv("ground_truth.csv")

model = torch.load("PINN_app_model.pth")
#model = torch.load("ref_model.pth")
model.eval()

x = file['x'].values
u = file['u'].values
t = file['t'].values

inputs = np.concatenate((x.reshape(-1,1), t.reshape(-1,1)), axis=1)
inputs = torch.tensor(inputs, dtype=torch.float32)
outputs = model(inputs.to(device))

x_unique = np.unique(x)
t_unique = np.unique(t)

x_grid, t_grid = np.meshgrid(x_unique, t_unique)
h_grid = u.reshape(len(t_unique), len(x_unique))
o_grid = outputs.cpu().detach().numpy().reshape(len(t_unique), len(x_unique))

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5))

contour1 = ax1.contourf(t_grid, x_grid, h_grid, levels=50, cmap='coolwarm')
cbar1 = fig.colorbar(contour1, ax=ax1)
cbar1.set_label('Color map', fontsize=12)

contour2 = ax2.contourf(t_grid, x_grid, o_grid, levels=50, cmap='coolwarm')
cbar2 = fig.colorbar(contour2, ax=ax2)
cbar2.set_label('Color map', fontsize=12)

ax1.set_title("Ground Truth", fontdict={'fontsize': 18, 'weight':'bold'})
ax1.set_xlabel("time", fontdict={'fontsize': 18, 'weight':'bold'})
ax1.set_ylabel("position", fontdict={'fontsize': 18, 'weight':'bold'})

ax2.set_title("Model Prediction", fontdict={'fontsize': 18, 'weight':'bold'})
ax2.set_xlabel("time", fontdict={'fontsize': 18, 'weight':'bold'})
ax2.set_ylabel("position", fontdict={'fontsize': 18, 'weight':'bold'})

plt.rcParams["axes.linewidth"] = 7
plt.tick_params(axis='both', direction='in',width=2, length=5,labelsize=14,pad=5)
plt.xticks(fontweight='bold')
plt.yticks(fontweight='bold')

plt.tight_layout()
plt.show()