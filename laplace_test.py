import numpy as np
import pandas as pd
import torch
import matplotlib.pyplot as plt


np.random.seed(0)
torch.manual_seed(0)

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
file = pd.read_csv("laplace.csv")

#model = torch.load("laplace_model.pth")
model = torch.load("ref_laplace_model.pth")
model.eval()

x = file['x'].values
y = file['y'].values
u = file['u'].values

inputs = np.concatenate((x.reshape(-1,1), y.reshape(-1,1)), axis=1)
inputs = torch.tensor(inputs, dtype=torch.float32)
outputs = model(inputs.to(device))

x_unique = np.unique(x)
y_unique = np.unique(y)

x_grid, y_grid = np.meshgrid(x_unique, y_unique)
h_grid = u.reshape(len(x_unique), len(y_unique))
o_grid = outputs.cpu().detach().numpy().reshape(len(x_unique), len(y_unique))

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5))

contour1 = ax1.contourf(x_grid, y_grid, h_grid, levels=50)
cbar1 = fig.colorbar(contour1, ax=ax1)
cbar1.set_label('Color map', fontsize=12)

contour2 = ax2.contourf(x_grid, y_grid, o_grid, levels=50)
cbar2 = fig.colorbar(contour2, ax=ax2)
cbar2.set_label('Color map', fontsize=12)

ax1.set_title("Ground Truth", fontdict={'fontsize': 18, 'weight':'bold'})
ax1.set_xlabel("x", fontdict={'fontsize': 18, 'weight':'bold'})
ax1.set_ylabel("y", fontdict={'fontsize': 18, 'weight':'bold'})

ax2.set_title("Model Prediction", fontdict={'fontsize': 18, 'weight':'bold'})
ax2.set_xlabel("x", fontdict={'fontsize': 18, 'weight':'bold'})
ax2.set_ylabel("y", fontdict={'fontsize': 18, 'weight':'bold'})

plt.rcParams["axes.linewidth"] = 7
plt.tick_params(axis='both', direction='in',width=2, length=5,labelsize=14,pad=5)
plt.xticks(fontweight='bold')
plt.yticks(fontweight='bold')

plt.tight_layout()
plt.show()