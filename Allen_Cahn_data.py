import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from utils import Allen_Cahn_PINN, Allen_Cahn_loss
from tqdm import tqdm
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.optim import Adam
import matplotlib.pyplot as plt

np.random.seed(1)
torch.manual_seed(1)

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
file = pd.read_csv("Allen_Cahn_data_train.csv")
test_file = pd.read_csv("Allen_Cahn_data.csv")

x = file['x'].values
y = file['y'].values
t = file['t'].values
u = file['u'].values
f = file['f'].values

test_x = test_file['x'].values
test_y = test_file['y'].values
test_t = test_file['t'].values
test_u = test_file['u'].values
test_f = test_file['f'].values

# MSELoss에 대한 데이터
inputs = np.concatenate((x.reshape(-1,1), y.reshape(-1,1), t.reshape(-1,1)), axis=1)
inputs = torch.tensor(inputs, dtype=torch.float32)
targets = np.concatenate((u.reshape(-1,1), f.reshape(-1,1)), axis=1)
targets = torch.tensor(targets, dtype=torch.float32)

# TestLoss에 대한 데이터
test_inputs = np.concatenate((test_x.reshape(-1,1), test_y.reshape(-1,1), test_t.reshape(-1,1)), axis=1)
test_inputs = torch.tensor(test_inputs, dtype=torch.float32)
test_targets = np.concatenate((test_u.reshape(-1,1), test_f.reshape(-1,1)), axis=1)
test_targets = torch.tensor(test_targets, dtype=torch.float32)

# 미분방정식 조건에 대한 데이터
all_x = np.linspace(0, 20, 20)
all_y = np.linspace(0, 20, 20)
all_t = np.linspace(0, 700, 70)

x_grid, y_grid, t_grid = np.meshgrid(all_x, all_y, all_t)
mesh_inputs = np.concatenate((x_grid.ravel().reshape(-1,1), y_grid.ravel().reshape(-1,1), t_grid.ravel().reshape(-1,1)), axis=1)
mesh_inputs = torch.tensor(mesh_inputs, dtype=torch.float32)

num_epochs = 100000
patience = 300

best_loss = float('inf')
best_epoch, best_LR = 0, 0
model = Allen_Cahn_PINN().to(device)
optimizer = Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=1e-3)
scheduler = ReduceLROnPlateau(optimizer, 'min', patience=patience)
loss_graph = []

for epoch in tqdm(range(num_epochs)):
    model.train()
    total_train_loss = 0

    optimizer.zero_grad()
    outputs, mu = model(inputs.to(device))
    targets = targets.to(device)

    mseloss = nn.MSELoss()(outputs, targets)
    mesh_x = mesh_inputs[:, 0].reshape(-1, 1).to(device)
    mesh_y = mesh_inputs[:, 1].reshape(-1, 1).to(device)
    mesh_t = mesh_inputs[:, 2].reshape(-1, 1).to(device)
    pinn_loss = Allen_Cahn_loss(model, mesh_x, mesh_y, mesh_t, device)

    loss = mseloss + pinn_loss

    loss.backward()
    optimizer.step()

    model.eval()
    with torch.no_grad():
        test_loss = 0

        test_outputs, test_mu = model(test_inputs.to(device))
        test_targets = test_targets.to(device)

        test_loss = torch.sqrt(nn.MSELoss()(test_outputs, test_targets)).item()
        loss_graph.append(test_loss)

        if test_loss < best_loss:
            best_loss = test_loss

            torch.save(model.state_dict(), 'best_Allen_Cahn_model.pth')
            best_Epoch, best_LR = epoch + 1, optimizer.param_groups[0]['lr']

            torch.save(model, 'Allen_Cahn_model.pth')

        scheduler.step(test_loss)

        if epoch == num_epochs - 1:
            print(model.mu.item())

print(f'\nTest Best Loss: {best_loss:.4f}, Epoch: {best_Epoch}, LR:{best_LR:.3e}\n')

plt.plot(loss_graph)
plt.show()