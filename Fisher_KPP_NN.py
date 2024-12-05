import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from utils import Fisher_KPP_PINN, physics_loss2, constranint_cond
from tqdm import tqdm
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.optim import Adam
import matplotlib.pyplot as plt

np.random.seed(0)
torch.manual_seed(0)

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
file = pd.read_csv("train.csv")
test_file = pd.read_csv("ground_truth.csv")

x = file['x'].values
u = file['u'].values
t = file['t'].values

test_x = test_file['x'].values
test_u = test_file['u'].values
test_t = test_file['t'].values

x_0 = np.linspace(-7.5, 7.5, 500)
t_0 = np.zeros_like(x_0)

all_x = np.linspace(-7.5, 7.5, 100)
all_t = np.linspace(0, 10, 20)
x_grid, t_grid = np.meshgrid(all_x, all_t)

# TestLoss에 대한 데이터
test_inputs = np.concatenate((test_x.reshape(-1,1), test_t.reshape(-1,1)), axis=1)
test_inputs = torch.tensor(test_inputs, dtype=torch.float32)
test_targets = torch.tensor(test_u.reshape(-1,1), dtype=torch.float32)

# MSELoss에 대한 데이터
inputs = np.concatenate((x.reshape(-1,1), t.reshape(-1,1)), axis=1)
inputs = torch.tensor(inputs, dtype=torch.float32)
targets = torch.tensor(u.reshape(-1,1), dtype=torch.float32)

# 초기 조건에 대한 데이터
initial_inputs = np.concatenate((x_0.reshape(-1,1), t_0.reshape(-1,1)), axis=1)
initial_inputs = torch.tensor(initial_inputs, dtype=torch.float32)

# 미분방정식 조건에 대한 데이터
mesh_inputs = np.concatenate((x_grid.ravel().reshape(-1,1), t_grid.ravel().reshape(-1,1)), axis=1)
mesh_inputs = torch.tensor(mesh_inputs, dtype=torch.float32)

num_epochs = 100000
patience = 300

best_loss = float('inf')
best_epoch, best_LR = 0, 0
model = Fisher_KPP_PINN().to(device)
optimizer = Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=1e-3)
scheduler = ReduceLROnPlateau(optimizer, 'min', patience=patience)
loss_graph = []

for epoch in tqdm(range(num_epochs)):
    model.train()
    total_train_loss = 0

    optimizer.zero_grad()
    outputs = model(inputs.to(device))
    targets = targets.to(device)

    mseloss = nn.MSELoss()(outputs, targets)
    constraintloss = constranint_cond(model, initial_inputs, device)

    mesh_x = mesh_inputs[:, 0].reshape(-1, 1).to(device)
    mesh_t = mesh_inputs[:, 1].reshape(-1, 1).to(device)
    pinn_loss = physics_loss2(model, mesh_x, mesh_t, device)

    loss = mseloss

    loss.backward()
    optimizer.step()

    model.eval()
    with torch.no_grad():
        test_loss = 0

        test_outputs = model(test_inputs.to(device))
        test_targets = test_targets.to(device)

        test_loss = torch.sqrt(nn.MSELoss()(test_outputs, test_targets)).item()
        loss_graph.append(test_loss)

        if test_loss < best_loss:
            best_loss = test_loss

            torch.save(model.state_dict(), 'best_PINN_app_model.pth')
            best_Epoch, best_LR = epoch + 1, optimizer.param_groups[0]['lr']

            torch.save(model, 'PINN_app_model.pth')

        scheduler.step(test_loss)

print(f'\nTest Best Loss: {best_loss:.4f}, Epoch: {best_Epoch}, LR:{best_LR:.3e}\n')

plt.plot(loss_graph)
plt.show()

