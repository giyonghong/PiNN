import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from utils import laplace_PINN, laplace_initial_cond, laplace_physics_loss
from tqdm import tqdm
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.optim import Adam
import matplotlib.pyplot as plt

np.random.seed(0)
torch.manual_seed(0)

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
file = pd.read_csv("laplace_train.csv")
test_file = pd.read_csv("laplace.csv")

x = file['x'].values
y = file['y'].values
u = file['u'].values

test_x = test_file['x'].values
test_y = test_file['y'].values
test_u = test_file['u'].values

# MSELoss에 대한 데이터
inputs = np.concatenate((x.reshape(-1,1), y.reshape(-1,1)), axis=1)
inputs = torch.tensor(inputs, dtype=torch.float32)
targets = torch.tensor(u.reshape(-1,1), dtype=torch.float32)

# TestLoss에 대한 데이터
test_inputs = np.concatenate((test_x.reshape(-1,1), test_y.reshape(-1,1)), axis=1)
test_inputs = torch.tensor(test_inputs, dtype=torch.float32)
test_targets = torch.tensor(test_u.reshape(-1,1), dtype=torch.float32)

# u(x, 0) = 0
x_bottom = np.linspace(0, 1, 50)
y_bottom = np.zeros_like(x_bottom)
bottom_inputs = np.concatenate((x_bottom.reshape(-1,1), y_bottom.reshape(-1,1)), axis=1)
bottom_inputs = torch.tensor(bottom_inputs, dtype=torch.float32)

# u(x, 0) = 1
x_top = np.linspace(0, 1, 50)
y_top = np.ones_like(x_top)
top_inputs = np.concatenate((x_top.reshape(-1,1), y_top.reshape(-1,1)), axis=1)
top_inputs = torch.tensor(top_inputs, dtype=torch.float32)

# u(0, y) = 0
y_left = np.linspace(0, 1, 50)
x_left = np.zeros_like(y_left)
left_inputs = np.concatenate((x_left.reshape(-1,1), y_left.reshape(-1,1)), axis=1)
left_inputs = torch.tensor(left_inputs, dtype=torch.float32)

# u(1, y) = 0
y_right = np.linspace(0, 1, 50)
x_right = np.ones_like(y_left)
right_inputs = np.concatenate((x_right.reshape(-1,1), y_right.reshape(-1,1)), axis=1)
right_inputs = torch.tensor(right_inputs, dtype=torch.float32)

# 미분방정식 조건에 대한 데이터
all_x = np.linspace(0, 1, 50)
all_y = np.linspace(0, 1, 50)
x_grid, y_grid = np.meshgrid(all_x, all_y)
mesh_inputs = np.concatenate((x_grid.ravel().reshape(-1,1), y_grid.ravel().reshape(-1,1)), axis=1)
mesh_inputs = torch.tensor(mesh_inputs, dtype=torch.float32)

num_epochs = 100000
patience = 300

best_loss = float('inf')
best_epoch, best_LR = 0, 0
model = laplace_PINN().to(device)
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
    constraintloss = laplace_initial_cond(model, bottom_inputs, top_inputs, left_inputs, right_inputs, device)

    mesh_x = mesh_inputs[:, 0].reshape(-1, 1).to(device)
    mesh_y = mesh_inputs[:, 1].reshape(-1, 1).to(device)
    pinn_loss = laplace_physics_loss(model, mesh_x, mesh_y, device)

    loss = mseloss + constraintloss + pinn_loss

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

            #torch.save(model.state_dict(), 'best_laplace_model.pth')
            torch.save(model.state_dict(), 'best_ref_laplace_model.pth')
            best_Epoch, best_LR = epoch + 1, optimizer.param_groups[0]['lr']

            #torch.save(model, 'laplace_model.pth')
            torch.save(model, 'ref_laplace_model.pth')

        scheduler.step(test_loss)

print(f'\nTest Best Loss: {best_loss:.4f}, Epoch: {best_Epoch}, LR:{best_LR:.3e}\n')

plt.plot(loss_graph)
plt.show()

