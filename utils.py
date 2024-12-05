import torch
import torch.nn as nn
import numpy as np

class PINN(nn.Module):
    def __init__(self):
        super(PINN, self).__init__()
        self.layers1 = nn.Linear(1, 30)
        self.tanh1 = nn.Tanh()
        self.layers2 = nn.Linear(30, 30)
        self.tanh2 = nn.Tanh()
        self.layers3 = nn.Linear(30, 30)
        self.tanh3 = nn.Tanh()
        self.layers4 = nn.Linear(30, 1)

        nn.init.kaiming_normal_(self.layers1.weight, mode='fan_out', nonlinearity='tanh')
        nn.init.kaiming_normal_(self.layers2.weight, mode='fan_out', nonlinearity='tanh')

    def forward(self, x):
        y = self.layers1(x)
        y = self.tanh1(y)
        y = self.layers2(y)
        y = self.tanh2(y)
        y = self.layers3(y)
        y = self.tanh3(y)
        y = self.layers4(y)

        return y

class Fisher_KPP_PINN(nn.Module):
    def __init__(self):
        super(Fisher_KPP_PINN, self).__init__()
        self.layers1 = nn.Linear(2, 30)
        self.tanh1 = nn.Tanh()
        self.layers2 = nn.Linear(30, 30)
        self.tanh2 = nn.Tanh()
        self.layers3 = nn.Linear(30, 30)
        self.tanh3 = nn.Tanh()
        self.layers4 = nn.Linear(30, 1)

        nn.init.kaiming_normal_(self.layers1.weight, mode='fan_out', nonlinearity='tanh')
        nn.init.kaiming_normal_(self.layers2.weight, mode='fan_out', nonlinearity='tanh')
        nn.init.kaiming_normal_(self.layers3.weight, mode='fan_out', nonlinearity='tanh')

    def forward(self, x):
        y = self.layers1(x)
        y = self.tanh1(y)
        y = self.layers2(y)
        y = self.tanh2(y)
        y = self.layers3(y)
        y = self.tanh3(y)
        y = self.layers4(y)

        return y

class laplace_PINN(nn.Module):
    def __init__(self):
        super(laplace_PINN, self).__init__()
        self.layers1 = nn.Linear(2, 30)
        self.tanh1 = nn.Tanh()
        self.layers2 = nn.Linear(30, 30)
        self.tanh2 = nn.Tanh()
        self.layers3 = nn.Linear(30, 30)
        self.tanh3 = nn.Tanh()
        self.layers4 = nn.Linear(30, 1)

        nn.init.kaiming_normal_(self.layers1.weight, mode='fan_out', nonlinearity='tanh')
        nn.init.kaiming_normal_(self.layers2.weight, mode='fan_out', nonlinearity='tanh')
        nn.init.kaiming_normal_(self.layers3.weight, mode='fan_out', nonlinearity='tanh')

    def forward(self, x):
        y = self.layers1(x)
        y = self.tanh1(y)
        y = self.layers2(y)
        y = self.tanh2(y)
        y = self.layers3(y)
        y = self.tanh3(y)
        y = self.layers4(y)

        return y

class Allen_Cahn_PINN(nn.Module):
    def __init__(self):
        super(Allen_Cahn_PINN, self).__init__()
        self.layers1 = nn.Linear(3, 40)
        self.tanh1 = nn.Tanh()
        self.layers2 = nn.Linear(40, 40)
        self.tanh2 = nn.Tanh()
        self.layers3 = nn.Linear(40, 40)
        self.tanh3 = nn.Tanh()
        self.layers4 = nn.Linear(40, 2)

        nn.init.kaiming_normal_(self.layers1.weight, mode='fan_out', nonlinearity='tanh')
        nn.init.kaiming_normal_(self.layers2.weight, mode='fan_out', nonlinearity='tanh')
        nn.init.kaiming_normal_(self.layers3.weight, mode='fan_out', nonlinearity='tanh')

        self.mu = nn.Parameter(torch.tensor(2.0, requires_grad=True))

    def forward(self, x):
        y = self.layers1(x)
        y = self.tanh1(y)
        y = self.layers2(y)
        y = self.tanh2(y)
        y = self.layers3(y)
        y = self.tanh3(y)
        y = self.layers4(y)

        return y, self.mu
def physics_loss(model, t, device):
    t = t.to(device)
    t.requires_grad = True

    u = model(t)
    u_t = torch.autograd.grad(u, t, grad_outputs=torch.ones_like(u), create_graph=True, allow_unused=True)[0]
    u_tt = torch.autograd.grad(u_t, t, grad_outputs=torch.ones_like(u_t), create_graph=True, allow_unused=True)[0]

    f = 2 * u_tt + 3 * u_t + 6 * u - torch.cos(t)
    pinn_loss = torch.mean(f**2, dim=0)

    return pinn_loss

def physics_loss2(model, x, t, device):
    t = t.to(device)
    x = x.to(device)

    t.requires_grad = True
    x.requires_grad = True

    input = torch.cat([x, t], dim=1)
    u = model(input.to(device))

    u_t = torch.autograd.grad(u, t, grad_outputs=torch.ones_like(u), create_graph=True, allow_unused=True)[0]
    u_x = torch.autograd.grad(u, x, grad_outputs=torch.ones_like(u), create_graph=True, allow_unused=True)[0]
    u_xx = torch.autograd.grad(u_x, x, grad_outputs=torch.ones_like(u_x), create_graph=True, allow_unused=True)[0]

    f = 0.01 * u_xx + u * (1 - u) - u_t
    pinn_loss = torch.mean(f**2, dim=0)

    return pinn_loss

def laplace_physics_loss(model, x, y, device):
    x = x.to(device)
    y = y.to(device)

    x.requires_grad = True
    y.requires_grad = True

    input = torch.cat([x, y], dim=1)
    u = model(input.to(device))

    u_x = torch.autograd.grad(u, x, grad_outputs=torch.ones_like(u), create_graph=True, allow_unused=True)[0]
    u_xx = torch.autograd.grad(u_x, x, grad_outputs=torch.ones_like(u_x), create_graph=True, allow_unused=True)[0]

    u_y = torch.autograd.grad(u, y, grad_outputs=torch.ones_like(u), create_graph=True, allow_unused=True)[0]
    u_yy = torch.autograd.grad(u_y, y, grad_outputs=torch.ones_like(u_y), create_graph=True, allow_unused=True)[0]

    f = u_xx + u_yy
    pinn_loss = torch.mean(f**2, dim=0)

    return pinn_loss

def Allen_Cahn_loss(model, x, y, t, device):
    x = x.to(device)
    y = y.to(device)
    t = t.to(device)

    x.requires_grad = True
    y.requires_grad = True
    t.requires_grad = True

    input = torch.cat([x, y, t], dim=1)
    output, pred_mu = model(input.to(device))
    u = output[:, 0].reshape(-1, 1).to(device)
    f = output[:, 1].reshape(-1, 1).to(device)

    u_x = torch.autograd.grad(u, x, grad_outputs=torch.ones_like(u), create_graph=True, allow_unused=True)[0]
    u_xx = torch.autograd.grad(u_x, x, grad_outputs=torch.ones_like(u_x), create_graph=True, allow_unused=True)[0]

    u_y = torch.autograd.grad(u, y, grad_outputs=torch.ones_like(u), create_graph=True, allow_unused=True)[0]
    u_yy = torch.autograd.grad(u_y, y, grad_outputs=torch.ones_like(u_y), create_graph=True, allow_unused=True)[0]

    u_t = torch.autograd.grad(u, t, grad_outputs=torch.ones_like(u), create_graph=True, allow_unused=True)[0]

    ff = pred_mu * (u_xx + u_yy) - f - u_t
    pinn_loss = torch.mean(ff**2, dim=0)

    return pinn_loss

def constranint_cond(model, x, device):
    u = model(x.to(device))
    xx = x[:,0].reshape(-1, 1).to(device)

    f = u - torch.exp(-1 * (xx**2))
    constranint_loss = torch.mean(f ** 2, dim=0)

    return constranint_loss

def laplace_initial_cond(model, X_bottom, X_top, X_left, X_right, device):
    u_bottom = model(X_bottom.to(device))
    u_top = model(X_top.to(device))
    u_left = model(X_left.to(device))
    u_right = model(X_right.to(device))

    f_bottom = u_bottom **2
    f_top = (u_top - 1) **2
    f_left = u_left ** 2
    f_right = u_right ** 2

    f = f_bottom + f_top + f_left + f_right

    constranint_loss = torch.mean(f, dim=0)

    return constranint_loss
