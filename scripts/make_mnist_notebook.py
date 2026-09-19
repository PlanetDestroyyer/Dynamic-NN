import os
import json

def create_mnist_notebook():
    cells = []
    
    # 1. Imports and Dataset loading
    data_cell = """import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
from torchvision import datasets, transforms

# 1. Download and format Split-MNIST
print("Downloading and processing MNIST dataset...")
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,))
])

train_dataset = datasets.MNIST('./data', train=True, download=True, transform=transform)
test_dataset = datasets.MNIST('./data', train=False, download=True, transform=transform)

tasks = []
pairs = [(0, 1), (2, 3), (4, 5), (6, 7), (8, 9)]

for (c0, c1) in pairs:
    train_mask = (train_dataset.targets == c0) | (train_dataset.targets == c1)
    X_train = train_dataset.data[train_mask].float() / 255.0
    X_train = (X_train - 0.1307) / 0.3081
    X_train = X_train.view(-1, 28*28)
    y_train = (train_dataset.targets[train_mask] == c1).float().view(-1, 1)
    
    test_mask = (test_dataset.targets == c0) | (test_dataset.targets == c1)
    X_test = test_dataset.data[test_mask].float() / 255.0
    X_test = (X_test - 0.1307) / 0.3081
    X_test = X_test.view(-1, 28*28)
    y_test = (test_dataset.targets[test_mask] == c1).float().view(-1, 1)
    
    name = f"Digits {c0} vs {c1}"
    tasks.append((name, X_train, y_train, X_test, y_test))

print(f"Created {len(tasks)} tasks.")
"""
    cells.append({"cell_type": "code", "metadata": {}, "execution_count": None, "outputs": [], "source": [line + "\n" for line in data_cell.split('\n')]})

    # 2. PyTorchDynamicNetwork (from gpu_dynamic_nn.py)
    with open(os.path.join(os.path.dirname(__file__), '../src/dynamic_network.py'), 'r') as f:
        network_code = f.read()
        network_code = network_code.replace("import torch\nimport torch.nn as nn\nimport numpy as np\n", "")
    cells.append({"cell_type": "code", "metadata": {}, "execution_count": None, "outputs": [], "source": [line + "\n" for line in network_code.split('\n')]})

    # 3. ReplayBuffer and Helper
    replay_cell = """class ReplayBuffer:
    def __init__(self, capacity: int = 500):
        self.capacity = capacity
        self.X = None
        self.y = None

    def add_data(self, X_new: torch.Tensor, y_new: torch.Tensor, num_samples: int = 100):
        idx = torch.randperm(X_new.size(0))[:num_samples]
        X_sub = X_new[idx].clone().detach()
        y_sub = y_new[idx].clone().detach()

        if self.X is None:
            self.X = X_sub
            self.y = y_sub
        else:
            self.X = torch.cat([self.X, X_sub], dim=0)
            self.y = torch.cat([self.y, y_sub], dim=0)

            if self.X.size(0) > self.capacity:
                keep = torch.randperm(self.X.size(0))[:self.capacity]
                self.X = self.X[keep]
                self.y = self.y[keep]

    def sample(self, batch_size: int = 64) -> tuple:
        if self.X is None:
            return None, None
        size = self.X.size(0)
        idx = torch.randint(0, size, (min(batch_size, size),))
        return self.X[idx], self.y[idx]

    def has_data(self) -> bool:
        return self.X is not None and self.X.size(0) > 0

def _reset_adam_for_neuron(optimizer: torch.optim.Adam, net: nn.Module, new_idx: int):
    if net.W in optimizer.state:
        s_W = optimizer.state[net.W]
        if 'exp_avg' in s_W:
            s_W['exp_avg'][new_idx, :] = 0.0
            s_W['exp_avg'][:, new_idx] = 0.0
            s_W['exp_avg_sq'][new_idx, :] = 0.0
            s_W['exp_avg_sq'][:, new_idx] = 0.0
            
    if net.b in optimizer.state:
        s_b = optimizer.state[net.b]
        if 'exp_avg' in s_b:
            s_b['exp_avg'][new_idx] = 0.0
            s_b['exp_avg_sq'][new_idx] = 0.0
"""
    cells.append({"cell_type": "code", "metadata": {}, "execution_count": None, "outputs": [], "source": [line + "\n" for line in replay_cell.split('\n')]})

    # 4. Baseline Training
    baseline_cell = """device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Running on {device}...")

class BaselineNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(784, 100),
            nn.ReLU(),
            nn.Linear(100, 1)
        )
    def forward(self, x):
        return torch.sigmoid(self.net(x))

print("\\n" + "="*50)
print("--- Training Baseline (Standard Neural Network) ---")
print("="*50)

baseline_net = BaselineNet().to(device)
optimizer_b = torch.optim.Adam(baseline_net.parameters(), lr=1e-3)
criterion_b = nn.BCELoss()

baseline_accuracies = {}

for task_id, (name, X_train, y_train, X_test, y_test) in enumerate(tasks):
    print(f"\\nTraining Baseline on {name}...")
    
    dataset = torch.utils.data.TensorDataset(X_train, y_train)
    loader = torch.utils.data.DataLoader(dataset, batch_size=256, shuffle=True)
    
    for epoch in range(5): 
        baseline_net.train()
        for bx, by in loader:
            bx, by = bx.to(device), by.to(device)
            optimizer_b.zero_grad()
            loss = criterion_b(baseline_net(bx), by)
            loss.backward()
            optimizer_b.step()
            
    baseline_net.eval()
    print(f"Accuracy after {name}:")
    baseline_accuracies[name] = []
    with torch.no_grad():
        for eval_id in range(task_id + 1):
            t_name, _, _, t_X_test, t_y_test = tasks[eval_id]
            t_X_test_d = t_X_test.to(device)
            t_y_test_d = t_y_test.to(device)
            acc = ((baseline_net(t_X_test_d) > 0.5).float() == t_y_test_d).float().mean().item()
            baseline_accuracies[name].append((t_name, acc))
            print(f"  {t_name}: {acc*100:.1f}%")
"""
    cells.append({"cell_type": "code", "metadata": {}, "execution_count": None, "outputs": [], "source": [line + "\n" for line in baseline_cell.split('\n')]})

    # 5. Dynamic Training
    dynamic_cell = """print("\\n" + "="*50)
print("--- Training Our Dynamic Network ---")
print("="*50)

dynamic_net = PyTorchDynamicNetwork(input_dim=784, output_dim=1, max_neurons=2000).to(device)
replay = ReplayBuffer(capacity=1000)
optimizer = torch.optim.Adam(dynamic_net.parameters(), lr=1e-3)
criterion = nn.MSELoss()

dynamic_accuracies = {}

for task_id, (name, X_train, y_train, X_test, y_test) in enumerate(tasks):
    print(f"\\nTraining Dynamic Network on {name}...")
    
    dataset = torch.utils.data.TensorDataset(X_train, y_train)
    loader = torch.utils.data.DataLoader(dataset, batch_size=256, shuffle=True)
    
    loss_ema = 0.5
    stress_ema = 0.0
    batches_since_grow = 0
    
    for epoch in range(5):
        dynamic_net.train()
        
        for bx, by in loader:
            bx, by = bx.to(device), by.to(device)
            optimizer.zero_grad()
            
            pred = dynamic_net(bx)
            loss = criterion(pred, by)
            
            if replay.has_data():
                rx, ry = replay.sample(128)
                rx, ry = rx.to(device), ry.to(device)
                r_pred = dynamic_net(rx)
                loss_r = criterion(r_pred, ry)
                
                grads_task = torch.autograd.grad(loss, dynamic_net.W, retain_graph=True, allow_unused=True)[0]
                grads_replay = torch.autograd.grad(loss_r, dynamic_net.W, retain_graph=True, allow_unused=True)[0]
                
                if grads_task is not None and grads_replay is not None:
                    dynamic_net.update_stress(grads_task, grads_replay)
                    
            loss.backward()
            optimizer.step()
            
            loss_ema = 0.9 * loss_ema + 0.1 * loss.item()
            stress_ema = dynamic_net.get_network_stress()
            batches_since_grow += 1
            
            # Freeze condition
            if stress_ema > 0.3:
                n_frozen = dynamic_net.stress_freeze(threshold=0.3)
                if n_frozen > 0:
                    dynamic_net.grow_neuron(num_connections=20)
                    _reset_adam_for_neuron(optimizer, dynamic_net, dynamic_net.active_neurons - 1)
                    batches_since_grow = 0
                    print(f"  [Batch] FREEZE: {n_frozen} conns. Grew 1. active={dynamic_net.active_neurons}")
            
            # Grow condition (loss)
            elif loss_ema > 0.2 and batches_since_grow > 20 and stress_ema <= 0.3:
                dynamic_net.grow_neuron(num_connections=20)
                _reset_adam_for_neuron(optimizer, dynamic_net, dynamic_net.active_neurons - 1)
                batches_since_grow = 0
                print(f"  [Batch] GROW(Loss): active={dynamic_net.active_neurons} loss={loss_ema:.4f}")
        
        print(f"  [Epoch {epoch+1:2d}] active={dynamic_net.active_neurons} loss={loss_ema:.4f}")
        
    replay.add_data(X_train, y_train, num_samples=200)
    
    dynamic_net.eval()
    print(f"Accuracy after {name}:")
    dynamic_accuracies[name] = []
    with torch.no_grad():
        for eval_id in range(task_id + 1):
            t_name, _, _, t_X_test, t_y_test = tasks[eval_id]
            t_X_test_d = t_X_test.to(device)
            t_y_test_d = t_y_test.to(device)
            acc = ((dynamic_net(t_X_test_d) > 0.5).float() == t_y_test_d).float().mean().item()
            dynamic_accuracies[name].append((t_name, acc))
            print(f"  {t_name}: {acc*100:.1f}%")
    
    print(f"  Active Neurons: {dynamic_net.active_neurons} (trainable={dynamic_net.n_trainable()}, frozen={dynamic_net.n_frozen()})")
"""
    cells.append({"cell_type": "code", "metadata": {}, "execution_count": None, "outputs": [], "source": [line + "\n" for line in dynamic_cell.split('\n')]})

    notebook = {
        "cells": cells,
        "metadata": {
            "kernelspec": {
                "display_name": "Python 3",
                "language": "python",
                "name": "python3"
            },
            "language_info": {
                "name": "python"
            }
        },
        "nbformat": 4,
        "nbformat_minor": 4
    }

    with open('mnist_benchmark.ipynb', 'w') as f:
        json.dump(notebook, f, indent=1)
    print("Notebook 'mnist_benchmark.ipynb' created successfully!")

if __name__ == "__main__":
    create_mnist_notebook()
