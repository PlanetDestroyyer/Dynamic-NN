import os
import json

def create_hybrid_notebook():
    cells = []
    
    # 1. Imports and Dataset loading
    data_cell = """import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, TensorDataset

# 1. Download and format Split-MNIST
print("Downloading and processing MNIST dataset for CNN (1x28x28)...")
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,))
])

train_dataset = datasets.MNIST('./data', train=True, download=True, transform=transform)
test_dataset = datasets.MNIST('./data', train=False, download=True, transform=transform)

# Keep the original 10-class dataset for pre-training the CNN
pretrain_loader = DataLoader(train_dataset, batch_size=256, shuffle=True)

tasks = []
pairs = [(0, 1), (2, 3), (4, 5), (6, 7), (8, 9)]

for (c0, c1) in pairs:
    train_mask = (train_dataset.targets == c0) | (train_dataset.targets == c1)
    X_train = train_dataset.data[train_mask].float() / 255.0
    X_train = (X_train - 0.1307) / 0.3081
    X_train = X_train.view(-1, 1, 28, 28)
    y_train = (train_dataset.targets[train_mask] == c1).float().view(-1, 1)
    
    test_mask = (test_dataset.targets == c0) | (test_dataset.targets == c1)
    X_test = test_dataset.data[test_mask].float() / 255.0
    X_test = (X_test - 0.1307) / 0.3081
    X_test = X_test.view(-1, 1, 28, 28)
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

    # 3. CNN, ReplayBuffer and Helper
    cnn_replay_cell = """class CNNFeatureExtractor(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 16, kernel_size=3, padding=1)
        self.relu1 = nn.ReLU()
        self.pool1 = nn.MaxPool2d(2) # 14x14
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, padding=1)
        self.relu2 = nn.ReLU()
        self.pool2 = nn.MaxPool2d(2) # 7x7
        self.flatten = nn.Flatten()
        self.fc = nn.Linear(32 * 7 * 7, 64)
        self.relu3 = nn.ReLU()

    def forward(self, x):
        x = self.pool1(self.relu1(self.conv1(x)))
        x = self.pool2(self.relu2(self.conv2(x)))
        x = self.flatten(x)
        x = self.relu3(self.fc(x))
        return x

class HybridModel(nn.Module):
    def __init__(self, cnn, device):
        super().__init__()
        self.cnn = cnn # Shared, frozen CNN
        self.dynamic = PyTorchDynamicNetwork(input_dim=64, output_dim=1, max_neurons=500).to(device)
        
    def forward(self, x):
        features = self.cnn(x)
        return self.dynamic(features)

class ReplayBuffer:
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
            
    if hasattr(net, 'b') and net.b in optimizer.state:
        s_b = optimizer.state[net.b]
        if 'exp_avg' in s_b:
            s_b['exp_avg'][new_idx] = 0.0
            s_b['exp_avg_sq'][new_idx] = 0.0
"""
    cells.append({"cell_type": "code", "metadata": {}, "execution_count": None, "outputs": [], "source": [line + "\n" for line in cnn_replay_cell.split('\n')]})

    # 4. Pre-Train CNN
    pretrain_cell = """device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Running on {device}...")

print("\\n" + "="*50)
print("--- Pre-Training CNN Feature Extractor ---")
print("="*50)

# Create CNN and temporary 10-class classifier
shared_cnn = CNNFeatureExtractor().to(device)
classifier = nn.Linear(64, 10).to(device)

optimizer_cnn = torch.optim.Adam(list(shared_cnn.parameters()) + list(classifier.parameters()), lr=1e-3)
criterion_cnn = nn.CrossEntropyLoss()

# Pre-train for 2 epochs on the standard MNIST 10-class task
shared_cnn.train()
classifier.train()
for epoch in range(2):
    total_loss = 0
    correct = 0
    total = 0
    for bx, by in pretrain_loader:
        bx, by = bx.to(device), by.to(device)
        optimizer_cnn.zero_grad()
        
        features = shared_cnn(bx)
        out = classifier(features)
        
        loss = criterion_cnn(out, by)
        loss.backward()
        optimizer_cnn.step()
        
        total_loss += loss.item()
        preds = out.argmax(dim=1)
        correct += (preds == by).sum().item()
        total += by.size(0)
        
    print(f"Pre-Train Epoch {epoch+1}: Loss = {total_loss/len(pretrain_loader):.4f}, Acc = {100.0 * correct / total:.2f}%")

# FREEZE THE CNN!
for param in shared_cnn.parameters():
    param.requires_grad = False
shared_cnn.eval()
print("CNN Feature Extractor is now perfectly trained and permanently FROZEN.")
"""
    cells.append({"cell_type": "code", "metadata": {}, "execution_count": None, "outputs": [], "source": [line + "\n" for line in pretrain_cell.split('\n')]})

    # 5. Baseline Training
    baseline_cell = """class BaselineNet(nn.Module):
    def __init__(self, cnn):
        super().__init__()
        self.cnn = cnn # Frozen
        self.fc = nn.Sequential(
            nn.Linear(64, 100),
            nn.ReLU(),
            nn.Linear(100, 1),
            nn.Sigmoid()
        )
    def forward(self, x):
        return self.fc(self.cnn(x))

print("\\n" + "="*50)
print("--- Training Baseline (Frozen CNN + Static Network) ---")
print("="*50)

baseline_net = BaselineNet(shared_cnn).to(device)
# Only optimize the FC layers!
optimizer_b = torch.optim.Adam(baseline_net.fc.parameters(), lr=1e-3)
criterion_b = nn.BCELoss()

baseline_accuracies = {}

for task_id, (name, X_train, y_train, X_test, y_test) in enumerate(tasks):
    print(f"\\nTraining Baseline on {name}...")
    
    dataset = TensorDataset(X_train, y_train)
    loader = DataLoader(dataset, batch_size=256, shuffle=True)
    
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

    # 6. Dynamic Training
    dynamic_cell = """print("\\n" + "="*50)
print("--- Training Our Hybrid (Frozen CNN + Dynamic Network) ---")
print("="*50)

hybrid_model = HybridModel(shared_cnn, device)
replay = ReplayBuffer(capacity=1000)

# ONLY optimize the Dynamic Network parameters!
optimizer = torch.optim.Adam(hybrid_model.dynamic.parameters(), lr=1e-3)
criterion = nn.MSELoss()

dynamic_accuracies = {}

for task_id, (name, X_train, y_train, X_test, y_test) in enumerate(tasks):
    print(f"\\nTraining Hybrid Network on {name}...")
    
    dataset = TensorDataset(X_train, y_train)
    loader = DataLoader(dataset, batch_size=256, shuffle=True)
    
    loss_ema = 0.5
    stress_ema = 0.0
    batches_since_grow = 0
    
    for epoch in range(5):
        hybrid_model.train()
        # Ensure frozen CNN stays in eval mode (e.g. for BatchNorm if we had it)
        hybrid_model.cnn.eval() 
        
        for bx, by in loader:
            bx, by = bx.to(device), by.to(device)
            optimizer.zero_grad()
            
            pred = hybrid_model(bx)
            loss = criterion(pred, by)
            
            if replay.has_data():
                rx, ry = replay.sample(128)
                rx, ry = rx.to(device), ry.to(device)
                r_pred = hybrid_model(rx)
                loss_r = criterion(r_pred, ry)
                
                # Compute gradient stress ONLY on the Dynamic Network's weights
                grads_task = torch.autograd.grad(loss, hybrid_model.dynamic.W, retain_graph=True, allow_unused=True)[0]
                grads_replay = torch.autograd.grad(loss_r, hybrid_model.dynamic.W, retain_graph=True, allow_unused=True)[0]
                
                if grads_task is not None and grads_replay is not None:
                    hybrid_model.dynamic.update_stress(grads_task, grads_replay)
                    
            loss.backward()
            optimizer.step()
            
            loss_ema = 0.9 * loss_ema + 0.1 * loss.item()
            stress_ema = hybrid_model.dynamic.get_network_stress()
            batches_since_grow += 1
            
            # Freeze condition
            if stress_ema > 0.3:
                n_frozen = hybrid_model.dynamic.stress_freeze(threshold=0.3)
                if n_frozen > 0:
                    hybrid_model.dynamic.grow_neuron(num_connections=15)
                    _reset_adam_for_neuron(optimizer, hybrid_model.dynamic, hybrid_model.dynamic.active_neurons - 1)
                    batches_since_grow = 0
                    print(f"  [Batch] FREEZE: {n_frozen} conns. Grew 1. active={hybrid_model.dynamic.active_neurons}")
            
            # Grow condition (loss)
            elif loss_ema > 0.2 and batches_since_grow > 20 and stress_ema <= 0.3:
                hybrid_model.dynamic.grow_neuron(num_connections=15)
                _reset_adam_for_neuron(optimizer, hybrid_model.dynamic, hybrid_model.dynamic.active_neurons - 1)
                batches_since_grow = 0
                print(f"  [Batch] GROW(Loss): active={hybrid_model.dynamic.active_neurons} loss={loss_ema:.4f}")
        
        print(f"  [Epoch {epoch+1:2d}] active={hybrid_model.dynamic.active_neurons} loss={loss_ema:.4f}")
        
    replay.add_data(X_train, y_train, num_samples=200)
    
    hybrid_model.eval()
    print(f"Accuracy after {name}:")
    dynamic_accuracies[name] = []
    with torch.no_grad():
        for eval_id in range(task_id + 1):
            t_name, _, _, t_X_test, t_y_test = tasks[eval_id]
            t_X_test_d = t_X_test.to(device)
            t_y_test_d = t_y_test.to(device)
            acc = ((hybrid_model(t_X_test_d) > 0.5).float() == t_y_test_d).float().mean().item()
            dynamic_accuracies[name].append((t_name, acc))
            print(f"  {t_name}: {acc*100:.1f}%")
    
    print(f"  Active Neurons: {hybrid_model.dynamic.active_neurons} (trainable={hybrid_model.dynamic.n_trainable()}, frozen={hybrid_model.dynamic.n_frozen()})")
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

    with open('hybrid_benchmark.ipynb', 'w') as f:
        json.dump(notebook, f, indent=1)
    print("Notebook 'hybrid_benchmark.ipynb' created successfully!")

if __name__ == "__main__":
    create_hybrid_notebook()
