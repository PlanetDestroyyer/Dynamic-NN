import os
import json

def create_gen2_notebook():
    cells = []
    
    # 1. Imports and Dataset
    data_cell = """import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
from torchvision import datasets, transforms
import torchvision.models as models
from torch.utils.data import DataLoader, TensorDataset

print("Preparing Dataset...")
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,))
])

train_dataset = datasets.MNIST('./data', train=True, download=True, transform=transform)
test_dataset = datasets.MNIST('./data', train=False, download=True, transform=transform)

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
    
    tasks.append((f"Digits {c0} vs {c1}", X_train, y_train, X_test, y_test))
"""
    cells.append({"cell_type": "code", "metadata": {}, "execution_count": None, "outputs": [], "source": [line + "\n" for line in data_cell.split('\n')]})

    # 2. Gen2 PyTorchDynamicNetwork
    with open(os.path.join(os.path.dirname(__file__), '../src/dynamic_network.py'), 'r') as f:
        network_code = f.read()
        network_code = network_code.replace("import torch\nimport torch.nn as nn\nimport numpy as np\n", "")
    cells.append({"cell_type": "code", "metadata": {}, "execution_count": None, "outputs": [], "source": [line + "\n" for line in network_code.split('\n')]})

    # 3. Hybrid Model and Replay
    hybrid_cell = """class HybridModel(nn.Module):
    def __init__(self, cnn, device):
        super().__init__()
        self.cnn = cnn
        # Gen 2: We start with output_dim=1 for the first task
        self.dynamic = PyTorchDynamicNetwork(input_dim=512, output_dim=1, max_neurons=1000).to(device)
        
    def forward(self, x):
        features = self.cnn(x)
        # Returns [batch, num_heads]
        return self.dynamic(features)

class ReplayBuffer:
    def __init__(self, capacity: int = 500):
        self.capacity = capacity
        self.X = None
        self.y = None
        self.task_ids = None

    def add_data(self, X_new: torch.Tensor, y_new: torch.Tensor, task_id: int, num_samples: int = 100):
        idx = torch.randperm(X_new.size(0))[:num_samples]
        X_sub = X_new[idx].clone().detach()
        y_sub = y_new[idx].clone().detach()
        t_sub = torch.full((num_samples,), task_id, dtype=torch.long)

        if self.X is None:
            self.X = X_sub
            self.y = y_sub
            self.task_ids = t_sub
        else:
            self.X = torch.cat([self.X, X_sub], dim=0)
            self.y = torch.cat([self.y, y_sub], dim=0)
            self.task_ids = torch.cat([self.task_ids, t_sub], dim=0)

            if self.X.size(0) > self.capacity:
                keep = torch.randperm(self.X.size(0))[:self.capacity]
                self.X = self.X[keep]
                self.y = self.y[keep]
                self.task_ids = self.task_ids[keep]

    def sample(self, batch_size: int = 64) -> tuple:
        if self.X is None:
            return None, None, None
        size = self.X.size(0)
        idx = torch.randint(0, size, (min(batch_size, size),))
        return self.X[idx], self.y[idx], self.task_ids[idx]

    def has_data(self) -> bool:
        return self.X is not None and self.X.size(0) > 0

def _reset_adam_for_neuron(optimizer, net, new_idx):
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
    cells.append({"cell_type": "code", "metadata": {}, "execution_count": None, "outputs": [], "source": [line + "\n" for line in hybrid_cell.split('\n')]})

    # 4. Pretrain CNN
    pretrain_cell = """device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Running on {device}...")

print("\\n" + "="*50)
print("--- Pre-Training CNN Feature Extractor ---")
print("="*50)

shared_cnn = models.resnet18(weights='DEFAULT').to(device)
# Adapt for 1-channel grayscale MNIST
shared_cnn.conv1 = nn.Conv2d(1, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False).to(device)
shared_cnn.fc = nn.Identity() # Output is now 512-dim feature vector

classifier = nn.Linear(512, 10).to(device)
optimizer_cnn = torch.optim.Adam(list(shared_cnn.parameters()) + list(classifier.parameters()), lr=1e-3)
criterion_cnn = nn.CrossEntropyLoss()

shared_cnn.train()
for epoch in range(2):
    for bx, by in pretrain_loader:
        bx, by = bx.to(device), by.to(device)
        optimizer_cnn.zero_grad()
        out = classifier(shared_cnn(bx))
        loss = criterion_cnn(out, by)
        loss.backward()
        optimizer_cnn.step()
    print(f"Pre-Train Epoch {epoch+1} finished.")

for param in shared_cnn.parameters():
    param.requires_grad = False
shared_cnn.eval()
print("CNN Frozen.")
"""
    cells.append({"cell_type": "code", "metadata": {}, "execution_count": None, "outputs": [], "source": [line + "\n" for line in pretrain_cell.split('\n')]})

    # 5. Gen 2 Dynamic Training
    dynamic_cell = """print("\\n" + "="*50)
print("--- Training Generation 2 (Multi-Head & Local Stress) ---")
print("="*50)

hybrid_model = HybridModel(shared_cnn, device)
replay = ReplayBuffer(capacity=1000)
optimizer = torch.optim.Adam(hybrid_model.dynamic.parameters(), lr=1e-3)
criterion = nn.MSELoss()

dynamic_accuracies = {}

for task_id, (name, X_train, y_train, X_test, y_test) in enumerate(tasks):
    print(f"\\nTraining Task {task_id}: {name}...")
    
    # Generation 2: Multi-Head Output Expansion!
    if task_id > 0:
        new_out_idx = hybrid_model.dynamic.grow_output_head(num_connections=20)
        _reset_adam_for_neuron(optimizer, hybrid_model.dynamic, new_out_idx)
        print(f"  [Task Boundary] Spawned new Output Head #{task_id}. active={hybrid_model.dynamic.active_neurons}")
    
    dataset = TensorDataset(X_train, y_train)
    loader = DataLoader(dataset, batch_size=256, shuffle=True)
    
    stream_loss_ema = 0.5
    batches_since_grow = 0
    
    for epoch in range(5):
        hybrid_model.train()
        hybrid_model.cnn.eval()
        
        for bx, by in loader:
            bx, by = bx.to(device), by.to(device)
            optimizer.zero_grad()
            
            # Use only the output head for the current task
            pred = hybrid_model(bx)[:, task_id].unsqueeze(1)
            loss = criterion(pred, by)
            
            if replay.has_data():
                rx, ry, rt = replay.sample(128)
                rx, ry, rt = rx.to(device), ry.to(device), rt.to(device)
                
                # We must route the replay samples to their respective output heads!
                r_pred_all_heads = hybrid_model(rx)
                # Select the correct head for each replay sample
                r_pred = r_pred_all_heads[torch.arange(rx.size(0)), rt].unsqueeze(1)
                
                loss_r = criterion(r_pred, ry)
                
                grads_task = torch.autograd.grad(loss, hybrid_model.dynamic.W, retain_graph=True, allow_unused=True)[0]
                grads_replay = torch.autograd.grad(loss_r, hybrid_model.dynamic.W, retain_graph=True, allow_unused=True)[0]
                
                if grads_task is not None and grads_replay is not None:
                    # Generation 2: Update Local Per-Neuron Stress!
                    hybrid_model.dynamic.update_stress(grads_task, grads_replay)
                    
            loss.backward()
            optimizer.step()
            
            # Anomaly Detection: Update the network's loss EMA
            hybrid_model.dynamic.update_loss_ema(loss.item())
            
            stream_loss_ema = 0.9 * stream_loss_ema + 0.1 * loss.item()
            # Gen 2: We check the MAX local stress among all neurons
            max_stress = hybrid_model.dynamic.get_max_neuron_stress()
            batches_since_grow += 1
            
            # Calculate what the threshold currently is based on loss EMA
            capped_loss = min(hybrid_model.dynamic.loss_ema, 1.0)
            current_threshold = 0.5 - (0.4 * capped_loss)
            
            if max_stress > current_threshold:
                # Generation 2: Localized Freeze using Dynamic Threshold
                n_frozen, actual_thresh = hybrid_model.dynamic.stress_freeze(base_threshold=0.5, sensitivity_factor=0.4)
                if n_frozen > 0:
                    hybrid_model.dynamic.grow_neuron(num_connections=15)
                    _reset_adam_for_neuron(optimizer, hybrid_model.dynamic, hybrid_model.dynamic.active_neurons - 1)
                    batches_since_grow = 0
                    print(f"  [Batch] LOCAL FREEZE: {n_frozen} conns. active={hybrid_model.dynamic.active_neurons} | Threshold dropped to {actual_thresh:.2f} due to Anomaly!")
            
            elif stream_loss_ema > 0.2 and batches_since_grow > 20 and max_stress <= current_threshold:
                hybrid_model.dynamic.grow_neuron(num_connections=15)
                _reset_adam_for_neuron(optimizer, hybrid_model.dynamic, hybrid_model.dynamic.active_neurons - 1)
                batches_since_grow = 0
                print(f"  [Batch] GROW(Loss): active={hybrid_model.dynamic.active_neurons} loss={stream_loss_ema:.4f} loss_ema={hybrid_model.dynamic.loss_ema:.2f}")
        
        print(f"  [Epoch {epoch+1:2d}] active={hybrid_model.dynamic.active_neurons} loss={stream_loss_ema:.4f} loss_ema={hybrid_model.dynamic.loss_ema:.2f} threshold={current_threshold:.2f}")
        
    replay.add_data(X_train, y_train, task_id, num_samples=200)
    
    hybrid_model.eval()
    print(f"Accuracy after {name}:")
    with torch.no_grad():
        for eval_id in range(task_id + 1):
            t_name, _, _, t_X_test, t_y_test = tasks[eval_id]
            t_X_test_d = t_X_test.to(device)
            t_y_test_d = t_y_test.to(device)
            # Route evaluation through its respective Output Head
            preds = hybrid_model(t_X_test_d)[:, eval_id].unsqueeze(1)
            acc = ((preds > 0.5).float() == t_y_test_d).float().mean().item()
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

    with open('gen2_benchmark.ipynb', 'w') as f:
        json.dump(notebook, f, indent=1)
    print("Notebook 'gen2_benchmark.ipynb' created successfully!")

if __name__ == "__main__":
    create_gen2_notebook()
