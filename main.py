import argparse
import torch
import torch.nn as nn
import numpy as np
import math
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, TensorDataset
import time

from src.dynamic_network import PyTorchDynamicNetwork

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
            self.X, self.y, self.task_ids = X_sub, y_sub, t_sub
        else:
            self.X = torch.cat([self.X, X_sub], dim=0)
            self.y = torch.cat([self.y, y_sub], dim=0)
            self.task_ids = torch.cat([self.task_ids, t_sub], dim=0)
            if self.X.size(0) > self.capacity:
                keep = torch.randperm(self.X.size(0))[:self.capacity]
                self.X, self.y, self.task_ids = self.X[keep], self.y[keep], self.task_ids[keep]

    def sample(self, batch_size: int = 64):
        if self.X is None: return None, None, None
        size = self.X.size(0)
        idx = torch.randint(0, size, (min(batch_size, size),))
        return self.X[idx], self.y[idx], self.task_ids[idx]
    
    def has_data(self):
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

# ==========================================
# 1. TOY 2D BENCHMARK
# ==========================================
def run_toy_benchmark():
    print("\\n" + "="*50)
    print("--- BENCHMARK 1: TOY 2D SEQUENTIAL TASKS ---")
    print("="*50)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Generate 3 Tasks
    def get_circles():
        X = torch.rand(1000, 2) * 2 - 1
        y = ((X[:, 0]**2 + X[:, 1]**2) < 0.5).float().view(-1, 1)
        return X.to(device), y.to(device)
    
    def get_linear():
        X = torch.rand(1000, 2) * 2 - 1
        y = (X[:, 0] > X[:, 1]).float().view(-1, 1)
        return X.to(device), y.to(device)
        
    def get_xor():
        X = torch.rand(1000, 2) * 2 - 1
        y = ((X[:, 0] * X[:, 1]) > 0).float().view(-1, 1)
        return X.to(device), y.to(device)

    tasks = [("Inner/Outer Circles", *get_circles()), 
             ("Linear Boundary", *get_linear()), 
             ("XOR Quadrants", *get_xor())]

    # --- Train Standard NN Baseline ---
    print("\\n[1/2] Training Standard Static Neural Network (Baseline)...")
    baseline = nn.Sequential(nn.Linear(2, 64), nn.ReLU(), nn.Linear(64, 64), nn.ReLU(), nn.Linear(64, 1)).to(device)
    opt_base = torch.optim.Adam(baseline.parameters(), lr=1e-2)
    crit_base = nn.MSELoss()
    
    for t_id, (name, X, y) in enumerate(tasks):
        print(f"  Training Task {t_id+1}: {name}...")
        for epoch in range(500):
            opt_base.zero_grad()
            loss = crit_base(baseline(X), y)
            loss.backward()
            opt_base.step()
            
    print("  Evaluating Baseline Memory Retention:")
    for t_id, (name, X, y) in enumerate(tasks):
        acc = ((baseline(X) > 0.5).float() == y).float().mean().item()
        print(f"    Task {t_id+1} ({name}): {acc*100:.1f}%")

    # --- Train Dynamic NN ---
    print("\\n[2/2] Training Dynamic Neural Network...")
    dynamic = PyTorchDynamicNetwork(input_dim=2, output_dim=1, max_neurons=500).to(device)
    opt_dyn = torch.optim.Adam(dynamic.parameters(), lr=1e-2)
    replay = ReplayBuffer(capacity=500)
    
    for t_id, (name, X, y) in enumerate(tasks):
        print(f"  Training Task {t_id+1}: {name}...")
        
        if t_id > 0:
            new_head = dynamic.grow_output_head(num_connections=10)
            _reset_adam_for_neuron(opt_dyn, dynamic, new_head)
            
        loss_ema = 0.5
        batches_since_grow = 0
        
        for epoch in range(500):
            opt_dyn.zero_grad()
            # Random batch of 128
            idx = torch.randperm(X.size(0))[:128]
            bx, by = X[idx], y[idx]
            
            pred = dynamic(bx)[:, t_id].unsqueeze(1)
            loss = crit_base(pred, by)
            
            if replay.has_data():
                rx, ry, rt = replay.sample(128)
                rx, ry, rt = rx.to(device), ry.to(device), rt.to(device)
                r_pred_all = dynamic(rx)
                r_pred = r_pred_all[torch.arange(rx.size(0)), rt].unsqueeze(1)
                loss_r = crit_base(r_pred, ry)
                
                g_t = torch.autograd.grad(loss, dynamic.W, retain_graph=True, allow_unused=True)[0]
                g_r = torch.autograd.grad(loss_r, dynamic.W, retain_graph=True, allow_unused=True)[0]
                if g_t is not None and g_r is not None:
                    dynamic.update_stress(g_t, g_r)
            
            loss.backward()
            opt_dyn.step()
            
            loss_ema = 0.9 * loss_ema + 0.1 * loss.item()
            batches_since_grow += 1
            max_stress = dynamic.get_max_neuron_stress()
            
            if max_stress > 0.3:
                n_frozen = dynamic.stress_freeze(0.3)
                if n_frozen > 0:
                    dynamic.grow_neuron(10)
                    _reset_adam_for_neuron(opt_dyn, dynamic, dynamic.active_neurons - 1)
                    batches_since_grow = 0
            elif loss_ema > 0.1 and batches_since_grow > 100 and max_stress <= 0.3:
                dynamic.grow_neuron(10)
                _reset_adam_for_neuron(opt_dyn, dynamic, dynamic.active_neurons - 1)
                batches_since_grow = 0
                
        replay.add_data(X, y, t_id, 200)

    print("  Evaluating Dynamic NN Memory Retention:")
    for t_id, (name, X, y) in enumerate(tasks):
        acc = ((dynamic(X)[:, t_id].unsqueeze(1) > 0.5).float() == y).float().mean().item()
        print(f"    Task {t_id+1} ({name}): {acc*100:.1f}%")
    print(f"    Total Neurons Built: {dynamic.active_neurons}")

# ==========================================
# 2. MNIST BENCHMARK
# ==========================================
class CNNFeatureExtractor(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 16, 3, 1)
        self.relu = nn.ReLU()
        self.pool = nn.MaxPool2d(2)
        self.flatten = nn.Flatten()
        self.fc = nn.Linear(16 * 13 * 13, 64)
    def forward(self, x):
        return self.fc(self.flatten(self.pool(self.relu(self.conv1(x)))))

def run_mnist_benchmark():
    print("\\n" + "="*50)
    print("--- BENCHMARK 2: HIGH-DIMENSIONAL SPLIT-MNIST ---")
    print("="*50)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))])
    train_ds = datasets.MNIST('./data', train=True, download=True, transform=transform)
    test_ds = datasets.MNIST('./data', train=False, download=True, transform=transform)
    
    tasks = []
    pairs = [(0,1), (2,3), (4,5), (6,7), (8,9)]
    for (c0, c1) in pairs:
        mask_tr = (train_ds.targets == c0) | (train_ds.targets == c1)
        X_tr = train_ds.data[mask_tr].float().unsqueeze(1) / 255.0
        X_tr = (X_tr - 0.1307) / 0.3081
        y_tr = (train_ds.targets[mask_tr] == c1).float().view(-1, 1)
        
        mask_te = (test_ds.targets == c0) | (test_ds.targets == c1)
        X_te = test_ds.data[mask_te].float().unsqueeze(1) / 255.0
        X_te = (X_te - 0.1307) / 0.3081
        y_te = (test_ds.targets[mask_te] == c1).float().view(-1, 1)
        tasks.append((f"Digits {c0} vs {c1}", X_tr, y_tr, X_te, y_te))

    # Pre-train frozen CNN (simulated quickly for demo purposes)
    cnn = CNNFeatureExtractor().to(device)
    for p in cnn.parameters(): p.requires_grad = False

    # --- Train Standard NN Baseline ---
    print("\\n[1/2] Training Standard Static Neural Network (Baseline)...")
    # For a fair comparison, the baseline gets a multi-head output too
    class MultiHeadBaseline(nn.Module):
        def __init__(self):
            super().__init__()
            self.hidden = nn.Linear(64, 128)
            self.heads = nn.ModuleList([nn.Linear(128, 1) for _ in range(5)])
        def forward(self, x, task_id):
            return self.heads[task_id](torch.relu(self.hidden(x)))
            
    baseline = MultiHeadBaseline().to(device)
    opt_base = torch.optim.Adam(baseline.parameters(), lr=1e-3)
    crit = nn.MSELoss()
    
    for t_id, (name, X_tr, y_tr, X_te, y_te) in enumerate(tasks):
        print(f"  Training Task {t_id+1}: {name}...")
        ds = TensorDataset(X_tr.to(device), y_tr.to(device))
        dl = DataLoader(ds, batch_size=256, shuffle=True)
        for epoch in range(3):
            for bx, by in dl:
                opt_base.zero_grad()
                pred = baseline(cnn(bx), t_id)
                loss = crit(pred, by)
                loss.backward()
                opt_base.step()
                
    print("  Evaluating Baseline Memory Retention:")
    for t_id, (name, _, _, X_te, y_te) in enumerate(tasks):
        with torch.no_grad():
            preds = baseline(cnn(X_te.to(device)), t_id)
            acc = ((preds > 0.5).float() == y_te.to(device)).float().mean().item()
            print(f"    Task {t_id+1} ({name}): {acc*100:.1f}%")

    # --- Train Dynamic NN ---
    print("\\n[2/2] Training Dynamic Neural Network...")
    dynamic = PyTorchDynamicNetwork(input_dim=64, output_dim=1, max_neurons=500).to(device)
    opt_dyn = torch.optim.Adam(dynamic.parameters(), lr=1e-3)
    replay = ReplayBuffer(1000)
    
    for t_id, (name, X_tr, y_tr, X_te, y_te) in enumerate(tasks):
        print(f"  Training Task {t_id+1}: {name}...")
        if t_id > 0:
            new_head = dynamic.grow_output_head(num_connections=20)
            _reset_adam_for_neuron(opt_dyn, dynamic, new_head)
            
        ds = TensorDataset(X_tr.to(device), y_tr.to(device))
        dl = DataLoader(ds, batch_size=256, shuffle=True)
        loss_ema = 0.5
        batches_since = 0
        
        for epoch in range(3):
            for bx, by in dl:
                opt_dyn.zero_grad()
                pred = dynamic(cnn(bx))[:, t_id].unsqueeze(1)
                loss = crit(pred, by)
                
                if replay.has_data():
                    rx, ry, rt = replay.sample(128)
                    rx, ry, rt = rx.to(device), ry.to(device), rt.to(device)
                    r_pred = dynamic(cnn(rx))[torch.arange(rx.size(0)), rt].unsqueeze(1)
                    loss_r = crit(r_pred, ry)
                    g_t = torch.autograd.grad(loss, dynamic.W, retain_graph=True, allow_unused=True)[0]
                    g_r = torch.autograd.grad(loss_r, dynamic.W, retain_graph=True, allow_unused=True)[0]
                    if g_t is not None and g_r is not None:
                        dynamic.update_stress(g_t, g_r)
                        
                loss.backward()
                opt_dyn.step()
                loss_ema = 0.9 * loss_ema + 0.1 * loss.item()
                batches_since += 1
                
                max_stress = dynamic.get_max_neuron_stress()
                if max_stress > 0.3:
                    n_frozen = dynamic.stress_freeze(0.3)
                    if n_frozen > 0:
                        dynamic.grow_neuron(15)
                        _reset_adam_for_neuron(opt_dyn, dynamic, dynamic.active_neurons - 1)
                        batches_since = 0
                elif loss_ema > 0.2 and batches_since > 20 and max_stress <= 0.3:
                    dynamic.grow_neuron(15)
                    _reset_adam_for_neuron(opt_dyn, dynamic, dynamic.active_neurons - 1)
                    batches_since = 0
                    
        replay.add_data(X_tr, y_tr, t_id, 200)

    print("  Evaluating Dynamic NN Memory Retention:")
    for t_id, (name, _, _, X_te, y_te) in enumerate(tasks):
        with torch.no_grad():
            preds = dynamic(cnn(X_te.to(device)))[:, t_id].unsqueeze(1)
            acc = ((preds > 0.5).float() == y_te.to(device)).float().mean().item()
            print(f"    Task {t_id+1} ({name}): {acc*100:.1f}%")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run Continuous Learning Benchmarks")
    parser.add_argument('--benchmark', type=str, default='all', choices=['toy', 'mnist', 'llm', 'all'])
    args = parser.parse_args()
    
    if args.benchmark in ['toy', 'all']: run_toy_benchmark()
    if args.benchmark in ['mnist', 'all']: run_mnist_benchmark()
    if args.benchmark in ['llm', 'all']: print("\\n[Notice] LLM benchmark is best viewed inside notebooks/baby_llm.ipynb due to text generation rendering.")
