import torch
import torch.nn as nn
import numpy as np
import os
import sys

# Add src to path robustly
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../src')))
from dynamic_network import PyTorchDynamicNetwork

class ReplayBuffer:
    def __init__(self, capacity: int = 2000):
        self.capacity = capacity
        self.X = None
        self.y = None
        self.task_ids = None

    def add_data(self, X_new: torch.Tensor, y_new: torch.Tensor, task_id: int, num_samples: int = 16):
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

    def sample(self, batch_size: int = 128) -> tuple:
        if self.X is None:
            return None, None, None
        size = self.X.size(0)
        idx = torch.randint(0, size, (min(batch_size, size),))
        return self.X[idx], self.y[idx], self.task_ids[idx]

    def has_data(self) -> bool:
        return self.X is not None and self.X.size(0) > 0

def run_experiment(seed, use_replay, use_growth, use_freeze):
    torch.manual_seed(seed)
    np.random.seed(seed)
    
    device = torch.device('cpu') # Keep it simple for script

    # 1. Generate Data
    def make_circles(n_samples, noise=0.05, factor=0.5):
        theta = np.random.uniform(0, 2*np.pi, n_samples)
        r = np.where(np.random.rand(n_samples) > 0.5, 1.0, factor)
        x = r * np.cos(theta) + np.random.randn(n_samples) * noise
        y = r * np.sin(theta) + np.random.randn(n_samples) * noise
        labels = (r == 1.0).astype(int)
        return torch.FloatTensor(np.c_[x, y]), torch.FloatTensor(labels).unsqueeze(1)

    def make_linear(n_samples, noise=0.05):
        x = np.random.uniform(-1.5, 1.5, n_samples)
        y = np.random.uniform(-1.5, 1.5, n_samples)
        labels = (x + y > 0).astype(int)
        X = np.c_[x + np.random.randn(n_samples)*noise, y + np.random.randn(n_samples)*noise]
        return torch.FloatTensor(X), torch.FloatTensor(labels).unsqueeze(1)

    def make_xor(n_samples, noise=0.05):
        x = np.random.uniform(-1.5, 1.5, n_samples)
        y = np.random.uniform(-1.5, 1.5, n_samples)
        labels = ((x > 0) ^ (y > 0)).astype(int)
        X = np.c_[x + np.random.randn(n_samples)*noise, y + np.random.randn(n_samples)*noise]
        return torch.FloatTensor(X), torch.FloatTensor(labels).unsqueeze(1)

    tasks = [
        ("Inner/Outer Circles", *make_circles(1000)),
        ("Linear Boundary", *make_linear(1000)),
        ("XOR Quadrants", *make_xor(1000))
    ]

    # Create stream
    stream_X = torch.cat([t[1] for t in tasks])
    stream_y = torch.cat([t[2] for t in tasks])
    
    # Model
    dynamic = PyTorchDynamicNetwork(input_dim=2, output_dim=1, max_neurons=200).to(device)
    optimizer = torch.optim.Adam(dynamic.parameters(), lr=0.01)
    criterion = nn.MSELoss()
    replay = ReplayBuffer(capacity=500)
    
    active_head = 0
    stream_loss_ema = 0.5
    total_batches = 0
    
    for i in range(0, stream_X.size(0), 16):
        bx = stream_X[i:i+16].to(device)
        by = stream_y[i:i+16].to(device)
        
        optimizer.zero_grad()
        
        pred_all_heads = dynamic(bx)
        pred = pred_all_heads[:, active_head].unsqueeze(1)
        loss = criterion(pred, by)
        
        # Replay Buffer
        if use_replay and replay.has_data():
            rx, ry, r_heads = replay.sample(64)
            rx, ry, r_heads = rx.to(device), ry.to(device), r_heads.to(device)
            r_pred_all = dynamic(rx)
            r_pred = r_pred_all[torch.arange(rx.size(0)), r_heads].unsqueeze(1)
            loss_r = criterion(r_pred, ry)
            
            g_t = torch.autograd.grad(loss, dynamic.W, retain_graph=True, allow_unused=True)[0]
            g_r = torch.autograd.grad(loss_r, dynamic.W, retain_graph=True, allow_unused=True)[0]
            if g_t is not None and g_r is not None:
                dynamic.update_stress(g_t, g_r)
                
            loss = loss + loss_r
            
        loss.backward()
        optimizer.step()
        
        dynamic.update_loss_ema(loss.item())
        stream_loss_ema = 0.9 * stream_loss_ema + 0.1 * loss.item()
        total_batches += 1
        max_stress = dynamic.get_max_neuron_stress()
        
        # Autonomous Shift Detection
        if use_growth:
            shift_detected = dynamic.detect_task_shift(max_stress, margin=0.15, stress_threshold=0.2)
            if shift_detected and total_batches > 100:
                active_head += 1
                new_head = dynamic.grow_output_head(num_connections=10)
                dynamic.fast_loss_ema = stream_loss_ema
                dynamic.slow_loss_ema = stream_loss_ema
                total_batches = 0 # reset burn-in
                
        # Freezing
        if use_freeze:
            capped_loss = min(dynamic.loss_ema, 1.0)
            current_threshold = max(0.05, 0.5 - (1.0 * capped_loss))
            if max_stress > current_threshold:
                dynamic.stress_freeze(base_threshold=0.5, sensitivity_factor=1.0)
                
        if use_replay and torch.rand(1).item() < 0.10:
            replay.add_data(bx, by, active_head, num_samples=16)
            
    # Evaluation
    accs = []
    for t_id, (name, X, y) in enumerate(tasks):
        eval_head = min(t_id, active_head)
        acc = ((dynamic(X)[:, eval_head].unsqueeze(1) > 0.5).float() == y).float().mean().item()
        accs.append(acc)
        
    return np.mean(accs)

if __name__ == "__main__":
    print("Running Rigorous Ablation Study (Average over 3 seeds)")
    print("-" * 60)
    
    configs = [
        ("Static MLP (No Replay, No Growth, No Freeze)", False, False, False),
        ("Static MLP + Replay", True, False, False),
        ("Dynamic Net (Growth + MultiHead) + Replay", True, True, False),
        ("Full Model (Growth + Freeze) + Replay", True, True, True)
    ]
    
    for name, use_replay, use_growth, use_freeze in configs:
        scores = []
        for seed in [42, 100, 999]:
            score = run_experiment(seed, use_replay, use_growth, use_freeze)
            scores.append(score)
        print(f"{name:45s} -> {np.mean(scores)*100:.1f}% ± {np.std(scores)*100:.1f}%")
