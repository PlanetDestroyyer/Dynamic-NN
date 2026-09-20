import torch
import torch.nn as nn
import numpy as np
import os
import sys
import json

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../src')))
from dynamic_network import PyTorchDynamicNetwork

class ReplayBuffer:
    def __init__(self, capacity: int = 500):
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

    def sample(self, batch_size: int = 64) -> tuple:
        if self.X is None:
            return None, None, None
        size = self.X.size(0)
        idx = torch.randint(0, size, (min(batch_size, size),))
        return self.X[idx], self.y[idx], self.task_ids[idx]

    def has_data(self) -> bool:
        return self.X is not None and self.X.size(0) > 0

def run_experiment(seed, config_name, use_replay, use_growth, use_freeze, use_compress, initial_capacity=4):
    torch.manual_seed(seed)
    np.random.seed(seed)
    
    device = torch.device('cpu')

    # 1. Generate Data Stream
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
        ("Circles", *make_circles(4000)),
        ("Linear", *make_linear(4000)),
        ("XOR", *make_xor(4000))
    ]

    stream_X = torch.cat([t[1] for t in tasks])
    stream_y = torch.cat([t[2] for t in tasks])
    
    # Track true task boundaries (batches)
    batch_size = 16
    task_boundaries = [
        0, 
        4000 // batch_size, 
        8000 // batch_size
    ]
    
    # Strict Single Head Constraint
    dynamic = PyTorchDynamicNetwork(input_dim=2, output_dim=1, max_neurons=500).to(device)
    
    # Initialize with the requested capacity
    if initial_capacity > 4:
        # We start with 4 hidden neurons by default in __init__, so we grow the rest
        for _ in range(initial_capacity - 4):
            dynamic.grow_neuron(10)
            
    initial_neurons = dynamic.active_neurons
    peak_active = initial_neurons
    total_created = 0
    total_pruned = 0
    
    # Add weight decay so useless weights naturally decay towards zero, allowing them to be pruned!
    optimizer = torch.optim.Adam(dynamic.parameters(), lr=0.01, weight_decay=1e-4)
    criterion = nn.MSELoss()
    replay = ReplayBuffer(capacity=500)
    
    stream_loss_ema = 0.5
    batches_since_grow = 0
    total_batches = 0
    
    # Metrics
    eval_matrix = [] # [batch_idx, task0_acc, task1_acc, task2_acc]
    detection_delays = []
    current_true_task = 0
    
    def evaluate():
        accs = []
        for t_id, (name, X, y) in enumerate(tasks):
            # Strict Single Head Evaluation (No Task ID Leakage)
            acc = ((dynamic(X)[:, 0].unsqueeze(1) > 0.5).float() == y).float().mean().item()
            accs.append(acc)
        return accs

    for i in range(0, stream_X.size(0), batch_size):
        if total_batches in task_boundaries:
            if total_batches > 0:
                current_true_task += 1
                
        bx = stream_X[i:i+batch_size].to(device)
        by = stream_y[i:i+batch_size].to(device)
        
        current_active = dynamic.active_neurons - len(getattr(dynamic, 'free_neurons', []))
        if current_active > peak_active:
            peak_active = current_active
            
        optimizer.zero_grad()
        
        pred_all_heads = dynamic(bx)
        # All tasks forced through the same shared bottleneck
        pred = pred_all_heads[:, 0].unsqueeze(1)
        loss = criterion(pred, by)
        
        if use_replay and replay.has_data():
            rx, ry, _ = replay.sample(64) # We don't care about task IDs from replay anymore
            rx, ry = rx.to(device), ry.to(device)
            r_pred_all = dynamic(rx)
            
            # Replay also forced through shared head
            r_pred = r_pred_all[:, 0].unsqueeze(1)
                
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
        batches_since_grow += 1
        max_stress = dynamic.get_max_neuron_stress()
        
        # Autonomous Shift Detection
        if use_growth:
            shift_detected = dynamic.detect_task_shift(max_stress, margin=0.15, stress_threshold=0.2)
            
            if shift_detected and total_batches > 50 and batches_since_grow > 50:
                # Record delay
                if current_true_task > 0 and current_true_task < len(task_boundaries):
                    delay = total_batches - task_boundaries[current_true_task]
                    if delay > 0 and (len(detection_delays) < current_true_task):
                        detection_delays.append(delay)
                
                dynamic.fast_loss_ema = stream_loss_ema
                dynamic.slow_loss_ema = stream_loss_ema
                batches_since_grow = 0
                
        # Freezing
        if use_freeze:
            capped_loss = min(dynamic.loss_ema, 1.0)
            current_threshold = max(0.05, 0.5 - (1.0 * capped_loss))
            dynamic.current_threshold = current_threshold
            
            if max_stress > current_threshold:
                n_frozen, _ = dynamic.stress_freeze(base_threshold=0.5, sensitivity_factor=1.0)
                if n_frozen > 0:
                    ret = dynamic.grow_neuron(10)
                    if ret != -1:
                        total_created += 1
                    batches_since_grow = 0
                    
        # Capacity Expansion
        if use_growth and stream_loss_ema > 0.2 and batches_since_grow > 20 and max_stress <= getattr(dynamic, 'current_threshold', 0.5):
            ret = dynamic.grow_neuron(10)
            if ret != -1:
                total_created += 1
            batches_since_grow = 0
                
        if use_replay and torch.rand(1).item() < 0.10:
            replay.add_data(bx, by, 0, num_samples=16)
            
        # Periodic Evaluation (every 20 batches)
        if total_batches % 20 == 0:
            eval_matrix.append(evaluate())
            
        # Periodic Compression
        if use_compress and total_batches % 100 == 0:
            p, m = dynamic.compress_network(prune_threshold=0.15, similarity_threshold=0.50)
            total_pruned += (p + m)
            
    # Final Evaluation
    final_accs = evaluate()
    eval_matrix = np.array(eval_matrix)
    
    # Calculate Backward Transfer
    # For each task, BT = final_acc - max_acc_during_its_training_window
    bt = 0.0
    for t_id in range(len(tasks)):
        # Max acc achieved during the entire run
        max_acc = np.max(eval_matrix[:, t_id])
        bt += (final_accs[t_id] - max_acc)
    bt /= len(tasks)
    
    metrics = {
        'final_accuracy': np.mean(final_accs),
        'backward_transfer': bt,
        'net_growth': (dynamic.active_neurons - len(getattr(dynamic, 'free_neurons', []))) - initial_neurons,
        'peak_growth': peak_active - initial_neurons,
        'total_created': total_created,
        'total_pruned': total_pruned,
        'detection_delay_avg': np.mean(detection_delays) if detection_delays else -1,
        'profile': dynamic.profile_network()
    }
    
    return metrics

if __name__ == "__main__":
    configs = [
        ("Static MLP (4 neurons) + Replay", True, False, False, False, 4),
        ("Static MLP (64 neurons) + Replay", True, False, False, False, 64),
        ("Static MLP (128 neurons) + Replay", True, False, False, False, 128),
        ("Dynamic (Growth Only) + Replay", True, True, False, False, 4),
        ("Full Dynamic (Growth+Freeze) + Replay", True, True, True, False, 4),
        ("Full Dynamic + Compress", True, True, True, True, 4)
    ]
    
    results = {}
    
    for name, ur, ug, uf, uc, ic in configs:
        print(f"\\nRunning: {name}")
        metrics_acc = []
        metrics_bt = []
        metrics_net = []
        metrics_peak = []
        metrics_created = []
        metrics_pruned = []
        metrics_delay = []
        
        for seed in [42, 100, 999]:
            m = run_experiment(seed, name, ur, ug, uf, uc, initial_capacity=ic)
            metrics_acc.append(m['final_accuracy'])
            metrics_bt.append(m['backward_transfer'])
            metrics_net.append(m['net_growth'])
            metrics_peak.append(m['peak_growth'])
            metrics_created.append(m['total_created'])
            metrics_pruned.append(m['total_pruned'])
            if m['detection_delay_avg'] != -1:
                metrics_delay.append(m['detection_delay_avg'])
                
        results[name] = {
            'Accuracy': f"{np.mean(metrics_acc)*100:.1f}% ± {np.std(metrics_acc)*100:.1f}%",
            'Forgetting': f"{np.mean(metrics_bt)*100:.1f}% ± {np.std(metrics_bt)*100:.1f}%",
            'Net Growth': f"{np.mean(metrics_net):.1f}",
            'Peak Growth': f"{np.mean(metrics_peak):.1f}",
            'Created': f"{np.mean(metrics_created):.1f}",
            'Pruned': f"{np.mean(metrics_pruned):.1f}",
            'Delay': f"{np.mean(metrics_delay):.1f}" if len(metrics_delay) > 0 else "N/A"
        }
        
        # Save hardware profile for the last seed
        results[name]['profile'] = m['profile']
    
    print("\\n\\n### Comprehensive Benchmark Results\\n")
    print("| Configuration | Final Accuracy | Forgetting (BT) | Net Growth | Peak Growth | Created | Pruned | Shift Detection Delay |")
    print("|--------------|----------------|----------------|------------|-------------|---------|--------|-----------------------|")
    for name, r in results.items():
        print(f"| {name} | {r['Accuracy']} | {r['Forgetting']} | {r['Net Growth']} | {r['Peak Growth']} | {r['Created']} | {r['Pruned']} | {r['Delay']} |")

    print("\\n\\n### Hardware Profiling (Physical vs Logical Capacity)\\n")
    print("| Configuration | Logical Parameters | Allocated Memory (Parameters) | Logical FLOPs / inference | Allocated FLOPs / inference |")
    print("|--------------|--------------------|-------------------------------|---------------------------|-----------------------------|")
    for name, r in results.items():
        prof = r['profile']
        print(f"| {name} | {prof['logical_parameters']} | {prof['allocated_parameters']} | {prof['logical_flops_per_inference']} | {prof['allocated_flops_per_inference']} |")
