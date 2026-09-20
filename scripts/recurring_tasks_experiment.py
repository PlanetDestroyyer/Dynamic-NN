import torch
import torch.nn as nn
import numpy as np
import os
import sys
import matplotlib.pyplot as plt

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
    labels = ((x * y) > 0).astype(int)
    X = np.c_[x + np.random.randn(n_samples)*noise, y + np.random.randn(n_samples)*noise]
    return torch.FloatTensor(X), torch.FloatTensor(labels).unsqueeze(1)

def run_experiment(seed=42):
    torch.manual_seed(seed)
    np.random.seed(seed)
    device = torch.device('cpu')

    # A -> B -> A -> C -> B -> A
    samples_per_phase = 4000
    phases = [
        ("A (Circles)", *make_circles(samples_per_phase)),
        ("B (Linear)", *make_linear(samples_per_phase)),
        ("A (Circles)", *make_circles(samples_per_phase)),
        ("C (XOR)", *make_xor(samples_per_phase)),
        ("B (Linear)", *make_linear(samples_per_phase)),
        ("A (Circles)", *make_circles(samples_per_phase))
    ]
    
    # Store clean evaluation sets
    eval_sets = {
        "A": make_circles(1000),
        "B": make_linear(1000),
        "C": make_xor(1000)
    }

    stream_X = torch.cat([p[1] for p in phases])
    stream_y = torch.cat([p[2] for p in phases])
    
    batch_size = 16
    task_boundaries = [i * (samples_per_phase // batch_size) for i in range(len(phases))]
    
    # Network Setup
    dynamic = PyTorchDynamicNetwork(input_dim=2, output_dim=1, max_neurons=1000).to(device)
    
    # Use Weight Decay to allow compression!
    optimizer = torch.optim.Adam(dynamic.parameters(), lr=0.01, weight_decay=1e-4)
    criterion = nn.MSELoss()
    replay = ReplayBuffer(capacity=500)
    
    stream_loss_ema = 0.5
    batches_since_grow = 0
    total_batches = 0
    
    total_created = 0
    total_pruned = 0
    
    # Tracking
    history = {
        'batch': [],
        'active_neurons': [],
        'created': [],
        'pruned': [],
        'acc_A': [],
        'acc_B': [],
        'acc_C': []
    }
    
    def evaluate():
        accs = {}
        for name, (X, y) in eval_sets.items():
            acc = ((dynamic(X)[:, 0].unsqueeze(1) > 0.5).float() == y).float().mean().item()
            accs[name] = acc
        return accs

    current_phase = 0
    
    print(f"Starting 6-Phase Plateau Experiment (Seed {seed})")
    
    for i in range(0, stream_X.size(0), batch_size):
        if total_batches in task_boundaries:
            if current_phase > 0:
                print(f"--- End of Phase {current_phase} ---")
            current_phase += 1
            print(f"\\nStarting Phase {current_phase}: {phases[current_phase-1][0]}")
            
        bx = stream_X[i:i+batch_size].to(device)
        by = stream_y[i:i+batch_size].to(device)
        
        optimizer.zero_grad()
        
        # Forward pass
        pred_all_heads = dynamic(bx)
        pred = pred_all_heads[:, 0].unsqueeze(1)
        loss = criterion(pred, by)
        
        # Replay
        if replay.has_data():
            rx, ry, _ = replay.sample(64)
            rx, ry = rx.to(device), ry.to(device)
            r_pred_all = dynamic(rx)
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
        shift_detected = dynamic.detect_task_shift(max_stress, margin=0.15, stress_threshold=0.2)
        if shift_detected and total_batches > 50 and batches_since_grow > 50:
            dynamic.fast_loss_ema = stream_loss_ema
            dynamic.slow_loss_ema = stream_loss_ema
            batches_since_grow = 0
                
        # Freezing
        capped_loss = min(dynamic.loss_ema, 1.0)
        current_threshold = max(0.05, 0.5 - (1.0 * capped_loss))
        dynamic.current_threshold = current_threshold
        if max_stress > current_threshold:
            n_frozen, _ = dynamic.stress_freeze(base_threshold=0.5, sensitivity_factor=1.0)
            if n_frozen > 0:
                ret = dynamic.grow_neuron(10)
                if ret != -1: total_created += 1
                batches_since_grow = 0
                
        # Capacity Expansion
        if stream_loss_ema > 0.2 and batches_since_grow > 20 and max_stress <= current_threshold:
            ret = dynamic.grow_neuron(10)
            if ret != -1: total_created += 1
            batches_since_grow = 0
                
        if torch.rand(1).item() < 0.10:
            replay.add_data(bx, by, 0, num_samples=16)

        # Compression
        if total_batches % 100 == 0:
            p, m = dynamic.compress_network(prune_threshold=0.15, similarity_threshold=0.50)
            total_pruned += (p + m)

        # Tracking
        if total_batches % 50 == 0:
            history['batch'].append(total_batches)
            history['active_neurons'].append(dynamic.active_neurons - len(dynamic.free_neurons))
            history['created'].append(total_created)
            history['pruned'].append(total_pruned)
            
            accs = evaluate()
            history['acc_A'].append(accs['A'])
            history['acc_B'].append(accs['B'])
            history['acc_C'].append(accs['C'])

    print(f"\\n--- End of Experiment ---")
    print(f"Final Active Neurons: {history['active_neurons'][-1]}")
    print(f"Total Created: {total_created}")
    print(f"Total Pruned: {total_pruned}")
    
    return history, task_boundaries

def plot_experiment(history, task_boundaries):
    os.makedirs('../output', exist_ok=True)
    
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10), sharex=True)
    
    # Subplot 1: Structural Lifecycle
    ax1.plot(history['batch'], history['active_neurons'], label='Active Neurons (Net)', color='purple', linewidth=2)
    ax1.plot(history['batch'], history['created'], label='Cumulative Created', color='green', linestyle='--')
    ax1.plot(history['batch'], history['pruned'], label='Cumulative Pruned', color='red', linestyle='--')
    
    for tb in task_boundaries:
        ax1.axvline(x=tb, color='gray', linestyle=':', alpha=0.5)
        
    ax1.set_ylabel('Number of Neurons')
    ax1.set_title('Structural Lifecycle (A -> B -> A -> C -> B -> A)')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Subplot 2: Accuracies
    ax2.plot(history['batch'], history['acc_A'], label='Task A (Circles)', color='blue')
    ax2.plot(history['batch'], history['acc_B'], label='Task B (Linear)', color='orange')
    ax2.plot(history['batch'], history['acc_C'], label='Task C (XOR)', color='red')
    
    for tb in task_boundaries:
        ax2.axvline(x=tb, color='gray', linestyle=':', alpha=0.5)
        
    ax2.set_ylabel('Accuracy')
    ax2.set_xlabel('Batch')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # Phase Labels
    phase_names = ['A', 'B', 'A', 'C', 'B', 'A']
    for i, tb in enumerate(task_boundaries):
        ax2.text(tb + 50, 0.45, f"Phase: {phase_names[i]}", fontsize=10, verticalalignment='bottom')
        
    plt.tight_layout()
    plt.savefig('../output/plateau_experiment.png', dpi=150)
    print("\\nPlot saved to output/plateau_experiment.png")

if __name__ == "__main__":
    history, boundaries = run_experiment(seed=42)
    plot_experiment(history, boundaries)
