import os
import sys
import torch
import torch.nn as nn
import numpy as np

# Add src to path robustly
if os.path.exists('src/dynamic_network.py'):
    sys.path.append(os.path.abspath('src'))
elif os.path.exists('../src/dynamic_network.py'):
    sys.path.append(os.path.abspath('../src'))
from dynamic_network import PyTorchDynamicNetwork

class ReplayBuffer:
    def __init__(self, capacity: int = 500):
        self.capacity = capacity
        self.X = None
        self.y = None
        self.head_ids = None

    def add_data(self, X_new: torch.Tensor, y_new: torch.Tensor, head_id: int, num_samples: int = 100):
        idx = torch.randperm(X_new.size(0))[:num_samples]
        X_sub = X_new[idx].clone().detach()
        y_sub = y_new[idx].clone().detach()
        h_sub = torch.full((num_samples,), head_id, dtype=torch.long)

        if self.X is None:
            self.X, self.y, self.head_ids = X_sub, y_sub, h_sub
        else:
            self.X = torch.cat([self.X, X_sub], dim=0)
            self.y = torch.cat([self.y, y_sub], dim=0)
            self.head_ids = torch.cat([self.head_ids, h_sub], dim=0)
            if self.X.size(0) > self.capacity:
                keep = torch.randperm(self.X.size(0))[:self.capacity]
                self.X, self.y, self.head_ids = self.X[keep], self.y[keep], self.head_ids[keep]

    def sample(self, batch_size: int = 64):
        if self.X is None: return None, None, None
        size = self.X.size(0)
        idx = torch.randint(0, size, (min(batch_size, size),))
        return self.X[idx], self.y[idx], self.head_ids[idx]
    
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

print("\\n" + "="*50)
print("--- TASK-FREE CONTINUOUS LEARNING BENCHMARK ---")
print("="*50)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# 1. Generate the Continuous Data Stream
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

# 2. Initialize the Network
dynamic = PyTorchDynamicNetwork(input_dim=2, output_dim=1, max_neurons=500).to(device)
optimizer = torch.optim.Adam(dynamic.parameters(), lr=1e-2)
criterion = nn.MSELoss()
replay = ReplayBuffer(capacity=500)

print("\\n[Streaming Data...] The network does NOT know when tasks change!")

# Variables for the stream
stream_loss_ema = 0.5
batches_since_grow = 0
total_batches = 0
active_head = 0

# To prove it works, we will simulate a stream by iterating over the 3 datasets blindly
for true_task_id, (name, X, y) in enumerate(tasks):
    # In a true stream, it wouldn't even know this loop boundary exists.
    # We just use this loop to feed the data in chunks.
    print(f"\\n-- Stream Segment (Secretly Task {true_task_id+1}: {name}) --")
    
    # We train on this segment for a while (e.g. 500 batches)
    for batch_idx in range(500):
        optimizer.zero_grad()
        
        # Sample a random batch from the current stream segment
        idx = torch.randperm(X.size(0))[:128]
        bx, by = X[idx], y[idx]
        
        # 1. Forward Pass using the CURRENT ACTIVE HEAD
        # The network has no idea what task this is, it just uses its latest head!
        pred_all_heads = dynamic(bx)
        pred = pred_all_heads[:, active_head].unsqueeze(1)
        loss = criterion(pred, by)
        
        # 2. Replay Buffer (Memory Consolidation)
        if replay.has_data():
            rx, ry, r_heads = replay.sample(128)
            rx, ry, r_heads = rx.to(device), ry.to(device), r_heads.to(device)
            r_pred_all = dynamic(rx)
            # Route replay samples to the specific head they were recorded with
            r_pred = r_pred_all[torch.arange(rx.size(0)), r_heads].unsqueeze(1)
            loss_r = criterion(r_pred, ry)
            
            g_t = torch.autograd.grad(loss, dynamic.W, retain_graph=True, allow_unused=True)[0]
            g_r = torch.autograd.grad(loss_r, dynamic.W, retain_graph=True, allow_unused=True)[0]
            if g_t is not None and g_r is not None:
                dynamic.update_stress(g_t, g_r)
        
        loss.backward()
        optimizer.step()
        
        # 3. Anomaly Detection: Update Loss EMA and Check for Task Shift
        dynamic.update_loss_ema(loss.item(), beta=0.9, fast_beta=0.5, slow_beta=0.99)
        stream_loss_ema = 0.9 * stream_loss_ema + 0.1 * loss.item()
        batches_since_grow += 1
        total_batches += 1
        
        # Calculate max_stress early so we can use it for Autonomous Detection
        max_stress = dynamic.get_max_neuron_stress()
        
        # AUTONOMOUS TASK SHIFT DETECTION (MACD + Gradient Stress)
        # We enforce a 100-batch Burn-In period so the EMAs can calibrate to the baseline loss
        shift_detected = dynamic.detect_task_shift(max_stress, margin=0.15, stress_threshold=0.2)
        
        if shift_detected and total_batches > 100 and batches_since_grow > 100:
            print(f"  [SHIFT DETECTED] (MACD Fast: {dynamic.fast_loss_ema:.2f}, Slow: {dynamic.slow_loss_ema:.2f} | Max Stress: {max_stress:.2f})")
            print(f"  [SHIFT DETECTED] Data distribution shift triggered head expansion.")
            
            new_head = dynamic.grow_output_head(num_connections=10)
            _reset_adam_for_neuron(optimizer, dynamic, new_head)
            active_head += 1
            
            # Reset EMAs and stress so it doesn't trigger again immediately
            dynamic.fast_loss_ema = stream_loss_ema
            dynamic.slow_loss_ema = stream_loss_ema
            dynamic.neuron_stress.zero_()
            batches_since_grow = 0
            
        # 4. Localized Stress Freeze
        capped_loss = min(dynamic.loss_ema, 1.0)
        current_threshold = max(0.05, 0.5 - (1.0 * capped_loss)) # Highly sensitive to stress
        
        if max_stress > current_threshold:
            n_frozen, actual_thresh = dynamic.stress_freeze(base_threshold=0.5, sensitivity_factor=1.0)
            if n_frozen > 0:
                dynamic.grow_neuron(10)
                _reset_adam_for_neuron(optimizer, dynamic, dynamic.active_neurons - 1)
                batches_since_grow = 0
                if batch_idx % 100 == 0:
                    print(f"  [Batch {batch_idx}] LOCAL FREEZE: {n_frozen} conns. Threshold={actual_thresh:.2f}")
        
        # 5. Autonomous Growth (Capacity Expansion)
        elif stream_loss_ema > 0.1 and batches_since_grow > 50 and max_stress <= current_threshold:
            dynamic.grow_neuron(10)
            _reset_adam_for_neuron(optimizer, dynamic, dynamic.active_neurons - 1)
            batches_since_grow = 0
            if batch_idx % 100 == 0:
                print(f"  [Batch {batch_idx}] GROW(Loss): active={dynamic.active_neurons} loss={stream_loss_ema:.4f}")
                
        # 6. Continuous Reservoir Sampling
        # 10% chance every batch to save a few samples into the rolling memory buffer
        if torch.rand(1).item() < 0.10:
            replay.add_data(bx, by, active_head, num_samples=16)

print("\\n[Stream Finished]")
print(f"Total Output Heads Spawned Autonomously: {active_head + 1} (Expected: 3)")
print(f"Total Neurons Built: {dynamic.active_neurons}")

print("\\n--- Evaluating Final Memory Retention ---")
# To evaluate retention, we must map the evaluation tasks to the heads the network spawned.
# Since it spawned heads sequentially, Task 0 should be Head 0, Task 1 -> Head 1, etc.
for t_id, (name, X, y) in enumerate(tasks):
    eval_head = min(t_id, active_head) # Safety check if it didn't spawn enough heads
    acc = ((dynamic(X)[:, eval_head].unsqueeze(1) > 0.5).float() == y).float().mean().item()
    print(f"  Task {t_id+1} ({name}) -> Evaluated on Head {eval_head}: {acc*100:.1f}%")
