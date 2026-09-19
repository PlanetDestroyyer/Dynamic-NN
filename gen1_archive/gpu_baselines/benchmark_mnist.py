import torch
import torch.nn as nn
import numpy as np
from torchvision import datasets, transforms
from gpu_dynamic_nn import PyTorchDynamicNetwork
from gpu_sequential_cl import ReplayBuffer

def get_mnist_tasks():
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])
    
    train_dataset = datasets.MNIST('./data', train=True, download=True, transform=transform)
    test_dataset = datasets.MNIST('./data', train=False, download=True, transform=transform)
    
    tasks = []
    pairs = [(0, 1), (2, 3), (4, 5), (6, 7), (8, 9)]
    
    for (c0, c1) in pairs:
        # Train data
        train_mask = (train_dataset.targets == c0) | (train_dataset.targets == c1)
        X_train = train_dataset.data[train_mask].float() / 255.0
        X_train = (X_train - 0.1307) / 0.3081
        X_train = X_train.view(-1, 28*28)
        y_train = (train_dataset.targets[train_mask] == c1).float().view(-1, 1)
        
        # Test data
        test_mask = (test_dataset.targets == c0) | (test_dataset.targets == c1)
        X_test = test_dataset.data[test_mask].float() / 255.0
        X_test = (X_test - 0.1307) / 0.3081
        X_test = X_test.view(-1, 28*28)
        y_test = (test_dataset.targets[test_mask] == c1).float().view(-1, 1)
        
        name = f"Digits {c0} vs {c1}"
        tasks.append((name, X_train, y_train, X_test, y_test))
        
    return tasks

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

def _reset_adam_for_neuron(optimizer: torch.optim.Adam, net: nn.Module, new_idx: int):
    for group in optimizer.param_groups:
        for p in group['params']:
            if p is net.W or p is net.b:
                state = optimizer.state[p]
                if 'exp_avg' in state:
                    state['exp_avg'][new_idx, :] = 0.0
                    state['exp_avg'][:, new_idx] = 0.0
                    state['exp_avg_sq'][new_idx, :] = 0.0
                    state['exp_avg_sq'][:, new_idx] = 0.0

def train_batched_dynamic_gpu(net, loader, replay, optimizer, criterion, device):
    net.train()
    total_loss = 0
    batches = 0
    
    for bx, by in loader:
        bx, by = bx.to(device), by.to(device)
        optimizer.zero_grad()
        
        pred = net(bx)
        loss = criterion(pred, by)
        
        if replay.has_data():
            rx, ry = replay.sample(64)
            rx, ry = rx.to(device), ry.to(device)
            r_pred = net(rx)
            loss_r = criterion(r_pred, ry)
            
            grads_task = torch.autograd.grad(loss, net.W, retain_graph=True, allow_unused=True)[0]
            grads_replay = torch.autograd.grad(loss_r, net.W, retain_graph=True, allow_unused=True)[0]
            
            if grads_task is not None and grads_replay is not None:
                net.update_stress(grads_task, grads_replay)
                
        loss.backward()
        net.update_fisher()
        optimizer.step()
        
        total_loss += loss.item()
        batches += 1
        
    return total_loss / batches

def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Running MNIST Benchmark on {device}...")
    
    tasks = get_mnist_tasks()
    
    print("\n" + "="*50)
    print("--- Training Baseline (Standard Neural Network) ---")
    print("="*50)
    
    baseline_net = BaselineNet().to(device)
    optimizer_b = torch.optim.Adam(baseline_net.parameters(), lr=1e-3)
    criterion_b = nn.BCELoss()
    
    for task_id, (name, X_train, y_train, X_test, y_test) in enumerate(tasks):
        print(f"\nTraining Baseline on {name}...")
        
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
        with torch.no_grad():
            for eval_id in range(task_id + 1):
                t_name, _, _, t_X_test, t_y_test = tasks[eval_id]
                t_X_test_d = t_X_test.to(device)
                t_y_test_d = t_y_test.to(device)
                acc = ((baseline_net(t_X_test_d) > 0.5).float() == t_y_test_d).float().mean().item()
                print(f"  {t_name}: {acc*100:.1f}%")
                
    print("\n" + "="*50)
    print("--- Training Our Dynamic Network ---")
    print("="*50)
    
    dynamic_net = PyTorchDynamicNetwork(input_dim=784, output_dim=1, max_neurons=500).to(device)
    replay = ReplayBuffer(capacity=500)
    optimizer = torch.optim.Adam(dynamic_net.parameters(), lr=1e-3)
    criterion = nn.BCELoss()
    
    for task_id, (name, X_train, y_train, X_test, y_test) in enumerate(tasks):
        print(f"\nTraining Dynamic Network on {name}...")
        
        dataset = torch.utils.data.TensorDataset(X_train, y_train)
        loader = torch.utils.data.DataLoader(dataset, batch_size=256, shuffle=True)
        
        loss_ema = 0.5
        stress_ema = 0.0
        batches_since_grow = 0
        
        for epoch in range(10):
            dynamic_net.train()
            
            for bx, by in loader:
                bx, by = bx.to(device), by.to(device)
                optimizer.zero_grad()
                
                pred = dynamic_net(bx)
                loss = criterion(pred, by)
                
                if replay.has_data():
                    rx, ry = replay.sample(64)
                    rx, ry = rx.to(device), ry.to(device)
                    r_pred = dynamic_net(rx)
                    loss_r = criterion(r_pred, ry)
                    
                    grads_task = torch.autograd.grad(loss, dynamic_net.W, retain_graph=True, allow_unused=True)[0]
                    grads_replay = torch.autograd.grad(loss_r, dynamic_net.W, retain_graph=True, allow_unused=True)[0]
                    
                    if grads_task is not None and grads_replay is not None:
                        dynamic_net.update_stress(grads_task, grads_replay)
                        
                loss.backward()
                dynamic_net.update_fisher()
                optimizer.step()
                
                loss_ema = 0.9 * loss_ema + 0.1 * loss.item()
                stress_ema = dynamic_net.get_network_stress()
                batches_since_grow += 1
                
                # Freeze condition
                if stress_ema > 0.3:
                    n_frozen = dynamic_net.stress_freeze(threshold=0.3)
                    if n_frozen > 0:
                        dynamic_net.grow_neuron(num_connections=10)
                        _reset_adam_for_neuron(optimizer, dynamic_net, dynamic_net.active_neurons - 1)
                        batches_since_grow = 0
                        print(f"  [Batch] FREEZE: {n_frozen} conns. Grew 1. active={dynamic_net.active_neurons}")
                
                # Grow condition (loss)
                elif loss_ema > 0.2 and batches_since_grow > 20 and stress_ema <= 0.3:
                    dynamic_net.grow_neuron(num_connections=10)
                    _reset_adam_for_neuron(optimizer, dynamic_net, dynamic_net.active_neurons - 1)
                    batches_since_grow = 0
                    print(f"  [Batch] GROW(Loss): active={dynamic_net.active_neurons} loss={loss_ema:.4f}")
            
            print(f"  [Epoch {epoch+1:2d}] active={dynamic_net.active_neurons} loss={loss_ema:.4f}")
            
        replay.add_data(X_train, y_train, num_samples=100)
        
        dynamic_net.eval()
        print(f"Accuracy after {name}:")
        with torch.no_grad():
            for eval_id in range(task_id + 1):
                t_name, _, _, t_X_test, t_y_test = tasks[eval_id]
                t_X_test_d = t_X_test.to(device)
                t_y_test_d = t_y_test.to(device)
                acc = ((dynamic_net(t_X_test_d) > 0.5).float() == t_y_test_d).float().mean().item()
                print(f"  {t_name}: {acc*100:.1f}%")
        
        print(f"  Active Neurons: {dynamic_net.active_neurons} (trainable={dynamic_net.n_trainable()}, frozen={dynamic_net.n_frozen()})")

if __name__ == "__main__":
    main()
