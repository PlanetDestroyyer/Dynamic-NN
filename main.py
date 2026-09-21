"""
Autonomous Structural Plasticity for Continual Learning
This script demonstrates the "Sprout & Freeze" architecture on Split-MNIST.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms

# ---------------------------------------------------------
# THE DYNAMIC SPROUT & FREEZE BRAIN
# ---------------------------------------------------------
class DynamicBrain(nn.Module):
    def __init__(self, input_dim=784, output_dim=10):
        super().__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        
        # Start with a tiny structural seed (10 neurons)
        self.hidden_W = nn.Parameter(torch.randn(10, input_dim) * 0.1)
        self.hidden_b = nn.Parameter(torch.zeros(10))
        self.out_W = nn.Parameter(torch.randn(output_dim, 10) * 0.1)
        self.out_b = nn.Parameter(torch.zeros(output_dim))
        
        self.frozen_neurons = 0
        self.active_neurons = 10

    def forward(self, x):
        h = F.relu(F.linear(x, self.hidden_W, self.hidden_b))
        return F.linear(h, self.out_W, self.out_b)
        
    def zero_frozen_grads(self):
        """
        The mathematical freezing mechanism that protects old memories.
        Must be called immediately before optimizer.step()
        """
        if self.frozen_neurons > 0:
            if self.hidden_W.grad is not None:
                self.hidden_W.grad[:self.frozen_neurons, :] = 0
                self.hidden_b.grad[:self.frozen_neurons] = 0
            if self.out_W.grad is not None:
                self.out_W.grad[:, :self.frozen_neurons] = 0

    def grow_and_freeze(self, num_neurons=20):
        """Locks the existing brain and sprouts new capacity."""
        self.frozen_neurons = self.active_neurons
        self.active_neurons += num_neurons
        
        with torch.no_grad():
            new_h_W = torch.randn(num_neurons, self.input_dim, device=self.out_b.device) * 0.1
            new_h_b = torch.zeros(num_neurons, device=self.out_b.device)
            new_o_W = torch.randn(self.output_dim, num_neurons, device=self.out_b.device) * 0.1
            
            self.hidden_W = nn.Parameter(torch.cat([self.hidden_W, new_h_W], dim=0))
            self.hidden_b = nn.Parameter(torch.cat([self.hidden_b, new_h_b], dim=0))
            self.out_W = nn.Parameter(torch.cat([self.out_W, new_o_W], dim=1))

# ---------------------------------------------------------
# THE PAIN RECEPTOR: Memory Aware Synapses (MAS)
# ---------------------------------------------------------
class MASSignal:
    def __init__(self, grow_margin=1.05, fast_alpha=0.1, slow_alpha=0.001, burnin=50):
        self.grow_margin = grow_margin
        self.fast_alpha = fast_alpha
        self.slow_alpha = slow_alpha
        self.burnin = burnin
        self.fast_ema = None
        self.slow_ema = None
        self.step = 0

    def reset(self):
        self.fast_ema = None
        self.slow_ema = None
        self.step = 0

    def score(self, network, X):
        network.zero_grad()
        pred = network(X)
        out_mag = pred.pow(2).mean()
        
        grads = torch.autograd.grad(out_mag, network.hidden_W, retain_graph=True, allow_unused=True)[0]
        if grads is None: return {'mas': 0.0, 'decision': 'HOLD'}
            
        mas_val = torch.norm(grads).item()
        
        if self.fast_ema is None:
            self.fast_ema = mas_val
            self.slow_ema = mas_val
        else:
            self.fast_ema = (1 - self.fast_alpha) * self.fast_ema + self.fast_alpha * mas_val
            self.slow_ema = (1 - self.slow_alpha) * self.slow_ema + self.slow_alpha * mas_val
            
        self.step += 1
        
        if self.step < self.burnin: return {'mas': mas_val, 'decision': 'BURNIN'}
        if self.fast_ema > self.slow_ema * self.grow_margin: return {'mas': mas_val, 'decision': 'GROW'}
        return {'mas': mas_val, 'decision': 'HOLD'}


if __name__ == "__main__":
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Running Continual Learning on device: {device}")

    # Dataset Setup
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Lambda(lambda x: x.view(-1))
    ])
    
    trainset = torchvision.datasets.MNIST(root='./data', train=True, download=True, transform=transform)
    testset = torchvision.datasets.MNIST(root='./data', train=False, download=True, transform=transform)

    def get_task_data(dataset, class_a, class_b, max_samples=3000):
        idx = (dataset.targets == class_a) | (dataset.targets == class_b)
        X = dataset.data[idx].float() / 255.0
        X = X.view(X.size(0), -1)
        y = dataset.targets[idx]
        perm = torch.randperm(X.size(0))[:max_samples]
        return X[perm], y[perm]

    tasks_train = [get_task_data(trainset, i*2, i*2+1) for i in range(5)]
    tasks_test = [get_task_data(testset, i*2, i*2+1, max_samples=1000) for i in range(5)]

    def evaluate(net, tasks_test, current_task_idx):
        net.eval()
        accs = []
        with torch.no_grad():
            for i in range(current_task_idx + 1):
                X_test, y_test = tasks_test[i]
                X_test, y_test = X_test.to(device), y_test.to(device)
                pred = net(X_test)
                
                valid_classes = [i * 2, i * 2 + 1]
                mask = torch.ones_like(pred, dtype=torch.bool)
                mask[:, valid_classes] = False
                pred[mask] = -float('inf')
                
                acc = (pred.argmax(dim=1) == y_test).float().mean().item()
                accs.append(acc)
        net.train()
        return accs

    # Initialize Architecture
    dynamic_net = DynamicBrain(input_dim=784, output_dim=10).to(device)
    opt_dyn = torch.optim.Adam(dynamic_net.parameters(), lr=0.005)
    crit = nn.CrossEntropyLoss()
    
    # We use Margin 1.05 for standard MNIST. 
    # For the harder FashionMNIST, use the Evolutionary Apex Predator genes:
    # signal = MASSignal(grow_margin=1.265, fast_alpha=0.141, slow_alpha=0.0001, burnin=30)
    signal = MASSignal(grow_margin=1.05, burnin=50)

    total_batches = 0
    print("\\nStarting Sequential Training (5 Tasks)...")
    for task_idx, (X_train, y_train) in enumerate(tasks_train):
        print(f"\\n--- Training Task {task_idx+1}/5 ---")
        since_grow = 0
        
        for epoch in range(2):
            for i in range(0, X_train.size(0), 64):
                bx = X_train[i:i+64].to(device)
                by = y_train[i:i+64].to(device)
                if bx.size(0) < 16: continue
                    
                res = signal.score(dynamic_net, bx)
                
                opt_dyn.zero_grad()
                pred = dynamic_net(bx)
                
                # Forward Interference Masking
                active_classes = [task_idx * 2, task_idx * 2 + 1]
                loss_mask = torch.ones_like(pred, dtype=torch.bool)
                loss_mask[:, active_classes] = False
                pred[loss_mask] = -float('inf')
                
                loss = crit(pred, by)
                loss.backward()
                
                dynamic_net.zero_frozen_grads() 
                opt_dyn.step()
                
                if res['decision'] == 'GROW' and since_grow > 20:
                    print(f"  Batch {total_batches:04d} | MAS Spiked! Sprouting new capacity...")
                    dynamic_net.grow_and_freeze(num_neurons=20)
                    opt_dyn = torch.optim.Adam(dynamic_net.parameters(), lr=0.005)
                    signal.reset()
                    since_grow = 0
                    print(f"  -> Network grew to {dynamic_net.active_neurons} neurons. Old knowledge FROZEN.")
                    
                total_batches += 1
                since_grow += 1
                
        accs = evaluate(dynamic_net, tasks_test, task_idx)
        print(f"End of Task {task_idx+1} Accuracies:")
        for i, a in enumerate(accs):
            print(f"  Task {i+1}: {a*100:.1f}%")
