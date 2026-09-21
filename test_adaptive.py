import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms

class DynamicBrain(nn.Module):
    def __init__(self, input_dim=784, output_dim=10):
        super().__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
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
        if self.frozen_neurons > 0:
            if self.hidden_W.grad is not None:
                self.hidden_W.grad[:self.frozen_neurons, :] = 0
                self.hidden_b.grad[:self.frozen_neurons] = 0
            if self.out_W.grad is not None:
                self.out_W.grad[:, :self.frozen_neurons] = 0

    def grow_and_freeze(self, num_neurons=20):
        self.frozen_neurons = self.active_neurons
        self.active_neurons += num_neurons
        with torch.no_grad():
            new_h_W = torch.randn(num_neurons, self.input_dim, device=self.out_b.device) * 0.1
            new_h_b = torch.zeros(num_neurons, device=self.out_b.device)
            new_o_W = torch.randn(self.output_dim, num_neurons, device=self.out_b.device) * 0.1
            self.hidden_W = nn.Parameter(torch.cat([self.hidden_W, new_h_W], dim=0))
            self.hidden_b = nn.Parameter(torch.cat([self.hidden_b, new_h_b], dim=0))
            self.out_W = nn.Parameter(torch.cat([self.out_W, new_o_W], dim=1))

class AdaptiveMASSignal:
    def __init__(self, z_threshold=3.0, burnin=50):
        self.z_threshold = z_threshold
        self.burnin = burnin
        self.step = 0
        self.mean = 0.0
        self.M2 = 0.0

    def reset(self):
        self.step = 0
        self.mean = 0.0
        self.M2 = 0.0

    def score(self, network, X):
        network.zero_grad()
        pred = network(X)
        out_mag = pred.pow(2).mean()
        grads = torch.autograd.grad(out_mag, network.hidden_W, retain_graph=True, allow_unused=True)[0]
        if grads is None: return {'mas': 0.0, 'decision': 'HOLD'}
        
        mas_val = torch.norm(grads).item()
        
        self.step += 1
        delta = mas_val - self.mean
        self.mean += delta / self.step
        delta2 = mas_val - self.mean
        self.M2 += delta * delta2
        
        if self.step <= self.burnin:
            return {'mas': mas_val, 'decision': 'BURNIN'}
            
        variance = self.M2 / (self.step - 1)
        std_dev = variance ** 0.5
        
        z_score = (mas_val - self.mean) / std_dev if std_dev > 1e-8 else 0.0
            
        if z_score > self.z_threshold:
            return {'mas': mas_val, 'decision': 'GROW'}
            
        return {'mas': mas_val, 'decision': 'HOLD'}

device = 'cuda' if torch.cuda.is_available() else 'cpu'
transform = transforms.Compose([transforms.ToTensor(), transforms.Lambda(lambda x: x.view(-1))])

def run_test(dataset_class, name):
    print(f"\n--- Testing {name} ---")
    dataset = dataset_class(root='./data', train=True, download=True, transform=transform)
    
    def get_task(c1, c2):
        idx = (dataset.targets == c1) | (dataset.targets == c2)
        X = dataset.data[idx].float() / 255.0
        X = X.view(X.size(0), -1)
        y = dataset.targets[idx]
        perm = torch.randperm(X.size(0))[:1500]
        return X[perm], y[perm]
        
    tasks = [get_task(i*2, i*2+1) for i in range(5)]
    net = DynamicBrain().to(device)
    opt = torch.optim.Adam(net.parameters(), lr=0.005)
    crit = nn.CrossEntropyLoss()
    signal = AdaptiveMASSignal(z_threshold=4.0, burnin=50) # try 4 sigma
    
    spikes = 0
    total_b = 0
    for t, (X_train, y_train) in enumerate(tasks):
        since_grow = 0
        for _ in range(2):
            for i in range(0, X_train.size(0), 64):
                bx, by = X_train[i:i+64].to(device), y_train[i:i+64].to(device)
                if bx.size(0) < 16: continue
                
                res = signal.score(net, bx)
                opt.zero_grad()
                pred = net(bx)
                active = [t*2, t*2+1]
                mask = torch.ones_like(pred, dtype=torch.bool)
                mask[:, active] = False
                pred[mask] = -float('inf')
                crit(pred, by).backward()
                net.zero_frozen_grads()
                opt.step()
                
                if res['decision'] == 'GROW' and since_grow > 20:
                    print(f"Task {t+1} Spike at batch {total_b}")
                    net.grow_and_freeze(20)
                    opt = torch.optim.Adam(net.parameters(), lr=0.005)
                    signal.reset()
                    since_grow = 0
                    spikes += 1
                since_grow += 1
                total_b += 1
    print(f"Total Spikes for {name}: {spikes}")

run_test(torchvision.datasets.MNIST, "MNIST")
run_test(torchvision.datasets.FashionMNIST, "FashionMNIST")
