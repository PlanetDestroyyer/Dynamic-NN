import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms
import random
import copy

device = 'cuda' if torch.cuda.is_available() else 'cpu'

# ---------------------------------------------------------
# CORE CLASSES
# ---------------------------------------------------------
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
        
        if self.fast_ema > self.slow_ema * self.grow_margin: 
            return {'mas': mas_val, 'decision': 'GROW'}
        return {'mas': mas_val, 'decision': 'HOLD'}


# ---------------------------------------------------------
# EVOLUTIONARY ALGORITHM LOGIC
# ---------------------------------------------------------

# Truncated fast dataset (3 tasks, 500 samples each)
transform = transforms.Compose([transforms.ToTensor(), transforms.Lambda(lambda x: x.view(-1))])
fashion_train = torchvision.datasets.FashionMNIST(root='./data', train=True, download=True, transform=transform)
fashion_test = torchvision.datasets.FashionMNIST(root='./data', train=False, download=True, transform=transform)

def get_task(dataset, c1, c2, samples=3000):
    idx = (dataset.targets == c1) | (dataset.targets == c2)
    X = dataset.data[idx].float() / 255.0
    X = X.view(X.size(0), -1)
    y = dataset.targets[idx]
    perm = torch.randperm(X.size(0))[:samples]
    return X[perm], y[perm]

# Train on all 5 tasks (full dataset) since you have a Colab GPU!
tasks_train = [get_task(fashion_train, i*2, i*2+1, 3000) for i in range(5)]
tasks_test = [get_task(fashion_test, i*2, i*2+1, 1000) for i in range(5)]

def evaluate_genome(genome):
    net = DynamicBrain().to(device)
    opt = torch.optim.Adam(net.parameters(), lr=0.005)
    crit = nn.CrossEntropyLoss()
    signal = MASSignal(grow_margin=genome['margin'], fast_alpha=genome['fast'], slow_alpha=genome['slow'], burnin=30)
    
    total_spikes = 0
    for t_idx, (X_train, y_train) in enumerate(tasks_train):
        since_grow = 0
        for _ in range(2): # 2 epochs
            for i in range(0, X_train.size(0), 64):
                bx, by = X_train[i:i+64].to(device), y_train[i:i+64].to(device)
                if bx.size(0) < 16: continue
                
                res = signal.score(net, bx)
                opt.zero_grad()
                pred = net(bx)
                
                mask = torch.ones_like(pred, dtype=torch.bool)
                mask[:, [t_idx*2, t_idx*2+1]] = False
                pred[mask] = -float('inf')
                
                crit(pred, by).backward()
                net.zero_frozen_grads()
                opt.step()
                
                if res['decision'] == 'GROW' and since_grow > 15:
                    net.grow_and_freeze(20)
                    opt = torch.optim.Adam(net.parameters(), lr=0.005)
                    signal.reset()
                    since_grow = 0
                    total_spikes += 1
                since_grow += 1
                
    # Evaluate accuracy on all 5 tasks
    net.eval()
    task_accs = []
    with torch.no_grad():
        for i in range(5):
            X_test, y_test = tasks_test[i]
            X_test, y_test = X_test.to(device), y_test.to(device)
            pred = net(X_test)
            mask = torch.ones_like(pred, dtype=torch.bool)
            mask[:, [i*2, i*2+1]] = False
            pred[mask] = -float('inf')
            acc = (pred.argmax(dim=1) == y_test).float().mean().item()
            task_accs.append(acc)
            
    avg_acc = sum(task_accs) / len(task_accs)
    
    # FITNESS FUNCTION:
    # High accuracy is good. Sprouting infinitely is bad. 
    # For 5 tasks, we should exactly transition 4 times (4 spikes).
    spike_penalty = abs(total_spikes - 4) * 0.05
    fitness = avg_acc - spike_penalty
    
    return fitness, avg_acc, total_spikes

def mutate(genome):
    # Add Gaussian noise to genes to explore the mathematical landscape
    new_genome = copy.deepcopy(genome)
    new_genome['margin'] += random.gauss(0, 0.05)
    new_genome['fast'] += random.gauss(0, 0.05)
    new_genome['slow'] += random.gauss(0, 0.005)
    
    # Clip to valid mathematical ranges
    new_genome['margin'] = max(1.0, min(new_genome['margin'], 2.0))
    new_genome['fast'] = max(0.01, min(new_genome['fast'], 0.5))
    new_genome['slow'] = max(0.0001, min(new_genome['slow'], 0.1))
    return new_genome

def run_evolution():
    POP_SIZE = 8
    GENERATIONS = 5
    
    # Initialize random primitive population
    population = []
    for _ in range(POP_SIZE):
        population.append({
            'margin': random.uniform(1.01, 1.5),
            'fast': random.uniform(0.01, 0.2),
            'slow': random.uniform(0.0005, 0.05)
        })
        
    print("Beginning Evolutionary Training...")
    
    for gen in range(GENERATIONS):
        print(f"\\n--- Generation {gen+1} ---")
        
        results = []
        for i, genome in enumerate(population):
            fitness, acc, spikes = evaluate_genome(genome)
            results.append((fitness, acc, spikes, genome))
            print(f" Organism {i+1} | Fit: {fitness:.3f} | Acc: {acc:.3f} | Spikes: {spikes} | Genes: [M:{genome['margin']:.3f}, F:{genome['fast']:.3f}, S:{genome['slow']:.4f}]")
            
        # Sort by survival fitness
        results.sort(key=lambda x: x[0], reverse=True)
        elites = results[:3] # Top 3 survive and breed
        
        print(f" => Gen {gen+1} Elite Average Fitness: {sum(r[0] for r in elites)/3:.3f}")
        
        if gen == GENERATIONS - 1:
            best = elites[0]
            print(f"\\n🏆 EVOLUTION COMPLETE. THE APEX PREDATOR HAS BEEN FOUND!")
            print(f"Best Genes: Margin={best[3]['margin']:.4f}, Fast_Alpha={best[3]['fast']:.4f}, Slow_Alpha={best[3]['slow']:.5f}")
            print(f"Achieved Accuracy: {best[1]*100:.1f}%, Spikes: {best[2]}")
            break
            
        # Breeding (Crossover & Mutation)
        new_population = [elites[0][3], elites[1][3], elites[2][3]] # Keep elites unmodified
        
        while len(new_population) < POP_SIZE:
            # Pick a random elite parent
            parent = random.choice(elites)[3]
            child = mutate(parent)
            new_population.append(child)
            
        population = new_population

if __name__ == '__main__':
    run_evolution()
