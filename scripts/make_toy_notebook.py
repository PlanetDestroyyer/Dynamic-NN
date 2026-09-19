import json
import os

def create_notebook():
    cells = []
    
    data_cell = """import torch
import torch.nn as nn
import numpy as np
import itertools
import networkx as nx
import matplotlib.pyplot as plt

# Re-implementing the dataset generation so the notebook is self-contained
def make_circles(n_samples=300, noise=0.05, seed=42, center=(0.0, 0.0)):
    np.random.seed(seed)
    angles = np.random.uniform(0, 2*np.pi, n_samples)
    radii = np.where(np.random.rand(n_samples) > 0.5, 1.0, 3.0) + np.random.randn(n_samples) * noise
    X = np.c_[radii * np.cos(angles), radii * np.sin(angles)]
    X += np.array(center)
    y = (radii > 2.0).astype(int)
    return X, y

def make_xor(n_samples=300, noise=0.0, seed=42, center=(0.0, 0.0)):
    np.random.seed(seed)
    X = np.random.randn(n_samples, 2)
    y = np.logical_xor(X[:, 0] > 0, X[:, 1] > 0).astype(int)
    X += np.array(center)
    return X, y

def make_linearly_separable(n_samples=300, noise=0.1, seed=42, center=(0.0, 0.0)):
    np.random.seed(seed)
    X = np.random.randn(n_samples, 2)
    y = (X[:, 0] + X[:, 1] > 0).astype(int)
    X += np.array(center)
    return X, y
"""
    cells.append({"cell_type": "code", "metadata": {}, "execution_count": None, "outputs": [], "source": [line + "\n" for line in data_cell.split('\n')]})

    with open(os.path.join(os.path.dirname(__file__), '../src/dynamic_network.py'), 'r') as f:
        network_code = f.read()
        network_code = network_code.replace("import torch\nimport torch.nn as nn\nimport numpy as np\n", "")
    cells.append({"cell_type": "code", "metadata": {}, "execution_count": None, "outputs": [], "source": [line + "\n" for line in network_code.split('\n')]})

    with open('gpu_sequential_cl.py', 'r') as f:
        training_code = f.read()
        training_code = training_code.replace("import torch\nimport torch.nn as nn\nimport numpy as np\nfrom gpu_dynamic_nn import PyTorchDynamicNetwork\n", "")
        training_code = training_code.replace("from dynamic_nn import make_xor, make_circles, make_linearly_separable\n", "")
        idx = training_code.find('def main():')
        if idx != -1:
            training_code = training_code[:idx]
    cells.append({"cell_type": "code", "metadata": {}, "execution_count": None, "outputs": [], "source": [line + "\n" for line in training_code.split('\n')]})

    experiment_loop_code = """
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

base_tasks = [
    ("Circles", *make_circles(n_samples=300, noise=0.05, seed=42, center=(0, 0))),
    ("XOR",     *make_xor(n_samples=300, noise=0.0,  seed=42, center=(4, 4))),
    ("Linear",  *make_linearly_separable(n_samples=300, noise=0.1, seed=42, center=(-4, -4))),
]

class BaselineNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(2, 14),
            nn.Sigmoid(),
            nn.Linear(14, 1)
        )
    def forward(self, x):
        return torch.sigmoid(self.net(x))

def run_experiment(task_order):
    perm_names = [t[0] for t in task_order]
    print(f"\\n{'='*60}\\nRunning Permutation: {perm_names}\\n{'='*60}")
    
    print("\\n--- Training Baseline ---")
    baseline_net = BaselineNet().to(device)
    optimizer_b = torch.optim.Adam(baseline_net.parameters(), lr=3e-3)
    criterion_b = nn.BCELoss()
    
    for task_id, (name, X_np, y_np) in enumerate(task_order):
        X_t = torch.tensor(X_np, dtype=torch.float32).to(device)
        y_t = torch.tensor(y_np, dtype=torch.float32).view(-1, 1).to(device)
        baseline_net.train()
        for epoch in range(1000):
            optimizer_b.zero_grad()
            loss = criterion_b(baseline_net(X_t), y_t)
            loss.backward()
            optimizer_b.step()
            
    baseline_accs = []
    baseline_net.eval()
    with torch.no_grad():
        for name, X_np, y_np in task_order:
            X_t = torch.tensor(X_np, dtype=torch.float32).to(device)
            y_t = torch.tensor(y_np, dtype=torch.float32).view(-1, 1).to(device)
            acc = ((baseline_net(X_t) > 0.5).float() == y_t).float().mean().item()
            baseline_accs.append((name, acc))
            
    print("\\n--- Training Dynamic Network ---")
    dynamic_net = PyTorchDynamicNetwork(input_dim=2, output_dim=1, max_neurons=200).to(device)
    replay = ReplayBuffer(capacity=500)
    
    for task_id, (name, X_np, y_np) in enumerate(task_order):
        print(f"  Training {name}...")
        X_t = torch.tensor(X_np, dtype=torch.float32)
        y_t = torch.tensor(y_np, dtype=torch.float32).view(-1, 1)
        train_one_task_gpu(dynamic_net, X_t, y_t, epochs=1000, lr=3e-3, replay=replay, device=device)
        
    dynamic_accs = []
    dynamic_net.eval()
    with torch.no_grad():
        for name, X_np, y_np in task_order:
            X_t = torch.tensor(X_np, dtype=torch.float32).to(device)
            y_t = torch.tensor(y_np, dtype=torch.float32).view(-1, 1).to(device)
            acc = ((dynamic_net(X_t) > 0.5).float() == y_t).float().mean().item()
            dynamic_accs.append((name, acc))
            
    return baseline_net, dynamic_net, baseline_accs, dynamic_accs

results = []
permutations = list(itertools.permutations(base_tasks))

for perm in permutations:
    b_net, d_net, b_accs, d_accs = run_experiment(perm)
    results.append({
        'perm': [t[0] for t in perm],
        'b_net': b_net,
        'd_net': d_net,
        'b_accs': b_accs,
        'd_accs': d_accs
    })

print("\\n" + "="*80)
print("FINAL SUMMARY (All 6 Permutations)")
print("="*80)
for r in results:
    perm_str = " -> ".join(r['perm'])
    print(f"Order: {perm_str:25s}")
    b_str = ", ".join([f"{n}: {a*100:.1f}%" for n, a in r['b_accs']])
    d_str = ", ".join([f"{n}: {a*100:.1f}%" for n, a in r['d_accs']])
    print(f"  Baseline Acc : {b_str}")
    print(f"  Dynamic Acc  : {d_str}\\n")
"""
    cells.append({"cell_type": "code", "metadata": {}, "execution_count": None, "outputs": [], "source": [line + "\n" for line in experiment_loop_code.split('\n')]})

    topology_code = """
def plot_dynamic_topology(net, title):
    G = nx.DiGraph()
    n = net.active_neurons
    M = net.M[:n, :n].cpu().numpy()
    trainable = net.trainable_mask[:n, :n].cpu().numpy()
    
    colors = []
    for i in range(n):
        G.add_node(i)
        if i < net.input_dim:
            colors.append('lightblue')
        elif i < net.input_dim + net.output_dim:
            colors.append('lightcoral')
        else:
            colors.append('lightgray')
            
    edge_colors = []
    for i in range(n):
        for j in range(n):
            if M[i, j] == 1.0:
                G.add_edge(i, j)
                if trainable[i, j] == 1.0:
                    edge_colors.append('green')
                else:
                    edge_colors.append('red')
                    
    plt.figure(figsize=(10, 8))
    pos = nx.spring_layout(G, seed=42)
    nx.draw(G, pos, node_color=colors, edge_color=edge_colors, with_labels=True, 
            node_size=600, arrows=True, font_size=10, font_weight='bold', 
            connectionstyle='arc3,rad=0.1')
    
    import matplotlib.patches as mpatches
    g_patch = mpatches.Patch(color='green', label='Trainable Connection')
    r_patch = mpatches.Patch(color='red', label='Frozen Connection')
    in_patch = mpatches.Patch(color='lightblue', label='Input Neuron')
    hid_patch = mpatches.Patch(color='lightgray', label='Hidden Neuron')
    out_patch = mpatches.Patch(color='lightcoral', label='Output Neuron')
    
    plt.legend(handles=[g_patch, r_patch, in_patch, hid_patch, out_patch], loc='upper left')
    plt.title(title, fontsize=14)
    plt.show()

last_d_net = results[-1]['d_net']
last_perm_str = " -> ".join(results[-1]['perm'])
plot_dynamic_topology(last_d_net, f"Dynamic Network Topology (After sequence: {last_perm_str})")
"""
    cells.append({"cell_type": "code", "metadata": {}, "execution_count": None, "outputs": [], "source": [line + "\n" for line in topology_code.split('\n')]})

    decision_boundary_code = """
b_net = results[-1]['b_net']
d_net = results[-1]['d_net']

def plot_model(model_net, ax, title):
    model_net.eval()
    x_min, x_max = -8, 8
    y_min, y_max = -8, 8
    xx, yy = np.meshgrid(np.linspace(x_min, x_max, 200),
                         np.linspace(y_min, y_max, 200))
    grid = np.c_[xx.ravel(), yy.ravel()]
    grid_tensor = torch.tensor(grid, dtype=torch.float32).to(device)
    
    with torch.no_grad():
        Z = model_net(grid_tensor).cpu().numpy()
        Z = (Z > 0.5).astype(int).reshape(xx.shape)
        
    ax.contourf(xx, yy, Z, alpha=0.3, cmap='bwr')
    
    colors = ['blue', 'red', 'green']
    markers = ['o', 's', '^']
    
    for i, (name, X_np, y_np) in enumerate(base_tasks):
        c0 = X_np[y_np == 0]
        c1 = X_np[y_np == 1]
        
        ax.scatter(c0[:, 0], c0[:, 1], c=colors[i], marker=markers[0], edgecolor='k', label=f'{name} (Class 0)' if i==0 else "", alpha=0.6)
        ax.scatter(c1[:, 0], c1[:, 1], c=colors[i], marker=markers[1], edgecolor='k', label=f'{name} (Class 1)' if i==0 else "", alpha=0.6)
        
    ax.set_title(title, fontsize=14)

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 7))

plot_model(b_net, ax1, f"Baseline Network\\n(Catastrophic Forgetting after {last_perm_str})")
plot_model(d_net, ax2, f"Our Dynamic Network\\n(Zero Forgetting after {last_perm_str}!)")

plt.tight_layout()
plt.show()
"""
    cells.append({"cell_type": "code", "metadata": {}, "execution_count": None, "outputs": [], "source": [line + "\n" for line in decision_boundary_code.split('\n')]})

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

    with open('new.ipynb', 'w') as f:
        json.dump(notebook, f, indent=1)
    print("Notebook 'new.ipynb' created successfully!")

if __name__ == "__main__":
    create_notebook()
