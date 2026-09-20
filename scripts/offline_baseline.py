import torch
import torch.nn as nn
import numpy as np

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

class StaticMLP(nn.Module):
    def __init__(self, input_dim=2, hidden_dim=256, output_dim=1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim)
        )
    def forward(self, x):
        return self.net(x)

def run_offline_baseline(seed=42):
    torch.manual_seed(seed)
    np.random.seed(seed)
    
    # 1. Generate Datasets
    n_per_task = 4000
    X_A, y_A = make_circles(n_per_task)
    X_B, y_B = make_linear(n_per_task)
    X_C, y_C = make_xor(n_per_task)
    
    # Concatenate and Shuffle
    X_all = torch.cat([X_A, X_B, X_C])
    y_all = torch.cat([y_A, y_B, y_C])
    
    perm = torch.randperm(X_all.size(0))
    X_all = X_all[perm]
    y_all = y_all[perm]
    
    # 2. Train Large Static MLP
    model = StaticMLP(hidden_dim=256)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    criterion = nn.MSELoss()
    
    batch_size = 64
    epochs = 20
    
    print(f"Training Offline Oracle (Seed {seed})...")
    for epoch in range(epochs):
        model.train()
        total_loss = 0
        for i in range(0, X_all.size(0), batch_size):
            bx = X_all[i:i+batch_size]
            by = y_all[i:i+batch_size]
            
            optimizer.zero_grad()
            pred = model(bx)
            loss = criterion(pred, by)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            
    # 3. Evaluate on Clean Test Sets
    model.eval()
    with torch.no_grad():
        acc_A = ((model(X_A) > 0.5).float() == y_A).float().mean().item()
        acc_B = ((model(X_B) > 0.5).float() == y_B).float().mean().item()
        acc_C = ((model(X_C) > 0.5).float() == y_C).float().mean().item()
        
    avg_acc = (acc_A + acc_B + acc_C) / 3.0
    
    return avg_acc, acc_A, acc_B, acc_C

if __name__ == "__main__":
    avg_accs = []
    for seed in [42, 100, 999]:
        avg, a, b, c = run_offline_baseline(seed)
        avg_accs.append(avg)
        print(f"Seed {seed}: Average={avg*100:.1f}% (A={a*100:.1f}%, B={b*100:.1f}%, C={c*100:.1f}%)")
        
    final_acc = np.mean(avg_accs)
    std_acc = np.std(avg_accs)
    
    print("\n" + "="*50)
    print("OFFLINE JOINT-TRAINING UPPER BOUND")
    print("="*50)
    print(f"Final Accuracy: {final_acc*100:.1f}% ± {std_acc*100:.1f}%")
