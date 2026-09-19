import json

def create_llm_notebook():
    cells = []
    
    # 1. Imports and Dataset loading
    data_cell = """import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader, TensorDataset

print("Preparing Character-Level Dataset...")

text_t1 = "O Romeo, Romeo! wherefore art thou Romeo? Deny thy father and refuse thy name."
text_t2 = "def calculate_loss(pred, target): return torch.mean((pred - target) ** 2)"

# Create shared vocabulary
chars = sorted(list(set(text_t1 + text_t2)))
vocab_size = len(chars)
char_to_idx = {c: i for i, c in enumerate(chars)}
idx_to_char = {i: c for i, c in enumerate(chars)}

print(f"Vocabulary Size: {vocab_size} unique characters.")

CONTEXT_SIZE = 5

def create_dataset(text):
    X, y = [], []
    for i in range(len(text) - CONTEXT_SIZE):
        context = text[i:i + CONTEXT_SIZE]
        target = text[i + CONTEXT_SIZE]
        X.append([char_to_idx[c] for c in context])
        y.append(char_to_idx[target])
    return torch.tensor(X, dtype=torch.long), torch.tensor(y, dtype=torch.long)

X_t1, y_t1 = create_dataset(text_t1)
X_t2, y_t2 = create_dataset(text_t2)

tasks = [
    ("Shakespeare", X_t1, y_t1),
    ("Python Code", X_t2, y_t2)
]
"""
    cells.append({"cell_type": "code", "metadata": {}, "execution_count": None, "outputs": [], "source": [line + "\n" for line in data_cell.split('\n')]})

    # 2. PyTorchDynamicNetwork (from gpu_dynamic_nn.py)
    with open('gpu_dynamic_nn.py', 'r') as f:
        network_code = f.read()
        network_code = network_code.replace("import torch\nimport torch.nn as nn\nimport numpy as np\n", "")
    cells.append({"cell_type": "code", "metadata": {}, "execution_count": None, "outputs": [], "source": [line + "\n" for line in network_code.split('\n')]})

    # 3. BabyLLM, ReplayBuffer and Helper
    llm_replay_cell = """class BabyLLM(nn.Module):
    def __init__(self, vocab_size, embed_dim, context_size, device):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim).to(device)
        self.context_size = context_size
        self.embed_dim = embed_dim
        # input is context_size * embed_dim, output is vocab_size
        self.dynamic = PyTorchDynamicNetwork(input_dim=context_size * embed_dim, output_dim=vocab_size, max_neurons=1000).to(device)
        
    def forward(self, x):
        # x shape: [batch_size, context_size]
        emb = self.embedding(x) # [batch, context, embed_dim]
        emb = emb.view(x.size(0), -1) # [batch, context * embed_dim]
        return self.dynamic(emb)

class ReplayBuffer:
    def __init__(self, capacity: int = 500):
        self.capacity = capacity
        self.X = None
        self.y = None

    def add_data(self, X_new: torch.Tensor, y_new: torch.Tensor, num_samples: int = 100):
        idx = torch.randperm(X_new.size(0))[:num_samples]
        X_sub = X_new[idx].clone().detach()
        y_sub = y_new[idx].clone().detach()

        if self.X is None:
            self.X = X_sub
            self.y = y_sub
        else:
            self.X = torch.cat([self.X, X_sub], dim=0)
            self.y = torch.cat([self.y, y_sub], dim=0)

            if self.X.size(0) > self.capacity:
                keep = torch.randperm(self.X.size(0))[:self.capacity]
                self.X = self.X[keep]
                self.y = self.y[keep]

    def sample(self, batch_size: int = 64) -> tuple:
        if self.X is None:
            return None, None
        size = self.X.size(0)
        idx = torch.randint(0, size, (min(batch_size, size),))
        return self.X[idx], self.y[idx]

    def has_data(self) -> bool:
        return self.X is not None and self.X.size(0) > 0

def _reset_adam_for_neuron(optimizer: torch.optim.Adam, net: nn.Module, new_idx: int):
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

def generate_text(model, seed_text, num_chars, device):
    model.eval()
    context = seed_text
    output = seed_text
    with torch.no_grad():
        for _ in range(num_chars):
            # Encode context
            x = [char_to_idx[c] for c in context[-CONTEXT_SIZE:]]
            x_tensor = torch.tensor([x], dtype=torch.long).to(device)
            # Predict
            logits = model(x_tensor)
            pred_idx = logits.argmax(dim=1).item()
            pred_char = idx_to_char[pred_idx]
            # Update
            output += pred_char
            context += pred_char
    return output
"""
    cells.append({"cell_type": "code", "metadata": {}, "execution_count": None, "outputs": [], "source": [line + "\n" for line in llm_replay_cell.split('\n')]})

    # 4. Pre-Train Embedding
    pretrain_cell = """device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Running on {device}...")

print("\\n" + "="*50)
print("--- Initializing Baby LLM ---")
print("="*50)

# 16-dimensional character embeddings
EMBED_DIM = 16
llm = BabyLLM(vocab_size, EMBED_DIM, CONTEXT_SIZE, device)
replay = ReplayBuffer(capacity=500)

# We will optimize BOTH the embedding and the dynamic network simultaneously!
# Wait, to prevent the embedding from drifting wildly and destroying memories, 
# we should freeze it after Task 1.
"""
    cells.append({"cell_type": "code", "metadata": {}, "execution_count": None, "outputs": [], "source": [line + "\n" for line in pretrain_cell.split('\n')]})

    # 5. Dynamic Training Loop
    dynamic_cell = """optimizer = torch.optim.Adam(llm.parameters(), lr=1e-3)
criterion = nn.CrossEntropyLoss()

for task_id, (name, X_train, y_train) in enumerate(tasks):
    print(f"\\nTraining Baby LLM on {name}...")
    
    # Freeze Embedding after Task 1 to protect semantic meaning of characters!
    if task_id == 1:
        print("Freezing Character Embeddings...")
        for param in llm.embedding.parameters():
            param.requires_grad = False
    
    dataset = TensorDataset(X_train, y_train)
    loader = DataLoader(dataset, batch_size=16, shuffle=True)
    
    loss_ema = 2.0
    stress_ema = 0.0
    batches_since_grow = 0
    
    for epoch in range(10): # Train for 10 epochs
        llm.train()
        
        for bx, by in loader:
            bx, by = bx.to(device), by.to(device)
            optimizer.zero_grad()
            
            pred = llm(bx)
            loss = criterion(pred, by)
            
            if replay.has_data():
                rx, ry = replay.sample(32)
                rx, ry = rx.to(device), ry.to(device)
                r_pred = llm(rx)
                loss_r = criterion(r_pred, ry)
                
                # Compute gradient stress ONLY on the Dynamic Network's weights
                grads_task = torch.autograd.grad(loss, llm.dynamic.W, retain_graph=True, allow_unused=True)[0]
                grads_replay = torch.autograd.grad(loss_r, llm.dynamic.W, retain_graph=True, allow_unused=True)[0]
                
                if grads_task is not None and grads_replay is not None:
                    llm.dynamic.update_stress(grads_task, grads_replay)
                    
            loss.backward()
            optimizer.step()
            
            loss_ema = 0.9 * loss_ema + 0.1 * loss.item()
            stress_ema = llm.dynamic.get_network_stress()
            batches_since_grow += 1
            
            # Freeze condition
            if stress_ema > 0.3:
                n_frozen = llm.dynamic.stress_freeze(threshold=0.3)
                if n_frozen > 0:
                    llm.dynamic.grow_neuron(num_connections=15)
                    _reset_adam_for_neuron(optimizer, llm.dynamic, llm.dynamic.active_neurons - 1)
                    batches_since_grow = 0
                    print(f"  [Batch] FREEZE: {n_frozen} conns. Grew 1. active={llm.dynamic.active_neurons}")
            
            # Grow condition (loss)
            elif loss_ema > 0.5 and batches_since_grow > 10 and stress_ema <= 0.3:
                llm.dynamic.grow_neuron(num_connections=15)
                _reset_adam_for_neuron(optimizer, llm.dynamic, llm.dynamic.active_neurons - 1)
                batches_since_grow = 0
                print(f"  [Batch] GROW(Loss): active={llm.dynamic.active_neurons} loss={loss_ema:.4f}")
        
        print(f"  [Epoch {epoch+1:2d}] active={llm.dynamic.active_neurons} loss={loss_ema:.4f}")
        
    replay.add_data(X_train, y_train, num_samples=200)
    
    print(f"\\n--- Generation Test after {name} ---")
    gen_s = generate_text(llm, "O Rom", num_chars=30, device=device)
    print(f"Prompt 'O Rom' -> {gen_s}")
    
    gen_c = generate_text(llm, "def c", num_chars=30, device=device)
    print(f"Prompt 'def c' -> {gen_c}")
    
    print(f"  Active Neurons: {llm.dynamic.active_neurons} (trainable={llm.dynamic.n_trainable()}, frozen={llm.dynamic.n_frozen()})")
"""
    cells.append({"cell_type": "code", "metadata": {}, "execution_count": None, "outputs": [], "source": [line + "\n" for line in dynamic_cell.split('\n')]})

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

    with open('baby_llm.ipynb', 'w') as f:
        json.dump(notebook, f, indent=1)
    print("Notebook 'baby_llm.ipynb' created successfully!")

if __name__ == "__main__":
    create_llm_notebook()
