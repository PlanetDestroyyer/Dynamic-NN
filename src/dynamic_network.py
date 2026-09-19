import torch
import torch.nn as nn
import numpy as np

class PyTorchDynamicNetwork(nn.Module):
    def __init__(self, input_dim: int, output_dim: int, max_neurons: int = 2000, steps: int = 3):
        super().__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.max_neurons = max_neurons
        self.steps = steps

        # Generation 2: Output indices track which neurons are output heads
        self.output_indices = list(range(input_dim, input_dim + output_dim))
        
        # Start with input + output + 4 hidden neurons
        self.active_neurons = input_dim + output_dim + 4

        # Adjacency Matrix (Structure)
        self.register_buffer('M', torch.zeros(max_neurons, max_neurons))
        # Trainable Mask (Memory Protection)
        self.register_buffer('trainable_mask', torch.ones(max_neurons, max_neurons))
        
        # Generation 2: Localized Per-Neuron Stress
        self.register_buffer('neuron_stress', torch.zeros(max_neurons))

        # Weight Matrix and Bias
        self.W = nn.Parameter(torch.zeros(max_neurons, max_neurons))
        self.b = nn.Parameter(torch.zeros(max_neurons))

        self._hook_registered = False
        self._init_random_connections()

    def _init_random_connections(self):
        with torch.no_grad():
            for i in range(self.active_neurons):
                for j in range(self.active_neurons):
                    # Neurons can't connect to themselves, and inputs receive no connections
                    if i != j and i >= self.input_dim:
                        if torch.rand(1).item() > 0.5:
                            self.M[i, j] = 1.0
            
            # Initialize weights where connections exist
            mask = self.M[:self.active_neurons, :self.active_neurons].bool()
            self.W[:self.active_neurons, :self.active_neurons][mask] = torch.randn(mask.sum()) * 0.1

    def _register_hooks(self):
        def _w_hook(grad):
            return grad * self.trainable_mask * self.M

        def _b_hook(grad):
            # If a neuron's incoming connections are ALL frozen, freeze its bias too
            b_mask = torch.ones_like(grad)
            for i in range(self.active_neurons):
                # If it has incoming connections but ALL of them are frozen (trainable=0)
                if self.M[i, :].sum() > 0 and (self.M[i, :] * self.trainable_mask[i, :]).sum() == 0:
                    b_mask[i] = 0.0
            return grad * b_mask

        self.W.register_hook(_w_hook)
        self.b.register_hook(_b_hook)
        self._hook_registered = True

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if not self._hook_registered:
            self._register_hooks()

        batch = x.size(0)
        state = torch.zeros(batch, self.max_neurons, device=x.device)
        state[:, :self.input_dim] = x

        W_eff = self.W * self.M
        for _ in range(self.steps):
            new_state = torch.relu(torch.matmul(state, W_eff.T) + self.b)
            # Only update non-input neurons
            state = torch.cat([x, new_state[:, self.input_dim:]], dim=1)

        # Generation 2: Return output based on flexible indices
        return state[:, self.output_indices]

    def update_stress(self, grads_task: torch.Tensor, grads_replay: torch.Tensor, beta: float = 0.9):
        """Generation 2: Computes stress PER NEURON based on incoming gradient conflict."""
        with torch.no_grad():
            # conflict > 0 means gradients are pushing in opposite directions
            # CRITICAL: We must clamp this to >= 0 so stress doesn't become negative when gradients align!
            conflict = torch.relu(-(grads_task * grads_replay))
            
            # Only consider active, trainable connections
            active_mask = self.M * self.trainable_mask
            
            # Sum the conflict for all incoming connections to a neuron
            conflict_sum = (conflict * active_mask).sum(dim=1)
            
            # Count how many active, trainable connections each neuron has
            num_incoming = active_mask.sum(dim=1)
            
            # Local stress is the average conflict per incoming connection
            local_stress = conflict_sum / (num_incoming + 1e-8)
            
            # Update Exponential Moving Average
            self.neuron_stress = beta * self.neuron_stress + (1 - beta) * local_stress

    def get_max_neuron_stress(self) -> float:
        """Returns the highest stress level among all active neurons."""
        if self.active_neurons == 0: return 0.0
        return self.neuron_stress[:self.active_neurons].max().item()

    def stress_freeze(self, threshold: float = 0.3) -> int:
        """Generation 2: Freezes ONLY the neurons whose local stress exceeds the threshold."""
        n_frozen = 0
        with torch.no_grad():
            stressed_neurons = (self.neuron_stress > threshold) & (torch.arange(self.max_neurons, device=self.neuron_stress.device) < self.active_neurons)
            
            for i in torch.where(stressed_neurons)[0]:
                i = i.item()
                # Find its incoming connections that are currently trainable
                incoming = (self.trainable_mask[i, :] == 1) & (self.M[i, :] == 1)
                num_to_freeze = incoming.sum().item()
                
                if num_to_freeze > 0:
                    self.trainable_mask[i, incoming] = 0.0
                    n_frozen += num_to_freeze
                    # Reset stress since it is now protected
                    self.neuron_stress[i] = 0.0
                    
        return n_frozen

    def grow_neuron(self, num_connections: int = 15):
        """Grows a single hidden neuron and wires it up."""
        if self.active_neurons >= self.max_neurons:
            return -1
            
        new_idx = self.active_neurons
        self.active_neurons += 1
        
        with torch.no_grad():
            self.W[new_idx, :] = 0.0
            
            # 1. Incoming connections: from inputs and other hidden neurons
            valid_sources = list(range(self.input_dim)) + [i for i in range(new_idx) if i not in self.output_indices]
            if len(valid_sources) > 0:
                k = min(num_connections, len(valid_sources))
                chosen = torch.tensor(valid_sources)[torch.randperm(len(valid_sources))[:k]]
                self.M[new_idx, chosen] = 1.0
                self.W[new_idx, chosen] = torch.randn(k, device=self.W.device) * 0.1
                
            # 2. Outgoing connections: attach it to ALL current output heads so it's useful immediately!
            for out_idx in self.output_indices:
                self.M[out_idx, new_idx] = 1.0
                self.W[out_idx, new_idx] = torch.randn(1, device=self.W.device).item() * 0.1
                
        return new_idx

    def grow_output_head(self, num_connections: int = 15):
        """Generation 2: Dynamically spawns a brand new Output Neuron (Multi-Head)."""
        if self.active_neurons >= self.max_neurons:
            return -1
            
        new_idx = self.active_neurons
        self.active_neurons += 1
        self.output_dim += 1
        self.output_indices.append(new_idx)
        
        with torch.no_grad():
            self.W[new_idx, :] = 0.0
            
            # Incoming connections: from hidden neurons only (or inputs)
            hidden_and_input = list(range(self.input_dim)) + [i for i in range(new_idx) if i not in self.output_indices[:-1]]
            
            if len(hidden_and_input) > 0:
                k = min(num_connections, len(hidden_and_input))
                chosen = torch.tensor(hidden_and_input)[torch.randperm(len(hidden_and_input))[:k]]
                self.M[new_idx, chosen] = 1.0
                self.W[new_idx, chosen] = torch.randn(k, device=self.W.device) * 0.1
                
        return new_idx

    def n_trainable(self) -> int:
        return int((self.M[:self.active_neurons, :self.active_neurons] * self.trainable_mask[:self.active_neurons, :self.active_neurons]).sum().item())
        
    def n_frozen(self) -> int:
        return int((self.M[:self.active_neurons, :self.active_neurons] * (1 - self.trainable_mask[:self.active_neurons, :self.active_neurons])).sum().item())
