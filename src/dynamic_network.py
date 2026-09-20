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
        self.free_neurons = set()

        # Adjacency Matrix (Structure)
        self.register_buffer('M', torch.zeros(max_neurons, max_neurons))
        # Trainable Mask (Memory Protection)
        self.register_buffer('trainable_mask', torch.ones(max_neurons, max_neurons))
        
        # Generation 2: Localized Per-Neuron Stress
        self.register_buffer('neuron_stress', torch.zeros(max_neurons))
        
        # Anomaly Detection: Loss Exponential Moving Averages
        self.loss_ema = 0.0
        self.fast_loss_ema = 0.0
        self.slow_loss_ema = 0.0

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
            # Measure how much the new gradients destroy the old knowledge (negative dot product)
            conflict = -(grads_task * grads_replay)
            conflict = torch.relu(conflict)
            
            # Sum over incoming connections (dim=1 in target-source matrix)
            neuron_conflict = conflict.sum(dim=1)
            
            # EMA Update
            self.neuron_stress = beta * self.neuron_stress + (1 - beta) * neuron_conflict

    def get_max_neuron_stress(self) -> float:
        """Returns the highest stress level among all active neurons."""
        if self.active_neurons == 0: return 0.0
        return self.neuron_stress[:self.active_neurons].max().item()

    def update_loss_ema(self, current_loss: float, beta: float = 0.9, fast_beta: float = 0.5, slow_beta: float = 0.99):
        """Updates the running loss EMAs for anomaly detection."""
        self.loss_ema = beta * self.loss_ema + (1 - beta) * current_loss
        self.fast_loss_ema = fast_beta * self.fast_loss_ema + (1 - fast_beta) * current_loss
        self.slow_loss_ema = slow_beta * self.slow_loss_ema + (1 - slow_beta) * current_loss

    def detect_task_shift(self, max_stress: float, margin: float = 0.2, stress_threshold: float = 0.2) -> bool:
        """
        Anomaly Detection:
        If the Fast EMA of loss diverges from the Slow EMA (MACD), OR if Gradient 
        Conflict (max_stress) exceeds the threshold, a distribution shift is detected.
        """
        # Wait until the slow EMA has built up some history (Burn-In)
        if self.slow_loss_ema < 0.01: 
            return False
            
        macd_spike = (self.fast_loss_ema - self.slow_loss_ema) > margin
        stress_spike = max_stress > stress_threshold
        
        return macd_spike or stress_spike

    def stress_freeze(self, base_threshold: float = 0.5, sensitivity_factor: float = 0.4) -> tuple:
        """
        Localized Parameter Freezing:
        Freezes incoming weights for neurons experiencing high gradient conflict.
        The threshold dynamically lowers as the running loss increases (higher sensitivity).
        Returns: (n_frozen, dynamic_threshold)
        """
        n_frozen = 0
        
        # Calculate dynamic threshold: Higher loss = Lower threshold (higher sensitivity)
        capped_loss = min(self.loss_ema, 1.0)
        dynamic_threshold = max(0.05, base_threshold - (sensitivity_factor * capped_loss))
        
        with torch.no_grad():
            stressed_neurons = (self.neuron_stress > dynamic_threshold) & (torch.arange(self.max_neurons, device=self.neuron_stress.device) < self.active_neurons)
            
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
                    
        return n_frozen, dynamic_threshold

    def grow_neuron(self, num_connections: int = 5):
        if len(self.free_neurons) > 0:
            new_idx = self.free_neurons.pop()
        else:
            if self.active_neurons >= self.max_neurons:
                return -1
            new_idx = self.active_neurons
            self.active_neurons += 1
            
        with torch.no_grad():
            self.M[new_idx, :] = 0.0
            self.M[:, new_idx] = 0.0
            self.trainable_mask[new_idx, :] = 1.0
            self.trainable_mask[:, new_idx] = 1.0
            self.neuron_stress[new_idx] = 0.0
            
            # Incoming connections from valid sources
            valid_sources = list(range(self.input_dim)) + [
                n for n in range(self.input_dim + self.output_dim, self.max_neurons) 
                if n < self.active_neurons and n not in self.free_neurons and n != new_idx
            ]
            if len(valid_sources) > 0:
                n_conn = min(num_connections, len(valid_sources))
                sources = torch.randperm(len(valid_sources))[:n_conn]
                for s in sources:
                    self.M[new_idx, valid_sources[s.item()]] = 1.0
            
            # Outgoing connections to valid targets
            valid_targets = [
                n for n in range(self.input_dim, self.max_neurons)
                if n < self.active_neurons and n not in self.free_neurons and n != new_idx
            ]
            if len(valid_targets) > 0:
                n_conn = min(num_connections, len(valid_targets))
                targets = torch.randperm(len(valid_targets))[:n_conn]
                for t in targets:
                    self.M[valid_targets[t.item()], new_idx] = 1.0

            # Initialize weights
            in_mask = self.M[new_idx, :].bool()
            self.W[new_idx, in_mask] = torch.randn(in_mask.sum()) * 0.1
            out_mask = self.M[:, new_idx].bool()
            self.W[out_mask, new_idx] = torch.randn(out_mask.sum()) * 0.1
            self.b[new_idx] = 0.0
            
        return new_idx

    def prune_dead_neurons(self, threshold: float = 1e-3) -> int:
        n_pruned = 0
        with torch.no_grad():
            effective_W = self.W * self.M
            for idx in range(self.input_dim + self.output_dim, self.active_neurons):
                if idx in self.free_neurons:
                    continue
                # Calculate L1 norm of incoming weights
                incoming_l1 = torch.sum(torch.abs(effective_W[idx, :])).item()
                if incoming_l1 < threshold:
                    # Turn off the neuron
                    self.M[idx, :] = 0.0
                    self.M[:, idx] = 0.0
                    self.free_neurons.add(idx)
                    n_pruned += 1
        return n_pruned

    def merge_redundant_neurons(self, similarity_threshold: float = 0.95) -> int:
        n_merged = 0
        with torch.no_grad():
            effective_W = self.W * self.M
            active_hiddens = [
                idx for idx in range(self.input_dim + self.output_dim, self.active_neurons)
                if idx not in self.free_neurons
            ]
            
            for i in range(len(active_hiddens)):
                idx_i = active_hiddens[i]
                if idx_i in self.free_neurons:
                    continue
                
                vec_i = effective_W[idx_i, :]
                norm_i = torch.norm(vec_i)
                if norm_i == 0:
                    continue
                    
                for j in range(i + 1, len(active_hiddens)):
                    idx_j = active_hiddens[j]
                    if idx_j in self.free_neurons:
                        continue
                        
                    vec_j = effective_W[idx_j, :]
                    norm_j = torch.norm(vec_j)
                    if norm_j == 0:
                        continue
                        
                    sim = torch.dot(vec_i, vec_j) / (norm_i * norm_j)
                    if sim > similarity_threshold:
                        # Combine output weights of j into i
                        self.W[:, idx_i] += self.W[:, idx_j]
                        # Turn off neuron j
                        self.M[idx_j, :] = 0.0
                        self.M[:, idx_j] = 0.0
                        self.free_neurons.add(idx_j)
                        n_merged += 1
                        
        return n_merged

    def compress_network(self, prune_threshold: float = 1e-3, similarity_threshold: float = 0.95) -> tuple:
        n_pruned = self.prune_dead_neurons(prune_threshold)
        n_merged = self.merge_redundant_neurons(similarity_threshold)
        return n_pruned, n_merged

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

    def profile_network(self) -> dict:
        """
        Profiles the physical vs logical capacity of the network.
        Returns a dictionary containing parameter and theoretical FLOP counts.
        """
        logical_connections = self.M.sum().item()
        allocated_connections = self.W.numel()
        
        # Biases for active neurons
        active_biases = self.active_neurons
        allocated_biases = self.max_neurons
        
        logical_params = logical_connections + active_biases
        allocated_params = allocated_connections + allocated_biases
        
        # FLOPs per inference
        # Logical: 2 ops (mul+add) per active connection + 1 activation per active neuron
        logical_flops = (2 * logical_connections) + self.active_neurons
        
        # Allocated: 2 ops per element in the dense weight matrix + 1 activation per max neuron
        allocated_flops = (2 * allocated_connections) + self.max_neurons
        
        return {
            'logical_parameters': int(logical_params),
            'allocated_parameters': int(allocated_params),
            'logical_flops_per_inference': int(logical_flops),
            'allocated_flops_per_inference': int(allocated_flops)
        }
