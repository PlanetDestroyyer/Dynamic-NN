import torch
import torch.nn as nn
import numpy as np


class PyTorchDynamicNetwork(nn.Module):
    """
    Adjacency-matrix dynamic network with hard-freeze pathway separation.

    Architecture
    ────────────
    • W[max×max]          : weight matrix (single nn.Parameter)
    • M[max×max]          : structural mask (which connections exist)
    • trainable_mask[max×max] : gradient mask (which connections can learn)

    The gradient hook ensures:
        W.grad *= trainable_mask
    so frozen weights receive EXACTLY ZERO gradient — no optimizer state update,
    no momentum, nothing.  They are physically frozen.

    Lifecycle
    ─────────
    1. Network starts small (input + output + 4 hidden).
       All existing connections have trainable_mask = 1 → fully plastic.

    2. Train on Task 1.  Network learns.  Fisher (EMA of g²) builds up
       for weights that are actively used.

    3. When a new distribution arrives (detected by rising EWC stress):
       a. autonomous_freeze() sets trainable_mask = 0 for ALL current connections
          that have significant Fisher importance.
       b. grow_neurons() adds new neurons with trainable_mask = 1.
       c. New neurons connect to inputs AND to the output → they form a
          parallel pathway that can learn the new task without any gradient
          flowing through old frozen weights.

    4. Replay ensures the new pathway doesn't accidentally counteract the
       old pathway's contribution to the output.

    Why this works where EWC/SI/AGEM failed
    ────────────────────────────────────────
    Those methods tried to DISCOURAGE the optimizer from changing old weights
    (via penalties or gradient scaling).  But in a recurrent forward pass,
    the optimizer always found paths through unprotected weights.

    Hard-freeze doesn't discourage — it PREVENTS.  Zero gradient = zero change.
    The optimizer has no choice but to use the new neurons.
    """

    def __init__(self, input_dim: int, output_dim: int,
                 max_neurons: int = 200, steps: int = 3):
        super().__init__()
        self.input_dim   = input_dim
        self.output_dim  = output_dim
        self.max_neurons = max_neurons
        self.steps       = steps

        self.active_neurons = input_dim + output_dim + 4

        # ── Learnable parameters ──────────────────────────────────────────
        self.W = nn.Parameter(torch.randn(max_neurons, max_neurons) * 0.1)
        self.b = nn.Parameter(torch.zeros(max_neurons))

        # ── Structural mask (which connections exist) ─────────────────────
        self.register_buffer('M', torch.zeros(max_neurons, max_neurons))

        # ── Gradient mask (which connections can learn) ───────────────────
        # 1.0 = trainable,  0.0 = frozen (hard zero gradient)
        self.register_buffer('trainable_mask',
                             torch.ones(max_neurons, max_neurons))

        # ── Bias freeze mask ─────────────────────────────────────────────
        self.register_buffer('bias_trainable',
                             torch.ones(max_neurons))

        # ── Per-weight gradient conflict statistics ───────────────────────────
        #
        # We explicitly measure if the CURRENT task is fighting REPLAY memory.
        #
        # stress_num: EMA of (-g_task * g_replay).
        #             Positive if they pull in opposite directions.
        #
        # stress_den: EMA of (|g_task| * |g_replay|).
        #             Normalization factor.
        #
        # stress = stress_num / (stress_den + 1e-8)
        #
        #   stress ≈ -1.0  →  Consistent agreement (single task learning)
        #   stress ≈ 0.0   →  Independent noise (converged at minimum)
        #   stress > 0.0   →  True conflict (new task destroying old knowledge)
        #
        self.register_buffer('stress_num', torch.zeros(max_neurons, max_neurons))
        self.register_buffer('stress_den', torch.zeros(max_neurons, max_neurons))

        # ── Gradient hooks (registered once) ─────────────────────────────
        self._hook_registered = False

        self._init_random_connections()

    # ─────────────────────────────────────────────────────────────────────
    def _init_random_connections(self):
        for i in range(self.active_neurons):
            for j in range(self.active_neurons):
                if i != j and i >= self.input_dim:
                    if torch.rand(1).item() > 0.5:
                        self.M[i, j] = 1.0

    # ─────────────────────────────────────────────────────────────────────
    def _register_hooks(self):
        """Register gradient hooks that enforce the trainable_mask."""
        if self._hook_registered:
            return

        def _w_hook(grad):
            # Hard-zero gradients for frozen connections
            return grad * self.trainable_mask * self.M

        def _b_hook(grad):
            return grad * self.bias_trainable

        self.W.register_hook(_w_hook)
        self.b.register_hook(_b_hook)
        self._hook_registered = True

    # ─────────────────────────────────────────────────────────────────────
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Ensure hooks are registered on first forward
        if not self._hook_registered:
            self._register_hooks()

        batch = x.size(0)
        state = torch.zeros(batch, self.max_neurons, device=x.device)
        state[:, :self.input_dim] = x

        W_eff = self.W * self.M

        for _ in range(self.steps):
            new_state = torch.relu(torch.matmul(state, W_eff.T) + self.b)
            state = torch.cat([x, new_state[:, self.input_dim:]], dim=1)

        return state[:, self.input_dim: self.input_dim + self.output_dim]

    # ─────────────────────────────────────────────────────────────────────
    def update_fisher(self, beta: float = 0.999):
        """
        Update Fisher (EMA of grad²) as a diagnostic signal.
        Called every batch AFTER backward().
        """
        with torch.no_grad():
            if self.W.grad is not None:
                grad_sq = (self.W.grad ** 2) * self.M
                self.fisher.mul_(beta).add_((1 - beta) * grad_sq)

    # ─────────────────────────────────────────────────────────────────────
    def update_stress(self, g_task: torch.Tensor, g_replay: torch.Tensor, beta: float = 0.99):
        """
        Update explicit task-vs-replay conflict statistics.
        
        Args:
            g_task: Gradient of W computed on the current task's batch.
            g_replay: Gradient of W computed on a batch from the replay buffer.
        """
        with torch.no_grad():
            # Only consider active connections that are trainable
            active = self.M * self.trainable_mask
            g_t = (g_task * active).detach()
            g_r = (g_replay * active).detach()
            
            # The product is negative if they point in opposite directions.
            # We negate it so that conflict is POSITIVE.
            conflict = -(g_t * g_r)
            magnitude = g_t.abs() * g_r.abs()
            
            self.stress_num.mul_(beta).add_((1 - beta) * conflict)
            self.stress_den.mul_(beta).add_((1 - beta) * magnitude)

    # ─────────────────────────────────────────────────────────────────────
    def get_stress_matrix(self) -> torch.Tensor:
        """
        Per-weight conflict fraction in [-1.0, 1.0].
        
        Only active trainable connections have non-zero stress.
        """
        with torch.no_grad():
            stress = self.stress_num / (self.stress_den + 1e-8)
            # Clip safely
            stress = stress.clamp(min=-1.0, max=1.0)
            return stress * self.M * self.trainable_mask

    # ─────────────────────────────────────────────────────────────────────
    def get_network_stress(self, conflict_threshold: float = 0.3) -> float:
        """
        Fraction of active trainable connections that are in high conflict.
        
        Returns a value in [0, 1]:
          0.0  →  No connections are conflicted (stable or just noise)
          ~1.0 →  All trainable connections are conflicted
        """
        with torch.no_grad():
            stress = self.get_stress_matrix()
            active_trainable = (self.M * self.trainable_mask).bool()
            if active_trainable.sum() == 0:
                return 0.0

            high_conflict = (stress[active_trainable] > conflict_threshold).float()
            return high_conflict.mean().item()

    # ─────────────────────────────────────────────────────────────────────
    def stress_freeze(self, threshold: float = 0.3) -> int:
        """
        Autonomously freeze the most-conflicted trainable connections.
        
        Args:
            threshold: Any trainable connection with a stress > threshold
                       is permanently frozen.
        """
        with torch.no_grad():
            stress = self.get_stress_matrix()
            active_trainable = (self.M * self.trainable_mask).bool()
            if active_trainable.sum() == 0:
                return 0

            freeze_mask = (stress >= threshold) & active_trainable
            n_frozen = int(freeze_mask.sum().item())

            self.trainable_mask[freeze_mask] = 0.0

            # Also freeze biases of neurons whose incoming connections
            # are now mostly frozen (they are 'old' neurons)
            for i in range(self.input_dim, self.active_neurons):
                incoming = self.M[i, :self.active_neurons]
                if incoming.sum() == 0:
                    continue
                frozen_frac = (
                    (1 - self.trainable_mask[i, :self.active_neurons])
                    [incoming.bool()].mean()
                )
                if frozen_frac > 0.5:
                    self.bias_trainable[i] = 0.0

            return n_frozen

    # ─────────────────────────────────────────────────────────────────────
    def get_ewc_stress(self) -> float:
        """
        Compute how much "stress" the network is under.

        Stress = mean Fisher of currently-trainable connections.
        High stress = existing trainable weights are receiving large gradients
        = the current task is trying hard to change them = potential conflict.
        """
        active_trainable = self.trainable_mask * self.M
        if active_trainable.sum() == 0:
            return 0.0
        return (self.fisher * active_trainable).sum().item() / active_trainable.sum().item()

    # ─────────────────────────────────────────────────────────────────────
    def autonomous_freeze(self, fisher_threshold_percentile: float = 50.0):
        """
        AUTONOMOUSLY freeze connections that have high Fisher importance.

        This is NOT manual task-boundary freezing.  It is triggered by the
        training loop when stress exceeds a threshold.

        Process:
        1. Look at Fisher values for currently-trainable connections.
        2. Connections with Fisher above the percentile threshold → freeze
           (trainable_mask = 0).
        3. Connections below threshold → stay trainable (they weren't
           important, so they can be reused).

        After freezing:
        - The network CANNOT modify these connections anymore (gradient = 0).
        - New neurons must be grown to provide fresh pathway capacity.
        """
        with torch.no_grad():
            # Only consider currently-trainable + active connections
            active = (self.trainable_mask * self.M).bool()
            if active.sum() == 0:
                return 0

            fisher_vals = self.fisher[active]
            if fisher_vals.numel() == 0:
                return 0

            threshold = torch.quantile(fisher_vals, fisher_threshold_percentile / 100.0)

            # Freeze connections with Fisher >= threshold
            freeze_mask = (self.fisher >= threshold) & active
            n_frozen = freeze_mask.sum().item()

            self.trainable_mask[freeze_mask] = 0.0

            # Also freeze biases of neurons whose incoming connections are
            # mostly frozen (they are "old" neurons now)
            for i in range(self.input_dim, self.active_neurons):
                incoming = self.M[i, :self.active_neurons]
                if incoming.sum() == 0:
                    continue
                frozen_frac = (1 - self.trainable_mask[i, :self.active_neurons])[incoming.bool()].mean()
                if frozen_frac > 0.5:
                    self.bias_trainable[i] = 0.0

            return n_frozen

    # ─────────────────────────────────────────────────────────────────────
    def grow_neuron(self, num_connections: int = 5):
        """
        Activate next pre-allocated neuron.

        New neuron has:
          • trainable_mask = 1 for all its connections → fully plastic
          • Fisher = 0 → not considered important (yet)
          • Small random weights → doesn't disrupt existing output
          • Connects to BOTH inputs and outputs → forms a parallel pathway

        The output neuron connections to the new neuron are TRAINABLE,
        while output connections to old neurons remain FROZEN.
        """
        if self.active_neurons >= self.max_neurons:
            print("  [network] Max capacity reached.")
            return

        new_idx = self.active_neurons
        self.active_neurons += 1

        with torch.no_grad():
            # Clear the new neuron's slot
            self.W.data[new_idx, :] = 0.0
            self.W.data[:, new_idx] = 0.0
            self.b.data[new_idx]    = 0.0

            # New neuron is fully trainable
            self.trainable_mask[new_idx, :] = 1.0
            self.trainable_mask[:, new_idx] = 1.0
            self.bias_trainable[new_idx]    = 1.0

            # Reset stress buffers for the new neuron
            self.stress_num[new_idx, :] = 0.0
            self.stress_num[:, new_idx] = 0.0
            self.stress_den[new_idx, :] = 0.0
            self.stress_den[:, new_idx] = 0.0

        # Wire incoming: from inputs + existing hidden neurons
        sources = [s for s in range(new_idx)
                   if s < self.input_dim or
                   s >= self.input_dim + self.output_dim]
        if sources:
            # Prefer connecting to input neurons (direct fresh signal)
            input_sources = [s for s in sources if s < self.input_dim]
            hidden_sources = [s for s in sources if s >= self.input_dim + self.output_dim]

            # Always connect to all inputs
            for s in input_sources:
                self.M[new_idx, s] = 1.0
                with torch.no_grad():
                    self.W.data[new_idx, s] = torch.randn(1).item() * 0.1

            # Connect to a few random hidden neurons
            if hidden_sources:
                n_hidden_conn = min(num_connections, len(hidden_sources))
                chosen = np.random.choice(hidden_sources, n_hidden_conn, replace=False)
                for s in chosen:
                    self.M[new_idx, s] = 1.0
                    with torch.no_grad():
                        self.W.data[new_idx, s] = torch.randn(1).item() * 0.05

        # Wire outgoing: to output neurons AND a few hidden neurons
        output_indices = list(range(self.input_dim, self.input_dim + self.output_dim))
        for out_idx in output_indices:
            self.M[out_idx, new_idx] = 1.0
            self.trainable_mask[out_idx, new_idx] = 1.0  # explicitly trainable
            with torch.no_grad():
                self.W.data[out_idx, new_idx] = 0.0  # start at 0 — no disruption

        # Also connect to a few other hidden neurons
        hidden_targets = [t for t in range(self.input_dim + self.output_dim, new_idx)]
        if hidden_targets:
            n_out = min(num_connections, len(hidden_targets))
            chosen = np.random.choice(hidden_targets, n_out, replace=False)
            for t in chosen:
                self.M[t, new_idx] = 1.0
                with torch.no_grad():
                    self.W.data[t, new_idx] = torch.randn(1).item() * 0.05

    # ─────────────────────────────────────────────────────────────────────
    def n_frozen(self) -> int:
        """Number of frozen connections."""
        active = self.M.bool()
        return int((active & ~self.trainable_mask.bool()).sum().item())

    def n_trainable(self) -> int:
        """Number of trainable connections."""
        return int((self.M * self.trainable_mask).sum().item())
