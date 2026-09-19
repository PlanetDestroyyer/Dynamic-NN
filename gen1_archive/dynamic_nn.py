"""
Dynamic Neural Network — Phase 1-3

A neural network represented as a directed graph where:
- Neurons are nodes (can be input, hidden, or output)
- Connections are weighted directed edges
- Forward pass uses topological sort
- Backward pass uses manual gradient computation through the graph
- Connections are PRUNED when weight magnitude drops below threshold
- Neurons are GROWN when loss plateaus

Pure NumPy. No PyTorch. KISS.
"""

import numpy as np
from dataclasses import dataclass, field
from enum import Enum
from typing import Optional
from collections import defaultdict


# ─────────────────────────────────────────────
# Data Structures
# ─────────────────────────────────────────────

class NeuronType(Enum):
    INPUT = "input"
    HIDDEN = "hidden"
    OUTPUT = "output"


@dataclass
class Neuron:
    """A single neuron in the graph."""
    id: int
    neuron_type: NeuronType
    bias: float = 0.0

    # Runtime values (set during forward/backward)
    activation: float = 0.0      # output after activation function
    pre_activation: float = 0.0  # weighted sum before activation function
    gradient: float = 0.0        # dL/d(pre_activation) for backprop
    bias_grad: float = 0.0       # accumulated gradient for bias


@dataclass
class Connection:
    """A directed weighted edge between two neurons."""
    from_id: int
    to_id: int
    weight: float

    # Runtime values
    weight_grad: float = 0.0     # accumulated gradient for this weight
    avg_delta: float = 0.0       # EMA of recent weight changes (for stability)


# ─────────────────────────────────────────────
# Activation Functions
# ─────────────────────────────────────────────

def sigmoid(x: float) -> float:
    """Sigmoid activation. Numerically stable version."""
    if x >= 0:
        return 1.0 / (1.0 + np.exp(-x))
    else:
        ex = np.exp(x)
        return ex / (1.0 + ex)


def sigmoid_derivative(activation: float) -> float:
    """Derivative of sigmoid given the OUTPUT of sigmoid (not the input)."""
    return activation * (1.0 - activation)


# ─────────────────────────────────────────────
# The Network
# ─────────────────────────────────────────────

class DynamicNetwork:
    """
    A neural network represented as a directed acyclic graph.

    Neurons are nodes, connections are weighted edges.
    Forward pass: topological sort → compute activations.
    Backward pass: reverse topological order → compute gradients.
    """

    def __init__(self):
        self.neurons: dict[int, Neuron] = {}
        self.connections: dict[tuple[int, int], Connection] = {}
        self._next_neuron_id: int = 0
        self._topo_order: list[int] = []  # cached topological sort
        self._topo_dirty: bool = True     # flag to recompute sort

        # Training history
        self.loss_history: list[float] = []
        self.neuron_count_history: list[int] = []
        self.connection_count_history: list[int] = []
        # Average absolute weight-change magnitude per update step.
        # High values = weights are being pulled hard = interference signal.
        self.weight_delta_history: list[float] = []

    # ─── Neuron Management ───

    def add_neuron(self, neuron_type: NeuronType, bias: float = 0.0) -> int:
        """Add a neuron to the network. Returns the neuron ID."""
        nid = self._next_neuron_id
        self._next_neuron_id += 1
        self.neurons[nid] = Neuron(id=nid, neuron_type=neuron_type, bias=bias)
        self._topo_dirty = True
        return nid

    def remove_neuron(self, nid: int) -> None:
        """Remove a neuron and all its connections."""
        if nid not in self.neurons:
            return
        # Remove all connections involving this neuron
        to_remove = [
            key for key in self.connections
            if key[0] == nid or key[1] == nid
        ]
        for key in to_remove:
            del self.connections[key]
        del self.neurons[nid]
        self._topo_dirty = True

    # ─── Connection Management ───

    def add_connection(self, from_id: int, to_id: int, weight: Optional[float] = None) -> None:
        """Add a weighted directed connection between two neurons."""
        assert from_id in self.neurons, f"Neuron {from_id} not found"
        assert to_id in self.neurons, f"Neuron {to_id} not found"
        assert from_id != to_id, "Self-connections not allowed"

        if weight is None:
            # Xavier-like initialization for a single connection
            weight = np.random.randn() * 0.5

        self.connections[(from_id, to_id)] = Connection(
            from_id=from_id, to_id=to_id, weight=weight
        )
        self._topo_dirty = True

    def remove_connection(self, from_id: int, to_id: int) -> None:
        """Remove a connection between two neurons."""
        key = (from_id, to_id)
        if key in self.connections:
            del self.connections[key]
            self._topo_dirty = True

    # ─── Topology ───

    def _topological_sort(self) -> list[int]:
        """
        Compute topological ordering of neurons using Kahn's algorithm.
        Input neurons come first, output neurons come last.
        """
        # Build adjacency and in-degree
        in_degree = defaultdict(int)
        adjacency = defaultdict(list)  # from → [to]

        for nid in self.neurons:
            in_degree[nid] = 0

        for (from_id, to_id) in self.connections:
            adjacency[from_id].append(to_id)
            in_degree[to_id] += 1

        # Start with neurons that have no incoming connections
        queue = [nid for nid in self.neurons if in_degree[nid] == 0]
        order = []

        while queue:
            # Sort for determinism
            queue.sort()
            node = queue.pop(0)
            order.append(node)
            for neighbor in adjacency[node]:
                in_degree[neighbor] -= 1
                if in_degree[neighbor] == 0:
                    queue.append(neighbor)

        if len(order) != len(self.neurons):
            raise ValueError("Graph has a cycle! Cannot do topological sort.")

        return order

    def _ensure_topo_order(self) -> None:
        """Recompute topological sort if the graph has changed."""
        if self._topo_dirty:
            self._topo_order = self._topological_sort()
            self._topo_dirty = False

    # ─── Forward Pass ───

    def forward(self, inputs: np.ndarray) -> np.ndarray:
        """
        Forward pass through the graph.

        Args:
            inputs: array of input values, one per input neuron

        Returns:
            array of output values, one per output neuron
        """
        self._ensure_topo_order()

        # Identify input and output neurons
        input_neurons = [n for n in self.neurons.values() if n.neuron_type == NeuronType.INPUT]
        output_neurons = [n for n in self.neurons.values() if n.neuron_type == NeuronType.OUTPUT]

        # Sort by ID for consistent ordering
        input_neurons.sort(key=lambda n: n.id)
        output_neurons.sort(key=lambda n: n.id)

        assert len(inputs) == len(input_neurons), \
            f"Expected {len(input_neurons)} inputs, got {len(inputs)}"

        # Reset all activations
        for neuron in self.neurons.values():
            neuron.activation = 0.0
            neuron.pre_activation = 0.0
            neuron.gradient = 0.0

        # Set input neuron activations (no activation function on inputs)
        for i, neuron in enumerate(input_neurons):
            neuron.activation = inputs[i]
            neuron.pre_activation = inputs[i]

        # Build incoming connections map for efficient lookup
        incoming = defaultdict(list)  # to_id → [Connection]
        for conn in self.connections.values():
            incoming[conn.to_id].append(conn)

        # Forward pass in topological order
        for nid in self._topo_order:
            neuron = self.neurons[nid]
            if neuron.neuron_type == NeuronType.INPUT:
                continue  # already set

            # Compute weighted sum of inputs
            pre_act = neuron.bias
            for conn in incoming[nid]:
                from_neuron = self.neurons[conn.from_id]
                pre_act += from_neuron.activation * conn.weight

            neuron.pre_activation = pre_act
            neuron.activation = sigmoid(pre_act)

        # Collect outputs
        outputs = np.array([n.activation for n in output_neurons])
        return outputs

    # ─── Backward Pass ───

    def backward(self, targets: np.ndarray) -> float:
        """
        Backward pass: compute gradients using reverse topological order.

        Uses binary cross-entropy loss for classification.

        Args:
            targets: array of target values, one per output neuron

        Returns:
            the loss value
        """
        self._ensure_topo_order()

        output_neurons = sorted(
            [n for n in self.neurons.values() if n.neuron_type == NeuronType.OUTPUT],
            key=lambda n: n.id
        )

        assert len(targets) == len(output_neurons), \
            f"Expected {len(output_neurons)} targets, got {len(targets)}"

        # ─── Compute loss (binary cross-entropy) ───
        eps = 1e-8
        loss = 0.0
        for i, neuron in enumerate(output_neurons):
            y = targets[i]
            a = np.clip(neuron.activation, eps, 1 - eps)
            loss += -(y * np.log(a) + (1 - y) * np.log(1 - a))

        # ─── Compute output gradients ───
        # dL/d(pre_activation) for output neurons
        # For BCE + sigmoid: gradient simplifies to (a - y)
        for i, neuron in enumerate(output_neurons):
            neuron.gradient = neuron.activation - targets[i]

        # Build outgoing connections map
        outgoing = defaultdict(list)  # from_id → [Connection]
        for conn in self.connections.values():
            outgoing[conn.from_id].append(conn)

        # ─── Backpropagate in reverse topological order ───
        for nid in reversed(self._topo_order):
            neuron = self.neurons[nid]
            if neuron.neuron_type == NeuronType.OUTPUT:
                # Gradient already set above
                # Accumulate bias gradient
                neuron.bias_grad += neuron.gradient
                # Compute weight gradients for incoming connections
                for conn_key, conn in self.connections.items():
                    if conn.to_id == nid:
                        from_neuron = self.neurons[conn.from_id]
                        conn.weight_grad += neuron.gradient * from_neuron.activation
                continue

            if neuron.neuron_type == NeuronType.INPUT:
                continue  # no parameters to update

            # Hidden neuron: accumulate gradient from all outgoing connections
            grad_sum = 0.0
            for conn in outgoing[nid]:
                to_neuron = self.neurons[conn.to_id]
                grad_sum += conn.weight * to_neuron.gradient

            # Apply activation derivative (sigmoid)
            neuron.gradient = grad_sum * sigmoid_derivative(neuron.activation)

            # Accumulate bias gradient
            neuron.bias_grad += neuron.gradient

            # Compute weight gradients for incoming connections
            for conn_key, conn in self.connections.items():
                if conn.to_id == nid:
                    from_neuron = self.neurons[conn.from_id]
                    conn.weight_grad += neuron.gradient * from_neuron.activation

        return loss

    # ─── Weight Update ───

    def update_weights(self, lr: float, use_stability: bool = False) -> None:
        """Apply gradients using SGD. Also tracks avg |Δw| for interference detection."""
        total_delta = 0.0
        count = 0
        
        # Stability hyperparameters
        beta = 0.9      # EMA decay for avg_delta
        alpha = 100.0   # How strongly stability reduces learning rate

        for conn in self.connections.values():
            base_delta = lr * conn.weight_grad
            
            if use_stability:
                # Stability: high weight magnitude & low recent variance = stable
                stability = abs(conn.weight) / (conn.avg_delta + 1e-5)
                effective_lr = lr / (1.0 + alpha * stability)
                delta = effective_lr * conn.weight_grad
                
                # Track the "gradient pressure" as the raw delta it WANTED to apply
                conn.avg_delta = beta * conn.avg_delta + (1 - beta) * abs(base_delta)
            else:
                delta = base_delta

            total_delta += abs(delta)
            conn.weight -= delta
            conn.weight_grad = 0.0
            count += 1

        for neuron in self.neurons.values():
            if neuron.neuron_type != NeuronType.INPUT:
                neuron.bias -= lr * neuron.bias_grad
                neuron.bias_grad = 0.0

        if count > 0:
            self.weight_delta_history.append(total_delta / count)

    # ─── Phase 2: Pruning (weight-based) ───

    def prune_connections(self, tau_prune: float = 0.01) -> int:
        """
        Remove connections whose weight magnitude is below threshold.

        Args:
            tau_prune: weight magnitude threshold

        Returns:
            number of connections removed
        """
        to_remove = [
            key for key, conn in self.connections.items()
            if abs(conn.weight) < tau_prune
        ]
        for key in to_remove:
            del self.connections[key]
        if to_remove:
            self._topo_dirty = True
        return len(to_remove)

    def prune_dead_neurons(self) -> int:
        """
        Remove hidden neurons that have no incoming OR no outgoing connections.
        (They can't contribute to the output, so they're dead weight.)

        Returns:
            number of neurons removed
        """
        removed = 0
        while True:
            to_remove = []
            for nid, neuron in self.neurons.items():
                if neuron.neuron_type != NeuronType.HIDDEN:
                    continue
                has_incoming = any(c.to_id == nid for c in self.connections.values())
                has_outgoing = any(c.from_id == nid for c in self.connections.values())
                if not has_incoming or not has_outgoing:
                    to_remove.append(nid)

            if not to_remove:
                break

            for nid in to_remove:
                self.remove_neuron(nid)
                removed += 1

        return removed

    def prune(self, tau_prune: float = 0.01) -> tuple[int, int]:
        """
        Full pruning step: remove weak connections, then dead neurons.

        Returns:
            (connections_removed, neurons_removed)
        """
        conn_removed = self.prune_connections(tau_prune)
        neuron_removed = self.prune_dead_neurons()
        return conn_removed, neuron_removed

    # ─── Phase 3: Growing (capacity expansion) ───

    def is_under_interference(self, window: int = 20, threshold: float = 0.04) -> bool:
        """
        Detect whether existing weights are being pulled hard in a new direction.

        High average |Δw| over recent updates means the new task is aggressively
        overwriting existing representations — the signal to grow new capacity
        instead of continuing to overwrite.

        Args:
            window:    how many recent update steps to look at
            threshold: avg |Δw| above this = interference detected

        Returns:
            True if the network should grow new neurons to absorb the conflict
        """
        if len(self.weight_delta_history) < window:
            return False
        recent = self.weight_delta_history[-window:]
        return (sum(recent) / len(recent)) > threshold

    def should_grow(self, patience: int = 50, min_improvement: float = 0.001) -> bool:
        """
        Check if loss has plateaued (not improved enough over recent history).

        Args:
            patience: number of recent epochs to look at
            min_improvement: minimum relative improvement expected

        Returns:
            True if network should grow
        """
        if len(self.loss_history) < patience:
            return False

        recent = self.loss_history[-patience:]
        old_loss = recent[0]
        new_loss = recent[-1]

        if old_loss == 0:
            return False

        improvement = (old_loss - new_loss) / abs(old_loss)
        return improvement < min_improvement

    def grow_neuron(self, max_connections: int = 4) -> int:
        """
        Add a new hidden neuron with random connections.

        Strategy (KISS): connect the new neuron to a random subset of
        existing input/hidden neurons (incoming) and to a random subset
        of existing hidden/output neurons (outgoing).

        Args:
            max_connections: max connections to create per direction

        Returns:
            the new neuron's ID
        """
        # Create new hidden neuron
        new_id = self.add_neuron(NeuronType.HIDDEN, bias=np.random.randn() * 0.1)

        # Possible source neurons (input or hidden, not the new one)
        sources = [
            n.id for n in self.neurons.values()
            if n.neuron_type in (NeuronType.INPUT, NeuronType.HIDDEN)
            and n.id != new_id
        ]

        # Possible target neurons (hidden or output, not the new one)
        targets = [
            n.id for n in self.neurons.values()
            if n.neuron_type in (NeuronType.HIDDEN, NeuronType.OUTPUT)
            and n.id != new_id
        ]

        # Pick random subsets
        n_in = min(max_connections, len(sources))
        n_out = min(max_connections, len(targets))

        if n_in > 0:
            chosen_sources = np.random.choice(sources, size=n_in, replace=False)
            for src in chosen_sources:
                self.add_connection(int(src), new_id, weight=np.random.randn() * 0.1)

        if n_out > 0:
            chosen_targets = np.random.choice(targets, size=n_out, replace=False)
            for tgt in chosen_targets:
                # Avoid creating cycles: only connect forward
                # (new neuron → target) if target is not a source of new neuron
                try:
                    self.add_connection(new_id, int(tgt))
                    # Verify no cycle was created
                    self._topological_sort()
                except (ValueError, AssertionError):
                    # Cycle detected — remove the connection
                    self.remove_connection(new_id, int(tgt))

        return new_id

    # ─── Utilities ───

    def get_input_neurons(self) -> list[Neuron]:
        return sorted(
            [n for n in self.neurons.values() if n.neuron_type == NeuronType.INPUT],
            key=lambda n: n.id
        )

    def get_hidden_neurons(self) -> list[Neuron]:
        return sorted(
            [n for n in self.neurons.values() if n.neuron_type == NeuronType.HIDDEN],
            key=lambda n: n.id
        )

    def get_output_neurons(self) -> list[Neuron]:
        return sorted(
            [n for n in self.neurons.values() if n.neuron_type == NeuronType.OUTPUT],
            key=lambda n: n.id
        )

    def summary(self) -> str:
        """Print a summary of the network."""
        n_input = len(self.get_input_neurons())
        n_hidden = len(self.get_hidden_neurons())
        n_output = len(self.get_output_neurons())
        n_conn = len(self.connections)
        lines = [
            f"DynamicNetwork Summary:",
            f"  Neurons:     {len(self.neurons)} (input={n_input}, hidden={n_hidden}, output={n_output})",
            f"  Connections: {n_conn}",
            f"  Connections detail:"
        ]
        for (f, t), conn in sorted(self.connections.items()):
            fn = self.neurons[f].neuron_type.value
            tn = self.neurons[t].neuron_type.value
            lines.append(f"    n{f}({fn}) → n{t}({tn}): w={conn.weight:.4f}")
        return "\n".join(lines)

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predict class (0 or 1) for multiple samples."""
        predictions = []
        for x in X:
            out = self.forward(x)
            predictions.append((out > 0.5).astype(int))
        return np.array(predictions)

    def accuracy(self, X: np.ndarray, y: np.ndarray) -> float:
        """Compute classification accuracy."""
        preds = self.predict(X)
        # Flatten for comparison
        return np.mean(preds.flatten() == y.flatten())


# ─────────────────────────────────────────────
# Data Generation
# ─────────────────────────────────────────────

def make_linearly_separable(n_samples: int = 200, noise: float = 0.1, seed: int = 42, center=(0.0, 0.0)) -> tuple:
    """
    Generate a simple linearly separable 2D dataset.

    Class 0: points below the line y = x
    Class 1: points above the line y = x
    """
    rng = np.random.RandomState(seed)
    X = rng.randn(n_samples, 2)
    y = (X[:, 1] > X[:, 0] + rng.randn(n_samples) * noise).astype(float)
    X += np.array(center)
    return X, y


def make_circles(n_samples: int = 200, noise: float = 0.05, seed: int = 42, center=(0.0, 0.0)) -> tuple:
    """
    Generate concentric circles dataset (NOT linearly separable).

    Class 0: inner circle
    Class 1: outer circle
    """
    rng = np.random.RandomState(seed)
    n_each = n_samples // 2

    # Inner circle
    theta_inner = rng.uniform(0, 2 * np.pi, n_each)
    r_inner = 0.5 + rng.randn(n_each) * noise
    X_inner = np.column_stack([r_inner * np.cos(theta_inner), r_inner * np.sin(theta_inner)])

    # Outer circle
    theta_outer = rng.uniform(0, 2 * np.pi, n_each)
    r_outer = 1.5 + rng.randn(n_each) * noise
    X_outer = np.column_stack([r_outer * np.cos(theta_outer), r_outer * np.sin(theta_outer)])

    X = np.vstack([X_inner, X_outer])
    X += np.array(center)
    y = np.array([0.0] * n_each + [1.0] * n_each)

    # Shuffle
    idx = rng.permutation(n_samples)
    return X[idx], y[idx]


def make_xor(n_samples: int = 200, noise: float = 0.15, seed: int = 42, center=(0.0, 0.0)) -> tuple:
    """
    Generate XOR-like dataset (NOT linearly separable).

    Class 1: top-left and bottom-right quadrants
    Class 0: top-right and bottom-left quadrants
    """
    rng = np.random.RandomState(seed)
    X = rng.randn(n_samples, 2)
    y = ((X[:, 0] * X[:, 1]) > 0).astype(float)
    X += np.array(center)
    return X, y


# ─────────────────────────────────────────────
# Training
# ─────────────────────────────────────────────

def train(
    net: DynamicNetwork,
    X: np.ndarray,
    y: np.ndarray,
    epochs: int = 1000,
    lr: float = 0.5,
    print_every: int = 100,
) -> DynamicNetwork:
    """
    Train the network on a dataset (static — no grow/prune).

    Args:
        net: the network to train
        X: input data, shape (n_samples, n_features)
        y: target labels, shape (n_samples,)
        epochs: number of full passes through the data
        lr: learning rate
        print_every: print loss every N epochs
    """
    n_samples = len(X)

    for epoch in range(epochs):
        # Shuffle data each epoch
        idx = np.random.permutation(n_samples)
        epoch_loss = 0.0

        for i in idx:
            # Forward
            output = net.forward(X[i])

            # Backward
            loss = net.backward(np.array([y[i]]))
            epoch_loss += loss

            # Update
            net.update_weights(lr)

        avg_loss = epoch_loss / n_samples

        # Track history
        net.loss_history.append(avg_loss)
        net.neuron_count_history.append(len(net.get_hidden_neurons()))
        net.connection_count_history.append(len(net.connections))

        if (epoch + 1) % print_every == 0 or epoch == 0:
            acc = net.accuracy(X, y)
            print(f"Epoch {epoch+1:4d} | Loss: {avg_loss:.4f} | Acc: {acc:.2%} | "
                  f"Neurons: {len(net.neurons)} | Connections: {len(net.connections)}")

    return net


def train_dynamic(
    net: DynamicNetwork,
    X: np.ndarray,
    y: np.ndarray,
    epochs: int = 2000,
    lr: float = 0.5,
    print_every: int = 100,
    # Pruning params
    prune_every: int = 50,
    tau_prune: float = 0.01,
    # Growing params
    grow_check_every: int = 100,
    grow_patience: int = 50,
    grow_min_improvement: float = 0.001,
    max_hidden_neurons: int = 10,
    max_connections_per_growth: int = 4,
    # Complexity penalty
    lambda_neurons: float = 0.001,
    lambda_connections: float = 0.0005,
) -> DynamicNetwork:
    """
    Train with dynamic grow/prune.

    The network starts small and:
    - PRUNES connections with |w| < tau_prune every `prune_every` epochs
    - GROWS a new neuron when loss plateaus (checked every `grow_check_every` epochs)
    - Pays a complexity penalty for each neuron and connection

    Args:
        net: the network to train
        X, y: training data
        epochs: max training epochs
        lr: learning rate
        prune_every: run pruning every N epochs
        tau_prune: weight magnitude threshold for pruning
        grow_check_every: check for growth every N epochs
        grow_patience: epochs of plateau before growing
        grow_min_improvement: minimum relative improvement to not be "plateau"
        max_hidden_neurons: cap on hidden neurons to prevent unbounded growth
        max_connections_per_growth: max connections when adding a neuron
        lambda_neurons: complexity penalty per hidden neuron
        lambda_connections: complexity penalty per connection
    """
    n_samples = len(X)

    for epoch in range(epochs):
        # Shuffle data each epoch
        idx = np.random.permutation(n_samples)
        epoch_loss = 0.0

        for i in idx:
            output = net.forward(X[i])
            loss = net.backward(np.array([y[i]]))
            epoch_loss += loss
            net.update_weights(lr)

        avg_loss = epoch_loss / n_samples

        # Add complexity penalty (for reporting; doesn't affect gradients directly)
        n_hidden = len(net.get_hidden_neurons())
        n_conn = len(net.connections)
        total_loss = avg_loss + lambda_neurons * n_hidden + lambda_connections * n_conn

        # Track history
        net.loss_history.append(avg_loss)
        net.neuron_count_history.append(n_hidden)
        net.connection_count_history.append(n_conn)

        # ─── Pruning step ───
        if (epoch + 1) % prune_every == 0 and epoch > 0:
            conn_removed, neuron_removed = net.prune(tau_prune)
            if conn_removed > 0 or neuron_removed > 0:
                print(f"  [PRUNE] Epoch {epoch+1}: removed {conn_removed} connections, "
                      f"{neuron_removed} neurons")

        # ─── Growing step ───
        if (epoch + 1) % grow_check_every == 0:
            current_hidden = len(net.get_hidden_neurons())
            if current_hidden < max_hidden_neurons and net.should_grow(grow_patience, grow_min_improvement):
                new_id = net.grow_neuron(max_connections_per_growth)
                print(f"  [GROW]  Epoch {epoch+1}: added neuron n{new_id} "
                      f"(now {len(net.get_hidden_neurons())} hidden neurons, "
                      f"{len(net.connections)} connections)")

        # ─── Logging ───
        if (epoch + 1) % print_every == 0 or epoch == 0:
            acc = net.accuracy(X, y)
            print(f"Epoch {epoch+1:4d} | Loss: {avg_loss:.4f} (total: {total_loss:.4f}) | "
                  f"Acc: {acc:.2%} | Hidden: {n_hidden} | Conn: {n_conn}")

    return net


# ─────────────────────────────────────────────
# Build & Run
# ─────────────────────────────────────────────

def build_minimal_network() -> DynamicNetwork:
    """
    Build the simplest possible network:
    2 input neurons → 1 output neuron (direct connections, no hidden layer)
    """
    net = DynamicNetwork()

    # 2 input neurons
    i0 = net.add_neuron(NeuronType.INPUT)
    i1 = net.add_neuron(NeuronType.INPUT)

    # 1 output neuron
    o0 = net.add_neuron(NeuronType.OUTPUT, bias=0.0)

    # Direct connections: each input → output
    net.add_connection(i0, o0)
    net.add_connection(i1, o0)

    return net


def build_network_with_hidden(n_hidden: int = 2) -> DynamicNetwork:
    """
    Build a network with a hidden layer:
    2 input → n_hidden hidden → 1 output
    """
    net = DynamicNetwork()

    # Input
    i0 = net.add_neuron(NeuronType.INPUT)
    i1 = net.add_neuron(NeuronType.INPUT)

    # Hidden
    hidden_ids = []
    for _ in range(n_hidden):
        hid = net.add_neuron(NeuronType.HIDDEN, bias=np.random.randn() * 0.1)
        hidden_ids.append(hid)

    # Output
    o0 = net.add_neuron(NeuronType.OUTPUT, bias=0.0)

    # Connect input → hidden
    for h in hidden_ids:
        net.add_connection(i0, h)
        net.add_connection(i1, h)

    # Connect hidden → output
    for h in hidden_ids:
        net.add_connection(h, o0)

    return net


# ─────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────

if __name__ == "__main__":
    np.random.seed(42)

    # ═══════════════════════════════════════════
    # Phase 1: Static network on linear data
    # ═══════════════════════════════════════════
    print("=" * 60)
    print("Phase 1: Static Graph Network — Linearly Separable Data")
    print("=" * 60)

    X_lin, y_lin = make_linearly_separable(n_samples=200, noise=0.1)
    print(f"\nDataset: {len(X_lin)} samples, 2 features")
    print(f"Class distribution: {np.mean(y_lin):.1%} positive\n")

    net1 = build_minimal_network()
    print(net1.summary())
    print()
    net1 = train(net1, X_lin, y_lin, epochs=500, lr=0.5, print_every=100)
    print(f"\nFinal accuracy: {net1.accuracy(X_lin, y_lin):.2%}")

    # ═══════════════════════════════════════════
    # Phase 2: Pruning demo
    # ═══════════════════════════════════════════
    print("\n" + "=" * 60)
    print("Phase 2: Pruning — Start fully connected, prune to minimal")
    print("=" * 60)

    # Build an over-connected network for linear data
    net2 = build_network_with_hidden(n_hidden=4)
    print(f"\nBefore training:")
    print(net2.summary())
    print()

    # Train first
    net2 = train(net2, X_lin, y_lin, epochs=300, lr=0.5, print_every=100)
    acc_before = net2.accuracy(X_lin, y_lin)
    print(f"\nAccuracy before pruning: {acc_before:.2%}")
    print(f"Connections before pruning: {len(net2.connections)}")

    # Now prune
    conn_removed, neuron_removed = net2.prune(tau_prune=0.1)
    print(f"\nPruned: {conn_removed} connections, {neuron_removed} neurons")
    print(f"Connections after pruning: {len(net2.connections)}")
    print(f"Hidden neurons after pruning: {len(net2.get_hidden_neurons())}")

    # Fine-tune after pruning
    net2 = train(net2, X_lin, y_lin, epochs=200, lr=0.3, print_every=100)
    acc_after = net2.accuracy(X_lin, y_lin)
    print(f"\nAccuracy after pruning + fine-tuning: {acc_after:.2%}")
    print(net2.summary())

    # ═══════════════════════════════════════════
    # Phase 3: Dynamic growing on non-linear data
    # ═══════════════════════════════════════════
    print("\n" + "=" * 60)
    print("Phase 3: Dynamic Growing — XOR data (needs hidden neurons)")
    print("=" * 60)

    X_xor, y_xor = make_xor(n_samples=300, noise=0.0)
    print(f"\nDataset: {len(X_xor)} samples, XOR pattern")
    print(f"Class distribution: {np.mean(y_xor):.1%} positive\n")

    # Start with the minimal network (no hidden neurons!)
    net3 = build_minimal_network()
    print("Starting with MINIMAL network (no hidden neurons):")
    print(net3.summary())
    print()

    # Train dynamically — the network should grow hidden neurons to solve XOR
    net3 = train_dynamic(
        net3, X_xor, y_xor,
        epochs=2000,
        lr=0.5,
        print_every=200,
        prune_every=100,
        tau_prune=0.01,
        grow_check_every=100,
        grow_patience=50,
        grow_min_improvement=0.005,
        max_hidden_neurons=8,
        max_connections_per_growth=4,
        lambda_neurons=0.001,
        lambda_connections=0.0005,
    )

    final_acc = net3.accuracy(X_xor, y_xor)
    print(f"\nFinal accuracy: {final_acc:.2%}")
    print(net3.summary())

    # ─── Also try circles ───
    print("\n" + "=" * 60)
    print("Phase 3b: Dynamic Growing — Circles data")
    print("=" * 60)

    X_circ, y_circ = make_circles(n_samples=300, noise=0.05)
    print(f"\nDataset: {len(X_circ)} samples, concentric circles")
    print(f"Class distribution: {np.mean(y_circ):.1%} positive\n")

    net4 = build_minimal_network()
    print("Starting with MINIMAL network:")
    print(net4.summary())
    print()

    net4 = train_dynamic(
        net4, X_circ, y_circ,
        epochs=2000,
        lr=0.5,
        print_every=200,
        prune_every=100,
        tau_prune=0.01,
        grow_check_every=100,
        grow_patience=50,
        grow_min_improvement=0.005,
        max_hidden_neurons=8,
        max_connections_per_growth=4,
    )

    final_acc4 = net4.accuracy(X_circ, y_circ)
    print(f"\nFinal accuracy: {final_acc4:.2%}")
    print(net4.summary())
