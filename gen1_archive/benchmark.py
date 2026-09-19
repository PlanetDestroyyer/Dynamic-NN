"""
Dynamic Neural Network — Phase 5: Benchmarking

Compare dynamic growing network vs fixed MLP on same datasets.

Metrics:
- Final accuracy
- Parameters used (neurons × connections as proxy)
- Epochs to reach threshold accuracy
- Training stability (loss variance)
"""

import numpy as np
import time
from dataclasses import dataclass

from dynamic_nn import (
    DynamicNetwork, NeuronType,
    build_minimal_network, build_network_with_hidden,
    make_xor, make_circles, make_linearly_separable,
    train, train_dynamic,
)


# ─────────────────────────────────────────────
# Result Container
# ─────────────────────────────────────────────

@dataclass
class BenchmarkResult:
    name: str
    dataset: str
    final_accuracy: float
    final_hidden_neurons: int
    final_connections: int
    total_neurons: int          # including input/output
    epochs_to_90pct: int        # -1 if never reached
    epochs_to_95pct: int        # -1 if never reached
    final_loss: float
    loss_variance_last100: float # training stability
    train_time_sec: float

    def n_params(self) -> int:
        """Approximate parameter count: connections + biases (excl. input)."""
        return self.final_connections + (self.total_neurons - 2)  # rough

    def __str__(self) -> str:
        return (
            f"\n{'─'*56}\n"
            f"  {self.name} on {self.dataset}\n"
            f"{'─'*56}\n"
            f"  Accuracy:        {self.final_accuracy:.2%}\n"
            f"  Hidden neurons:  {self.final_hidden_neurons}\n"
            f"  Connections:     {self.final_connections}\n"
            f"  ~Parameters:     {self.n_params()}\n"
            f"  Epochs to 90%:   {self.epochs_to_90pct if self.epochs_to_90pct != -1 else 'never'}\n"
            f"  Epochs to 95%:   {self.epochs_to_95pct if self.epochs_to_95pct != -1 else 'never'}\n"
            f"  Final loss:      {self.final_loss:.4f}\n"
            f"  Loss stability:  {self.loss_variance_last100:.6f} (lower=better)\n"
            f"  Train time:      {self.train_time_sec:.1f}s\n"
        )


# ─────────────────────────────────────────────
# Instrumented Training (tracks milestones)
# ─────────────────────────────────────────────

def train_with_milestones(
    net: DynamicNetwork,
    X: np.ndarray,
    y: np.ndarray,
    epochs: int,
    lr: float,
    name: str,
    dataset: str,
    dynamic: bool = False,
    **dynamic_kwargs,
) -> BenchmarkResult:
    """Train and track accuracy milestones."""
    start = time.time()
    n_samples = len(X)
    epoch_to_90 = -1
    epoch_to_95 = -1

    # Dynamic training params
    prune_every = dynamic_kwargs.get('prune_every', 100)
    tau_prune = dynamic_kwargs.get('tau_prune', 0.01)
    grow_check_every = dynamic_kwargs.get('grow_check_every', 100)
    grow_patience = dynamic_kwargs.get('grow_patience', 50)
    grow_min_improvement = dynamic_kwargs.get('grow_min_improvement', 0.005)
    max_hidden = dynamic_kwargs.get('max_hidden_neurons', 10)
    max_conn_growth = dynamic_kwargs.get('max_connections_per_growth', 4)

    for epoch in range(epochs):
        idx = np.random.permutation(n_samples)
        epoch_loss = 0.0

        for i in idx:
            net.forward(X[i])
            loss = net.backward(np.array([y[i]]))
            epoch_loss += loss
            net.update_weights(lr)

        avg_loss = epoch_loss / n_samples
        net.loss_history.append(avg_loss)
        net.neuron_count_history.append(len(net.get_hidden_neurons()))
        net.connection_count_history.append(len(net.connections))

        # Dynamic: prune + grow
        if dynamic:
            if (epoch + 1) % prune_every == 0 and epoch > 0:
                net.prune(tau_prune)
            if (epoch + 1) % grow_check_every == 0:
                if len(net.get_hidden_neurons()) < max_hidden:
                    if net.should_grow(grow_patience, grow_min_improvement):
                        net.grow_neuron(max_conn_growth)

        # Track milestones (check every 10 epochs for speed)
        if (epoch + 1) % 10 == 0:
            acc = net.accuracy(X, y)
            if epoch_to_90 == -1 and acc >= 0.90:
                epoch_to_90 = epoch + 1
            if epoch_to_95 == -1 and acc >= 0.95:
                epoch_to_95 = epoch + 1

    elapsed = time.time() - start
    final_acc = net.accuracy(X, y)
    final_loss = net.loss_history[-1] if net.loss_history else 0.0
    loss_var = float(np.var(net.loss_history[-100:])) if len(net.loss_history) >= 100 else 0.0

    return BenchmarkResult(
        name=name,
        dataset=dataset,
        final_accuracy=final_acc,
        final_hidden_neurons=len(net.get_hidden_neurons()),
        final_connections=len(net.connections),
        total_neurons=len(net.neurons),
        epochs_to_90pct=epoch_to_90,
        epochs_to_95pct=epoch_to_95,
        final_loss=final_loss,
        loss_variance_last100=loss_var,
        train_time_sec=elapsed,
    )


# ─────────────────────────────────────────────
# Run One Benchmark (multiple seeds for reliability)
# ─────────────────────────────────────────────

def benchmark_dataset(
    X: np.ndarray,
    y: np.ndarray,
    dataset_name: str,
    epochs: int = 1500,
    lr: float = 0.5,
    n_runs: int = 3,
) -> dict[str, list[BenchmarkResult]]:
    """
    Run 3 models on the same dataset n_runs times with different seeds.

    Models:
      1. Dynamic  — starts 2→1, grows/prunes as needed
      2. Small MLP — 2→4→1 (small fixed)
      3. Large MLP — 2→16→1 (large fixed, typical overkill)
    """
    results = {
        "Dynamic (2→?→1)": [],
        "Fixed Small (2→4→1)": [],
        "Fixed Large (2→16→1)": [],
    }

    for run in range(n_runs):
        seed = 100 + run * 7
        np.random.seed(seed)
        print(f"  Run {run+1}/{n_runs} (seed={seed})...")

        # ─── 1. Dynamic network ───
        net_dyn = build_minimal_network()
        r = train_with_milestones(
            net_dyn, X, y, epochs, lr,
            name="Dynamic (2→?→1)", dataset=dataset_name,
            dynamic=True,
            prune_every=100, tau_prune=0.01,
            grow_check_every=100, grow_patience=50,
            grow_min_improvement=0.005,
            max_hidden_neurons=10, max_connections_per_growth=4,
        )
        results["Dynamic (2→?→1)"].append(r)

        # ─── 2. Small fixed MLP ───
        np.random.seed(seed)  # same seed for fair comparison
        net_small = build_network_with_hidden(n_hidden=4)
        r = train_with_milestones(
            net_small, X, y, epochs, lr,
            name="Fixed Small (2→4→1)", dataset=dataset_name,
            dynamic=False,
        )
        results["Fixed Small (2→4→1)"].append(r)

        # ─── 3. Large fixed MLP ───
        np.random.seed(seed)
        net_large = build_network_with_hidden(n_hidden=16)
        r = train_with_milestones(
            net_large, X, y, epochs, lr,
            name="Fixed Large (2→16→1)", dataset=dataset_name,
            dynamic=False,
        )
        results["Fixed Large (2→16→1)"].append(r)

    return results


# ─────────────────────────────────────────────
# Aggregate & Print Results
# ─────────────────────────────────────────────

def avg(vals, default=-1):
    """Average of a list, ignoring -1 (never reached)."""
    valid = [v for v in vals if v != -1]
    return sum(valid) / len(valid) if valid else default


def print_comparison_table(all_results: dict[str, list[BenchmarkResult]], dataset: str):
    """Print a clean comparison table for one dataset."""
    print(f"\n{'═'*72}")
    print(f"  Dataset: {dataset}")
    print(f"{'═'*72}")
    print(f"  {'Model':<24} {'Acc':>7} {'Hidden':>7} {'Conn':>6} {'~Params':>8} "
          f"{'→90%':>7} {'→95%':>7} {'Stability':>11}")
    print(f"  {'─'*24} {'─'*7} {'─'*7} {'─'*6} {'─'*8} {'─'*7} {'─'*7} {'─'*11}")

    for model_name, runs in all_results.items():
        acc   = avg([r.final_accuracy for r in runs])
        hidden = avg([r.final_hidden_neurons for r in runs])
        conn  = avg([r.final_connections for r in runs])
        params = avg([r.n_params() for r in runs])
        to90  = avg([r.epochs_to_90pct for r in runs])
        to95  = avg([r.epochs_to_95pct for r in runs])
        stab  = avg([r.loss_variance_last100 for r in runs])

        to90_str = f"{int(to90)}" if to90 != -1 else "never"
        to95_str = f"{int(to95)}" if to95 != -1 else "never"

        print(f"  {model_name:<24} {acc:>7.2%} {hidden:>7.1f} {conn:>6.1f} "
              f"{params:>8.1f} {to90_str:>7} {to95_str:>7} {stab:>11.6f}")

    print()


# ─────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────

if __name__ == "__main__":
    EPOCHS = 1500
    LR = 0.5
    N_RUNS = 3
    N_SAMPLES = 300

    print("=" * 72)
    print("  Phase 5: Benchmarking — Dynamic vs Fixed MLP")
    print(f"  {EPOCHS} epochs | {N_SAMPLES} samples | {N_RUNS} runs each")
    print("=" * 72)

    all_tables = {}

    # ─── Dataset 1: Linearly Separable ───
    print("\n[1/3] Linearly Separable data...")
    X_lin, y_lin = make_linearly_separable(n_samples=N_SAMPLES, noise=0.1, seed=42)
    results_lin = benchmark_dataset(X_lin, y_lin, "Linear", EPOCHS, LR, N_RUNS)
    all_tables["Linear (easy)"] = results_lin

    # ─── Dataset 2: XOR ───
    print("\n[2/3] XOR data...")
    X_xor, y_xor = make_xor(n_samples=N_SAMPLES, noise=0.0, seed=42)
    results_xor = benchmark_dataset(X_xor, y_xor, "XOR", EPOCHS, LR, N_RUNS)
    all_tables["XOR (non-linear)"] = results_xor

    # ─── Dataset 3: Circles ───
    print("\n[3/3] Circles data...")
    X_circ, y_circ = make_circles(n_samples=N_SAMPLES, noise=0.05, seed=42)
    results_circ = benchmark_dataset(X_circ, y_circ, "Circles", EPOCHS, LR, N_RUNS)
    all_tables["Circles (non-linear)"] = results_circ

    # ─── Print all tables ───
    print("\n\n" + "=" * 72)
    print("  RESULTS SUMMARY")
    print("=" * 72)

    for dataset_label, results in all_tables.items():
        print_comparison_table(results, dataset_label)

    # ─── Efficiency summary ───
    print("=" * 72)
    print("  EFFICIENCY INSIGHT")
    print("=" * 72)
    print("""
  The key question: does the dynamic network achieve competitive accuracy
  while using fewer parameters than the fixed large MLP?

  Parameter efficiency ratio = Dynamic params / Large MLP params
  If < 1.0, dynamic network is more parameter-efficient.
""")

    for dataset_label, results in all_tables.items():
        dyn_params  = avg([r.n_params() for r in results["Dynamic (2→?→1)"]])
        large_params = avg([r.n_params() for r in results["Fixed Large (2→16→1)"]])
        dyn_acc     = avg([r.final_accuracy for r in results["Dynamic (2→?→1)"]])
        large_acc   = avg([r.final_accuracy for r in results["Fixed Large (2→16→1)"]])

        ratio = dyn_params / large_params if large_params > 0 else 0
        acc_gap = dyn_acc - large_acc

        print(f"  {dataset_label}:")
        print(f"    Dynamic:   {dyn_params:.0f} params, {dyn_acc:.2%} acc")
        print(f"    Fixed Big: {large_params:.0f} params, {large_acc:.2%} acc")
        print(f"    → Parameter ratio: {ratio:.2f}x  |  Accuracy gap: {acc_gap:+.2%}")
        print()
