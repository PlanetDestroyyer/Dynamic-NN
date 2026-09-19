"""
Dynamic Neural Network — Phase 6: Continual Learning

The key experiment: does a dynamic network resist catastrophic forgetting?

Protocol:
  1. Train on Task A (e.g. circles)
  2. Train on Task B (e.g. XOR) — same network, no reset
  3. Test on Task A again → measure forgetting

Compared against:
  - Fixed Small MLP (2→4→1)
  - Fixed Large MLP (2→16→1)
  - Dynamic network (starts 2→1, grows as needed)

Forgetting metric: accuracy_on_A_after_B - accuracy_on_A_before_B
  0.00 = no forgetting (ideal)
  -1.0 = total catastrophic forgetting
"""

import numpy as np
from dataclasses import dataclass

from dynamic_nn import (
    DynamicNetwork, NeuronType,
    build_minimal_network, build_network_with_hidden,
    make_xor, make_circles, make_linearly_separable,
)


# ─────────────────────────────────────────────
# Continual Training Loop
# ─────────────────────────────────────────────

def train_task(
    net: DynamicNetwork,
    X: np.ndarray,
    y: np.ndarray,
    epochs: int,
    lr: float,
    dynamic: bool = False,
    prune_every: int = 100,
    tau_prune: float = 0.01,
    grow_check_every: int = 100,
    grow_patience: int = 50,
    grow_min_improvement: float = 0.005,
    max_hidden_neurons: int = 12,
    max_connections_per_growth: int = 4,
) -> None:
    """Train a network on one task (modifies in-place)."""
    n_samples = len(X)

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

        if dynamic:
            if (epoch + 1) % prune_every == 0 and epoch > 0:
                net.prune(tau_prune)
            if (epoch + 1) % grow_check_every == 0:
                if len(net.get_hidden_neurons()) < max_hidden_neurons:
                    if net.should_grow(grow_patience, grow_min_improvement):
                        net.grow_neuron(max_connections_per_growth)


# ─────────────────────────────────────────────
# Result Container
# ─────────────────────────────────────────────

@dataclass
class ContinualResult:
    model_name: str
    # Accuracy on each task, measured at each stage
    acc_A_after_A: float    # after training on A
    acc_B_after_B: float    # after training on B
    acc_A_after_B: float    # retest A after training on B  ← key metric
    acc_A_after_AB: float   # after retraining on A (recovery)

    final_hidden: int
    final_connections: int

    @property
    def forgetting(self) -> float:
        """How much Task A accuracy dropped after training on B."""
        return self.acc_A_after_B - self.acc_A_after_A

    @property
    def recovery(self) -> float:
        """How much accuracy recovered after retraining on A."""
        return self.acc_A_after_AB - self.acc_A_after_B

    def __str__(self) -> str:
        bar_forget = "█" * int(abs(self.forgetting) * 20) if self.forgetting < 0 else ""
        bar_recover = "▓" * int(self.recovery * 20) if self.recovery > 0 else ""
        return (
            f"\n  {'─'*52}\n"
            f"  {self.model_name}\n"
            f"  {'─'*52}\n"
            f"  After Task A:          {self.acc_A_after_A:.2%}\n"
            f"  After Task B:          {self.acc_B_after_B:.2%} (on B)\n"
            f"  Task A retested:       {self.acc_A_after_B:.2%}  "
            f"{'← FORGOT' if self.forgetting < -0.05 else '← held'}\n"
            f"  Forgetting:            {self.forgetting:+.2%}  {bar_forget}\n"
            f"  After relearn A:       {self.acc_A_after_AB:.2%}\n"
            f"  Recovery:              {self.recovery:+.2%}  {bar_recover}\n"
            f"  Final hidden neurons:  {self.final_hidden}\n"
            f"  Final connections:     {self.final_connections}\n"
        )


# ─────────────────────────────────────────────
# Run One Continual Learning Trial
# ─────────────────────────────────────────────

def run_continual_trial(
    model_name: str,
    net: DynamicNetwork,
    X_A: np.ndarray, y_A: np.ndarray,
    X_B: np.ndarray, y_B: np.ndarray,
    epochs_per_task: int = 800,
    lr: float = 0.5,
    dynamic: bool = False,
) -> ContinualResult:
    """
    Run the full A → B → A continual learning protocol on one model.
    """
    # ── Step 1: Train on Task A ──
    print(f"    Training on Task A ({epochs_per_task} epochs)...", end="", flush=True)
    train_task(net, X_A, y_A, epochs_per_task, lr, dynamic=dynamic)
    acc_A_after_A = net.accuracy(X_A, y_A)
    print(f" Acc(A): {acc_A_after_A:.2%}")

    # ── Step 2: Train on Task B (same network, no reset!) ──
    print(f"    Training on Task B ({epochs_per_task} epochs)...", end="", flush=True)
    train_task(net, X_B, y_B, epochs_per_task, lr, dynamic=dynamic)
    acc_B_after_B = net.accuracy(X_B, y_B)
    acc_A_after_B = net.accuracy(X_A, y_A)
    print(f" Acc(B): {acc_B_after_B:.2%}  |  Acc(A retested): {acc_A_after_B:.2%}")

    # ── Step 3: Retrain on Task A (recovery test) ──
    print(f"    Retraining on Task A ({epochs_per_task} epochs)...", end="", flush=True)
    train_task(net, X_A, y_A, epochs_per_task, lr, dynamic=dynamic)
    acc_A_after_AB = net.accuracy(X_A, y_A)
    print(f" Acc(A): {acc_A_after_AB:.2%}")

    return ContinualResult(
        model_name=model_name,
        acc_A_after_A=acc_A_after_A,
        acc_B_after_B=acc_B_after_B,
        acc_A_after_B=acc_A_after_B,
        acc_A_after_AB=acc_A_after_AB,
        final_hidden=len(net.get_hidden_neurons()),
        final_connections=len(net.connections),
    )


# ─────────────────────────────────────────────
# Full Continual Learning Comparison
# ─────────────────────────────────────────────

def run_continual_experiment(
    X_A, y_A, X_B, y_B,
    task_a_name: str, task_b_name: str,
    epochs_per_task: int = 800,
    lr: float = 0.5,
    n_runs: int = 3,
) -> list[tuple[str, list[ContinualResult]]]:
    """
    Compare three models on the A→B→A continual learning protocol.
    Returns list of (model_name, [results across runs]).
    """
    all_results = {
        "Dynamic (2→?→1)":      [],
        "Fixed Small (2→4→1)":  [],
        "Fixed Large (2→16→1)": [],
    }

    for run in range(n_runs):
        seed = 42 + run * 13
        np.random.seed(seed)
        print(f"\n  Run {run+1}/{n_runs} (seed={seed})")

        # ─── Dynamic network ───
        print(f"  [Dynamic]")
        net_dyn = build_minimal_network()
        r = run_continual_trial(
            "Dynamic (2→?→1)", net_dyn, X_A, y_A, X_B, y_B,
            epochs_per_task, lr, dynamic=True,
        )
        all_results["Dynamic (2→?→1)"].append(r)

        # ─── Fixed small ───
        np.random.seed(seed)
        print(f"  [Fixed Small]")
        net_small = build_network_with_hidden(n_hidden=4)
        r = run_continual_trial(
            "Fixed Small (2→4→1)", net_small, X_A, y_A, X_B, y_B,
            epochs_per_task, lr, dynamic=False,
        )
        all_results["Fixed Small (2→4→1)"].append(r)

        # ─── Fixed large ───
        np.random.seed(seed)
        print(f"  [Fixed Large]")
        net_large = build_network_with_hidden(n_hidden=16)
        r = run_continual_trial(
            "Fixed Large (2→16→1)", net_large, X_A, y_A, X_B, y_B,
            epochs_per_task, lr, dynamic=False,
        )
        all_results["Fixed Large (2→16→1)"].append(r)

    return all_results


# ─────────────────────────────────────────────
# Print Summary Table
# ─────────────────────────────────────────────

def avg(vals):
    return sum(vals) / len(vals) if vals else 0.0


def print_continual_table(
    all_results: dict, task_a: str, task_b: str
):
    print(f"\n{'═'*72}")
    print(f"  Task A: {task_a}  →  Task B: {task_b}  →  Retest A")
    print(f"{'═'*72}")
    print(f"  {'Model':<24} {'A→A':>7} {'B→B':>7} {'A retested':>11} "
          f"{'Forget':>9} {'Recovery':>10}")
    print(f"  {'─'*24} {'─'*7} {'─'*7} {'─'*11} {'─'*9} {'─'*10}")

    for model_name, runs in all_results.items():
        a_after_a  = avg([r.acc_A_after_A  for r in runs])
        b_after_b  = avg([r.acc_B_after_B  for r in runs])
        a_after_b  = avg([r.acc_A_after_B  for r in runs])
        forget     = avg([r.forgetting      for r in runs])
        recovery   = avg([r.recovery        for r in runs])

        forget_str = f"{forget:+.2%}"
        print(f"  {model_name:<24} {a_after_a:>7.2%} {b_after_b:>7.2%} "
              f"{a_after_b:>11.2%} {forget_str:>9} {recovery:>+10.2%}")
    print()


# ─────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────

if __name__ == "__main__":
    EPOCHS_PER_TASK = 800
    LR = 0.5
    N_RUNS = 3
    N_SAMPLES = 300

    print("=" * 72)
    print("  Phase 6: Continual Learning — Catastrophic Forgetting Test")
    print(f"  Protocol: A → B → A  |  {EPOCHS_PER_TASK} epochs/task | {N_RUNS} runs")
    print("=" * 72)

    # Generate datasets
    X_circles, y_circles = make_circles(n_samples=N_SAMPLES, noise=0.05, seed=42)
    X_xor,     y_xor     = make_xor(n_samples=N_SAMPLES, noise=0.0, seed=42)
    X_lin,     y_lin     = make_linearly_separable(n_samples=N_SAMPLES, noise=0.1, seed=42)

    # ─── Experiment 1: Circles → XOR → Circles ───
    print("\n[Experiment 1] Circles → XOR → Circles")
    print("─" * 72)
    results_1 = run_continual_experiment(
        X_circles, y_circles, X_xor, y_xor,
        task_a_name="Circles", task_b_name="XOR",
        epochs_per_task=EPOCHS_PER_TASK, lr=LR, n_runs=N_RUNS,
    )

    # ─── Experiment 2: XOR → Circles → XOR ───
    print("\n[Experiment 2] XOR → Circles → XOR")
    print("─" * 72)
    results_2 = run_continual_experiment(
        X_xor, y_xor, X_circles, y_circles,
        task_a_name="XOR", task_b_name="Circles",
        epochs_per_task=EPOCHS_PER_TASK, lr=LR, n_runs=N_RUNS,
    )

    # ─── Experiment 3: Linear → Circles → Linear ───
    print("\n[Experiment 3] Linear → Circles → Linear")
    print("─" * 72)
    results_3 = run_continual_experiment(
        X_lin, y_lin, X_circles, y_circles,
        task_a_name="Linear", task_b_name="Circles",
        epochs_per_task=EPOCHS_PER_TASK, lr=LR, n_runs=N_RUNS,
    )

    # ─── Print all tables ───
    print("\n\n" + "=" * 72)
    print("  CONTINUAL LEARNING RESULTS SUMMARY")
    print("  (Forgetting = acc_on_A_after_B_training - acc_on_A_after_A_training)")
    print("  (Closer to 0 = less forgetting = better)")
    print("=" * 72)

    print_continual_table(results_1, "Circles", "XOR")
    print_continual_table(results_2, "XOR", "Circles")
    print_continual_table(results_3, "Linear", "Circles")

    # ─── Key summary ───
    print("=" * 72)
    print("  FORGETTING SUMMARY (averaged across all 3 experiments)")
    print("=" * 72)

    for model_name in ["Dynamic (2→?→1)", "Fixed Small (2→4→1)", "Fixed Large (2→16→1)"]:
        all_forgetting = (
            [r.forgetting for r in results_1[model_name]] +
            [r.forgetting for r in results_2[model_name]] +
            [r.forgetting for r in results_3[model_name]]
        )
        avg_forget = avg(all_forgetting)
        print(f"  {model_name:<28}: avg forgetting = {avg_forget:+.2%}")

    print()
