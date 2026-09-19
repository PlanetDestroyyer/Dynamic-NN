"""
Dynamic Neural Network — Sequential Continual Learning
=======================================================

Experiment: train on 3 tasks in order, evaluate ALL seen tasks after each.

Mechanisms tested:
  1. Horizontal scaling  — grow new neurons when |Δw| is high (interference detected)
  2. Replay buffer       — keep 30 samples per past task, mix them in every few epochs

Compared baselines:
  A. Fixed MLP (static, no growth, no replay)
  B. Dynamic (growth only — no replay)
  C. Dynamic + Replay    (both mechanisms)

Table printed after every task:

  After | T1 Acc | T2 Acc | T3 Acc | Neurons | Connections
  ─────────────────────────────────────────────────────────
  T1    |  100%  |   -    |   -    |    2    |      9
  T2    |   ??%  |  95%   |   -    |    ?    |      ?
  T3    |   ??%  |   ??%  |  99%  |    ?    |      ?

KISS. No EWC. No distillation. No attention.
"""

import numpy as np
from dataclasses import dataclass, field

from dynamic_nn import (
    DynamicNetwork, NeuronType,
    build_minimal_network, build_network_with_hidden,
    make_xor, make_circles, make_linearly_separable,
)


# ─────────────────────────────────────────────
# Replay Buffer
# ─────────────────────────────────────────────

class ReplayBuffer:
    """
    Stores a small random subset of samples from each past task.
    No fancy prioritisation — pure random selection. KISS.
    """

    def __init__(self, max_samples_per_task: int = 30):
        self.max_samples = max_samples_per_task
        self._store: dict[int, tuple[np.ndarray, np.ndarray]] = {}

    def add(self, task_id: int, X: np.ndarray, y: np.ndarray) -> None:
        n = min(self.max_samples, len(X))
        idx = np.random.choice(len(X), n, replace=False)
        self._store[task_id] = (X[idx], y[idx])

    def past_tasks(self, current_task_id: int):
        """Yield (X, y) for every task before current_task_id."""
        for tid, (X, y) in self._store.items():
            if tid < current_task_id:
                yield X, y

    def __len__(self):
        return len(self._store)


# ─────────────────────────────────────────────
# Training
# ─────────────────────────────────────────────

def train_one_task(
    net: DynamicNetwork,
    X: np.ndarray,
    y: np.ndarray,
    epochs: int,
    lr: float,
    *,
    dynamic: bool = True,
    replay_buffer: ReplayBuffer | None = None,
    task_id: int = 0,
    # Growth hyper-params
    grow_check_every: int = 50,          # check interference every N epochs
    interference_window: int = 20,       # |Δw| window for interference detection
    interference_threshold: float = 0.04,# |Δw| threshold to trigger growth
    loss_plateau_check_every: int = 100, # also keep the original plateau check
    loss_patience: int = 50,
    loss_min_improvement: float = 0.005,
    max_hidden: int = 20,
    max_connections_per_growth: int = 4,
    prune_every: int = 100,
    tau_prune: float = 0.01,
    # Replay hyper-params
    replay_every: int = 50,             # replay past tasks every N epochs
    replay_lr_factor: float = 0.3,      # smaller lr for replay (don't over-correct)
) -> None:
    """
    Train net on (X, y) for `epochs` epochs.
    Optionally: grow on interference, replay past tasks.

    Always resets loss_history and weight_delta_history so should_grow()
    and is_under_interference() get clean signals for the new task.
    """
    # ── Reset per-task histories for clean signals ──────────────────────────
    net.loss_history = []
    net.weight_delta_history = []

    n_samples = len(X)

    for epoch in range(epochs):

        # 1. Train one epoch on current task
        idx = np.random.permutation(n_samples)
        epoch_loss = 0.0
        for i in idx:
            net.forward(X[i])
            loss = net.backward(np.array([y[i]]))
            epoch_loss += loss
            net.update_weights(lr, use_stability=dynamic)
        avg_loss = epoch_loss / n_samples

        net.loss_history.append(avg_loss)
        net.neuron_count_history.append(len(net.get_hidden_neurons()))
        net.connection_count_history.append(len(net.connections))

        # 2. Replay past tasks (small random batches at a reduced lr)
        if replay_buffer and (epoch + 1) % replay_every == 0:
            for X_old, y_old in replay_buffer.past_tasks(task_id):
                old_idx = np.random.permutation(len(X_old))
                for i in old_idx:
                    net.forward(X_old[i])
                    net.backward(np.array([y_old[i]]))
                    net.update_weights(lr * replay_lr_factor, use_stability=dynamic)

        # 3. Dynamic growth
        if dynamic and len(net.get_hidden_neurons()) < max_hidden:

            # 3a. Horizontal scaling: interference detected → grow new neuron
            if (epoch + 1) % grow_check_every == 0:
                if net.is_under_interference(interference_window, interference_threshold):
                    net.grow_neuron(max_connections_per_growth)
                    net.weight_delta_history = []   # reset after growing (fresh start)

            # 3b. Original: grow if loss has plateaued on the current task
            if (epoch + 1) % loss_plateau_check_every == 0:
                if net.should_grow(loss_patience, loss_min_improvement):
                    net.grow_neuron(max_connections_per_growth)

        # 4. Prune weak connections occasionally
        if dynamic and (epoch + 1) % prune_every == 0:
            net.prune(tau_prune)


# ─────────────────────────────────────────────
# Sequential Experiment Runner
# ─────────────────────────────────────────────

@dataclass
class TaskSnapshot:
    """Accuracy snapshot across all tasks seen so far, captured after one task."""
    after_task: int                     # which task was just trained
    accuracies: dict[int, float]        # task_id → accuracy
    n_hidden: int
    n_connections: int


def run_sequential(
    model_tag: str,
    net: DynamicNetwork,
    tasks: list[tuple[str, np.ndarray, np.ndarray]],
    epochs: int = 800,
    lr: float = 0.5,
    dynamic: bool = True,
    use_replay: bool = True,
) -> list[TaskSnapshot]:
    """
    Train net sequentially on all tasks.
    After each task, evaluate accuracy on all tasks seen so far.

    Returns a list of TaskSnapshot (one per task).
    """
    replay_buf = ReplayBuffer(max_samples_per_task=30) if use_replay else None
    snapshots: list[TaskSnapshot] = []

    for task_id, (task_name, X, y) in enumerate(tasks):
        print(f"    [{model_tag}] Task {task_id+1} ({task_name})...", end="", flush=True)

        train_one_task(
            net, X, y, epochs, lr,
            dynamic=dynamic,
            replay_buffer=replay_buf,
            task_id=task_id,
        )

        # Add current task to replay buffer AFTER training (don't replay self)
        if replay_buf is not None:
            replay_buf.add(task_id, X, y)

        # Evaluate all tasks seen so far
        accs: dict[int, float] = {}
        for prev_id, (_, X_prev, y_prev) in enumerate(tasks[:task_id + 1]):
            accs[prev_id] = net.accuracy(X_prev, y_prev)

        snap = TaskSnapshot(
            after_task=task_id,
            accuracies=accs,
            n_hidden=len(net.get_hidden_neurons()),
            n_connections=len(net.connections),
        )
        snapshots.append(snap)
        print(
            f" acc={accs[task_id]:.1%} | "
            f"{snap.n_hidden}h {snap.n_connections}c"
        )

    return snapshots


# ─────────────────────────────────────────────
# Reporting
# ─────────────────────────────────────────────

def print_table(model_tag: str, snapshots: list[TaskSnapshot], task_names: list[str]) -> None:
    n_tasks = len(task_names)
    col_w = 10

    header_tasks = "".join(f"{'T'+str(i+1)+' ('+task_names[i][:4]+')':>{col_w}}" for i in range(n_tasks))
    print(f"\n  {model_tag}")
    print(f"  {'After':<8}" + header_tasks + f"  {'Neurons':>8}  {'Conn':>6}")
    print(f"  {'─'*8}" + '─'*col_w*n_tasks + f"  {'─'*8}  {'─'*6}")

    for snap in snapshots:
        row = f"  {'T'+str(snap.after_task+1):<8}"
        for i in range(n_tasks):
            if i in snap.accuracies:
                row += f"{snap.accuracies[i]:>{col_w}.1%}"
            else:
                row += f"{'—':>{col_w}}"
        row += f"  {snap.n_hidden:>8}  {snap.n_connections:>6}"
        print(row)


def print_forgetting(model_tag: str, snapshots: list[TaskSnapshot], task_names: list[str]) -> None:
    """
    Forgetting for task i = accuracy right after training task i
                           - accuracy after all subsequent tasks
    """
    print(f"\n  Forgetting ({model_tag}):")
    n_tasks = len(task_names)

    for task_id in range(n_tasks - 1):  # last task has no future tasks to forget it
        # accuracy right after learning this task
        acc_right_after = snapshots[task_id].accuracies[task_id]
        # accuracy after the final task
        acc_at_end = snapshots[-1].accuracies.get(task_id)
        if acc_at_end is not None:
            forgetting = acc_at_end - acc_right_after
            bar = "█" * int(abs(forgetting) * 20) if forgetting < 0 else ""
            print(f"    T{task_id+1} ({task_names[task_id]:<8}): "
                  f"{acc_right_after:.1%} → {acc_at_end:.1%}  "
                  f"[{forgetting:+.1%}] {bar}")


# ─────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────

if __name__ == "__main__":
    EPOCHS = 800
    LR = 0.5
    N_SAMPLES = 300
    SEED = 42
    np.random.seed(SEED)

    # ── Three diverse classification tasks ─────────────────────────────────
    tasks = [
        ("Circles",  *make_circles(n_samples=N_SAMPLES, noise=0.05, seed=SEED)),
        ("XOR",      *make_xor(n_samples=N_SAMPLES, noise=0.0, seed=SEED)),
        ("Linear",   *make_linearly_separable(n_samples=N_SAMPLES, noise=0.1, seed=SEED)),
    ]
    task_names = [t[0] for t in tasks]

    print("=" * 72)
    print("  Sequential Continual Learning  —  3 Tasks")
    print("  Mechanisms: Horizontal Scaling (|Δw| interference) + Replay Buffer")
    print(f"  Protocol: T1 → T2 → T3  |  {EPOCHS} epochs/task")
    print("=" * 72)

    # ── Run three configurations ────────────────────────────────────────────
    configs = [
        # (label,             net_factory,                  dynamic, replay)
        ("Fixed MLP (2→8→1)", lambda: build_network_with_hidden(8), False, False),
        ("Dynamic (no replay)",build_minimal_network,               True,  False),
        ("Dynamic + Replay",   build_minimal_network,               True,  True),
    ]

    all_results: dict[str, list[TaskSnapshot]] = {}

    for label, net_factory, dynamic, use_replay in configs:
        print(f"\n{'─'*72}")
        print(f"  {label}")
        print(f"{'─'*72}")
        np.random.seed(SEED)   # same seed for fair comparison
        net = net_factory()
        all_results[label] = run_sequential(
            label, net, tasks,
            epochs=EPOCHS, lr=LR,
            dynamic=dynamic,
            use_replay=use_replay,
        )

    # ── Print results ───────────────────────────────────────────────────────
    print("\n\n" + "=" * 72)
    print("  RESULTS")
    print("=" * 72)

    for label, snapshots in all_results.items():
        print_table(label, snapshots, task_names)

    print("\n\n" + "=" * 72)
    print("  FORGETTING ANALYSIS")
    print("=" * 72)

    for label, snapshots in all_results.items():
        print_forgetting(label, snapshots, task_names)

    print()
