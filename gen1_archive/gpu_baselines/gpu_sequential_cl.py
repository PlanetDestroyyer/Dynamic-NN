import torch
import torch.nn as nn
import numpy as np
from gpu_dynamic_nn import PyTorchDynamicNetwork

from dynamic_nn import make_xor, make_circles, make_linearly_separable


# ─────────────────────────────────────────────────────────────────────────────
class ReplayBuffer:
    """
    Flat reservoir-sampled episodic memory.

    Purpose: provide a stream of MIXED past+current data to the training loss.
    This is NOT used for forgetting detection — that job belongs to the
    network's own gradient stress signal.

    Reservoir sampling ensures every past sample has equal probability of
    being retained regardless of when it arrived.
    """

    def __init__(self, capacity: int = 500):
        self.capacity = capacity
        self.X: torch.Tensor | None = None
        self.y: torch.Tensor | None = None
        self.n_seen = 0

    def push(self, X: torch.Tensor, y: torch.Tensor):
        X, y = X.detach().cpu(), y.detach().cpu()
        for xi, yi in zip(X, y):
            xi, yi = xi.unsqueeze(0), yi.unsqueeze(0)
            if self.n_seen < self.capacity:
                self.X = xi if self.X is None else torch.cat([self.X, xi], 0)
                self.y = yi if self.y is None else torch.cat([self.y, yi], 0)
            else:
                j = np.random.randint(0, self.n_seen + 1)
                if j < self.capacity:
                    self.X[j] = xi[0]
                    self.y[j] = yi[0]
            self.n_seen += 1

    def sample(self, n: int, device):
        if self.X is None or len(self.X) == 0:
            return None, None
        n = min(n, len(self.X))
        idx = torch.randperm(len(self.X))[:n]
        return self.X[idx].to(device), self.y[idx].to(device)

    def __len__(self):
        return len(self.X) if self.X is not None else 0


# ─────────────────────────────────────────────────────────────────────────────
def _reset_adam_for_neuron(optimizer, net, ni: int):
    """Zero out Adam momentum/variance for a freshly grown neuron."""
    if net.W not in optimizer.state:
        return
    s_W = optimizer.state[net.W]
    for key in ('exp_avg', 'exp_avg_sq'):
        if key in s_W:
            s_W[key][ni, :] = 0.0
            s_W[key][:, ni] = 0.0
    if net.b in optimizer.state:
        s_b = optimizer.state[net.b]
        for key in ('exp_avg', 'exp_avg_sq'):
            if key in s_b:
                s_b[key][ni] = 0.0


# ─────────────────────────────────────────────────────────────────────────────
def train_one_task_gpu(
    net: PyTorchDynamicNetwork,
    X: torch.Tensor,
    y: torch.Tensor,
    epochs: int,
    lr: float,
    *,
    replay: ReplayBuffer = None,
    device='cpu'
):
    """
    Training loop driven by the network's own internal stress signal.

    ╔══════════════════════════════════════════════════════════════════╗
    ║  Core principle (no cheating, no task labels)                   ║
    ║                                                                  ║
    ║  Each weight tracks explicit gradient covariance:                ║
    ║    g_task   = gradient from current task's batch                ║
    ║    g_replay = gradient from replay memory batch                 ║
    ║                                                                  ║
    ║    stress   = EMA(-g_task * g_replay) / EMA(|g_task| * |g_replay|) 
    ║                                                                  ║
    ║  Interpretation:                                                 ║
    ║    stress < 0  →  Task and Replay agree (single task learning)  ║
    ║    stress ≈ 0  →  Independent noise (converged at minimum)      ║
    ║    stress > 0  →  True conflict (new task destroying old memory)║
    ║                                                                  ║
    ║  get_network_stress() returns the fraction of connections that  ║
    ║  have a stress > 0.3.                                           ║
    ║                                                                  ║
    ║  When conflict crosses the threshold → weights "feel" it and    ║
    ║  signal: "I can't hold both things — give me space."            ║
    ║    → stress_freeze(): lock the highly-conflicted connections     ║
    ║    → grow_neuron():   add fresh neurons for the new learning     ║
    ║                                                                  ║
    ║  Why this works                                                  ║
    ║    Task 1 only (no replay conflict):                             ║
    ║      Early: gradients point towards minimum. stress < 0.        ║
    ║      Late: gradients are noise. noise averages to 0. stress ≈ 0.║
    ║      → conflict never rises → no freeze fires.                  ║
    ║                                                                  ║
    ║    Task 2 starts (XOR fights Circles in replay):                 ║
    ║      Circles replay: pushes weight +                             ║
    ║      XOR forward:    pushes weight −                             ║
    ║      g_task * g_replay is consistently negative.                 ║
    ║      → stress → +1.0 → FREEZE + GROW                            ║
    ║                                                                  ║
    ║    After freeze+grow:                                            ║
    ║      Old connections: frozen, gradient = 0 → stress decays to 0 ║
    ║      New neurons: learn XOR freely, consistent direction         ║
    ║      → conflict falls → network stable again                    ║
    ╚══════════════════════════════════════════════════════════════════╝

    Replay is used for JOINT TRAINING only (mixing past data into the
    current loss) — NOT for forgetting detection.  The stress signal
    is what creates the conflict: Circles replay data creates opposing
    gradients when XOR is being learned, naturally raising stress on
    the shared weights.
    """
    optimizer = torch.optim.Adam(net.parameters(), lr=lr)
    criterion = nn.MSELoss()

    net.to(device)
    X, y = X.to(device), y.to(device)

    loss_ema = 0.25

    # ── Stress monitoring ─────────────────────────────────────────────────────
    # stress_ema: smoothed network-level conflict fraction (in [0, 1])
    # Starts at 0 (no conflict assumed at start of each call).
    stress_ema          = 0.0
    stress_ema_beta     = 0.9       # smooth over ~10 checks = ~100 batches
    stress_check_every  = 10        # batches between stress evaluations

    # Conflict threshold: importance-weighted conflict fraction above which
    # the network is considered to be "under stress from a new task."
    #
    # Task 1 alone (no replay):   conflict ≈ 0.00–0.15  (consistent gradients)
    # Task 2 (Circles vs XOR):    conflict → 0.40–0.90  (opposing gradients)
    #
    # Threshold at 0.3 sits well between these two regimes.
    # Increase if too many false freezes during Task 1.
    # Decrease if freeze fires too late into Task 2.
    conflict_threshold  = 0.3

    # Burn-in: don't allow any freeze in the first N batches.
    # Gives the network time to start learning (gradients are chaotic
    # at initialization regardless of task conflict).
    freeze_burnin       = 300       # ≈ 30 epochs with batch_size=32, n=300

    # Cooldown: minimum batches between successive freeze events.
    # Prevents rapid-fire freezing before new neurons have had time
    # to relieve the stress.
    freeze_cooldown     = 500       # ≈ 50 epochs
    batches_since_freeze = freeze_cooldown   # start ready
    neurons_per_freeze   = 3

    # ── Task-1 conservative growth ────────────────────────────────────────────
    # During Task 1 there is no replay conflict, so stress stays low and the
    # conflict-based freeze never fires.  Instead, we grow when the task loss
    # is still high (network genuinely doesn't have enough capacity).
    task1_grow_cooldown   = 200
    batches_since_t1_grow = task1_grow_cooldown
    task1_loss_threshold  = 0.15
    task1_max_neurons     = 20      # hard cap: prevents runaway growth on easy tasks

    total_batches = 0

    for epoch in range(epochs):
        net.train()
        perm = torch.randperm(X.size(0))

        for i in range(0, X.size(0), 32):
            idx     = perm[i:i + 32]
            batch_x = X[idx]
            batch_y = y[idx]

            optimizer.zero_grad()

            # ── 1. Task Gradient ──────────────────────────────────────────────
            optimizer.zero_grad()
            task_loss = criterion(net(batch_x), batch_y)
            task_loss.backward(retain_graph=True)
            g_task = net.W.grad.clone() if net.W.grad is not None else torch.zeros_like(net.W)

            # ── 2. Replay Gradient (creates the conflict signal) ──────────────
            optimizer.zero_grad()
            replay_val = torch.tensor(0.0, device=device)
            g_replay = torch.zeros_like(net.W)
            if replay is not None and len(replay) >= 16:
                rx, ry = replay.sample(16, device)
                if rx is not None:
                    replay_val = criterion(net(rx), ry)
                    replay_val.backward(retain_graph=True)
                    g_replay = net.W.grad.clone() if net.W.grad is not None else torch.zeros_like(net.W)

            # ── 3. Update Stress Signal ───────────────────────────────────────
            if replay is not None and len(replay) >= 16:
                net.update_stress(g_task, g_replay)

            # ── 4. Actual Optimizer Step (Combined) ───────────────────────────
            optimizer.zero_grad()
            combined_loss = task_loss + replay_val
            combined_loss.backward()
            optimizer.step()

            # Push batch to replay (AFTER step, so this batch's data is
            # available for future tasks — not this batch's training)
            if replay is not None:
                replay.push(batch_x, batch_y)

            loss_ema = 0.99 * loss_ema + 0.01 * task_loss.item()
            batches_since_freeze  += 1
            batches_since_t1_grow += 1

            # ── Stress check ──────────────────────────────────────────────────
            if total_batches % stress_check_every == 0:
                s = net.get_network_stress()
                stress_ema = stress_ema_beta * stress_ema + (1 - stress_ema_beta) * s

                # ── Conflict-triggered FREEZE + GROW ─────────────────────────
                if (total_batches >= freeze_burnin and
                        batches_since_freeze >= freeze_cooldown and
                        stress_ema > conflict_threshold):

                    n_frozen = net.stress_freeze(threshold=0.3)

                    if n_frozen > 0:
                        for _ in range(neurons_per_freeze):
                            net.grow_neuron(num_connections=3)

                        # Reset stress buffers for new neurons
                        with torch.no_grad():
                            new_trainable = net.trainable_mask * net.M
                            net.stress_num *= (1.0 - new_trainable)
                            net.stress_den *= (1.0 - new_trainable)

                        # Reset Adam state for new neurons
                        for ni in range(net.active_neurons - neurons_per_freeze,
                                        net.active_neurons):
                            _reset_adam_for_neuron(optimizer, net, ni)

                        batches_since_freeze = 0
                        # Snap stress_ema down: after freeze, conflict should
                        # reduce — don't let stale high value trigger another
                        # freeze immediately after cooldown expires.
                        stress_ema = 0.0

                        print(f"  [Epoch {epoch+1:4d}] STRESS-FREEZE: "
                              f"{n_frozen} conns frozen, grew {neurons_per_freeze} → "
                              f"active={net.active_neurons} "
                              f"(trainable={net.n_trainable()}, frozen={net.n_frozen()}) "
                              f"conflict={s:.3f}")

                # ── Loss-based growth (when there is no conflict) ─────────────
                # If loss is high, but stress is low, the network doesn't have 
                # enough capacity for the new task, but the new task isn't 
                # fighting the old ones (orthogonal). So just grow!
                elif (net.active_neurons < task1_max_neurons and
                      loss_ema > task1_loss_threshold and
                      batches_since_t1_grow >= task1_grow_cooldown and
                      stress_ema <= conflict_threshold):

                    net.grow_neuron(num_connections=3)
                    ni = net.active_neurons - 1
                    _reset_adam_for_neuron(optimizer, net, ni)
                    batches_since_t1_grow = 0
                    print(f"  [Epoch {epoch+1:4d}] GROW(Loss): "
                          f"active={net.active_neurons} loss={loss_ema:.4f}")

            total_batches += 1

    return net


# ─────────────────────────────────────────────────────────────────────────────
def evaluate(net, tasks, seen_up_to, device):
    net.eval()
    with torch.no_grad():
        for name, X_np, y_np in tasks[:seen_up_to + 1]:
            X_t = torch.tensor(X_np, dtype=torch.float32).to(device)
            y_t = torch.tensor(y_np, dtype=torch.float32).view(-1, 1).to(device)
            acc = ((net(X_t) > 0.5).float() == y_t).float().mean().item()
            print(f"  Accuracy on {name}: {acc * 100:.1f}%")


# ─────────────────────────────────────────────────────────────────────────────
def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Running on {device}...")

    EPOCHS    = 1000
    LR        = 3e-3
    N_SAMPLES = 300
    SEED      = 42
    np.random.seed(SEED)
    torch.manual_seed(SEED)

    tasks = [
        ("Circles", *make_circles(n_samples=N_SAMPLES, noise=0.05, seed=SEED, center=(0, 0))),
        ("XOR",     *make_xor(n_samples=N_SAMPLES, noise=0.0,  seed=SEED, center=(4, 4))),
        ("Linear",  *make_linearly_separable(n_samples=N_SAMPLES, noise=0.1, seed=SEED, center=(-4, -4))),
    ]

    net    = PyTorchDynamicNetwork(input_dim=2, output_dim=1, max_neurons=200)
    replay = ReplayBuffer(capacity=500)

    for task_id, (name, X_np, y_np) in enumerate(tasks):
        print(f"\n--- Training Task {task_id + 1}: {name} ---")
        X = torch.tensor(X_np, dtype=torch.float32)
        y = torch.tensor(y_np, dtype=torch.float32).view(-1, 1)

        train_one_task_gpu(net, X, y, epochs=EPOCHS, lr=LR,
                           replay=replay, device=device)
        evaluate(net, tasks, task_id, device)
        print(f"  Active Neurons: {net.active_neurons} "
              f"(trainable={net.n_trainable()}, frozen={net.n_frozen()}) "
              f"stress={net.get_network_stress():.3f}")

    return net, tasks, device


if __name__ == "__main__":
    main()
