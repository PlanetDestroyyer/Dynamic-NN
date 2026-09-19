"""
Dynamic Neural Network — Phase 4: Visualization

Visualize the network growing as it learns:
- Left: Network graph (nodes + edges, thickness = weight magnitude)
- Right: Decision boundary on the 2D data
- Bottom: Loss curve + neuron/connection count over time

Uses matplotlib. Generates both static snapshots and animated training.
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.collections import LineCollection
import os

from dynamic_nn import (
    DynamicNetwork, NeuronType, build_minimal_network,
    make_xor, make_circles, make_linearly_separable,
    train_dynamic,
)


# ─────────────────────────────────────────────
# Network Graph Visualization
# ─────────────────────────────────────────────

def plot_network(ax, net: DynamicNetwork, title: str = "Network Topology"):
    """
    Draw the network as a graph.

    - Input neurons: green (left)
    - Hidden neurons: blue (middle, arranged by depth)
    - Output neurons: red (right)
    - Edge thickness = weight magnitude
    - Edge color: blue for positive, red for negative
    """
    ax.clear()
    ax.set_title(title, fontsize=12, fontweight='bold')
    ax.set_xlim(-0.5, 3.5)
    ax.set_ylim(-2, 2)
    ax.set_aspect('equal')
    ax.axis('off')

    # Assign positions to neurons
    positions = {}

    input_neurons = net.get_input_neurons()
    hidden_neurons = net.get_hidden_neurons()
    output_neurons = net.get_output_neurons()

    # Input neurons: x=0
    for i, n in enumerate(input_neurons):
        y = (i - (len(input_neurons) - 1) / 2) * 0.8
        positions[n.id] = (0.0, y)

    # Hidden neurons: spread across x=1 to x=2
    if hidden_neurons:
        n_hidden = len(hidden_neurons)
        for i, n in enumerate(hidden_neurons):
            x = 1.0 + (i % 3) * 0.5  # spread horizontally
            y = (i - (n_hidden - 1) / 2) * 0.6
            positions[n.id] = (x, y)

    # Output neurons: x=3
    for i, n in enumerate(output_neurons):
        y = (i - (len(output_neurons) - 1) / 2) * 0.8
        positions[n.id] = (3.0, y)

    # Draw connections (edges)
    for conn in net.connections.values():
        if conn.from_id not in positions or conn.to_id not in positions:
            continue
        x1, y1 = positions[conn.from_id]
        x2, y2 = positions[conn.to_id]
        weight = conn.weight

        # Color: blue for positive, red for negative
        color = '#2196F3' if weight > 0 else '#F44336'
        # Thickness: proportional to magnitude (clamped)
        thickness = min(abs(weight) * 0.15, 4.0)
        alpha = min(0.3 + abs(weight) * 0.05, 0.9)

        ax.plot([x1, x2], [y1, y2], color=color, linewidth=max(thickness, 0.3),
                alpha=alpha, zorder=1)

    # Draw neurons (nodes)
    colors = {
        NeuronType.INPUT: '#4CAF50',   # green
        NeuronType.HIDDEN: '#2196F3',  # blue
        NeuronType.OUTPUT: '#FF5722',  # red/orange
    }

    for nid, pos in positions.items():
        neuron = net.neurons[nid]
        color = colors[neuron.neuron_type]
        circle = plt.Circle(pos, 0.15, color=color, ec='white', linewidth=2, zorder=2)
        ax.add_patch(circle)
        ax.text(pos[0], pos[1], f'n{nid}', ha='center', va='center',
                fontsize=7, fontweight='bold', color='white', zorder=3)

    # Legend
    legend_elements = [
        mpatches.Patch(color='#4CAF50', label=f'Input ({len(input_neurons)})'),
        mpatches.Patch(color='#2196F3', label=f'Hidden ({len(hidden_neurons)})'),
        mpatches.Patch(color='#FF5722', label=f'Output ({len(output_neurons)})'),
    ]
    ax.legend(handles=legend_elements, loc='lower left', fontsize=7, framealpha=0.8)

    # Stats text
    ax.text(3.3, -1.8, f'Connections: {len(net.connections)}',
            fontsize=8, ha='right', style='italic', color='#666')


# ─────────────────────────────────────────────
# Decision Boundary Visualization
# ─────────────────────────────────────────────

def plot_decision_boundary(ax, net: DynamicNetwork, X: np.ndarray, y: np.ndarray,
                           title: str = "Decision Boundary", resolution: int = 100):
    """
    Plot the decision boundary of the network on 2D data.
    """
    ax.clear()
    ax.set_title(title, fontsize=12, fontweight='bold')

    # Create a mesh grid
    x_min, x_max = X[:, 0].min() - 0.5, X[:, 0].max() + 0.5
    y_min, y_max = X[:, 1].min() - 0.5, X[:, 1].max() + 0.5
    xx, yy = np.meshgrid(
        np.linspace(x_min, x_max, resolution),
        np.linspace(y_min, y_max, resolution)
    )

    # Predict on mesh
    grid_points = np.column_stack([xx.ravel(), yy.ravel()])
    Z = []
    for point in grid_points:
        out = net.forward(point)
        Z.append(out[0])
    Z = np.array(Z).reshape(xx.shape)

    # Plot decision boundary as contour
    ax.contourf(xx, yy, Z, levels=50, cmap='RdYlBu', alpha=0.7)
    ax.contour(xx, yy, Z, levels=[0.5], colors='black', linewidths=2, linestyles='--')

    # Plot data points
    class_0 = y == 0
    class_1 = y == 1
    ax.scatter(X[class_0, 0], X[class_0, 1], c='#F44336', edgecolors='white',
               s=30, linewidths=0.5, label='Class 0', zorder=2)
    ax.scatter(X[class_1, 0], X[class_1, 1], c='#2196F3', edgecolors='white',
               s=30, linewidths=0.5, label='Class 1', zorder=2)

    ax.legend(fontsize=8, loc='upper left')
    ax.set_xlabel('x₁', fontsize=10)
    ax.set_ylabel('x₂', fontsize=10)


# ─────────────────────────────────────────────
# Training History Visualization
# ─────────────────────────────────────────────

def plot_training_history(axes, net: DynamicNetwork):
    """
    Plot loss curve and neuron/connection count over time.
    axes should be a list of 2 axes [ax_loss, ax_counts].
    """
    ax_loss, ax_counts = axes

    # Loss curve
    ax_loss.clear()
    ax_loss.set_title('Training Loss', fontsize=11, fontweight='bold')
    if net.loss_history:
        ax_loss.plot(net.loss_history, color='#FF5722', linewidth=1.5, alpha=0.8)
        ax_loss.set_xlabel('Epoch', fontsize=9)
        ax_loss.set_ylabel('Loss', fontsize=9)
        ax_loss.set_yscale('log')
        ax_loss.grid(True, alpha=0.3)

    # Neuron and connection counts
    ax_counts.clear()
    ax_counts.set_title('Network Complexity', fontsize=11, fontweight='bold')
    if net.neuron_count_history:
        epochs = range(len(net.neuron_count_history))
        ax_counts.plot(epochs, net.neuron_count_history, color='#2196F3',
                       linewidth=2, label='Hidden neurons', marker='', markersize=3)
        ax_counts.set_xlabel('Epoch', fontsize=9)
        ax_counts.set_ylabel('Hidden Neurons', fontsize=9, color='#2196F3')
        ax_counts.tick_params(axis='y', labelcolor='#2196F3')

        # Second y-axis for connections
        ax_conn = ax_counts.twinx()
        ax_conn.plot(epochs, net.connection_count_history, color='#4CAF50',
                     linewidth=2, label='Connections', linestyle='--')
        ax_conn.set_ylabel('Connections', fontsize=9, color='#4CAF50')
        ax_conn.tick_params(axis='y', labelcolor='#4CAF50')

        ax_counts.grid(True, alpha=0.3)


# ─────────────────────────────────────────────
# Full Dashboard
# ─────────────────────────────────────────────

def plot_full_dashboard(net: DynamicNetwork, X: np.ndarray, y: np.ndarray,
                        title: str = "Dynamic Neural Network", save_path: str = None):
    """
    Create a full 4-panel dashboard:
    - Top-left: Network topology
    - Top-right: Decision boundary
    - Bottom-left: Loss curve
    - Bottom-right: Network complexity over time
    """
    fig = plt.figure(figsize=(14, 10))
    fig.suptitle(title, fontsize=16, fontweight='bold', y=0.98)

    ax_net = fig.add_subplot(2, 2, 1)
    ax_boundary = fig.add_subplot(2, 2, 2)
    ax_loss = fig.add_subplot(2, 2, 3)
    ax_counts = fig.add_subplot(2, 2, 4)

    plot_network(ax_net, net, "Network Topology")
    plot_decision_boundary(ax_boundary, net, X, y, "Decision Boundary")
    plot_training_history([ax_loss, ax_counts], net)

    # Add accuracy annotation
    acc = net.accuracy(X, y)
    fig.text(0.5, 0.01, f'Final Accuracy: {acc:.1%} | '
             f'Hidden Neurons: {len(net.get_hidden_neurons())} | '
             f'Connections: {len(net.connections)}',
             ha='center', fontsize=12, style='italic',
             bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

    plt.tight_layout(rect=[0, 0.03, 1, 0.95])

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved to {save_path}")

    return fig


# ─────────────────────────────────────────────
# Training with Snapshots
# ─────────────────────────────────────────────

def train_dynamic_with_snapshots(
    net: DynamicNetwork,
    X: np.ndarray,
    y: np.ndarray,
    epochs: int = 2000,
    lr: float = 0.5,
    snapshot_every: int = 200,
    save_dir: str = "snapshots",
    **train_kwargs,
) -> DynamicNetwork:
    """
    Train with dynamic grow/prune and save dashboard snapshots at intervals.
    Returns the trained network.
    """
    os.makedirs(save_dir, exist_ok=True)

    n_samples = len(X)

    # Training params with defaults
    prune_every = train_kwargs.get('prune_every', 100)
    tau_prune = train_kwargs.get('tau_prune', 0.01)
    grow_check_every = train_kwargs.get('grow_check_every', 100)
    grow_patience = train_kwargs.get('grow_patience', 50)
    grow_min_improvement = train_kwargs.get('grow_min_improvement', 0.005)
    max_hidden_neurons = train_kwargs.get('max_hidden_neurons', 8)
    max_connections_per_growth = train_kwargs.get('max_connections_per_growth', 4)
    lambda_neurons = train_kwargs.get('lambda_neurons', 0.001)
    lambda_connections = train_kwargs.get('lambda_connections', 0.0005)

    # Save initial snapshot
    fig = plot_full_dashboard(net, X, y, f"Epoch 0 (initial)")
    plt.savefig(os.path.join(save_dir, f"snapshot_0000.png"), dpi=120, bbox_inches='tight')
    plt.close(fig)

    for epoch in range(epochs):
        idx = np.random.permutation(n_samples)
        epoch_loss = 0.0

        for i in idx:
            output = net.forward(X[i])
            loss = net.backward(np.array([y[i]]))
            epoch_loss += loss
            net.update_weights(lr)

        avg_loss = epoch_loss / n_samples
        n_hidden = len(net.get_hidden_neurons())
        n_conn = len(net.connections)

        net.loss_history.append(avg_loss)
        net.neuron_count_history.append(n_hidden)
        net.connection_count_history.append(n_conn)

        # Pruning
        if (epoch + 1) % prune_every == 0 and epoch > 0:
            conn_rm, neuron_rm = net.prune(tau_prune)
            if conn_rm > 0 or neuron_rm > 0:
                print(f"  [PRUNE] Epoch {epoch+1}: -{conn_rm} conn, -{neuron_rm} neurons")

        # Growing
        if (epoch + 1) % grow_check_every == 0:
            current_hidden = len(net.get_hidden_neurons())
            if current_hidden < max_hidden_neurons and net.should_grow(grow_patience, grow_min_improvement):
                new_id = net.grow_neuron(max_connections_per_growth)
                print(f"  [GROW]  Epoch {epoch+1}: +neuron n{new_id} "
                      f"({len(net.get_hidden_neurons())} hidden, {len(net.connections)} conn)")

        # Snapshot
        if (epoch + 1) % snapshot_every == 0:
            acc = net.accuracy(X, y)
            print(f"Epoch {epoch+1:4d} | Loss: {avg_loss:.4f} | Acc: {acc:.2%} | "
                  f"Hidden: {n_hidden} | Conn: {n_conn}")

            fig = plot_full_dashboard(
                net, X, y,
                f"Epoch {epoch+1} — Acc: {acc:.1%}"
            )
            plt.savefig(
                os.path.join(save_dir, f"snapshot_{epoch+1:04d}.png"),
                dpi=120, bbox_inches='tight'
            )
            plt.close(fig)

    return net


# ─────────────────────────────────────────────
# Main — Generate visualizations
# ─────────────────────────────────────────────

if __name__ == "__main__":
    np.random.seed(42)

    output_dir = os.path.join(os.path.dirname(__file__), "output")
    os.makedirs(output_dir, exist_ok=True)

    # ═══════════════════════════════════════
    # Experiment 1: Circles (non-linear)
    # ═══════════════════════════════════════
    print("=" * 60)
    print("Visualization: Circles dataset (dynamic growing)")
    print("=" * 60)

    X_circ, y_circ = make_circles(n_samples=300, noise=0.05)

    net_circ = build_minimal_network()
    print(f"Starting with: {net_circ.summary()}\n")

    snap_dir = os.path.join(output_dir, "circles_snapshots")
    net_circ = train_dynamic_with_snapshots(
        net_circ, X_circ, y_circ,
        epochs=1500,
        lr=0.5,
        snapshot_every=100,
        save_dir=snap_dir,
        grow_check_every=100,
        grow_patience=50,
        grow_min_improvement=0.005,
        max_hidden_neurons=6,
    )

    # Final dashboard
    fig = plot_full_dashboard(
        net_circ, X_circ, y_circ,
        "Dynamic Network — Circles (Final)",
        save_path=os.path.join(output_dir, "circles_final.png")
    )
    plt.close(fig)

    # ═══════════════════════════════════════
    # Experiment 2: XOR (non-linear)
    # ═══════════════════════════════════════
    print("\n" + "=" * 60)
    print("Visualization: XOR dataset (dynamic growing)")
    print("=" * 60)

    X_xor, y_xor = make_xor(n_samples=300, noise=0.0)

    net_xor = build_minimal_network()
    print(f"Starting with: {net_xor.summary()}\n")

    snap_dir = os.path.join(output_dir, "xor_snapshots")
    net_xor = train_dynamic_with_snapshots(
        net_xor, X_xor, y_xor,
        epochs=1500,
        lr=0.5,
        snapshot_every=100,
        save_dir=snap_dir,
        grow_check_every=100,
        grow_patience=50,
        grow_min_improvement=0.005,
        max_hidden_neurons=6,
    )

    fig = plot_full_dashboard(
        net_xor, X_xor, y_xor,
        "Dynamic Network — XOR (Final)",
        save_path=os.path.join(output_dir, "xor_final.png")
    )
    plt.close(fig)

    print("\n✅ All visualizations saved to:", output_dir)
    print("   Check the snapshots folders to see the network evolve!")
