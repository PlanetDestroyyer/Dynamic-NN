# Structural Plasticity via Memory Aware Synapses (MAS)

This repository implements a biologically-inspired neural architecture designed to mitigate **Catastrophic Forgetting** in Artificial Neural Networks during sequential task learning. 

The approach leverages dynamic structural plasticity—allocating new neurons dynamically when capacity constraints are detected—and explicit gradient masking to freeze consolidated knowledge representations.

## Methodology

### 1. Capacity Detection (MAS Signal)
The architecture employs a Memory Aware Synapses (MAS) gradient-tracking mechanism to autonomously detect task boundaries or distribution shifts. By monitoring the gradient norm of the output magnitude with respect to the hidden weights:
```python
grads = torch.autograd.grad(out_mag, network.hidden_W)[0]
```
The network maintains exponential moving averages (EMA) of this gradient magnitude. A significant deviation indicates that the network is struggling to map novel inputs into the existing parameter space, triggering structural expansion.

### 2. Structural Plasticity & Gradient Masking
When a capacity constraint is detected, the network executes a "Sprout and Freeze" operation:
*   **Expansion:** A predetermined number of new neurons are instantiated and initialized.
*   **Freezing:** The gradients of all pre-existing synapses are explicitly zeroed out (`zero_frozen_grads()`) during the backward pass. This mathematically guarantees that historical representations are immutable and immune to catastrophic weight overwriting.

### 3. Mitigating Forward Interference
To prevent newly instantiated neurons from aggressively suppressing the logits of previous tasks, the architecture applies an active-class mask to the output layer prior to loss calculation. By setting inactive class logits to negative infinity during the forward pass, the Softmax cross-entropy loss generates zero gradients for historical classes. This prevents the optimizer from driving new connections to extreme negative values, thereby preserving the integrity of frozen representations.

## Benchmark Results

The architecture was evaluated on the **Split-MNIST** continual learning benchmark (5 sequential binary classification tasks). The model's retention was compared against a standard Multi-Layer Perceptron (MLP) trained sequentially without rehearsal or regularization.

| Architecture | Task 1 (0/1) | Task 2 (2/3) | Task 3 (4/5) | Task 4 (6/7) | Task 5 (8/9) |
| :--- | :---: | :---: | :---: | :---: | :---: |
| Standard Baseline MLP | 52.5% | 49.3% | 53.0% | 48.7% | 96.6% |
| **Dynamic Sprout & Freeze** | **99.8%** | **97.2%** | **95.9%** | **99.4%** | **97.8%** |

**Results on Split-FashionMNIST (Hard Mode):**

| Architecture | Task 1 | Task 2 | Task 3 | Task 4 | Task 5 |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **Dynamic Sprout & Freeze** | **81.6%** | **90.2%** | **99.4%** | **100.0%** | **99.6%** |

## Repository Structure

*   `docs/theory.md`: A comprehensive breakdown of the biological hypothesis and the mathematical implementation of the architecture.
*   `main.py`: A standalone executable Python script containing the core architecture and training loop for easy reference.
*   `notebooks/split_mnist_benchmark.ipynb`: A self-contained Jupyter Notebook implementing the architectures, training loops, evaluation metrics, and comparative visualizations for both Split-MNIST and Split-FashionMNIST.

## Execution

The provided Jupyter Notebook can be executed in any standard environment (Google Colab, Jupyter Lab, VSCode). It autonomously downloads the necessary datasets, trains the models sequentially, and outputs performance trajectories illustrating the mitigation of catastrophic forgetting.
