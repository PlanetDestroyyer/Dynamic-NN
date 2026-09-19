# Self-Aware Dynamic Neural Network (Generation 1)

This repository contains a PyTorch-accelerated Continuous Learning (CL) architecture that actively prevents **Catastrophic Forgetting** without requiring task labels or explicit boundaries. 

Instead of relying on static architectures with weight penalties (like EWC), this network is structurally plastic: it monitors the gradient covariance between incoming tasks and its own replay buffer to detect "stress" (memory overwrites). When high stress is detected, the network autonomously **freezes** threatened pathways and **grows** new neurons to expand its capacity.

## Core Mechanisms
- **Gradient Covariance Stress:** The network computes an Exponential Moving Average (EMA) of `-(g_task * g_replay)` to measure true conflict.
- **Stress-Freeze:** If the global stress metric exceeds 30%, the network physically sets the gradients of high-stress connections to exactly `0.0`.
- **Autonomous Growth:** To compensate for frozen capacity, the network spawns new neurons (with random initialization) to create parallel pathways for the new task.

## Benchmarks & Findings
This architecture was put through three rigorous benchmarks to test its limits:

### 1. 2D Toy Benchmark (Circles, XOR, Linear)
- **Result:** Massive Success (100% memory retention).
- **Finding:** The network perfectly solved all 6 permutations of the 3 tasks, cleanly partitioning the 2D space without forgetting previously learned boundaries.

### 2. High-Dimensional Scaling (Split-MNIST)
- **Result:** Limitation Discovered.
- **Finding:** Scaling to 784 dimensions diluted the global stress metric. Because pixels are sparse, the global percentage of conflicted connections never reached the 30% threshold, preventing the network from freezing. This empirically proved the need for **Localized (Per-Neuron) Stress Metrics** in high-dimensional spaces.

### 3. Feature Extractor Hybrid (CNN + Dynamic Net)
- **Result:** The "Dead ReLU" Collapse.
- **Finding:** Jointly training a CNN alongside a dynamically growing backend destroyed the CNN due to chaotic structural gradients. Pre-training and freezing the CNN stabilized the features (yielding 95%+ accuracy on Task 1), but highlighted a second bottleneck: forcing all tasks through a **Single Shared Output Neuron** causes old memories to be overwritten even if hidden weights are frozen.

### 4. Baby LLM (Character-Level Next-Token Prediction)
- **Result:** Successful `STRESS-FREEZE`.
- **Finding:** By compressing characters into a dense 16D embedding space, the global stress metric successfully activated! The network recognized the linguistic shift between Shakespeare and Python code, permanently locking down 1,000+ connections to protect its poetry neurons.

## Future Work (Generation 2)
The empirical results from Gen 1 dictate the architectural requirements for Gen 2:
1. **Multi-Head Output:** Dynamically growing new output neurons to prevent task interference at the final bottleneck.
2. **Per-Neuron Local Stress:** Shifting the conflict metric from a global matrix average to a local evaluation, allowing individual neurons to freeze themselves independently.
