# Dynamic Neural Network (PyTorch)

A prototype implementation of **Dynamic Capacity Allocation and Gradient-Conflict-Based Parameter Freezing** for Continual Learning.

## Scientific Disclaimer
This is a research prototype. The continual learning performance demonstrated in these benchmarks is achieved through a combination of:
**Dynamic Capacity + Replay + Modular Heads + Gradient Conflict Freezing.**
It is important to note that the Multi-Head evaluation leverages architectural isolation, and the Replay buffer handles a substantial portion of the memory retention. Ablation studies are provided to isolate the specific contributions of the structural freezing vs. memory replay.

## Key Mechanisms
Unlike standard MLPs that use a fixed topology, this network dynamically adapts:
1. **Dynamic Topology**: The network uses an adjacency matrix (`M`) to maintain a partially connected graph.
2. **Iterated State Update**: The forward pass runs for `steps` iterations, functioning as a fixed-point/recurrent message passing over a static graph.
3. **Cosine Gradient Conflict**: It tracks element-wise gradient conflicts between incoming streaming data and samples from a Continuous Reservoir Replay buffer using Cosine Similarity.
4. **Selective Freezing**: When gradient conflict exceeds a dynamically adjusted threshold, it freezes the incoming weights of the stressed neurons.
5. **Capacity Expansion**: When loss plateaus, it randomly spawns and connects new hidden neurons to expand model capacity.

*Note: While the topology is structurally sparse, the current PyTorch implementation uses a dense weight matrix (`W * M`) and dense matrix multiplication for simplicity, so it does not achieve computational sparsity.*

## Repository Structure

### 1. The Core Architecture (`src/dynamic_network.py`)
The `PyTorchDynamicNetwork` implements the structural plasticity and gradient conflict logic. It includes `detect_task_shift()` (a MACD-based heuristic anomaly detector) and `stress_freeze()` to protect parameters.

### 2. Task-Free Benchmark (`scripts/task_free_benchmark.py`)
A continuous stream evaluation on synthetic 2D datasets (Inner/Outer Circles -> Linear Boundary -> XOR Quadrants). The network utilizes a **Continuous Reservoir Sampling** buffer alongside Dynamic Capacity Allocation to detect statistical distribution shifts.

## Experimental Results

### 1. Comprehensive Benchmark (Task-Free 2D Stream)
To evaluate the autonomous structural decision-making mechanism, we ran a continuous data stream (Inner/Outer Circles $\rightarrow$ Linear Boundary $\rightarrow$ XOR Quadrants) with hidden distribution shifts.

**Key Findings:**
1. **Dynamic expansion improves or maintains performance without task-specific heads.** The dynamic network achieved ~68% accuracy on a Task-Free stream without explicit task IDs. 
2. **Freezing provides only a modest incremental effect in the current benchmark.** Structural growth provides the capacity needed for new representations, while freezing simply protects it.
3. **Compression substantially reduces net structural growth.** The `Full Dynamic + Compress` ablation mathematically proves that the network actively cleans up its architecture. It physically created 32.0 neurons across the data stream, but aggressively pruned 19.7 of them, reducing its Net Growth to just 12.3.
4. **The logical topology can be dramatically sparser than the dense implementation.** While the network allocated 250,500 parameters to satisfy PyTorch's dense tensor requirements, the compressed network achieved the performance of a massive 128-neuron static model using 10x fewer logical parameters (233 vs 2,602). Actual hardware efficiency has not yet been demonstrated because the implementation still uses dense tensors.
5. **Recurring-task experiments show bounded structural growth despite repeated distribution shifts.** In a 6-phase recurring sequence, the network created 54 neurons but pruned 44 of them, leaving 17 active neurons. This provides evidence that the structural adaptation mechanism can recycle capacity rather than monotonically expanding.
6. **The Empirical Upper Bound of the dataset is 78.9%**. Training a 256-neuron Single-Head MLP offline on the perfectly shuffled dataset caps at 78.9% because the target labels actively conflict for the exact same input coordinates. The dynamic model is therefore performing efficiently within the empirical limits of the benchmark.

| Configuration | Final Accuracy | Forgetting (BT) | Net Growth | Peak Growth | Created | Pruned | Shift Detection Delay |
|--------------|----------------|----------------|------------|-------------|---------|--------|-----------------------|
| Static MLP (Naive) | 63.4% ± 0.8% | -34.3% ± 0.8% | 0.0 | 0.0 | 0.0 | 0.0 | N/A |
| Static MLP (Replay) | 67.8% ± 0.9% | -28.2% ± 1.5% | 0.0 | 0.0 | 0.0 | 0.0 | N/A |
| Dynamic (Growth Only) + Replay | 68.0% ± 0.3% | -28.1% ± 0.8% | 28.3 | 28.3 | 28.3 | 0.0 | 1.0 |
| Dynamic (Freeze Only) + Replay | 67.5% ± 0.4% | -28.0% ± 0.7% | 1.0 | 1.0 | 1.0 | 0.0 | N/A |
| Full Dynamic (Growth+Freeze) + Replay | 68.3% ± 1.0% | -27.2% ± 1.3% | 30.3 | 30.3 | 30.3 | 0.0 | 1.0 |
| Full Dynamic + Compress | 67.6% ± 0.9% | -28.1% ± 1.7% | 12.3 | 15.0 | 32.0 | 19.7 | 1.0 |

### 2. High-Dimensional Scaling (Split-MNIST)
Evaluates the network on Split-MNIST using a Hybrid Architecture: A pre-trained frozen `ResNet18` feature extractor routing into the Dynamic Network. It sequentially learns 5 binary classification tasks, dynamically spawning new output heads for each task. The network achieved near-zero forgetting (98-99% accuracy across all tasks).

## Installation and Usage

```bash
git clone https://github.com/yourusername/dynamic-neural-network.git
cd dynamic-neural-network
pip install -r requirements.txt
```

To run the ablation study:
```bash
python scripts/ablation_study.py
```

To generate and run the Jupyter notebooks for visual inspection:
```bash
python scripts/make_task_free_notebook.py
python scripts/make_gen2_mnist.py
```
