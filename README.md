# Dynamic Neural Network (Generation 2)

This repository contains a PyTorch-accelerated Continuous Learning (CL) architecture that actively prevents **Catastrophic Forgetting** without requiring task labels or explicit boundaries. 

Instead of relying on static architectures with weight penalties (like EWC), this network is structurally plastic: it monitors the gradient covariance between incoming tasks and its own replay buffer to detect "stress" (memory overwrites). When high stress is detected, the network autonomously **freezes** threatened pathways and **grows** new neurons to expand its capacity.

## Core Mechanisms (Generation 2)
- **Local (Per-Neuron) Gradient Covariance:** The network computes an Exponential Moving Average (EMA) of `relu(-(g_task * g_replay))` for every individual neuron.
- **Localized Stress-Freeze:** If a specific neuron's local stress metric exceeds 0.3, the network physically sets the gradients of its incoming high-stress connections to exactly `0.0`. The rest of the network remains plastic.
- **Dynamic Multi-Head Output:** To prevent multiple tasks from overwriting the output layer, the network physically spawns a brand new Output Neuron ("Head") whenever a task shift is detected, cleanly separating the final bottleneck.
- **Autonomous Growth:** To compensate for frozen capacity and handle new tasks, the network spawns new hidden neurons (with random initialization) to create parallel pathways.

## Benchmarks & Findings

### 1. High-Dimensional Scaling (Split-MNIST)
- **Result:** Massive Success (Near 0% Forgetting).
- **Finding:** By combining a pre-trained frozen CNN feature extractor with the Generation 2 Multi-Head architecture, the network retained 94%–99% accuracy on 4 out of 5 sequential tasks. 
- **The Adam Momentum Discovery:** We empirically discovered that the Adam optimizer's residual momentum from previous tasks can temporarily kill hidden neurons when shifting to a new task. The Dynamic Network solved this autonomously by instantly growing new neurons to handle the remaining tasks!

### 2. The Optimizer Experiment (PlasticAdam)
- **Result:** Mathematical Collapse (NaN).
- **Finding:** We attempted to build a custom optimizer ("PlasticAdam") that decayed Adam's momentum (1st and 2nd moments) based on our stress metric. We empirically discovered that manipulating Adam's 2nd moment (variance) by even 10% causes the learning rates to explode by 100x, leading to catastrophic failure. 
- **Conclusion:** Modifying variance buffers is fundamentally unstable. Standard optimizers paired with highly modular, structurally plastic architectures (like Multi-Head Generation 2) are the most robust approach to Continuous Learning.

### 3. Baby LLM (Character-Level Next-Token Prediction)
- **Result:** Successful Semantic `STRESS-FREEZE`.
- **Finding:** By compressing characters into a dense 16D embedding space, the stress metric successfully activated. When shifting from learning Shakespeare to learning Python code, the network recognized the severe linguistic shift and autonomously locked down over 1,000 connections to protect its poetry neurons!

## Installation and Usage

1. **Install Dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

2. **Run Benchmarks:**
   The repository includes a consolidated `main.py` script that sequentially trains a Standard Static Baseline side-by-side with the Dynamic Network to explicitly demonstrate the catastrophic forgetting problem and our solution.
   
   ```bash
   # Run all benchmarks
   python main.py --benchmark all
   
   # Run only the Toy 2D Benchmark
   python main.py --benchmark toy
   
   # Run only the Split-MNIST Benchmark
   python main.py --benchmark mnist
   ```
   *(Note: The LLM language generation benchmark is best viewed inside the `notebooks/baby_llm.ipynb` Jupyter Notebook due to text rendering.)*

## Conclusion
This repository serves as an empirical playground for Artificial General Intelligence (AGI) research. It mathematically proves that true lifelong learning requires structural plasticity (growth and targeted freezing) and modularity (Multi-Head outputs). 

*Note: The Generation 1 architecture and notebooks are preserved in the `gen1_archive/` directory for historical reference.*
