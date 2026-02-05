# 🔥 CUDA Fire Simulation 🔥

<p align="center">
  <img src="docs/demo.gif" alt="Simulation Demo">
  <br>
  </p>

## Overview
This project implements a high-performance **real-time particle system** simulating fire and smoke using **CUDA**. The goal was to leverage the parallel architecture of the GPU to overcome the computational limits of traditional CPU simulations.

Starting from a naive implementation, the project applies advanced optimization techniques, such as **Memory Coalescing (SoA)**, **Warp Divergence reduction**, and **Shared Memory** caching, achieving a **68% reduction** in kernel execution time and stable **144 FPS** with up to **2 million particles**.

## Key Features
* **Massive Parallelism:** Simulates millions of interacting particles in real-time.
* **Physical Accuracy:** Integration of turbulence, wind forces, and fluid-like behavior for smoke.
* **Optimized Memory Access:** Transition from Array of Structures (AoS) to Structure of Arrays (SoA) for perfectly coalesced memory access.
* **On-Chip Caching:** Extensive use of **Shared Memory** to reduce global memory latency.
* **Visual Rendering:** Direct interoperability with OpenGL (Vertex Buffer Objects) for rendering without CPU-GPU bus transfer overhead.

## Tech Stack
* **Language:** C++ / CUDA C
* **Graphics API:** OpenGL / GLFW
* [cite_start]**Hardware Target:** NVIDIA RTX 3060 (Compute Capability 8.6) [cite: 49]

---

## Optimization Journey

The core of this project is the incremental optimization strategy. The simulation evolved through several versions, each addressing a specific bottleneck identified via **NVIDIA Nsight Compute**.

### 1. Naive Implementation (Baseline)
* **Approach:** One thread per particle. Direct mapping.
* **Bottleneck:** **Memory Bound**. [cite_start]The naive use of `struct Particle` (AoS) caused uncoalesced memory accesses and significant latency (~137 cycles/instruction)[cite: 332].

### 2. Memory Optimization (SoA)
* **Solution:** Converted data structures from *Array of Structures* to *Structure of Arrays*.
* **Result:** Memory transactions became perfectly coalesced (1 transaction per warp instead of ~5). [cite_start]Latency dropped to **~41 cycles**[cite: 209].

### 3. Compute Optimization (Sub-Steps)
* **Solution:** Implemented sub-stepping physics.
* [cite_start]**Result:** Increased arithmetic intensity, allowing the scheduler to hide memory latency by executing useful calculations while waiting for data[cite: 232, 236].

### 4. Shared Memory & Interaction
* **Solution:** Used Shared Memory to cache particle data and manage thread interaction.
* [cite_start]**Feature:** Implemented a block-based interaction system where "wind" forces are calculated by a leader thread and shared via shared memory to simulate turbulence and smoke effects[cite: 295, 296].
* [cite_start]**Final Result:** The kernel became **Balanced** (Compute vs Memory) with a latency of just **~13 cycles**[cite: 332].

---

## Performance Analysis

### Kernel Optimization Impact
Comparison of execution metrics across different versions of the simulation:

<div align="center">

| Version | Latency (Cyc/Inst) | Warp Efficiency | Kernel Time (avg) | Improvement |
| :--- | :--- | :--- | :--- | :--- |
| **Naive (AoS)** | ~137 | Low (~0.15 warp) | 0.70 ms | - |
| **SoA (Optimized Mem)** | ~41 | Med (~0.30 warp) | 0.55 ms | 21% |
| **Final (Shared Mem)** | **~13** | **High (~11 warp)** | **0.24 ms** | **68%** |

</div>

[cite_start]*Data collected via NVIDIA Nsight Compute on RTX 3060.* [cite: 332]

### Rendering Scalability (FPS)
Performance scaling on a 144Hz monitor:

<div align="center">

| Particle Count | FPS | Visual Quality |
| :--- | :--- | :--- |
| **64K - 512K** | 144 FPS | Smooth, sparse fire |
| **1 Million** | 144 FPS | Good density & fluidity |
| **2 Million** | **144 FPS** | **Excellent result (Target)** |
| 16 Million | 76 FPS | Extremely dense |
| 64 Million | 24 FPS | GPU Saturation |

</div>

[cite_start]The architecture maintains maximum framerate (144 FPS) up to **2 million particles**, saturating the GPU limits only under extreme loads[cite: 342, 344].

---

## Project Structure

```text
├── src/
│   ├── kernels/           # CUDA kernel versions (Naive, Optimized, SharedMem)
│   ├── 4_interaction_particles # CUDA final kernel version
│   ├── glad.c             # OpenGL Loader
│   └── ...
├── docs/                  # Documentation and Analysis
└── README.md

