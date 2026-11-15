# NeuralNet VR — 3D Advanced Neural Network Visualizer (Meta Quest / Unity)

Interactive 3D VR visualization of advanced neural network algorithms built in **Unity (OpenXR + XR Interaction Toolkit)** for **Meta Quest 2 / 3 / Pro** (with automatic Desktop WASD + Mouse fallback).

## Visualized Algorithms
1. **Transformer Self-Attention (`TransformerAttentionVisualizer.cs`)**
   - 3D Token Embeddings, Query (`Q`) & Value (`V`) projection nodes, Multi-Head Attention beams, and live `softmax(Q·Kᵀ / (√dₖ · Temperature))` weighting.
2. **Backpropagation & Adam Optimizer (`BackpropAdamVisualizer.cs`)**
   - Forward activations (cyan) -> MSE Loss -> Backward gradients (magenta), with a live toggle comparing **Adam adaptive gradients** vs. **vanishing sigmoid gradients**.
3. **ResNet Residual Block & 3D Convolution (`ResNetConvVisualizer.cs`)**
   - 3D stacked feature maps, sliding `3×3` convolution kernel cube, and glowing **Identity Skip Connection (`F(x) + x`)** arcs.
4. **Mixture of Experts (`MixtureOfExpertsVisualizer.cs`)**
   - Token stream -> Gating Router `G(x)` -> Sparse **Top-2 Expert** dispatch (`Syntax`, `Math`, `Code`, `Memory`) -> Weighted Combine.

## Quick Start (Unity Hub -> Meta Quest)
1. Open this folder in **Unity Hub** (`2022.3 LTS` or `Unity 6`).
2. In Unity: run **`Tools > DiGiorgioLabs > 1. Build NeuralNet VR Scene`** and **`2. Configure Meta Quest Android Settings`**.
3. Open **Edit > Project Settings > XR Plug-in Management** (Android tab) and enable **OpenXR** + **Meta Quest Support**.
4. Switch Build Target to **Android**, connect your Quest via USB/ADB, and click **Build & Run** (or press **Play** in the Editor with WASD + Right-Mouse drag).
