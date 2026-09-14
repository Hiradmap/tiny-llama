# Tiny_Llama
> **A State-of-the-Art ~51M Parameter Open LLM Architecture Built from Scratch**  
> *Engineered with RoPE, RMSNorm, SwiGLU, QK-Norm, Flash Attention, Muon Matrix Optimization, and WSD Schedule.*

---

## 1. Overview

**`Tiny_Llama`** represents the culmination of research and training experiments from `Tiny_GPT` and `Tiny_GPT2`. While preserving battle-tested training dynamics (AMP mixed precision, Warmup-Stable-Decay scheduling, Tensor Core vocabulary padding, and Muon optimization), `Tiny_Llama` fully transitions the network from legacy 2019 GPT-2 to the modern **2024–2026 open LLM architectural standard** (identical to LLaMA 3, Mistral, and Qwen 2.5).

---

## 2. Architectural Comparison: GPT-2 vs. Tiny_Llama

| Component | Tiny_GPT2 (2019 Legacy) | Tiny_Llama (Modern 2024–2026) | Mathematical & Practical Benefit |
| :--- | :--- | :--- | :--- |
| **Positional Encoding** | Absolute Learned Table (`wpe`) | **RoPE (Rotary Position Embeddings)** | Rotates $Q$ & $K$ in 2D pairs. Encodes relative distance $(m - n)$; eliminates fixed position tables. |
| **Normalization** | `nn.LayerNorm` (mean + std + bias) | **RMSNorm** (bias-free) | Drops mean-centering: $\frac{x}{\text{RMS}(x)} \odot \gamma$. Faster, simpler, and more stable in mixed precision. |
| **Feed-Forward Block** | Standard MLP ($4\times$ GELU) | **SwiGLU** (Gated Linear Unit) | $(xW_{\text{gate}} \odot \text{SiLU}(xW_{\text{up}})) W_{\text{down}}$. Provides multiplicative gradient gating. |
| **Attention Stability** | Unnormalized $Q$ and $K$ | **QK-Norm** (RMSNorm on $Q$ and $K$) | Prevents attention logit drift and entropy collapse during long training runs. |
| **Linear Layer Biases** | Biases included (`bias=True`) | **Bias-Free** (`bias=False` everywhere)| Reduces parameter overhead, improves training throughput and stability. |
| **Vocabulary Alignment**| 50,257 tokens | **50,304 tokens** (Multiple of 64) | Aligns matrices for NVIDIA Tensor Core warp execution. |
| **Optimization** | Element-wise AdamW | **Muon (2D) + AdamW (1D/Embed)** | 5-step Newton-Schulz matrix orthogonalization for internal weights. |

---

## 3. Mathematical Formulations

### A. Rotary Position Embeddings (RoPE)
Given query and key vectors $x \in \mathbb{R}^{d_k}$ at sequence position $m$:
$$R_{\Theta, m} x = \begin{pmatrix} x_1 \cos(m\theta_1) - x_2 \sin(m\theta_1) \\ x_1 \sin(m\theta_1) + x_2 \cos(m\theta_1) \\ \vdots \end{pmatrix}, \quad \text{where } \theta_i = 10000^{-2(i-1)/d_k}$$

### B. Root Mean Square Normalization (RMSNorm)
$$\text{RMSNorm}(x) = \frac{x}{\sqrt{\frac{1}{d} \sum_{i=1}^d x_i^2 + \epsilon}} \odot \gamma$$

### C. SwiGLU Gated Feed-Forward Network
$$\text{SwiGLU}(x) = \Big( (x W_{\text{gate}}) \odot \text{SiLU}(x W_{\text{up}}) \Big) W_{\text{down}}$$
* Intermediate hidden dimension is scaled to $\approx \frac{8}{3} n_{\text{embd}} = 1,344$ channels.

---

## 4. Repository Structure

```
Tiny_Llama/
├── model.py            # Complete modern LLaMA architecture (RMSNorm, RoPE, QK-Norm, SwiGLU, Flash Attention)
├── muon.py             # Keller Jordan's Muon optimizer + parameter partitioning utility
├── dataset.py          # Memory-mapped binary token loader with auto-symlink support
├── train.py            # Dual-optimizer training loop (Muon + AdamW, WSD, AMP, GradAccum, Checkpointing)
├── generate.py         # Interactive CLI inference engine (temperature, top-k/top-p, repetition penalty)
├── benchmark.py        # Standardized benchmark observer & metrics recorder
└── README.md           # Visual showcase and documentation
```

---

## 5. Quickstart & Usage

### 1. Training
Launch the complete production training pipeline:
```bash
python train.py
```
Key configurable flags:
* `--max_steps 10000`: Total optimization steps.
* `--batch_size 16`: Micro-batch size per GPU step.
* `--grad_accum_steps 4`: Gradient accumulation multiplier (Effective Batch Size = 64).
* `--eval_interval 500`: Evaluation frequency and checkpoint save interval.

### 2. Interactive Generation
Chat with or prompt your trained model checkpoint:
```bash
# Interactive REPL mode
python generate.py --interactive

# Single prompt inference
python generate.py --prompt "Once upon a time in a ancient realm" --temperature 0.8 --top_k 50
```

### 3. Architecture Sanity Check
Inspect parameter counts and verify tensor shapes:
```bash
python model.py
```

---

## 6. Training Dynamics & Benchmark Targets

* **Optimizer Setup**:
  * **Muon**: Internal 2D weights ($lr = 0.02$).
  * **AdamW**: Embedding table & RMSNorm scale weights ($lr = 6\text{e-}4, \text{weight\_decay}=0.1$).
* **Schedule**: Warmup ($0 \to 200$), Stable ($200 \to 6000$), Cosine Decay ($6000 \to 10000$).
* **Target Milestones**:
  - **4.50 Validation Loss**: Reached rapidly during warmup/early stable phase.
  - **3.80 Validation Loss**: Stretch target achieved during full cosine cooldown.
