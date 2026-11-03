# SegFormer: Simple and Efficient Design for Semantic Segmentation with Transformers

**Paper:** [SegFormer: Simple and Efficient Design for Semantic Segmentation with Transformers](https://arxiv.org/abs/2105.15203) (Xie et al., NeurIPS 2021)

This document serves as the comprehensive technical guide for understanding, training, and debugging this repository's implementation of **SegFormer**. Rather than repeating boilerplate code, this guide focuses deeply on **the mathematical mechanics of the model**, **subtle discrepancies between the paper's equations and the actual tensor operations**, and **the exact training, regularization, and evaluation protocols** required to reproduce the paper's state-of-the-art results.

---

## Table of Contents

1. [Core Motivation: Why ViT & CNNs Fall Short](#1-core-motivation-why-vit--cnns-fall-short)
2. [Deep Dive: Mix Transformer (MiT) Encoder](#2-deep-dive-mix-transformer-mit-encoder)
   * [2.1 Overlapping Patch Merging](#21-overlapping-patch-merging)
   * [2.2 Efficient Self-Attention (Spatial Reduction)](#22-efficient-self-attention-spatial-reduction)
   * [2.3 Mix-FFN & Positional-Encoding-Free Design](#23-mix-ffn--positional-encoding-free-design)
3. [Deep Dive: Lightweight All-MLP Decoder](#3-deep-dive-lightweight-all-mlp-decoder)
   * [3.1 Step-by-Step Mathematical Formulation](#31-step-by-step-mathematical-formulation)
   * [3.2 Effective Receptive Field (ERF) Analysis: Why MLP is Enough](#32-effective-receptive-field-erf-analysis-why-mlp-is-enough)
4. [The Tricky Parts & Paper-to-Implementation Gotchas](#4-the-tricky-parts--paper-to-implementation-gotchas)
   * [Gotcha 1: Spatial Reduction — 1D Reshape in Theory vs. 2D Strided Conv in Practice](#gotcha-1-spatial-reduction--1d-reshape-in-theory-vs-2d-strided-conv-in-practice)
   * [Gotcha 2: Asymmetric Query vs. Key/Value Sequence Lengths](#gotcha-2-asymmetric-query-vs-keyvalue-sequence-lengths)
   * [Gotcha 3: Zero-Padding Boundary Leakage & Why Weight Decay Must Be Masked](#gotcha-3-zero-padding-boundary-leakage--why-weight-decay-must-be-masked)
   * [Gotcha 4: Dynamic $(H, W)$ Propagation for Non-Square Aspect Ratios](#gotcha-4-dynamic-h-w-propagation-for-non-square-aspect-ratios)
   * [Gotcha 5: `align_corners=False` in Bilinear Upsampling](#gotcha-5-align_cornersfalse-in-bilinear-upsampling)
   * [Gotcha 6: Missing Bias and Normalization Placement in Fusion Layers](#gotcha-6-missing-bias-and-normalization-placement-in-fusion-layers)
   * [Gotcha 7: Stochastic Depth (`DropPath`) Linear Scaling Rule](#gotcha-7-stochastic-depth-droppath-linear-scaling-rule)
5. [Complete Tensor Shape & Complexity Trace (MiT-B0 to MiT-B5)](#5-complete-tensor-shape--complexity-trace-mit-b0-to-mit-b5)
6. [Training Methodology & Optimization Recipe](#6-training-methodology--optimization-recipe)
   * [6.1 Two-Stage Paradigm: ImageNet-1K Pre-training $\rightarrow$ Fine-tuning](#61-two-stage-paradigm-imagenet-1k-pre-training-rightarrow-fine-tuning)
   * [6.2 Optimizer & Parameter-Specific Weight Decay Rules](#62-optimizer--parameter-specific-weight-decay-rules)
   * [6.3 Learning Rate Schedule: Linear Warmup + Polynomial Decay](#63-learning-rate-schedule-linear-warmup--polynomial-decay)
   * [6.4 Data Augmentation Pipeline by Dataset](#64-data-augmentation-pipeline-by-dataset)
   * [6.5 Loss Function & Full-Resolution Supervision](#65-loss-function--full-resolution-supervision)
   * [6.6 Inference Protocols: Whole vs. Sliding Window Evaluation](#66-inference-protocols-whole-vs-sliding-window-evaluation)
7. [Key Ablation Studies & Benchmark Summary](#7-key-ablation-studies--benchmark-summary)

---

## 1. Core Motivation: Why ViT & CNNs Fall Short

Semantic segmentation requires two competing properties simultaneously: **fine-grained spatial localization** (to trace sharp object boundaries like poles, curbs, and limbs) and **long-range global context** (to resolve ambiguous textures like a large wall vs. a road).

| Architecture Family | Representative Models | Primary Bottleneck in Semantic Segmentation |
| :--- | :--- | :--- |
| **Dilated CNNs** | FCN, DeepLabV3+, PSPNet | **Limited Effective Receptive Field (ERF):** Convolutions are inherently local. Expanding context requires heavy decoder heads (ASPP, pyramid pooling) and dilated convolutions that are memory-intensive and still decay exponentially from the center. |
| **Columnar ViTs** | ViT, SETR | **Single-Scale Coarse Resolution:** Splitting an image into $16 \times 16$ patches produces a single $\frac{1}{16}$-scale feature grid. High-frequency edge detail is destroyed in Layer 1, and quadratic $\mathcal{O}(N^2)$ attention prevents operating at $\frac{1}{4}$ resolution. |
| **Positional-Encoded ViTs** | DeiT, Swin, SETR | **Resolution Rigidity:** Fixed-size learned positional embeddings (or fixed window biases) must be interpolated when test resolution differs from training resolution, often causing a $1\%\text{–}4\%$ drop in mIoU. |

**SegFormer's Design Philosophy:**
1. **Multi-scale hierarchy in the backbone:** Generate CNN-like multi-level features ($\frac{1}{4}, \frac{1}{8}, \frac{1}{16}, \frac{1}{32}$) inside a pure Transformer encoder.
2. **Positional-encoding-free:** Use $3 \times 3$ depthwise convolutions with zero-padding to leak spatial position implicitly, allowing seamless multi-resolution training and inference.
3. **Ultra-lightweight decoder:** Because the Transformer encoder already captures a global receptive field across all stages, the decoder only needs to unify channel dimensions, upsample, concatenate, and project.

---

## 2. Deep Dive: Mix Transformer (MiT) Encoder

Given an input image $X \in \mathbb{R}^{B \times 3 \times H \times W}$, the MiT encoder processes the image through four sequential stages $i \in \{1, 2, 3, 4\}$. Each stage outputs a feature map $F_i$ of shape:

$$F_i \in \mathbb{R}^{B \times C_i \times \frac{H}{2^{i+1}} \times \frac{W}{2^{i+1}}}$$

```
Input Image: [B, 3, H, W]
  │
  ├──► Stage 1: OverlapPatchMerge (K=7, S=4, P=3) ──► L1 × MiT Block (SR=8) ──► F1: [B, C1,  H/4,  W/4]
  │
  ├──► Stage 2: OverlapPatchMerge (K=3, S=2, P=1) ──► L2 × MiT Block (SR=4) ──► F2: [B, C2,  H/8,  W/8]
  │
  ├──► Stage 3: OverlapPatchMerge (K=3, S=2, P=1) ──► L3 × MiT Block (SR=2) ──► F3: [B, C3, H/16, W/16]
  │
  └──► Stage 4: OverlapPatchMerge (K=3, S=2, P=1) ──► L4 × MiT Block (SR=1) ──► F4: [B, C4, H/32, W/32]
```

### 2.1 Overlapping Patch Merging
In ViT and Swin Transformer, patch merging splits or concatenates non-overlapping $S \times S$ regions ($K = S, P = 0$). This slices continuous object boundaries at patch borders.

SegFormer enforces **overlapping** receptive windows at the start of every stage by coupling kernel size $K$, stride $S$, and zero-padding $P$ via:

$$K = 2S - 1, \quad P = S - 1$$

Substituting into the standard 2D convolution spatial output equation:

$$H_{\text{out}} = \left\lfloor \frac{H_{\text{in}} + 2P - K}{S} \right\rfloor + 1 = \left\lfloor \frac{H_{\text{in}} + 2(S - 1) - (2S - 1)}{S} \right\rfloor + 1 = \left\lfloor \frac{H_{\text{in}} - 1}{S} \right\rfloor + 1$$

* **Stage 1 ($S = 4 \implies K = 7, P = 3$):** Downsamples $H \times W \rightarrow \frac{H}{4} \times \frac{W}{4}$ while each token aggregates a $7 \times 7$ pixel neighborhood (overlapping by 3 pixels with adjacent tokens).
* **Stages 2, 3, 4 ($S = 2 \implies K = 3, P = 1$):** Downsamples spatial dimensions by $2\times$ at each transition while each token aggregates a $3 \times 3$ neighborhood from the previous stage.

### 2.2 Efficient Self-Attention (Spatial Reduction)
At Stage 1, the token sequence length is $N_1 = \frac{H}{4} \times \frac{W}{4}$. For a $512 \times 512$ image, $N_1 = 16,384$; for a $1024 \times 2048$ Cityscapes image, $N_1 = 256 \times 512 = 131,072$. Standard self-attention requires an $N_1 \times N_1$ matrix ($1.7 \times 10^{10}$ elements per head on Cityscapes), which immediately exhausts GPU VRAM.

SegFormer introduces a **Spatial Reduction Ratio** $SR_i$ per spatial axis (yielding a total sequence length reduction ratio $R_i = SR_i^2$). While the Query $Q$ preserves all $N_i$ tokens, the Key $K$ and Value $V$ sequences are spatially compressed from $N_i$ tokens to $\frac{N_i}{R_i}$ tokens:

$$\hat{K} = \text{Reshape}\left(\frac{N_i}{R_i}, C_i \cdot R_i\right)(K)$$

$$K_{\text{red}} = \text{Linear}(C_i \cdot R_i, C_i)\big(\text{LayerNorm}(\hat{K})\big)$$

*(The identical reduction is applied to $V \rightarrow V_{\text{red}}$).* Multi-Head Self-Attention is then computed as:

$$\text{Attention}(Q, K_{\text{red}}, V_{\text{red}}) = \text{Softmax}\left(\frac{Q K_{\text{red}}^\top}{\sqrt{d_{\text{head}}}}\right) V_{\text{red}}$$

Across the four stages, SegFormer sets:
* $SR = [8, 4, 2, 1] \implies R = SR^2 = [64, 16, 4, 1]$
* Computational complexity drops from $\mathcal{O}(N_i^2)$ to $\mathcal{O}\left(\frac{N_i^2}{R_i}\right)$.

### 2.3 Mix-FFN & Positional-Encoding-Free Design
Following the Efficient Self-Attention block, each MiT layer applies a **Mix-FFN** module instead of a standard point-wise MLP:

$$X_{\text{out}} = \text{MLP}\Big(\text{GELU}\big(\text{Conv}_{3 \times 3}^{\text{DW}}(\text{MLP}(X_{\text{in}}))\big)\Big) + X_{\text{in}}$$

1. **Channel Expansion ($\text{MLP}_1$):** Projects token channels from $C_i \rightarrow E \cdot C_i$ (where expansion ratio $E = 4$).
2. **Local Spatial Mixing ($\text{Conv}_{3 \times 3}^{\text{DW}}$):** Tokens are reshaped into a 2D spatial grid $[B, 4C_i, H_i, W_i]$ and passed through a $3 \times 3$ **depthwise convolution** (`groups = 4 * C_i`, `padding = 1`).
3. **Non-Linearity & Projection ($\text{GELU} + \text{MLP}_2$):** Applies GELU activation and projects channels back from $4C_i \rightarrow C_i$.

Because a $3 \times 3$ depthwise convolution has only $9$ parameters per channel, it adds negligible parameter overhead ($<1\%$) while providing both local inductive bias and implicit positional awareness.

---

## 3. Deep Dive: Lightweight All-MLP Decoder

### 3.1 Step-by-Step Mathematical Formulation
Given the four multi-level feature maps $\{F_1, F_2, F_3, F_4\}$ from the MiT encoder where $F_i \in \mathbb{R}^{B \times C_i \times H_i \times W_i}$, the All-MLP decoder executes four sequential steps using a unified embedding dimension $C_{\text{dec}}$ ($256$ for B0–B1; $768$ for B2–B5):

1. **Channel Unification:** Each feature map $F_i$ is projected to $C_{\text{dec}}$ channels via a point-wise linear transformation:
   $$\hat{F}_i = \text{Linear}(C_i, C_{\text{dec}})(F_i) \quad \forall i \in \{1, 2, 3, 4\}$$
2. **Spatial Unification:** Each projected map $\hat{F}_i$ is bilinearly upsampled to match the spatial resolution of Stage 1 ($\frac{H}{4} \times \frac{W}{4}$):
   $$\tilde{F}_i = \text{Upsample}_{\frac{H}{4} \times \frac{W}{4}}(\hat{F}_i) \quad \forall i \in \{1, 2, 3, 4\}$$
3. **Multi-Scale Concatenation & Fusion:** The four aligned feature maps are concatenated along the channel dimension (yielding $4 C_{\text{dec}}$ channels) and fused back to $C_{\text{dec}}$ channels:
   $$F_{\text{fused}} = \text{ReLU}\Big(\text{BatchNorm2d}\big(\text{Linear}(4C_{\text{dec}}, C_{\text{dec}})(\text{Concat}[\tilde{F}_1, \tilde{F}_2, \tilde{F}_3, \tilde{F}_4])\big)\Big)$$
4. **Pixel-Wise Classification:** After spatial dropout, a final linear layer predicts $K_{\text{cls}}$ class logits at $\frac{1}{4}$ resolution:
   $$M_{\frac{1}{4}} = \text{Linear}(C_{\text{dec}}, K_{\text{cls}})(\text{Dropout}(F_{\text{fused}})) \in \mathbb{R}^{B \times K_{\text{cls}} \times \frac{H}{4} \times \frac{W}{4}}$$

### 3.2 Effective Receptive Field (ERF) Analysis: Why MLP is Enough
A natural question is: *Why can SegFormer get away with a $1 \times 1$ point-wise decoder when DeepLabV3+ needs $3 \times 3$ atrous convolutions with dilation rates $\{6, 12, 18\}$?*

In Section 3.3 of the paper, the authors visualize the **Effective Receptive Field (ERF)** of DeepLabV3+ (ResNet-101) vs. SegFormer:
* **CNN Backbones:** Even at Stage 4 of ResNet-101, the ERF is localized to a relatively small central Gaussian region. Therefore, a CNN **must** rely on its decoder head to aggregate multi-scale context via large kernels or dilated convolutions.
* **MiT Backbone:** Because Self-Attention connects every query token to keys across the entire spatial grid from Stage 1 onward, **every stage of MiT already possesses a global receptive field**. Stage 1 combines high-frequency local edges (via $3 \times 3$ Mix-FFN) with global context, and Stage 4 produces deeply semantic global representations.
* Consequently, a point-wise MLP decoder ($1 \times 1$ spatial footprint) is completely sufficient—it only needs to mix the *scales* of features at each pixel location, not expand the spatial receptive field.

---

## 4. The Tricky Parts & Paper-to-Implementation Gotchas

When reading the paper alongside the codebase, there are seven critical discrepancies and subtle engineering details you must understand.

### Gotcha 1: Spatial Reduction — 1D Reshape in Theory vs. 2D Strided Conv in Practice
In Section 3.1, Equation 2 writes spatial reduction as:
$$\hat{K} = \text{Reshape}\left(\frac{N}{R}, C \cdot R\right)(K), \quad K = \text{Linear}(C \cdot R, C)(\hat{K})$$

**Why a naive 1D `.reshape(B, N // R, C * R)` breaks your model:**
If a 2D feature map of size $H \times W$ is flattened row-by-row into $N = H \cdot W$ tokens, grouping $R = 64$ consecutive tokens in 1D groups a $1 \times 64$ **horizontal strip** of pixels (or wraps around across image rows) rather than an $8 \times 8$ **2D spatial patch**!

**How it is actually implemented:**
To group true $SR \times SR$ 2D neighborhoods (where $SR = \sqrt{R}$), the token sequence $[B, N, C]$ is reshaped back to a 2D spatial tensor $[B, C, H, W]$ and passed through a 2D convolution with `kernel_size=SR, stride=SR`:
* Mathematically, a `Conv2d(C, C, kernel_size=SR, stride=SR)` extracts non-overlapping $SR \times SR$ patches (containing $C \cdot SR^2 = C \cdot R$ values) and linearly projects them to $C$ channels—making it **100% equivalent** to a proper 2D block-wise reshape followed by `Linear(C * R, C)`.
* Furthermore, in implementation, **one shared `sr` Conv2d and `LayerNorm`** is applied to the input token tensor `x` *before* projecting into $K$ and $V$ via a combined `self.kv = nn.Linear(C, 2 * C)` layer, cutting the spatial reduction compute in half.

### Gotcha 2: Asymmetric Query vs. Key/Value Sequence Lengths
During Spatial Reduction Attention in Stages 1–3:
* $Q$ has shape $[B, \text{heads}, N, d_{\text{head}}]$
* $K, V$ have shape $[B, \text{heads}, N / R, d_{\text{head}}]$

When computing $A = \text{Softmax}\left(\frac{Q K^\top}{\sqrt{d_{\text{head}}}}\right)$, the attention map $A$ is **rectangular** of shape $[B, \text{heads}, N, N / R]$.
Multiplying $A \times V$ performs:

$$[B, \text{heads}, N, N / R] \times [B, \text{heads}, N / R, d_{\text{head}}] \longrightarrow [B, \text{heads}, N, d_{\text{head}}]$$

**Takeaway:** The spatial resolution of the feature map is **never downsampled** inside the Transformer blocks. Every single one of the $N$ query locations receives a distinct output vector; it merely summarizes context from $N / R$ compressed key-value regions.

### Gotcha 3: Zero-Padding Boundary Leakage & Why Weight Decay Must Be Masked
How does a translation-equivariant $3 \times 3$ depthwise convolution teach the network *where* a token is located in the image?
* At the top-left corner $(0, 0)$, the $3 \times 3$ kernel sees $5$ zero-padded values on the top and left.
* On the top edge $(0, w)$, it sees $3$ zero-padded values on top.
* In the interior $(h, w)$, it sees $0$ padded zeros.

Across stacked layers, this asymmetric boundary signal propagates inward, establishing a coordinate frame. **However**, if strong $L_2$ weight decay is applied to the depthwise convolution kernels or normalization biases, it suppresses the border-sensitive filter weights. This is why the training recipe **explicitly sets `weight_decay = 0.0` on all `dwconv`, `LayerNorm`, and `BatchNorm` parameters**.

### Gotcha 4: Dynamic $(H, W)$ Propagation for Non-Square Aspect Ratios
A frequent bug in custom implementations is inferring spatial dimensions inside `MixFFN` or `EfficientSelfAttention` via:
```python
# BUGGY ASSUMPTION: Only works when H == W!
H = W = int(math.sqrt(N))
```
While training crops on ADE20K are often square ($512 \times 512$), **validation and test images are almost always rectangular** (e.g., $512 \times 683$ on ADE20K whole-image inference, or $1024 \times 2048$ on Cityscapes).
* Therefore, `OverlapPatchEmbed.forward(x)` **must** dynamically inspect `H, W = x.shape[2], x.shape[3]` after the strided convolution and return `(x, H, W)`.
* Every `MiTBlock`, `EfficientSelfAttention`, and `MixFFN` call must accept `(x, H, W)` explicitly so `view(B, C, H, W)` works on arbitrary aspect ratios.

### Gotcha 5: `align_corners=False` in Bilinear Upsampling
In PyTorch, `F.interpolate(..., mode='bilinear')` behaves differently depending on `align_corners`:
* `align_corners=True` aligns the centers of the 4 corner pixels, which shifts pixel coordinates slightly when scaling across power-of-two pyramids ($16 \rightarrow 32 \rightarrow 64 \rightarrow 128$).
* SegFormer (following MMSegmentation conventions for strided patch hierarchies) strictly uses **`align_corners=False`** both when upsampling $\hat{F}_2, \hat{F}_3, \hat{F}_4$ to $\frac{H}{4} \times \frac{W}{4}$ inside the decoder and when upsampling the final logits $4\times$ to $H \times W$. Mixing `True` and `False` between training and inference degrades boundary mIoU by $0.3\%\text{–}0.6\%$.

### Gotcha 6: Missing Bias and Normalization Placement in Fusion Layers
In Section 3.2, the decoder equations only write $\text{Linear}(C_i, C)$ and $\text{Linear}(4C, C)$. In the actual implementation:
1. The four per-stage projections $\text{Linear}(C_i, C_{\text{dec}})$ are implemented as $1 \times 1$ convolutions (`nn.Conv2d(C_i, C_dec, kernel_size=1, bias=True)`) **without** normalization or activation.
2. The concatenation fusion layer `linear_fuse` is not a bare linear layer—it is a `Conv2d(4 * C_dec, C_dec, kernel_size=1, bias=False)` immediately followed by **`BatchNorm2d(C_dec)`** (or `SyncBatchNorm` in multi-GPU training) and **`ReLU(inplace=True)`**. Because `BatchNorm2d` centers the activations and has its own learnable affine shift $\beta$, `bias=False` is set on the fusion convolution.

### Gotcha 7: Stochastic Depth (`DropPath`) Linear Scaling Rule
While `MiT-B0` is shallow ($8$ total Transformer blocks) and can train with `drop_path_rate = 0.0`, deeper models like `MiT-B3` ($28$ blocks), `MiT-B4` ($41$ blocks), and `MiT-B5` ($52$ blocks) will overfit or suffer gradient instability without **Stochastic Depth (`DropPath`)**.
* Crucially, `DropPath` is **not** uniform across layers. It scales **linearly** from $0.0$ at the very first block of Stage 1 to `drop_path_rate` ( typically $0.1$) at the final block of Stage 4:
  $$p_l = \frac{l}{\sum_{i=1}^4 L_i - 1} \times p_{\max}, \quad l \in \left\{0, 1, \dots, \sum L_i - 1\right\}$$
* It is applied around the residual branches of both Attention and Mix-FFN:
  $$X = X + \text{DropPath}_l\big(\text{Attn}(\text{LN}(X))\big)$$
  $$X = X + \text{DropPath}_l\big(\text{MixFFN}(\text{LN}(X))\big)$$

---

## 5. Complete Tensor Shape & Complexity Trace (MiT-B0 to MiT-B5)

### Detailed Step-by-Step Tensor Trace (SegFormer-B0 on $512 \times 512$ Input, $K_{\text{cls}} = 19$)

| Module / Operation | Kernel / Stride / Pad | Representation | Tensor Shape | Notes |
| :--- | :---: | :---: | :--- | :--- |
| **Input RGB Image** | — | 2D Grid | $[B, 3, 512, 512]$ | Normalized RGB float tensor |
| **Stage 1 Patch Embed** | $K=7, S=4, P=3$ | 2D $\rightarrow$ 1D Seq | $[B, 16384, 32]$ | $H_1=128, W_1=128, N_1=16,384$ |
| **Stage 1 SRA ($\times 2$ blocks)** | $SR_1=8 \implies R_1=64$ | 1D Seq | $Q: [B, 1, 16384, 32]$<br>$K, V: [B, 1, 256, 32]$ | $\text{Heads}=1, d_{\text{head}}=32$<br>Attn matrix: $[B, 1, 16384, 256]$ |
| **Stage 1 Mix-FFN ($\times 2$ blocks)** | $K=3, S=1, P=1$ | 1D $\leftrightarrow$ 2D | $[B, 128, 128, 128]$ (inner) | Expanded $4\times$: $32 \rightarrow 128 \rightarrow 32$ |
| **Stage 1 Output ($F_1$)** | — | 2D Grid | $[B, 32, 128, 128]$ | $\frac{1}{4}$ spatial resolution |
| **Stage 2 Patch Embed** | $K=3, S=2, P=1$ | 2D $\rightarrow$ 1D Seq | $[B, 4096, 64]$ | $H_2=64, W_2=64, N_2=4,096$ |
| **Stage 2 SRA ($\times 2$ blocks)** | $SR_2=4 \implies R_2=16$ | 1D Seq | $Q: [B, 2, 4096, 32]$<br>$K, V: [B, 2, 256, 32]$ | $\text{Heads}=2, d_{\text{head}}=32$<br>Attn matrix: $[B, 2, 4096, 256]$ |
| **Stage 2 Output ($F_2$)** | — | 2D Grid | $[B, 64, 64, 64]$ | $\frac{1}{8}$ spatial resolution |
| **Stage 3 Patch Embed** | $K=3, S=2, P=1$ | 2D $\rightarrow$ 1D Seq | $[B, 1024, 160]$ | $H_3=32, W_3=32, N_3=1,024$ |
| **Stage 3 SRA ($\times 2$ blocks)** | $SR_3=2 \implies R_3=4$ | 1D Seq | $Q: [B, 5, 1024, 32]$<br>$K, V: [B, 5, 256, 32]$ | $\text{Heads}=5, d_{\text{head}}=32$<br>Attn matrix: $[B, 5, 1024, 256]$ |
| **Stage 3 Output ($F_3$)** | — | 2D Grid | $[B, 160, 32, 32]$ | $\frac{1}{16}$ spatial resolution |
| **Stage 4 Patch Embed** | $K=3, S=2, P=1$ | 2D $\rightarrow$ 1D Seq | $[B, 256, 256]$ | $H_4=16, W_4=16, N_4=256$ |
| **Stage 4 SRA ($\times 2$ blocks)** | $SR_4=1 \implies R_4=1$ | 1D Seq | $Q, K, V: [B, 8, 256, 32]$ | $\text{Heads}=8, d_{\text{head}}=32$<br>Full self-attention ($256 \times 256$) |
| **Stage 4 Output ($F_4$)** | — | 2D Grid | $[B, 256, 16, 16]$ | $\frac{1}{32}$ spatial resolution |
| **Decoder MLP Projections** | $K=1, S=1, P=0$ | 2D Grid | $4 \times [B, 256, H_i, W_i]$ | Unifies all 4 stages to $C_{\text{dec}}=256$ |
| **Decoder Upsample & Concat** | Bilinear ($\frac{H}{4} \times \frac{W}{4}$) | 2D Grid | $[B, 1024, 128, 128]$ | Concatenated along `dim=1` |
| **Decoder Linear Fuse** | $K=1, S=1$ + BN + ReLU | 2D Grid | $[B, 256, 128, 128]$ | Mixes multi-scale features |
| **Decoder Classifier** | $K=1, S=1$ | 2D Grid | $[B, 19, 128, 128]$ | Native $\frac{1}{4}$-scale class logits |
| **Final Upsampling** | Bilinear ($4\times$) | 2D Grid | $[B, 19, 512, 512]$ | Full-resolution pixel logits |

*Notice an elegant architectural symmetry in the table above:* Because $N_1 = 16,384, N_2 = 4,096, N_3 = 1,024, N_4 = 256$ and $R = [64, 16, 4, 1]$, the reduced Key/Value sequence length $N_i / R_i$ at $512 \times 512$ resolution is **identically $256$ tokens across every single stage of the network!**

### Architectural Specifications Across All Variants (B0 – B5)

All six variants share the exact same patch merging strides, spatial reduction ratios $SR = [8, 4, 2, 1]$, attention head counts $H = [1, 2, 5, 8]$ (so $d_{\text{head}}$ is always $32$ in B0 and $64$ in B1–B5), and Mix-FFN expansion ratio $E = 4$:

| Model Variant | Embed Dims $[C_1, C_2, C_3, C_4]$ | Layer Depths $[L_1, L_2, L_3, L_4]$ | Total Blocks | Head Dim $d_{\text{head}}$ | Decoder Dim $C_{\text{dec}}$ | Encoder Params | Decoder Params | FLOPs ($512^2$) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **SegFormer-B0** | $[32, 64, 160, 256]$ | $[2, 2, 2, 2]$ | $8$ | $32$ | $256$ | $3.4\text{M}$ | $0.4\text{M}$ | $8.4\text{G}$ |
| **SegFormer-B1** | $[64, 128, 320, 512]$ | $[2, 2, 2, 2]$ | $8$ | $64$ | $256$ | $13.1\text{M}$ | $0.6\text{M}$ | $15.9\text{G}$ |
| **SegFormer-B2** | $[64, 128, 320, 512]$ | $[3, 4, 6, 3]$ | $16$ | $64$ | $768$ | $24.2\text{M}$ | $3.3\text{M}$ | $62.4\text{G}$ |
| **SegFormer-B3** | $[64, 128, 320, 512]$ | $[3, 4, 18, 3]$ | $28$ | $64$ | $768$ | $44.0\text{M}$ | $3.3\text{M}$ | $79.0\text{G}$ |
| **SegFormer-B4** | $[64, 128, 320, 512]$ | $[3, 8, 27, 3]$ | $41$ | $64$ | $768$ | $60.8\text{M}$ | $3.3\text{M}$ | $95.7\text{G}$ |
| **SegFormer-B5** | $[64, 128, 320, 512]$ | $[3, 6, 40, 3]$ | $52$ | $64$ | $768$ | $81.4\text{M}$ | $3.3\text{M}$ | $106.7\text{G}$ |

---

## 6. Training Methodology & Optimization Recipe

Training Vision Transformers from scratch directly on segmentation datasets (which only have $2\text{K–}20\text{K}$ images) leads to severe underfitting because Transformers lack the strong translation invariance of CNNs. Reproducing the paper requires adhering strictly to the following training pipeline.

### 6.1 Two-Stage Paradigm: ImageNet-1K Pre-training $\rightarrow$ Fine-tuning
1. **Stage 1 (Encoder Pre-training):**
   * Attach a global average pooling layer and a $1000$-way linear classifier to Stage 4 output $F_4$ of the MiT encoder.
   * Pre-train on **ImageNet-1K** ($1.28\text{M}$ images, $224 \times 224$) for $300$ epochs using the DeiT training recipe (AdamW, cosine annealing LR, RandAugment, Mixup, CutMix, Label Smoothing $= 0.1$).
2. **Stage 2 (Segmentation Fine-tuning):**
   * Load the ImageNet-pretrained weights into the `MixVisionTransformer` backbone.
   * Initialize the `SegFormerHead` decoder weights randomly:
     * Convolutions: Kaiming Normal initialization (`mode='fan_out', nonlinearity='relu'`).
     * `LayerNorm` / `BatchNorm2d`: $\gamma = 1.0, \beta = 0.0$.
     * Linear layers: Truncated Normal ($\sigma = 0.02$), biases $= 0.0$.

### 6.2 Optimizer & Parameter-Specific Weight Decay Rules
* **Optimizer:** `AdamW` ($\beta_1 = 0.9, \beta_2 = 0.999$, base `weight_decay = 0.01`).
* **Differential Learning Rate & Parameter Grouping:**
  * **Decoder Head Multiplier:** In standard MMSegmentation recipes for SegFormer, the randomly initialized decoder head receives a **$10\times$ learning rate multiplier** ($6 \times 10^{-4}$) relative to the pre-trained encoder backbone ($6 \times 10^{-5}$) so the head adapts rapidly without destabilizing pre-trained backbone representations.
  * **No-Weight-Decay Parameter Group (`weight_decay = 0.0`):**
    * All bias parameters (`*.bias`)
    * All `LayerNorm` and `BatchNorm` parameters
    * All depthwise convolution weights inside `MixFFN` (`*.dwconv.weight`)

### 6.3 Learning Rate Schedule: Linear Warmup + Polynomial Decay
SegFormer is trained by **iterations** rather than epochs ($160,000$ iterations on ADE20K and Cityscapes; $80,000$ iterations on COCO-Stuff):

1. **Linear Warmup ($t \le T_{\text{warmup}} = 1,500$ iterations):**
   $$\eta_t = \eta_{\text{base}} \cdot \left(\alpha_{\text{warmup}} + (1 - \alpha_{\text{warmup}}) \frac{t}{T_{\text{warmup}}}\right), \quad \alpha_{\text{warmup}} = 10^{-6}$$
2. **Polynomial Decay ($T_{\text{warmup}} < t \le T_{\text{max}} = 160,000$ iterations):**
   $$\eta_t = \eta_{\text{base}} \cdot \left(1 - \frac{t - T_{\text{warmup}}}{T_{\text{max}} - T_{\text{warmup}}}\right)^p, \quad \text{power } p = 1.0$$

*(Note: Because $p = 1.0$, the post-warmup schedule is a strict linear decay to $0.0$ at $t = 160,000$).*

### 6.4 Data Augmentation Pipeline by Dataset
The exact training augmentations applied before feeding an image into the model are:

1. **Random Resize:** Scale the image randomly within ratio range $[0.5, 2.0]$ preserving aspect ratio.
   * Base scale: $2048 \times 512$ for ADE20K and COCO-Stuff; $2048 \times 1024$ for Cityscapes.
2. **Random Horizontal Flip:** Applied with probability $p = 0.5$.
3. **Random Photometric Distortion:** Sequential random brightness, contrast, saturation, and hue jittering.
4. **Random Crop (with Padding & Category Max Ratio):**
   * **ADE20K (B0–B4):** $512 \times 512$ crop (Batch size $= 16$ across 8 GPUs, i.e., 2 images/GPU).
   * **ADE20K (B5):** $640 \times 640$ crop (Batch size $= 16$).
   * **COCO-Stuff:** $512 \times 512$ crop (Batch size $= 16$).
   * **Cityscapes:** $1024 \times 1024$ crop (Batch size $= 8$ across 8 GPUs, i.e., 1 image/GPU).
   * If the rescaled image is smaller than the crop size, pad the image with ImageNet mean values and pad the ground-truth segmentation mask with the `ignore_index` value ($255$).
5. **ImageNet Normalization:** Subtract mean $\mu = [123.675, 116.28, 103.53]$ and divide by std $\sigma = [58.395, 57.12, 57.375]$ (in $0\text{–}255$ RGB space).

### 6.5 Loss Function & Full-Resolution Supervision
Unlike PSPNet or DeepLabV3 (which require an auxiliary FCN head on Stage 3 with loss weight $0.4$), or MaskFormer (which requires bipartite matching + Dice + Focal loss), SegFormer trains with **a single pixel-wise Cross-Entropy loss** at the output:

$$\mathcal{L}_{\text{seg}} = -\frac{1}{|\Omega|} \sum_{(h, w) \in \Omega} \log \frac{\exp\big(\hat{Y}_{Y_{h, w}, h, w}\big)}{\sum_{k=1}^{K_{\text{cls}}} \exp\big(\hat{Y}_{k, h, w}\big)}$$

* $\hat{Y} \in \mathbb{R}^{B \times K_{\text{cls}} \times H \times W}$ is the $4\times$ bilinearly upsampled logit tensor.
* $Y \in \{0, \dots, K_{\text{cls}}-1, 255\}^{B \times H \times W}$ is the full-resolution ground-truth mask.
* $\Omega = \{(h, w) \mid Y_{h, w} \neq 255\}$ is the set of valid (non-ignored) pixel coordinates.
* **Why upsample logits instead of downsampling the ground-truth mask?** Downsampling $Y$ by $4\times$ via nearest-neighbor interpolation introduces aliasing along object boundaries and erases thin objects ($<4$ pixels wide). Upsampling the logits bilinearly to $H \times W$ forces the $\frac{1}{4}$-scale features to learn sub-pixel linear combinations that reconstruct sharp boundaries.

### 6.6 Inference Protocols: Whole vs. Sliding Window Evaluation
During evaluation, you must match the paper's inference mode for the target dataset:
1. **Whole-Image Inference (ADE20K, COCO-Stuff):**
   * Rescale the test image so its shorter side equals the training crop size ($512$ for B0–B4; $640$ for B5) while keeping the original aspect ratio (divisible by $32$, or dynamically handled by passing $(H, W)$).
   * Run a single forward pass on the entire image.
2. **Sliding-Window Inference (Cityscapes $1024 \times 2048$):**
   * Because Cityscapes crops are trained at $1024 \times 1024$ and full images are $1024 \times 2048$, evaluation uses a **sliding window** of size $1024 \times 1024$ with stride $768 \times 768$.
   * Class probabilities (`softmax(logits, dim=1)`) from overlapping windows are averaged at each pixel before taking `argmax(dim=1)`.
3. **Multi-Scale + Flip Test-Time Augmentation (MS+Flip):**
   * Evaluates at scale ratios $\{0.5, 0.75, 1.0, 1.25, 1.5, 1.75\}$ with and without horizontal flipping ($12$ passes total), averaging the upsampled probability maps.

---

## 7. Key Ablation Studies & Benchmark Summary

### Why Each Design Choice Matters (Paper Ablations on ADE20K)
1. **Positional Encoding vs. Mix-FFN (Table 2 in Paper):**
   * When trained on $512 \times 512$ and tested on Cityscapes at $1024 \times 2048$, a model using learned Positional Embeddings drops by **$-3.3\%$ mIoU** ($74.1\% \rightarrow 70.8\%$) due to positional embedding interpolation.
   * With **Mix-FFN**, the model maintains **$76.2\%$ mIoU** across resolutions without any interpolation penalty.
2. **Decoder Channel Dimension $C_{\text{dec}}$ (Table 1c):**
   * Increasing $C_{\text{dec}}$ from $256 \rightarrow 768$ on larger backbones improves mIoU by $+0.9\%$ while adding only $2.7\text{M}$ parameters. Beyond $C_{\text{dec}} = 768$, gains saturate while FLOPs increase significantly.
3. **Multi-Stage Feature Fusion vs. Single Stage (Table 1d):**
   * Using only Stage 4 ($F_4$, $\frac{1}{32}$ scale) in the decoder yields **$46.3\%$ mIoU** (coarse boundaries).
   * Progressively adding $F_3 \rightarrow F_2 \rightarrow F_1$ boosts mIoU to **$49.0\% \rightarrow 49.8\% \rightarrow 50.3\%$**, proving that Stage 1's $\frac{1}{4}$-scale edge features and Stage 4's $\frac{1}{32}$-scale semantic features are mutually indispensable.

### Final Benchmark Reference Table

| Model | Backbone | Params (M) | ADE20K<br>mIoU (SS / MS) | Cityscapes<br>mIoU (SS / MS) | COCO-Stuff<br>mIoU (SS) |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **FCN** | ResNet-101 | $68.6$ | $39.9\% \text{ / } 41.4\%$ | $75.5\% \text{ / } 76.6\%$ | $32.6\%$ |
| **PSPNet** | ResNet-101 | $68.1$ | $44.4\% \text{ / } 45.4\%$ | $78.5\% \text{ / } 79.8\%$ | $38.9\%$ |
| **DeepLabV3+** | ResNet-101 | $62.7$ | $45.5\% \text{ / } 46.4\%$ | $80.1\% \text{ / } 81.1\%$ | $38.4\%$ |
| **SETR-MLA** | ViT-Large | $310.5$ | $48.6\% \text{ / } 50.3\%$ | $79.3\% \text{ / } 82.2\%$ | — |
| **SegFormer-B0** | MiT-B0 | **$3.8$** | $37.4\% \text{ / } 38.0\%$ | $76.2\% \text{ / } 78.1\%$ | $35.6\%$ |
| **SegFormer-B1** | MiT-B1 | $13.7$ | $42.2\% \text{ / } 43.1\%$ | $78.5\% \text{ / } 80.0\%$ | $40.2\%$ |
| **SegFormer-B2** | MiT-B2 | $27.5$ | $46.5\% \text{ / } 47.5\%$ | $81.0\% \text{ / } 82.2\%$ | $44.6\%$ |
| **SegFormer-B3** | MiT-B3 | $47.3$ | $49.4\% \text{ / } 50.0\%$ | $81.7\% \text{ / } 83.3\%$ | $45.5\%$ |
| **SegFormer-B4** | MiT-B4 | $64.1$ | $50.3\% \text{ / } 51.1\%$ | $82.3\% \text{ / } 83.9\%$ | $46.5\%$ |
| **SegFormer-B5** | MiT-B5 | $84.7$ | **$51.0\% \text{ / } 51.8\%$** | **$82.4\% \text{ / } 84.0\%$** | **$46.7\%$** |

*(SS = Single-Scale evaluation; MS = Multi-Scale + Flip evaluation).*