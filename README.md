# BareTensor

<p align="center"><img src="assets/BareTensor.png" alt="BareTensor" width="358"></p>

[![Python](https://img.shields.io/badge/Python-3.10%2B-blue)](https://www.python.org/)
[![NumPy](https://img.shields.io/badge/NumPy-✓-013243)](https://numpy.org/)
[![PyTorch Parity](https://img.shields.io/badge/PyTorch_Parity-≤_1e⁻⁴-green)](https://github.com/Omnivex3/BareTensor)
[![License](https://img.shields.io/badge/License-MIT-yellow)](LICENSE)
[![PyPI version](https://img.shields.io/pypi/v/baretensor)](https://pypi.org/project/baretensor/)

Autograd engine and deep learning framework in pure Python/NumPy — verified against PyTorch's C++ backend to ≤ 1e⁻⁴.

---

## Installation

```bash
pip install baretensor
```

Requires Python 3.10+ and NumPy ≥ 1.26. **Zero other dependencies.**

---

## Quick Start

```python
from baretensor import Tensor, Linear, Sequential, SGD

# Autograd
x = Tensor([[1.0, 2.0]], requires_grad=True)
w = Tensor([[0.5], [-0.3]], requires_grad=True)
y = x @ w
y.backward()
print(w.grad)  # exact analytical gradient

# Build models
model = Sequential(
    Linear(784, 256),
    Linear(256, 10),
)
```

---

## What's Here

### Autograd Engine
- Dynamic DAG, topological sort, reverse-mode differentiation
- Analytical Jacobians — Softmax Cross-Entropy, LayerNorm, Batched MatMul, GELU, MSE
- Gradient un-broadcasting across batch axes
- Full suite of activations: `relu`, `sigmoid`, `tanh`, `gelu`, `softmax`
- Element-wise math: `exp`, `log`, `+`, `-`, `*`, `/`, `@`, negation

### NN Modules
| Module | Description |
|---|---|
| `Linear` | Fully-connected: y = xW + b |
| `Conv2d` | im2col-based 2D convolution with stride/padding |
| `MaxPool2d` | Max pooling with stride/padding |
| `Dropout` | Inverted dropout regularization |
| `BatchNorm1d` | Batch normalization with running statistics |
| `LayerNorm` | Layer normalization with learnable γ, β |
| `RMSNorm` | LLaMA/Mistral-style RMS normalization (γ only) |
| `Embedding` | Token → dense vector lookup with scatter-add grad |
| `MultiHeadAttention` | Multi-head scaled dot-product attention |
| `TransformerEncoderBlock` | Attention + FFN + residual + LayerNorm |
| `Sequential` | Chain modules in order |

### Functional Ops
| Function | Description |
|---|---|
| `scaled_dot_product_attention(Q, K, V, mask)` | Attention with optional causal masking |
| `layer_norm(x, gamma, beta, eps)` | Layer normalization |
| `cross_entropy_loss(logits, targets)` | Fused softmax + NLL with analytical Jacobian |
| `mse_loss(y_pred, y_true)` | Fused MSE with analytical Jacobian |
| `rope(x, positions, base)` | Rotary Position Embeddings (RoPE) |
| `cat(tensors, axis)` | Concatenate along axis |

### Optimizers
| Optimizer | Features |
|---|---|
| `SGD` | Vanilla stochastic gradient descent |
| `Adam` | Adaptive, bias-corrected, optional L2 weight decay |
| `AdamW` | Decoupled weight decay (Loshchilov & Hutter 2019) |
| `clip_grad_norm_(params, max_norm)` | Global L2 gradient clipping |

### LR Schedulers
| Scheduler | Description |
|---|---|
| `StepLR(opt, step_size, gamma)` | Multiplicative decay every N epochs |
| `CosineAnnealingLR(opt, T_max, eta_min)` | Cosine schedule with warm restart support |

### Data
| Component | Description |
|---|---|
| `Dataset` | Abstract base class |
| `TensorDataset(*tensors)` | Wrap numpy arrays / Tensors |
| `DataLoader(dataset, batch_size, shuffle, drop_last)` | Minibatch iterator |
| `Subset(dataset, indices)` | Index-based dataset view |
| `random_split(dataset, lengths, seed)` | Deterministic train/val/test split |

### Infrastructure
| Feature | Description |
|---|---|
| `Module.save(path)` / `Module.load(path)` | Model checkpointing (`.npz`) |
| `Opt.state_dict()` / `Opt.load_state_dict()` | Optimizer checkpointing with index-stable keys |
| `Module.register_forward_pre_hook(hook)` | Pre-forward hooks |
| `Module.register_forward_hook(hook)` | Post-forward hooks |
| `Module.train()` / `Module.eval()` | Training/eval mode toggle |
| `Module.zero_grad()` | Zero all parameter gradients |

---

## Verified Against PyTorch — 35/35 Tests

| Feature | Test | Parity |
|---|---|---|
| Linear + ReLU autograd | `test_linear_relu_autograd` | ≤ 1e⁻⁵ |
| LayerNorm (2D, 3D, Module) | `test_layer_norm_*` | ≤ 1e⁻⁴ |
| Multi-Head Attention | `test_mha_autograd_parity` | ≤ 1e⁻⁴ |
| Softmax Cross-Entropy | `test_cross_entropy_parity` | ≤ 1e⁻⁴ |
| Causal Masking | `test_causal_mask_parity` | ≤ 1e⁻⁴ |
| Embedding scatter-add | `test_embedding_parity` | ≤ 1e⁻⁴ |
| Batched MatMul | `test_batched_matmul_parity` | ≤ 1e⁻⁴ |
| Reshape | `test_reshape_parity` | ≤ 1e⁻⁵ |
| Dropout | `test_dropout_parity` | ≤ 1e⁻⁶ |
| BatchNorm1d | `test_batchnorm1d_parity` | ≤ 1e⁻⁵ |
| Negation | `test_neg_parity` | ≤ 1e⁻⁶ |
| Division | `test_truediv_parity` | ≤ 1e⁻⁵ |
| Sigmoid | `test_sigmoid_parity` | ≤ 1e⁻⁶ |
| Tanh | `test_tanh_parity` | ≤ 1e⁻⁶ |
| GELU (tanh approx) | `test_gelu_parity` | ≤ 1e⁻⁵ |
| Exp | `test_exp_parity` | ≤ 1e⁻⁵ |
| Log | `test_log_parity` | ≤ 1e⁻⁵ |
| Sequential | `test_sequential_parity` | ≤ 1e⁻⁴ |
| MSE Loss | `test_mse_loss_parity` | ≤ 1e⁻⁶ |
| AdamW | `test_adamw_parity` | ≤ 1e⁻⁶ |
| Gradient Clipping | `test_clip_grad_norm_parity` | ≤ 1e⁻⁶ |
| Conv2d | `test_conv2d_parity` | ≤ 1e⁻⁴ |
| MaxPool2d | `test_maxpool2d_parity` | ≤ 1e⁻⁶ |
| RMSNorm | `test_rmsnorm_parity` | ≤ 1e⁻⁴ |
| RoPE | `test_rope_parity` | ≤ 1e⁻⁵ |
| StepLR | `test_step_lr_parity` | ≤ 1e⁻⁸ |
| CosineAnnealingLR | `test_cosine_lr_parity` | ≤ 1e⁻⁶ |
| Random Split | `test_random_split` | — |
| Optimizer state_dict | `test_optimizer_state_dict` | — |
| Forward Hooks | `test_forward_hooks` | — |

---

## Architecture

```python
class MicroGPT(Module):
    def __init__(self, vocab_size, d_model, num_heads):
        super().__init__()
        self.token_emb = Embedding(vocab_size, d_model)
        self.transformer = TransformerEncoderBlock(d_model, num_heads)
        self.lm_head = Linear(d_model, vocab_size)

    def forward(self, idx, mask=None):
        x = self.token_emb(idx)
        x = self.transformer(x, mask=mask)
        return self.lm_head(x)
```

See [examples/](https://github.com/Omnivex3/BareTensor/tree/main/examples) for full demos: XOR, MNIST (MLP + ConvNet), Transformer, Micro-GPT, Adam vs SGD, CartPole RL.

---

## License

MIT — see [LICENSE](LICENSE).
