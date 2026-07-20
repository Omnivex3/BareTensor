# Getting Started with BareTensor

**Version:** 0.3.1

---

## Installation

```bash
pip install baretensor
```

**Requirements:**
- Python 3.10+
- NumPy (any recent version)

**No GPU required.** BareTensor is CPU-only.

Verify installation:

```python
import baretensor
print(baretensor.__version__)   # 0.3.1
```

---

## Quick Tour

### 1. Create a Tensor

```python
from baretensor import Tensor

x = Tensor([1.0, 2.0, 3.0], requires_grad=True)
print(x)
# Tensor([1.0, 2.0, 3.0], requires_grad=True)

print(x.data)   # numpy array: [1. 2. 3.]
print(x.grad)   # zero buffer: [0. 0. 0.]
```

### 2. Perform Operations

```python
y = x * 2 + 1       # All operators participate in autograd
z = y.sum()
print(z)            # Tensor(11.0, requires_grad=True)
```

### 3. Backward Pass

```python
z.backward()
print(x.grad)       # [2. 2. 2.]
```

Derivative: `d(z)/d(x_i) = 2` for each `i`.

### 4. Reset Gradients

```python
x.grad = None
# or use an optimizer's zero_grad method
```

---

## Minimal Training Loop

```python
import numpy as np
from baretensor import Tensor
from baretensor.nn import Linear, MSELoss  # mse_loss is functional
from baretensor.optim import SGD

# Data
X = Tensor(np.random.randn(100, 10))
y = Tensor(np.random.randn(100, 1))

# Model
model = Linear(10, 1)
optimizer = SGD(model.parameters(), lr=0.01)

# Training loop
for epoch in range(100):
    y_pred = model(X)
    loss = mse_loss(y_pred, y)

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    if epoch % 10 == 0:
        print(f"Epoch {epoch}, loss: {loss.data:.4f}")
```

**Note:** `mse_loss` is imported as `from baretensor.nn import mse_loss` (a function, not a class).

---

## PyTorch User Map

| PyTorch | BareTensor | Notes |
|---|---|---|
| `torch.Tensor` | `baretensor.Tensor` | Same basic API. No `.to(device)`. |
| `torch.nn.Module` | `baretensor.nn.Module` | Same structure. `register_buffer` maps to plain attribute. |
| `torch.nn.Linear` | `baretensor.nn.Linear` | Same init. He init instead of default uniform. |
| `torch.nn.Conv2d` | `baretensor.nn.Conv2d` | im2col-based. Slower but numerically equivalent. |
| `torch.nn.MaxPool2d` | `baretensor.nn.MaxPool2d` | im2col-based. |
| `torch.nn.Dropout` | `baretensor.nn.Dropout` | Inverted dropout (same as PyTorch behavior). |
| `torch.nn.BatchNorm1d` | `baretensor.nn.BatchNorm1d` | Same running stats. |
| `torch.nn.LayerNorm` | `baretensor.nn.LayerNorm` | Same computation. |
| `torch.nn.RMSNorm` | `baretensor.nn.RMSNorm` | Same as LLaMA-style RMSNorm. |
| `torch.nn.Embedding` | `baretensor.nn.Embedding` | Same forward/backward. |
| `torch.nn.MultiheadAttention` | `baretensor.nn.MultiHeadAttention` | Non-fused (separate W per head). |
| `torch.nn.TransformerEncoderLayer` | `baretensor.nn.TransformerEncoderBlock` | Pre-norm, same components. |
| `torch.nn.Sequential` | `baretensor.nn.Sequential` | Same usage. |
| `torch.nn.functional.scaled_dot_product_attention` | `baretensor.nn.scaled_dot_product_attention` | Same signature minus dropout. |
| `torch.nn.functional.cross_entropy` | `baretensor.nn.cross_entropy_loss` | Expects targets as numpy array (not Tensor). |
| `torch.nn.functional.mse_loss` | `baretensor.nn.mse_loss` | Same. |
| `torch.cat` | `baretensor.nn.cat` | Same. |
| `torch.optim.SGD` | `baretensor.optim.SGD` | Fewer options (no momentum/Nesterov). |
| `torch.optim.Adam` | `baretensor.optim.Adam` | Same defaults. |
| `torch.optim.AdamW` | `baretensor.optim.AdamW` | Same defaults, decoupled weight decay. |
| `torch.nn.utils.clip_grad_norm_` | `baretensor.optim.clip_grad_norm_` | Same. |
| `torch.optim.lr_scheduler.StepLR` | `baretensor.optim.StepLR` | Same. |
| `torch.optim.lr_scheduler.CosineAnnealingLR` | `baretensor.optim.CosineAnnealingLR` | Same. |
| `torch.utils.data.Dataset` | `baretensor.data.Dataset` | Same abstract interface. |
| `torch.utils.data.TensorDataset` | `baretensor.data.TensorDataset` | Same. |
| `torch.utils.data.DataLoader` | `baretensor.data.DataLoader` | Single-process only. |
| `torch.utils.data.Subset` | `baretensor.data.Subset` | Same. |
| `torch.utils.data.random_split` | `baretensor.data.random_split` | Same, with `seed=` parameter. |
| `tensor.backward()` | `tensor.backward()` | Same (accumulates gradients). |
| `tensor.detach()` | No direct equivalent | Use `Tensor(x.data)` instead. |
| `tensor.view()` | `tensor.reshape()` | Same semantics. |
| `tensor.transpose()` | `tensor.transpose()` | Same. |
| `F.relu()` | `tensor.relu()` | Method, not function. |
| `reduction='mean'` | Always mean | Cross-entropy and MSE always average over the batch. |

**Key differences at a glance:**

| Capability | PyTorch | BareTensor |
|---|---|---|
| GPU | Yes | No |
| Mixed precision | Yes | No |
| Distributed training | Yes | No |
| In-place ops | Yes | No |
| Line count | Millions | ~800 |
| Install size | GBs | ~10 KB |
| Autograd graph | Dynamic tape | DAG + topological sort |
