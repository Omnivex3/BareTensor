# BareTensor: Architecture & Design

**Version:** 0.3.1 | **License:** MIT

BareTensor is a pure Python/NumPy autograd engine that provides a PyTorch-analogous tensor compute graph with automatic differentiation. It is designed for education, prototyping, and environments where CUDA/C extensions are unavailable.

---

## 1. Core Design Philosophy

BareTensor makes three deliberate design choices:

1. **Pure Python + NumPy** -- No C/CUDA extensions, no build step. Every operation runs through NumPy, making the library trivially installable (`pip install baretensor`) and debuggable with standard Python tools.
2. **DAG-based autograd** -- Each `Tensor` records its operation history in a directed acyclic graph. Calling `.backward()` topologically sorts the graph and propagates gradients via the chain rule.
3. **PyTorch-analogous API** -- The API surface mirrors PyTorch (Tensor, nn.Module, optim.SGD/Adam, DataLoader) to minimize friction for users familiar with PyTorch.

**Numerical guarantee:** All operations are verifiable against PyTorch to within 1e-4 tolerance.

---

## 2. The Tensor Class

### 2.1 Data Layout

```python
class Tensor:
    def __init__(self, data, requires_grad=False, parents=(), dtype=None):
```

- **Storage:** Every Tensor wraps a NumPy `ndarray` in `self.data`.
- **Shape/strides:** Inherited from the underlying ndarray. No custom memory layout.
- **dtype coercion:** If `requires_grad=True` and the input is not float, it is cast to `np.float32`. Providing `dtype` explicitly bypasses this heuristic.
- **Gradient buffer:** If `requires_grad=True`, `self.grad` is initialized as a zero array of the same shape as `self.data`.

### 2.2 The Autograd Graph

Each Tensor stores:
- `self.parents` -- a tuple of parent Tensors that produced it.
- `self._backward` -- a closure that implements the local gradient of the operation.

When an operation creates a new Tensor, it:
1. Runs the forward computation via NumPy.
2. Records `parents=(a, b)`.
3. Defines `_backward` which accumulates gradients into `a.grad` and `b.grad` using the chain rule.
4. Assigns `requires_grad=True` on the output.

Example (`__add__`):

```python
def __add__(self, other):
    other = other if isinstance(other, Tensor) else Tensor(other)
    out = Tensor(self.data + other.data, parents=(self, other), requires_grad=True)

    def _backward():
        if self.requires_grad:
            self.grad += self._unbroadcast(out.grad, self.data.shape)
        if other.requires_grad:
            other.grad += self._unbroadcast(out.grad, other.data.shape)

    out._backward = _backward
    return out
```

### 2.3 Gradient Broadcasting

The `_unbroadcast` helper handles NumPy-style broadcasting in reverse:

```python
def _unbroadcast(self, grad, target_shape):
    ndims_added = len(grad.shape) - len(target_shape)
    for _ in range(ndims_added):
        grad = grad.sum(axis=0)
    for i, dim in enumerate(target_shape):
        if dim == 1 and grad.shape[i] > 1:
            grad = grad.sum(axis=i, keepdims=True)
    return grad
```

This sums over dimensions that were broadcast, correctly reducing the gradient back to the shape of the parent.

### 2.4 How backward() Works

```python
def backward(self):
    topo = []
    visited = set()

    def build_topo(v):
        if v not in visited:
            visited.add(v)
            for parent in v.parents:
                build_topo(parent)
            topo.append(v)

    build_topo(self)
    self.grad = np.ones_like(self.data)        # seed: d(loss)/d(loss) = 1
    for node in reversed(topo):
        node._backward()                        # chain rule
```

Steps:
1. **Topological sort** -- Starting from the output (typically the loss), recursively visit all parent Tensors, appending each to `topo` only after its parents are processed. This produces a linear ordering where every node appears before all nodes that depend on it.
2. **Seed** -- Set the output's gradient to 1 (the derivative of the loss with respect to itself).
3. **Reverse traversal** -- Walk `topo` in reverse order, calling each node's `_backward` closure. Each closure computes local gradients and accumulates (`+=`) them into the `grad` arrays of its parent nodes.

**Key properties:**
- Gradient accumulation (not assignment) supports fan-out: when a Tensor is used in multiple operations, all gradient contributions are summed.
- The graph is consumed after a single backward pass; Tensor does not reset the graph automatically -- call `zero_grad()` between iterations.

---

## 3. Memory Model

- **No arena allocator.** BareTensor relies entirely on Python's garbage collector. Each Tensor owns its `data` and `grad` ndarrays.
- **No tensor pooling.** Tensors are created and freed per operation.
- **Graph memory.** The parent references and `_backward` closures keep the entire computation graph alive until the output Tensor is garbage collected.
- **Practical implication:** Long training loops with large models may accumulate Python memory pressure. Call `del loss` or let Tensors fall out of scope between iterations.

---

## 4. Module System

### 4.1 Module Base Class

```python
class Module:
    def __init__(self):
        self._modules = {}
        self.training = True
        self._forward_pre_hooks = []
        self._forward_hooks = []
```

**Automatic child tracking.** When a `Module` attribute is assigned (e.g. `self.linear = Linear(...)`), `__setattr__` detects `Module` instances and registers them in `self._modules`. This enables recursive parameter collection.

**Key methods:**

| Method | Purpose |
|---|---|
| `parameters()` | Recursively collects all `Tensor` parameters from child modules. Override in leaf modules to return their learnable weights. |
| `zero_grad()` | Zeros the grad of every parameter. |
| `train()/eval()` | Sets `self.training` flag on the module and all children. |
| `__call__()` | Runs forward hooks, then `forward()`, then backward hooks. |
| `save()/load()` | Serializes/deserializes all named parameters via `np.savez` / `np.load`. |
| `register_forward_pre_hook()` | Registers a callable invoked before `forward()`. |
| `register_forward_hook()` | Registers a callable invoked after `forward()`. |

### 4.2 Forward Hooks API

```python
def register_forward_pre_hook(self, hook):
    # hook(module, args) -> modified_args or None
    self._forward_pre_hooks.append(hook)
    return hook

def register_forward_hook(self, hook):
    # hook(module, args, output) -> modified_output or None
    self._forward_hooks.append(hook)
    return hook
```

Hooks are called in registration order. If a pre-hook returns a non-None value, it replaces the `args` tuple. If a forward hook returns a non-None value, it replaces the output.

### 4.3 Leaf Modules

Leaf modules (Linear, Conv2d, etc.) override `parameters()` to return a list of their learnable `Tensor` attributes. All are initialized with `requires_grad=True`.

- **Weight initialization:** Linear and Conv2d use He initialization (`np.sqrt(2.0 / fan_in)`). Embedding and transformer weights use `* 0.02` or `* 0.1`.

### 4.4 Module Containers

`Sequential` takes `*modules` and applies them in order. Child modules are stored in `self.layers` and registered via `setattr(f'_{i}', module)` for automatic parameter discovery.

---

## 5. Layer Implementations

### 5.1 Conv2d (im2col-based)

BareTensor implements 2D convolution via the **im2col** algorithm:

1. **im2col:** Extract sliding-window patches from the 4D input `(N, C, H, W)` into a 2D column matrix of shape `(N * out_h * out_w, C * kh * kw)`. Uses `np.lib.stride_tricks.as_strided` for zero-copy patch extraction.
2. **GEMM:** Flatten the weight to `(out_channels, C * kh * kw)` and perform matrix multiply: `cols @ W^T`.
3. **Reshape:** Reshape and transpose the result back to `(N, out_channels, out_h, out_w)`.
4. **col2im (backward):** The gradient with respect to the input is computed by reversing the im2col transform, scattering column gradients back into the original spatial layout.

### 5.2 MaxPool2d (im2col-based)

Max pooling uses the same im2col approach: extract patches, find the argmax within each channel's patch, and select those values. The backward pass scatter-adds gradients at the argmax positions.

### 5.3 MultiHeadAttention

Attention is implemented with explicit per-head weight matrices (not fused). Each head computes `scaled_dot_product_attention`, outputs are concatenated and projected through `W_o`.

### 5.4 TransformerEncoderBlock

A pre-norm-style transformer block:
1. Multi-head self-attention + residual.
2. Layer normalization (functional `layer_norm` with learnable per-element gamma/beta).
3. Position-wise FFN: `ReLU(x @ W1 + b1) @ W2 + b2`.
4. Layer normalization + residual.

### 5.5 RMSNorm

LLaMA-style RMS normalization: normalizes by `sqrt(mean(x^2) + eps)` without centering, then scales by learnable `gamma`.

### 5.6 RoPE (Rotary Position Embeddings)

The `rope()` function applies rotary position embeddings by rotating pairs of dimensions by frequency-scaled position angles: for each pair `(x_even, x_odd)` at position `pos`, the rotation is:

```
rot_even = x_even * cos(pos * theta) - x_odd * sin(pos * theta)
rot_odd  = x_even * sin(pos * theta) + x_odd * cos(pos * theta)
```

where `theta_i = base^{-2i/d}` for `i = 0, ..., d/2 - 1`.

---

## 6. Optimizers

| Optimizer | Algorithm | Key Features |
|---|---|---|
| `SGD` | Stochastic Gradient Descent | `p -= lr * grad` |
| `Adam` | Adaptive Moment Estimation | Bias correction, optional weight decay in gradient |
| `AdamW` | Decoupled Adam | Weight decay applied directly to params, not mixed into gradient |

All optimizers implement:
- `step()` -- update parameters using stored gradients.
- `zero_grad()` -- zero out all parameter gradients.
- `state_dict()` / `load_state_dict()` -- serialization for checkpointing.

### Learning Rate Schedulers

- `StepLR` -- multiplies LR by `gamma` every `step_size` epochs.
- `CosineAnnealingLR` -- cosine annealing from `base_lr` to `eta_min` over `T_max` epochs.

### Gradient Clipping

`clip_grad_norm_(parameters, max_norm)` computes the global L2 norm and scales all gradients if it exceeds `max_norm`. Returns the total norm before clipping.

---

## 7. Data Pipeline

| Class | Purpose |
|---|---|
| `Dataset` | Abstract base class requiring `__len__` and `__getitem__`. |
| `TensorDataset` | Wraps numpy arrays / Tensors into a dataset indexed along dim 0. |
| `DataLoader` | Single-process mini-batch iterator with shuffle and `drop_last`. Yields tuples of numpy arrays. |
| `Subset` | Indexed view into a parent dataset. |
| `random_split` | Deterministic seeded split into non-overlapping subsets. |

---

## 8. Design Tradeoffs vs PyTorch

| Feature | BareTensor | PyTorch | Implication |
|---|---|---|---|
| **Backend** | NumPy | ATen/CUDA | BareTensor is CPU-only, 10-100x slower for large models. |
| **Autograd** | Topological sort + closures | Dynamic tape (`autograd.grad`) | BareTensor's graph is simpler but can't handle in-place ops or graph modification. |
| **Gradient accumulation** | `+=` always | `+=` or `=` (depending on `retain_graph`) | BareTensor always accumulates; user must call `zero_grad()`. |
| **Memory** | Python GC | Reference counting + custom allocator | BareTensor has higher memory overhead. |
| **Device support** | CPU only | CPU/CUDA/MPS/XPU | No `.to('cuda')` in BareTensor. |
| **Sparse / quantized** | Not supported | Full support | BareTensor targets dense float32 only. |
| **JIT compilation** | None | torch.compile / TorchScript | No graph optimization. |
| **Numerical precision** | float32 only | float16/32/64/bfloat16 | Mixed precision not available. |
| **Serialization** | np.savez (weights only) | torch.save (full state) | No optimizer state in BareTensor's native format. |
| **Broadcasting** | NumPy rules | NumPy-compatible | Identical behavior. |
| **Install** | `pip install baretensor` | Multi-GB CUDA toolkit | BareTensor is ~10 KB and has no native dependencies beyond NumPy. |
| **Code readability** | ~800 lines total | Millions | BareTensor can be read in full in an afternoon. |

---

## 9. When to Use BareTensor

**Good fits:**
- Learning how autograd works (the entire library is ~800 lines).
- Prototyping small models (MLP, CNN, simple transformer).
- Teaching deep learning fundamentals.
- Environments where installing PyTorch is impractical.
- Reproducing research results at small scale.

**Poor fits:**
- Large-scale training (ImageNet, LLMs).
- Production inference.
- GPU-accelerated workloads.
- Models requiring advanced features (mixed precision, distributed, sparse).
