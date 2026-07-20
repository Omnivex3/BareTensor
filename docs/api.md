# BareTensor API Reference

**Version:** 0.3.1

This document covers every symbol exported from `baretensor.__init__`. Signatures reflect the actual source code.

---

## baretensor.\_\_version\_\_

```python
baretensor.__version__  # "0.3.1"
```

A string constant.

---

## baretensor.Tensor

The core autograd tensor class. Wraps a NumPy ndarray and records operations in a DAG for automatic differentiation.

### Constructor

```python
Tensor(data, requires_grad: bool = False, parents: tuple = (), dtype=None)
```

| Parameter | Type | Description |
|---|---|---|
| `data` | array-like | Initial data. Converted via `np.array()`. |
| `requires_grad` | bool | If `True`, allocate a gradient buffer and track this tensor in the autograd graph. Non-float data is cast to `float32`. |
| `parents` | tuple of Tensor | Parent tensors in the computation graph. Set automatically by operations; you almost never pass this directly. |
| `dtype` | dtype or None | NumPy dtype for the underlying array. If `None`, inferred from `data`. |

**Raises:** None (NumPy conversion errors propagate).

### Properties and Attributes

| Attribute | Type | Description |
|---|---|---|
| `data` | ndarray | The underlying numpy array. Read/write. |
| `grad` | ndarray or None | Gradient buffer (zero-filled if `requires_grad`, else `None`). |
| `requires_grad` | bool | Whether this tensor participates in autograd. |
| `parents` | tuple of Tensor | Parent tensors in the DAG. |

### Tensor Operations

**Arithmetic operators** -- all support broadcasting via NumPy rules. Non-Tensor scalars are auto-wrapped.

| Operator | Method | Backward behavior |
|---|---|---|
| `+` | `__add__(other)` | Upstream grad passed to both parents (unbroadcast). |
| `-` | `__sub__(other)` | Upstream grad to `self`; negated grad to `other`. |
| `*` | `__mul__(other)` | `grad * other.data` to `self`; `grad * self.data` to `other`. |
| `@` | `__matmul__(other)` | Matrix product via `swapaxes(-1, -2)`. |
| `-` (unary) | `__neg__()` | Negated upstream grad. |
| `/` | `__truediv__(other)` | Quotient rule backward. |
| `+` (reflected) | `__radd__(other)` | Delegates to `__add__`. |
| `-` (reflected) | `__rsub__(other)` | Delegates to `other.__sub__`. |
| `*` (reflected) | `__rmul__(other)` | Delegates to `__mul__`. |
| `@` (reflected) | `__rmatmul__(other)` | Delegates to `__matmul__`. |
| `/` (reflected) | `__rtruediv__(other)` | Delegates to `__truediv__`. |

**Activation functions:**

```python
t.relu()          # -> Tensor    ReLU: max(0, x)
t.sigmoid()       # -> Tensor    Sigmoid: 1 / (1 + exp(-x))
t.tanh()          # -> Tensor    Tanh
t.gelu()          # -> Tensor    GELU (tanh approximation variant)
t.softmax(axis=-1)# -> Tensor    Softmax along axis
```

| Method | Backward behavior |
|---|---|
| `relu()` | `grad * (x > 0)` |
| `sigmoid()` | `grad * sigmoid * (1 - sigmoid)` |
| `tanh()` | `grad * (1 - tanh^2)` |
| `gelu()` | Full analytical gradient of the tanh-approximation GELU. |
| `softmax(axis=-1)` | `probs * (grad - sum(grad * probs))` along axis. |

**Shape operations:**

```python
t.reshape(shape)                    # -> Tensor
t.transpose(axes=None)              # -> Tensor    axes tuple or None (= .T)
```

| Method | Backward behavior |
|---|---|
| `reshape(shape)` | Gradients reshaped back to original shape. |
| `transpose(axes)` | Gradients transposed with inverse permutation. |

**Math functions:**

```python
t.exp()                             # -> Tensor
t.log(eps=1e-7)                     # -> Tensor    Clips: max(x, eps)
t.sum()                             # -> Tensor    Scalar (sum of all elements)
```

| Method | Backward behavior |
|---|---|
| `exp()` | `grad * exp(x)` |
| `log(eps)` | `grad / max(x, eps)` |
| `sum()` | `grad * ones_like(x)` |

**Embedding lookup (used internally by `Embedding` module):**

```python
t.embedding(indices)                # -> Tensor
```

`indices`: integer array of token IDs. Backward uses `np.add.at` for sparse gradient accumulation.

### Autograd

```python
t.backward()
```

Computes gradients via reverse-mode automatic differentiation:
1. Topologically sorts the DAG starting from `t`.
2. Seeds `t.grad = ones_like(t.data)`.
3. Walks the sorted nodes in reverse, calling each `_backward` closure.

**Important:** Gradients are _accumulated_ (added to existing `.grad`). Call `optimizer.zero_grad()` or manually reset `.grad` between iterations.

### Representation

```python
repr(t)  # e.g. "Tensor([1.0, 2.0], requires_grad=True)"
```

---

## baretensor.nn Module

### nn.Module

Base class for all neural network modules.

```python
class Module:
    def __init__(self)
    def forward(self, *args, **kwargs)            # Override in subclasses
    def __call__(self, *args, **kwargs)           # Runs hooks + forward
    def parameters(self) -> list[Tensor]          # Recursively collect params
    def zero_grad(self)                           # Zero all param grads
    def train(self, mode: bool = True)            # Set training mode
    def eval(self)                                # Set eval mode
    def save(self, filepath: str)                 # Save to .npz
    def load(self, filepath: str)                 # Load from .npz
    def register_forward_pre_hook(self, hook)     # -> hook
    def register_forward_hook(self, hook)         # -> hook
```

**Hook signatures:**
```python
pre_hook(module, args) -> tuple | None            # Modify args before forward
hook(module, args, output) -> Tensor | None       # Modify output after forward
```

**Save/load format:** NumPy `.npz` file containing one array per named parameter. Raises `ValueError` on shape mismatch or missing keys.

---

### nn.Linear

```python
Linear(in_features: int, out_features: int)
```

Fully connected layer: `y = x @ W + b`.

| Parameter | Shape | Description |
|---|---|---|
| `in_features` | int | Number of input features. |
| `out_features` | int | Number of output features. |

**Weight init:** He normal: `randn(in, out) * sqrt(2/in)`.

**Parameters:**
- `weight`: Tensor of shape `(in_features, out_features)`
- `bias`: Tensor of shape `(out_features,)`

**Forward:**
```python
def forward(self, x: Tensor) -> Tensor
```
- `x`: `(..., in_features)`
- Returns: `(..., out_features)`

---

### nn.Conv2d

```python
Conv2d(in_channels: int, out_channels: int, kernel_size: int | tuple,
       stride: int | tuple = 1, padding: int | tuple = 0)
```

2D convolution via im2col. Input: `(N, C_in, H, W)`. Output: `(N, C_out, H_out, W_out)`.

| Parameter | Type | Description |
|---|---|---|
| `in_channels` | int | Number of input channels. |
| `out_channels` | int | Number of output filters. |
| `kernel_size` | int or (int, int) | Height and width of kernel. |
| `stride` | int or (int, int) | Convolution stride (default 1). |
| `padding` | int or (int, int) | Zero-padding on each side (default 0). |

**Weight init:** He normal: `randn(out, in, kh, kw) * sqrt(2 / (in * kh * kw))`.

**Parameters:**
- `weight`: Tensor of shape `(out_channels, in_channels, kh, kw)`
- `bias`: Tensor of shape `(out_channels,)`

**Output spatial dimensions:**
```
H_out = (H + 2*ph - kh) // sh + 1
W_out = (W + 2*pw - kw) // sw + 1
```

---

### nn.MaxPool2d

```python
MaxPool2d(kernel_size: int | tuple, stride: int | tuple = None,
          padding: int | tuple = 0)
```

2D max pooling via im2col. When `stride` is `None`, it defaults to `kernel_size`.

**Forward:** `(N, C, H, W)` -> `(N, C, H_out, W_out)`

No learnable parameters.

---

### nn.Dropout

```python
Dropout(p: float = 0.5)
```

Inverted dropout. During training, zeroes elements with probability `p` and scales survivors by `1/(1-p)`. During eval, acts as identity.

**Parameters:** None (no learnable weights).

---

### nn.BatchNorm1d

```python
BatchNorm1d(num_features: int, eps: float = 1e-5, momentum: float = 0.9)
```

Batch normalization for 2D inputs `(N, num_features)`.

| Parameter | Description |
|---|---|
| `num_features` | Number of features (channels). |
| `eps` | Small constant for numerical stability. |
| `momentum` | Running statistics momentum (`running = momentum * running + (1-momentum) * batch`). |

**Training:** Uses batch mean/var, updates running statistics.
**Eval:** Uses `self.running_mean` and `self.running_var` (initialized to zeros and ones).

**Parameters:**
- `gamma`: Tensor of shape `(num_features,)`
- `beta`: Tensor of shape `(num_features,)`

---

### nn.LayerNorm

```python
LayerNorm(normalized_shape: int | tuple, eps: float = 1e-5)
```

Layer normalization over the last dimension. Wraps the functional `layer_norm()`.

| Parameter | Description |
|---|---|
| `normalized_shape` | int or tuple. Shape of the normalization dimension(s). |
| `eps` | Small constant for numerical stability. |

**Parameters:**
- `gamma`: Tensor of shape `normalized_shape`
- `beta`: Tensor of shape `normalized_shape`

---

### nn.RMSNorm

```python
RMSNorm(normalized_shape: int | tuple, eps: float = 1e-6)
```

Root Mean Square Layer Normalization (Zhang & Sennrich, 2019). Used in LLaMA, Mistral, etc. Normalizes by `sqrt(mean(x^2) + eps)` without centering.

**Parameters:**
- `gamma`: Tensor of shape `normalized_shape`

---

### nn.Embedding

```python
Embedding(vocab_size: int, embedding_dim: int)
```

Maps integer token IDs to dense vectors. Internally calls `Tensor.embedding()`.

| Parameter | Description |
|---|---|
| `vocab_size` | Number of tokens in the vocabulary. |
| `embedding_dim` | Dimension of each embedding vector. |

**Weight init:** `randn(vocab_size, embedding_dim) * 0.02`

**Forward:**
```python
def forward(self, indices: ndarray) -> Tensor
```
- `indices`: integer array of token IDs of any shape.
- Returns: Tensor of shape `(*indices.shape, embedding_dim)`.

**Parameters:**
- `weight`: Tensor of shape `(vocab_size, embedding_dim)`

---

### nn.MultiHeadAttention

```python
MultiHeadAttention(d_model: int, num_heads: int)
```

Multi-head scaled dot-product attention with explicit per-head weight matrices (not fused).

| Parameter | Description |
|---|---|
| `d_model` | Total model dimension. Must be divisible by `num_heads`. |
| `num_heads` | Number of attention heads. Head dimension: `d_k = d_model // num_heads`. |

**Attention weights:** Separate `W_q[i]`, `W_k[i]`, `W_v[i]` for each head, each `(d_model, d_k)`.

**Parameters:**
- `self.W_q`: list of `num_heads` Tensors of shape `(d_model, d_k)`
- `self.W_k`: list of `num_heads` Tensors of shape `(d_model, d_k)`
- `self.W_v`: list of `num_heads` Tensors of shape `(d_model, d_k)`
- `self.W_o`: single Tensor of shape `(num_heads * d_k, d_model)`

**Forward:**
```python
def forward(self, X: Tensor, mask: Tensor = None) -> Tensor
```
- `X`: Tensor of shape `(seq_len, d_model)`
- `mask`: optional Tensor of shape `(seq_len, seq_len)` added to attention scores.
- Returns: Tensor of shape `(seq_len, d_model)`

---

### nn.TransformerEncoderBlock

```python
TransformerEncoderBlock(d_model: int, num_heads: int, d_ff: int)
```

A pre-norm transformer encoder block.

```
x -> MHA -> residual -> layer_norm -> FFN(ReLU) -> residual -> layer_norm -> out
```

| Parameter | Description |
|---|---|
| `d_model` | Model dimension. |
| `num_heads` | Number of attention heads. |
| `d_ff` | Feed-forward hidden dimension. |

**Parameters:** (all learnable)
- Sub-parameters of the internal `MultiHeadAttention`
- `gamma1`, `beta1`: LayerNorm after MHA
- `W_f1` `(d_model, d_ff)`, `b_f1` `(d_ff,)`: first FFN projection
- `W_f2` `(d_ff, d_model)`, `b_f2` `(d_model,)`: second FFN projection
- `gamma2`, `beta2`: LayerNorm after FFN

**Forward:**
```python
def forward(self, x: Tensor, mask: Tensor = None) -> Tensor
```
- `x`: Tensor of shape `(seq_len, d_model)`
- Returns: Tensor of shape `(seq_len, d_model)`

---

### nn.Sequential

```python
Sequential(*modules: Module)
```

A sequential container. Modules are called in the order they are passed.

**Forward:**
```python
def forward(self, x: Tensor) -> Tensor
```

Applies each module in order, passing the output of one as input to the next.

---

### nn.scaled_dot_product_attention

```python
scaled_dot_product_attention(Q: Tensor, K: Tensor, V: Tensor,
                              mask: Tensor = None) -> tuple[Tensor, Tensor]
```

Computes `softmax(Q @ K^T / sqrt(d_k)) @ V`.

| Parameter | Shape | Description |
|---|---|---|
| `Q` | `(seq_len, d_k)` | Query tensor. |
| `K` | `(seq_len, d_k)` | Key tensor. |
| `V` | `(seq_len, d_k)` | Value tensor. |
| `mask` | `(seq_len, seq_len)` or None | Optional additive mask. |

**Returns:** `(context, weights)` where both are Tensor of shape `(seq_len, d_k)` / `(seq_len, seq_len)`.

---

### nn.layer_norm

```python
layer_norm(x: Tensor, gamma: Tensor, beta: Tensor,
           eps: float = 1e-5) -> Tensor
```

Functional layer normalization.

| Parameter | Shape | Description |
|---|---|---|
| `x` | `(..., d_model)` | Input tensor. |
| `gamma` | `(d_model,)` | Learnable scale. |
| `beta` | `(d_model,)` | Learnable shift. |
| `eps` | float | Numerical stability constant. |

Normalizes: `gamma * (x - mean) / sqrt(var + eps) + beta` over the last axis.

---

### nn.cross_entropy_loss

```python
cross_entropy_loss(logits: Tensor, targets: ndarray) -> Tensor
```

Cross-entropy loss for classification.

| Parameter | Shape | Description |
|---|---|---|
| `logits` | `(batch_size, num_classes)` | Raw class scores. |
| `targets` | `(batch_size,)` | Integer class labels (numpy array). |

**Returns:** Scalar Tensor.

**Numerics:** Numerically stable softmax (subtracts max before exp). Uses `log(probs + 1e-7)`.

---

### nn.mse_loss

```python
mse_loss(y_pred: Tensor, y_true: Tensor | ndarray) -> Tensor
```

Mean squared error: `mean((y_pred - y_true)^2)`.

| Parameter | Shape | Description |
|---|---|---|
| `y_pred` | any | Prediction tensor. |
| `y_true` | same as `y_pred` | Target (Tensor or ndarray). |

**Returns:** Scalar Tensor.

---

### nn.cat

```python
cat(tensors: list[Tensor], axis: int = -1) -> Tensor
```

Concatenates a list of Tensors along a given axis. All tensors must have the same shape except in the concatenation dimension.

---

### nn.rope

```python
rope(x: Tensor, positions: ndarray, base: float = 10000.0) -> Tensor
```

Rotary Position Embeddings (Su et al., 2023). Rotates pairs of dimensions by frequency-scaled position angles.

| Parameter | Shape | Description |
|---|---|---|
| `x` | `(..., seq_len, d_model)` | Input. `d_model` must be even. |
| `positions` | `(seq_len,)` | Integer positions (numpy array). |
| `base` | float | Frequency base (default 10000.0). |

**Returns:** Tensor of same shape as `x`.

---

## baretensor.optim Module

### optim.SGD

```python
SGD(parameters: Iterable[Tensor], lr: float = 0.01)
```

Stochastic gradient descent: `p.data -= lr * p.grad`.

| Method | Description |
|---|---|
| `step()` | Apply SGD update to all parameters. |
| `zero_grad()` | Zero all parameter gradients. |
| `state_dict()` | Returns `{'lr': self.lr}`. |
| `load_state_dict(sd)` | Restores LR from dict. |

---

### optim.Adam

```python
Adam(parameters: Iterable[Tensor], lr: float = 0.001,
     betas: tuple = (0.9, 0.999), eps: float = 1e-8,
     weight_decay: float = 0.0)
```

Adam optimizer with bias correction (Kingma & Ba, 2015). When `weight_decay > 0`, L2 regularization is applied to the gradient before the Adam step.

| Method | Description |
|---|---|
| `step()` | Apply Adam update. Maintains first/second moment buffers per param. |
| `zero_grad()` | Zero all parameter gradients. |
| `state_dict()` | Serializes LR, betas, eps, weight_decay, timestep, and optimizer state. |
| `load_state_dict(sd)` | Restores all state from dict. |

---

### optim.AdamW

```python
AdamW(parameters: Iterable[Tensor], lr: float = 0.001,
      betas: tuple = (0.9, 0.999), eps: float = 1e-8,
      weight_decay: float = 0.01)
```

Adam with decoupled weight decay (Loshchilov & Hutter, 2019). Unlike Adam with L2 regularization, weight decay is applied directly to the parameters: `p.data *= (1 - lr * wd)` before the Adam step.

---

### optim.clip_grad_norm_

```python
clip_grad_norm_(parameters: Iterable[Tensor], max_norm: float) -> float
```

Clips global gradient norm in-place. Computes `total_norm = sqrt(sum(||grad_i||^2))`. If `total_norm > max_norm`, scales all gradients by `max_norm / total_norm`.

**Returns:** Total L2 norm before clipping (Python float).

---

### optim.StepLR

```python
StepLR(optimizer, step_size: int, gamma: float = 0.1)
```

Decays LR by `gamma` every `step_size` epochs.

| Method | Description |
|---|---|
| `step()` | Advances one epoch, updates optimizer LR. |
| `get_lr()` | Returns current learning rate. |

---

### optim.CosineAnnealingLR

```python
CosineAnnealingLR(optimizer, T_max: int, eta_min: float = 0.0)
```

Cosine annealing LR schedule (Loshchilov & Hutter, 2017). LR follows a cosine curve from `base_lr` down to `eta_min` over `T_max` epochs.

| Method | Description |
|---|---|
| `step()` | Advances one epoch, updates optimizer LR. |
| `get_lr()` | Returns current learning rate. |

---

## baretensor.data Module

### data.Dataset

```python
class Dataset:
    def __len__(self) -> int
    def __getitem__(self, idx) -> any
```

Abstract base class. Subclasses must override both methods.

---

### data.TensorDataset

```python
TensorDataset(*tensors: ndarray | Tensor)
```

Wraps numpy arrays or baretensor Tensors into a dataset indexed along dimension 0.

All tensors must have the same size in dimension 0. Non-Tensor inputs are converted to `float32` (integer/boolean dtypes preserved).

**Indexing:** Returns `tuple(t[idx] for t in self.tensors)`.

---

### data.DataLoader

```python
DataLoader(dataset: Dataset, batch_size: int = 1,
           shuffle: bool = False, drop_last: bool = False)
```

Single-process mini-batch iterator.

| Parameter | Description |
|---|---|
| `dataset` | A `Dataset` instance. Must have `.tensors` attribute (e.g. `TensorDataset`). |
| `batch_size` | Samples per batch. Must be >= 1. |
| `shuffle` | If `True`, permute indices at each epoch. |
| `drop_last` | If `True`, drop the last incomplete batch. |

**Iteration:** Yields tuples of numpy arrays (one per tensor in the dataset), sliced along dimension 0.

```python
len(dataloader)  # Number of batches per epoch
```

---

### data.Subset

```python
Subset(dataset: Dataset, indices: Sequence[int])
```

Indexed view into a parent dataset.

```python
len(subset)                         # len(indices)
subset[i]                           # dataset[indices[i]]
```

---

### data.random_split

```python
random_split(dataset: Dataset, lengths: list[int | float],
             seed: int = 42) -> list[Subset]
```

Randomly splits a dataset into non-overlapping subsets.

| Parameter | Description |
|---|---|
| `dataset` | Dataset to split. |
| `lengths` | List of subset sizes (integers), or fractions summing to 1.0. |
| `seed` | Random seed for reproducibility (via `np.random.RandomState`). |

**Returns:** list of `Subset` instances.

**Raises:** `AssertionError` if integer lengths don't sum to `len(dataset)`.
