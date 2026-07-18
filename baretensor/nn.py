import numpy as np
from .tensor import Tensor

class Module:
    """Base class for all neural network modules.

    Child modules assigned via self.xxx = Module() are automatically
    tracked for parameter collection.
    """

    def __init__(self):
        self._modules = {}
        self.training = True
        self._forward_pre_hooks = []
        self._forward_hooks = []

    def __setattr__(self, name, value):
        if isinstance(value, Module):
            self._modules[name] = value
        super().__setattr__(name, value)

    def train(self, mode=True):
        self.training = mode
        for child in self._modules.values():
            child.train(mode)

    def eval(self):
        self.train(False)

    def parameters(self):
        params = []
        for child in self._modules.values():
            params.extend(child.parameters())
        return params

    def zero_grad(self):
        """Zero all parameter gradients."""
        for param in self.parameters():
            param.grad = np.zeros_like(param.data)

    def _param_names(self, prefix=''):
        """Walk attributes to build named (key, param) pairs."""
        result = []
        for attr_name in sorted(vars(self)):
            attr = getattr(self, attr_name)
            full_name = f"{prefix}{attr_name}" if prefix else attr_name
            if isinstance(attr, Tensor) and attr.requires_grad:
                result.append((full_name, attr))
            elif isinstance(attr, Module):
                result.extend(attr._param_names(prefix=f"{full_name}."))
            elif isinstance(attr, list):
                for i, item in enumerate(attr):
                    indexed_name = f"{full_name}.{i}"
                    if isinstance(item, Tensor) and item.requires_grad:
                        result.append((indexed_name, item))
                    elif isinstance(item, Module):
                        result.extend(item._param_names(prefix=f"{indexed_name}."))
        return result

    def save(self, filepath):
        named_params = {name: param.data for name, param in self._param_names()}
        np.savez(filepath, **named_params)
        print(f"Model saved to {filepath}.npz ({len(named_params)} parameters)")

    def load(self, filepath):
        if not filepath.endswith('.npz'):
            filepath += '.npz'
        named_params = dict(self._param_names())
        with np.load(filepath) as data:
            for name, param in named_params.items():
                if name not in data:
                    raise ValueError(f"Missing parameter '{name}' in checkpoint")
                saved = data[name]
                if param.data.shape != saved.shape:
                    raise ValueError(f"Shape mismatch for '{name}': expected {param.data.shape}, got {saved.shape}")
                param.data = saved
        print(f"Model loaded from {filepath}")

    def __call__(self, *args, **kwargs):
        for hook in self._forward_pre_hooks:
            result = hook(self, args)
            if result is not None:
                args = result if isinstance(result, tuple) else (result,)

        output = self.forward(*args, **kwargs)

        for hook in self._forward_hooks:
            result = hook(self, args, output)
            if result is not None:
                output = result

        return output

    def forward(self, *args, **kwargs):
        raise NotImplementedError(f"{type(self).__name__} must implement forward()")

    def register_forward_pre_hook(self, hook):
        """Register a hook called before forward. hook(module, args) -> modified_args or None."""
        self._forward_pre_hooks.append(hook)
        return hook

    def register_forward_hook(self, hook):
        """Register a hook called after forward. hook(module, args, output) -> modified_output or None."""
        self._forward_hooks.append(hook)
        return hook

class Linear(Module):
    """Fully connected linear layer: y = x @ W + b."""

    def __init__(self, in_features, out_features):
        super().__init__()
        self.weight = Tensor(
            np.random.randn(in_features, out_features) * np.sqrt(2.0 / in_features),
            requires_grad=True,
        )
        self.bias = Tensor(np.zeros(out_features), requires_grad=True)

    def forward(self, x):
        return x @ self.weight + self.bias

    def parameters(self):
        return [self.weight, self.bias]

class Dropout(Module):
    """Inverted dropout regularization layer.

    During training: randomly zeroes elements with probability p and scales
    the remaining by 1/(1-p). During eval: acts as identity.
    """

    def __init__(self, p=0.5):
        super().__init__()
        self.p = p
        self.mask = None

    def forward(self, x):
        if not self.training:
            return x
        scale = 1.0 / (1.0 - self.p)
        self.mask = (np.random.rand(*x.data.shape) > self.p).astype(np.float32) * scale
        out_data = x.data * self.mask
        out = Tensor(out_data, parents=(x,), requires_grad=True)

        def _backward():
            x.grad += out.grad * self.mask

        out._backward = _backward
        return out


class Sequential(Module):
    """A sequential container of modules, called in order.

    Modules are automatically tracked for parameter collection.
    """

    def __init__(self, *modules):
        super().__init__()
        self.layers = list(modules)
        for i, module in enumerate(modules):
            setattr(self, f'_{i}', module)

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x

    def parameters(self):
        params = []
        for layer in self.layers:
            params.extend(layer.parameters())
        return params


class BatchNorm1d(Module):
    """Batch Normalization for 2D inputs (batch, features).

    Normalizes each feature to have zero mean and unit variance across the batch,
    then applies learnable affine transform (gamma * x_hat + beta).

    During training: uses batch statistics.
    During eval: uses running mean/variance.
    """

    def __init__(self, num_features, eps=1e-5, momentum=0.9):
        super().__init__()
        self.num_features = num_features
        self.eps = eps
        self.momentum = momentum

        # Learnable parameters
        self.gamma = Tensor(np.ones(num_features), requires_grad=True)
        self.beta = Tensor(np.zeros(num_features), requires_grad=True)

        # Running statistics (not learnable, no grad)
        self.running_mean = np.zeros(num_features, dtype=np.float32)
        self.running_var = np.ones(num_features, dtype=np.float32)

    def parameters(self):
        return [self.gamma, self.beta]

    def forward(self, x):
        if self.training:
            # Compute batch statistics
            batch_mean = np.mean(x.data, axis=0)  # shape: (num_features,)
            batch_var = np.var(x.data, axis=0)    # shape: (num_features,)

            # Update running statistics
            self.running_mean = self.momentum * self.running_mean + (1 - self.momentum) * batch_mean
            self.running_var = self.momentum * self.running_var + (1 - self.momentum) * batch_var

            # Normalize using batch stats
            mu = batch_mean
            var = batch_var
        else:
            mu = self.running_mean
            var = self.running_var

        N = x.data.shape[0]  # batch size
        std = np.sqrt(var + self.eps)
        x_hat = (x.data - mu) / std
        out_data = self.gamma.data * x_hat + self.beta.data
        out = Tensor(out_data, parents=(x, self.gamma, self.beta), requires_grad=True)

        def _backward():
            dy = out.grad  # shape: (N, C)

            if self.gamma.requires_grad:
                self.gamma.grad += np.sum(dy * x_hat, axis=0)
            if self.beta.requires_grad:
                self.beta.grad += np.sum(dy, axis=0)
            if x.requires_grad:
                # Gradient through batch normalization
                # Reference: https://arxiv.org/abs/1502.03167
                N = x.data.shape[0]
                dx_hat = dy * self.gamma.data
                # Sum over batch for the mean of dx_hat and dx_hat * x_hat
                sum_dx_hat = np.sum(dx_hat, axis=0)
                sum_dx_hat_x_hat = np.sum(dx_hat * x_hat, axis=0)
                dx = (1.0 / N) * (1.0 / std) * (
                    N * dx_hat - sum_dx_hat - x_hat * sum_dx_hat_x_hat
                )
                x.grad += dx

        out._backward = _backward
        return out


class LayerNorm(Module):
    """Layer Normalization with learnable affine parameters.

    Normalizes over the last dimension of the input.
    Wraps the functional :func:`layer_norm` for stateful use in modules.
    """

    def __init__(self, normalized_shape, eps=1e-5):
        super().__init__()
        if isinstance(normalized_shape, int):
            normalized_shape = (normalized_shape,)
        self.normalized_shape = normalized_shape
        self.eps = eps
        self.gamma = Tensor(np.ones(normalized_shape), requires_grad=True)
        self.beta = Tensor(np.zeros(normalized_shape), requires_grad=True)

    def forward(self, x):
        return layer_norm(x, self.gamma, self.beta, self.eps)

    def parameters(self):
        return [self.gamma, self.beta]

def cat(tensors, axis=-1):
    """Concatenates a list of Tensors along a specified axis."""
    out_data = np.concatenate([t.data for t in tensors], axis=axis)
    out = Tensor(out_data, parents=tuple(tensors), requires_grad=True)

    def _backward():
        split_sizes = [t.data.shape[axis] for t in tensors]
        indices = np.cumsum(split_sizes)[:-1]
        grads = np.split(out.grad, indices, axis=axis)
        for i, t in enumerate(tensors):
            if t.requires_grad:
                t.grad += grads[i]

    out._backward = _backward
    return out

def scaled_dot_product_attention(Q, K, V, mask=None):
    """
    Q: Query tensor (Sequence_Length x Hidden_Dim)
    K: Key tensor (Sequence_Length x Hidden_Dim)
    V: Value tensor (Sequence_Length x Hidden_Dim)
    mask: Optional Tensor of shape (Sequence_Length x Sequence_Length)
    """
    d_k = Q.data.shape[-1]
    k_t_axes = tuple(range(K.data.ndim - 2)) + (K.data.ndim - 1, K.data.ndim - 2)
    scores = Q @ K.transpose(k_t_axes)
    scaled_scores = scores * (1.0 / np.sqrt(d_k))
    if mask is not None:
        scaled_scores = scaled_scores + mask
    weights = scaled_scores.softmax(axis=-1)
    context = weights @ V
    return context, weights


def layer_norm(x, gamma, beta, eps=1e-5):
    """
    x: Tensor of shape (Sequence, d_model)
    gamma: Tensor of shape (d_model,) - Learnable scale
    beta: Tensor of shape (d_model,) - Learnable shift
    """
    mu = np.mean(x.data, axis=-1, keepdims=True)
    var = np.var(x.data, axis=-1, keepdims=True)
    std = np.sqrt(var + eps)
    x_hat = (x.data - mu) / std
    out_data = gamma.data * x_hat + beta.data
    out = Tensor(out_data, parents=(x, gamma, beta), requires_grad=True)

    def _backward():
        dy = out.grad
        if gamma.requires_grad:
            sum_axes = tuple(range(dy.ndim - 1))
            gamma.grad += np.sum(dy * x_hat, axis=sum_axes)
        if beta.requires_grad:
            sum_axes = tuple(range(dy.ndim - 1))
            beta.grad += np.sum(dy, axis=sum_axes)
        if x.requires_grad:
            dx_hat = dy * gamma.data
            mean_dx_hat = np.mean(dx_hat, axis=-1, keepdims=True)
            mean_dx_hat_x_hat = np.mean(dx_hat * x_hat, axis=-1, keepdims=True)
            dx = (1.0 / std) * (dx_hat - mean_dx_hat - x_hat * mean_dx_hat_x_hat)
            x.grad += dx

    out._backward = _backward
    return out

def cross_entropy_loss(logits, targets):
    """
    Cross-entropy loss for classification.

    logits: Tensor of shape (Batch_Size, Num_Classes)
    targets: NumPy array of shape (Batch_Size,) with integer class labels
    """
    N, C = logits.data.shape

    # Numerically stable softmax
    shifted = logits.data - np.max(logits.data, axis=-1, keepdims=True)
    exp_shifted = np.exp(shifted)
    probs = exp_shifted / np.sum(exp_shifted, axis=-1, keepdims=True)

    # Log probabilities with epsilon for numerical stability
    log_probs = np.log(probs + 1e-7)

    # Cross-entropy: pick log-prob of the correct class, average over batch
    loss_val = -np.mean(log_probs[np.arange(N), targets])

    out = Tensor(np.array(loss_val), parents=(logits,), requires_grad=True)

    def _backward():
        if logits.requires_grad:
            one_hot = np.zeros_like(probs)
            one_hot[np.arange(N), targets] = 1.0
            logits.grad += (probs - one_hot) / N * out.grad

    out._backward = _backward
    return out


def mse_loss(y_pred, y_true):
    """Mean Squared Error loss with analytical Jacobian.

    Args:
        y_pred: Tensor of shape (...)
        y_true: numpy array or Tensor of same shape

    Returns:
        Scalar Tensor with MSE loss value.
    """
    if not isinstance(y_true, Tensor):
        y_true = Tensor(y_true)
    diff_data = y_pred.data - y_true.data
    N = diff_data.size
    out = Tensor(np.array(np.mean(diff_data ** 2)), parents=(y_pred, y_true), requires_grad=True)

    def _backward():
        if y_pred.requires_grad:
            y_pred.grad += (2.0 / N) * diff_data * out.grad
        if y_true.requires_grad:
            y_true.grad += (-2.0 / N) * diff_data * out.grad

    out._backward = _backward
    return out

class MultiHeadAttention(Module):
    def __init__(self, d_model, num_heads):
        super().__init__()
        self.num_heads = num_heads
        self.d_k = d_model // num_heads
        assert d_model % num_heads == 0, "d_model must be divisible by num_heads"

        # Explicitly create separate weights for each head to be visually clear
        self.W_q = [Tensor(np.random.randn(d_model, self.d_k) * 0.1, requires_grad=True) for _ in range(num_heads)]
        self.W_k = [Tensor(np.random.randn(d_model, self.d_k) * 0.1, requires_grad=True) for _ in range(num_heads)]
        self.W_v = [Tensor(np.random.randn(d_model, self.d_k) * 0.1, requires_grad=True) for _ in range(num_heads)]
        self.W_o = Tensor(np.random.randn(num_heads * self.d_k, d_model) * 0.1, requires_grad=True)

    def forward(self, X, mask=None):
        heads = []
        for i in range(self.num_heads):
            Q_i = X @ self.W_q[i]
            K_i = X @ self.W_k[i]
            V_i = X @ self.W_v[i]
            head_out, _ = scaled_dot_product_attention(Q_i, K_i, V_i, mask=mask)
            heads.append(head_out)
        multi_head_out = cat(heads, axis=-1)
        output = multi_head_out @ self.W_o
        return output

    def parameters(self):
        return self.W_q + self.W_k + self.W_v + [self.W_o]

class TransformerEncoderBlock(Module):
    def __init__(self, d_model, num_heads, d_ff):
        super().__init__()
        self.mha = MultiHeadAttention(d_model, num_heads)
        self.gamma1 = Tensor(np.ones(d_model), requires_grad=True)
        self.beta1 = Tensor(np.zeros(d_model), requires_grad=True)
        self.W_f1 = Tensor(np.random.randn(d_model, d_ff) * 0.1, requires_grad=True)
        self.b_f1 = Tensor(np.zeros(d_ff), requires_grad=True)
        self.W_f2 = Tensor(np.random.randn(d_ff, d_model) * 0.1, requires_grad=True)
        self.b_f2 = Tensor(np.zeros(d_model), requires_grad=True)
        self.gamma2 = Tensor(np.ones(d_model), requires_grad=True)
        self.beta2 = Tensor(np.zeros(d_model), requires_grad=True)

    def forward(self, x, mask=None):
        attn_out = self.mha.forward(x, mask=mask)
        x = layer_norm(x + attn_out, self.gamma1, self.beta1)
        ffn_out = ((x @ self.W_f1) + self.b_f1).relu() @ self.W_f2 + self.b_f2
        out = layer_norm(x + ffn_out, self.gamma2, self.beta2)
        return out

    def parameters(self):
        params = self.mha.parameters()
        params += [self.gamma1, self.beta1, self.gamma2, self.beta2]
        params += [self.W_f1, self.b_f1, self.W_f2, self.b_f2]
        return params

class Embedding(Module):
    """Embedding layer for mapping token IDs to dense vectors."""
    def __init__(self, vocab_size, embedding_dim):
        super().__init__()
        self.weight = Tensor(np.random.randn(vocab_size, embedding_dim) * 0.02, requires_grad=True)

    def forward(self, indices):
        return self.weight.embedding(indices)

    def parameters(self):
        return [self.weight]


# ---------------------------------------------------------------------------
# Convolution helpers
# ---------------------------------------------------------------------------

def _im2col(x, kernel_h, kernel_w, stride, padding):
    """Extract sliding-window patches from a 4D input (N, C, H, W) into columns."""
    N, C, H, W = x.shape
    if padding > 0:
        x = np.pad(x, ((0, 0), (0, 0), (padding, padding), (padding, padding)), mode='constant')
    out_h = (H + 2 * padding - kernel_h) // stride + 1
    out_w = (W + 2 * padding - kernel_w) // stride + 1

    shape = (N, C, out_h, out_w, kernel_h, kernel_w)
    strides = (x.strides[0], x.strides[1],
               stride * x.strides[2], stride * x.strides[3],
               x.strides[2], x.strides[3])
    patches = np.lib.stride_tricks.as_strided(x, shape=shape, strides=strides)

    # (N, out_h, out_w, C, kernel_h, kernel_w) -> (N*out_h*out_w, C*kernel_h*kernel_w)
    cols = patches.transpose(0, 2, 3, 1, 4, 5).reshape(N * out_h * out_w, -1)
    return cols, out_h, out_w


def _col2im(cols, x_shape, kernel_h, kernel_w, stride, padding):
    """Reverse of _im2col: scatter columns back to a 4D image gradient."""
    N, C, H, W = x_shape
    out_h = (H + 2 * padding - kernel_h) // stride + 1
    out_w = (W + 2 * padding - kernel_w) // stride + 1
    padded_h = H + 2 * padding
    padded_w = W + 2 * padding

    patches = cols.reshape(N, out_h, out_w, C, kernel_h, kernel_w).transpose(0, 3, 1, 2, 4, 5)
    x_padded = np.zeros((N, C, padded_h, padded_w), dtype=cols.dtype)

    for i in range(kernel_h):
        for j in range(kernel_w):
            x_padded[:, :, i:i + stride * out_h:stride, j:j + stride * out_w:stride] += \
                patches[:, :, :, :, i, j]

    if padding > 0:
        return x_padded[:, :, padding:-padding, padding:-padding]
    return x_padded


# ---------------------------------------------------------------------------
# Conv2d
# ---------------------------------------------------------------------------

class Conv2d(Module):
    """2D convolution layer with im2col-based implementation.

    Args:
        in_channels: Number of input channels.
        out_channels: Number of output channels (filters).
        kernel_size: Height and width of the convolution kernel (int or tuple).
        stride: Stride (int or tuple). Default 1.
        padding: Zero-padding (int or tuple). Default 0.
    """

    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0):
        super().__init__()
        if isinstance(kernel_size, int):
            kernel_size = (kernel_size, kernel_size)
        if isinstance(stride, int):
            stride = (stride, stride)
        if isinstance(padding, int):
            padding = (padding, padding)

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding

        kh, kw = kernel_size
        fan_in = in_channels * kh * kw
        self.weight = Tensor(
            np.random.randn(out_channels, in_channels, kh, kw) * np.sqrt(2.0 / fan_in),
            requires_grad=True,
        )
        self.bias = Tensor(np.zeros(out_channels), requires_grad=True)

    def forward(self, x):
        kh, kw = self.kernel_size
        sh, sw = self.stride
        ph, pw = self.padding
        N, C, H, W = x.data.shape
        assert C == self.in_channels, f"Expected {self.in_channels} input channels, got {C}"

        out_h = (H + 2 * ph - kh) // sh + 1
        out_w = (W + 2 * pw - kw) // sw + 1

        # im2col: (N*out_h*out_w, C*kh*kw)
        cols, _, _ = _im2col(x.data, kh, kw, sh, ph)  # same for h/w since square padding assumed
        # Weight: (out_channels, C*kh*kw)
        w_flat = self.weight.data.reshape(self.out_channels, -1)

        # Forward: cols @ w_flat.T + bias → reshape
        out_data = cols @ w_flat.T  # (N*out_h*out_w, out_channels)
        out_data = out_data.reshape(N, out_h, out_w, self.out_channels).transpose(0, 3, 1, 2)
        out_data += self.bias.data.reshape(1, -1, 1, 1)

        out = Tensor(out_data, parents=(x, self.weight, self.bias), requires_grad=True)

        def _backward():
            dy = out.grad  # (N, out_channels, out_h, out_w)

            if self.bias.requires_grad:
                self.bias.grad += dy.sum(axis=(0, 2, 3))

            if self.weight.requires_grad:
                # dW = cols.T @ dy_flat
                dy_flat = dy.transpose(0, 2, 3, 1).reshape(-1, self.out_channels)
                dw_flat = cols.T @ dy_flat  # (C*kh*kw, out_channels)
                self.weight.grad += dw_flat.T.reshape(self.out_channels, C, kh, kw)

            if x.requires_grad:
                # dx = col2im(dy_flat @ w_flat)
                dy_flat = dy.transpose(0, 2, 3, 1).reshape(-1, self.out_channels)
                dx_cols = dy_flat @ w_flat  # (N*out_h*out_w, C*kh*kw)
                x.grad += _col2im(dx_cols, x.data.shape, kh, kw, sh, ph)

        out._backward = _backward
        return out

    def parameters(self):
        return [self.weight, self.bias]


# ---------------------------------------------------------------------------
# MaxPool2d
# ---------------------------------------------------------------------------

class MaxPool2d(Module):
    """2D max pooling layer.

    Args:
        kernel_size: Size of the pooling window (int or tuple).
        stride: Stride (defaults to kernel_size).
        padding: Zero-padding (int or tuple). Default 0.
    """

    def __init__(self, kernel_size, stride=None, padding=0):
        super().__init__()
        if isinstance(kernel_size, int):
            kernel_size = (kernel_size, kernel_size)
        self.kernel_size = kernel_size
        self.stride = stride if stride is not None else kernel_size
        if isinstance(self.stride, int):
            self.stride = (self.stride, self.stride)
        self.padding = padding if isinstance(padding, tuple) else (padding, padding)
        self._argmax = None

    def forward(self, x):
        kh, kw = self.kernel_size
        sh, sw = self.stride
        ph, pw = self.padding
        N, C, H, W = x.data.shape

        out_h = (H + 2 * ph - kh) // sh + 1
        out_w = (W + 2 * pw - kw) // sw + 1

        # Use im2col approach to extract patches, then max over each patch
        cols, oh, ow = _im2col(x.data, kh, kw, sh, ph)
        # cols: (N*out_h*out_w, C*kh*kw)

        # Reshape for per-channel max: (N*out_h*out_w, C, kh*kw)
        cols_rs = cols.reshape(N * out_h * out_w, C, kh * kw)
        argmax = np.argmax(cols_rs, axis=-1)  # (N*out_h*out_w, C)
        max_vals = np.take_along_axis(cols_rs, argmax[:, :, np.newaxis], axis=-1).squeeze(-1)
        # max_vals: (N*out_h*out_w, C)

        out_data = max_vals.reshape(N, out_h, out_w, C).transpose(0, 3, 1, 2)
        out = Tensor(out_data, parents=(x,), requires_grad=True)
        self._argmax = (argmax, N, C, out_h, out_w, kh, kw, sh, ph)

        def _backward():
            if x.requires_grad:
                argmax, N, C, out_h, out_w, kh, kw, sh, ph = self._argmax
                dy = out.grad  # (N, C, out_h, out_w)
                dy_flat = dy.transpose(0, 2, 3, 1).reshape(-1, C)  # (N*out_h*out_w, C)

                # Build grad cols: zeros, then scatter dy at argmax positions
                grad_cols = np.zeros((N * out_h * out_w, C, kh * kw), dtype=dy_flat.dtype)
                np.put_along_axis(grad_cols, argmax[:, :, np.newaxis], dy_flat[:, :, np.newaxis], axis=-1)
                grad_cols = grad_cols.reshape(N * out_h * out_w, C * kh * kw)

                x.grad += _col2im(grad_cols, x.data.shape, kh, kw, sh, ph)

        out._backward = _backward
        return out


# ---------------------------------------------------------------------------
# RMSNorm (LLaMA-style)
# ---------------------------------------------------------------------------

class RMSNorm(Module):
    """Root Mean Square Layer Normalization (Zhang & Sennrich, 2019).

    As used in LLaMA, Mistral, and other modern LLMs.
    Normalizes by RMS without centering, then applies learnable scale.
    """

    def __init__(self, normalized_shape, eps=1e-6):
        super().__init__()
        if isinstance(normalized_shape, int):
            normalized_shape = (normalized_shape,)
        self.normalized_shape = normalized_shape
        self.eps = eps
        self.gamma = Tensor(np.ones(normalized_shape), requires_grad=True)

    def forward(self, x):
        # RMS: sqrt(mean(x^2) + eps)
        rms = np.sqrt(np.mean(x.data ** 2, axis=-1, keepdims=True) + self.eps)
        x_hat = x.data / rms
        out_data = self.gamma.data * x_hat
        out = Tensor(out_data, parents=(x, self.gamma), requires_grad=True)

        def _backward():
            dy = out.grad
            if self.gamma.requires_grad:
                sum_axes = tuple(range(dy.ndim - 1))
                self.gamma.grad += np.sum(dy * x_hat, axis=sum_axes)
            if x.requires_grad:
                D = x.data.shape[-1]
                # d(x_hat)/dx = (1/rms) * (I - (1/D) * x_hat @ x_hat^T)
                dx_hat = dy * self.gamma.data
                mean_dx_hat_x_hat = np.mean(dx_hat * x_hat, axis=-1, keepdims=True)
                dx = (1.0 / rms) * (dx_hat - x_hat * mean_dx_hat_x_hat)
                x.grad += dx

        out._backward = _backward
        return out

    def parameters(self):
        return [self.gamma]


# ---------------------------------------------------------------------------
# RoPE (Rotary Position Embeddings)
# ---------------------------------------------------------------------------

def rope(x, positions, base=10000.0):
    """Apply Rotary Position Embeddings (Su et al., 2023) to a tensor.

    Rotates pairs of dimensions by frequency-scaled position angles.

    Args:
        x: Tensor of shape (..., seq_len, d_model) where d_model is even.
        positions: numpy array of shape (seq_len,) with integer positions.
        base: Base for frequency computation (default 10000.0).

    Returns:
        Tensor of same shape as x with RoPE applied.
    """
    d = x.data.shape[-1]
    assert d % 2 == 0, "d_model must be even for RoPE"

    # Compute frequencies: theta_i = base^(-2i/d)
    i = np.arange(0, d // 2, dtype=np.float32)
    theta = base ** (-2.0 * i / d)  # (d/2,)

    # Compute angles for each position
    pos = positions.astype(np.float32)  # (seq_len,)
    angles = np.outer(pos, theta)  # (seq_len, d/2)

    cos = np.cos(angles)
    sin = np.sin(angles)

    # Split x into even/odd pairs along last dim
    x_data = x.data
    x_even = x_data[..., 0::2]  # (..., seq_len, d/2)
    x_odd = x_data[..., 1::2]   # (..., seq_len, d/2)

    # Apply rotation
    rot_even = x_even * cos - x_odd * sin
    rot_odd = x_even * sin + x_odd * cos

    # Interleave
    out_data = np.empty_like(x_data)
    out_data[..., 0::2] = rot_even
    out_data[..., 1::2] = rot_odd

    out = Tensor(out_data, parents=(x,), requires_grad=True)

    def _backward():
        if x.requires_grad:
            dy = out.grad
            dy_even = dy[..., 0::2]
            dy_odd = dy[..., 1::2]
            # Inverse rotation: transpose the 2x2 rotation matrix (cos, -sin; sin, cos)^T = (cos, sin; -sin, cos)
            dx_even = dy_even * cos + dy_odd * sin
            dx_odd = -dy_even * sin + dy_odd * cos
            dx = np.empty_like(dy)
            dx[..., 0::2] = dx_even
            dx[..., 1::2] = dx_odd
            x.grad += dx

    out._backward = _backward
    return out

