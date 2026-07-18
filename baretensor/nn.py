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

class Linear(Module):
    """Fully connected linear layer: y = x @ W + b."""

    def __init__(self, in_features, out_features):
        super().__init__()
        self.weight = Tensor(
            np.random.randn(in_features, out_features) * np.sqrt(2.0 / in_features),
            requires_grad=True,
        )
        self.bias = Tensor(np.zeros(out_features), requires_grad=True)

    def __call__(self, x):
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

    def __call__(self, x):
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

    def __call__(self, x):
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

    def __call__(self, indices):
        return self.weight.embedding(indices)

    def parameters(self):
        return [self.weight]

