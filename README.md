# BareTensor 🧠

> **A purely educational, NumPy-only autograd engine and deep learning framework built from scratch to demystify how PyTorch actually works.**

---

## 📌 Why Does This Exist?

Modern frameworks like PyTorch and TensorFlow are incredible, but they operate as massive black boxes. When your model suffers from vanishing gradients, shape broadcasting mismatches, or computational graph errors, debugging becomes a nightmare if you don't understand the underlying mathematics of vector-Jacobian products (VJPs) and topological sorting.

**BareTensor is designed to strip away the magic.**

There are no pre-packaged layers. There are no pre-packaged optimizers. There is no C++ backend. By forcing you to explicitly define forward operations and construct the reverse-mode automatic differentiation graph using pure Python and NumPy, BareTensor proves that deep learning is just calculus, linear algebra, and smart caching.

If you can build a neural network here, you can debug anything PyTorch throws at you.

---

## 🚀 Core Features

- **Dynamic Computational Graph:** Automatically tracks operations and parent dependencies on the fly.
- **Reverse-Mode Autograd Engine:** Implements a custom Depth-First Search (DFS) topological sort to route gradients backward through complex, branching networks.
- **Multi-Dimensional Tensor Math:** Native support for matrix multiplications (`__matmul__`), transposes, and element-wise operations.
- **Intelligent Gradient Unbroadcasting:** Accurately collapses broadcasted dimensions during the backward pass (crucial for adding biases to batch data).
- **Transformer-ready Primitives:** Native support for transposes, element-wise ReLU, softmax, and multi-head attention.
- **Zero Dependencies:** Built entirely with standard library Python and NumPy. No CUDA, no PyTorch, no bloated binaries required.

---

## 💻 Quickstart: Training a Neural Network

Here is how you train a 2-layer Multi-Layer Perceptron (MLP) to solve the classic non-linear XOR problem using only raw tensor math and BareTensor's autograd engine:

```python
import numpy as np
from baretensor import Tensor, SGD

# 1. Define Data (XOR Gate)
X = Tensor([[0, 0], [0, 1], [1, 0], [1, 1]])
Y = Tensor([[0], [1], [1], [0]])

# 2. Initialize Weights (requires_grad=True tracks them in the graph)
np.random.seed(42)
W1 = Tensor(np.random.randn(2, 8) * 0.5, requires_grad=True)
b1 = Tensor(np.zeros((8,)), requires_grad=True)
W2 = Tensor(np.random.randn(8, 1) * 0.5, requires_grad=True)
b2 = Tensor(np.zeros((1,)), requires_grad=True)

# Define Mean Squared Error helper
def mse_loss(y_pred, y_true):
    diff = y_pred - y_true
    sq = diff * diff
    out = Tensor(np.mean(sq.data), parents=(sq,), requires_grad=True)
    def _backward():
        if sq.requires_grad:
            sq.grad += out.grad * (1.0 / sq.data.size)
    out._backward = _backward
    return out

# 3. The Raw Training Loop
optimizer = SGD([W1, b1, W2, b2], lr=0.05)

for epoch in range(1000):
    # Forward Pass (Linear -> ReLU -> Linear)
    hidden = (X @ W1 + b1).relu()
    y_pred = hidden @ W2 + b2

    # Loss Calculation
    loss = mse_loss(y_pred, Y)

    # Backward Pass
    loss.backward()

    # Optimization Step
    optimizer.step()
    optimizer.zero_grad()

    if (epoch + 1) % 100 == 0:
        print(f"Epoch {epoch + 1:4d} | Loss: {loss.data:.4f}")
```

---

## 🔬 Under the Hood: How the Autograd Works

BareTensor does not rely on numerical approximation. It computes exact analytical gradients. Every time you perform an operation on a `Tensor`, it returns a new `Tensor` that contains a pointer to the exact mathematical function (`_backward`) required to compute its local derivative (vector-Jacobian product).

When you call `loss.backward()`, the engine performs a topological sort:

```python
def backward(self):
    topo, visited = [], set()
    def build_topo(v):
        if v not in visited:
            visited.add(v)
            for parent in v.parents: 
                build_topo(parent)
            topo.append(v)
    build_topo(self)

    self.grad = np.ones_like(self.data)
    for node in reversed(topo): 
        node._backward()
```

---

## 🧪 Running the Verification Tests

A professional framework requires mathematical proof. BareTensor comes with a `pytest` suite that generates identical random weights, executes complex forward/backward passes in both PyTorch and BareTensor, and asserts that the resulting gradients match down to the 5th decimal place:

```bash
pytest tests/
```
