"""Compare Adam and SGD on a binary classification task.

Trains two identical Linear(10, 1) models — one with Adam, one with SGD —
on synthetic Gaussian data and prints side-by-side loss curves.
"""

import numpy as np
from baretensor import Tensor, Linear, SGD, Adam


def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))


# =============================================
# Generate synthetic binary classification data
# =============================================
rng = np.random.RandomState(42)
X = rng.randn(500, 10)
y = (X[:, 0] + X[:, 2] > 0).astype(np.float32).reshape(-1, 1)

# =============================================
# SGD Model
# =============================================
model_sgd = Linear(10, 1)
opt_sgd = SGD(model_sgd.parameters(), lr=0.01)

sgd_losses = []
print("=== SGD Training ===")
for epoch in range(20):
    probs = sigmoid(model_sgd(Tensor(X)).data)
    loss = float(
        -np.mean(
            y * np.log(probs + 1e-8) + (1 - y) * np.log(1 - probs + 1e-8)
        )
    )
    grad = (probs - y) / len(y)
    model_sgd.weight.grad = X.T @ grad
    model_sgd.bias.grad = grad.sum(axis=0)
    opt_sgd.step()
    opt_sgd.zero_grad()
    sgd_losses.append(loss)
    print(f"Epoch {epoch:2d}: loss={loss:.6f}")

# =============================================
# Adam Model
# =============================================
model_adam = Linear(10, 1)
opt_adam = Adam(model_adam.parameters(), lr=0.01)

adam_losses = []
print("\n=== Adam Training ===")
for epoch in range(20):
    probs = sigmoid(model_adam(Tensor(X)).data)
    loss = float(
        -np.mean(
            y * np.log(probs + 1e-8) + (1 - y) * np.log(1 - probs + 1e-8)
        )
    )
    grad = (probs - y) / len(y)
    model_adam.weight.grad = X.T @ grad
    model_adam.bias.grad = grad.sum(axis=0)
    opt_adam.step()
    opt_adam.zero_grad()
    adam_losses.append(loss)
    print(f"Epoch {epoch:2d}: loss={loss:.6f}")

# =============================================
# Side-by-side comparison
# =============================================
print("\n=== Comparison (loss per epoch) ===")
print(f"{'Epoch':>6}  {'SGD':>10}  {'Adam':>10}")
for i in range(20):
    print(f"{i:>6}  {sgd_losses[i]:>10.6f}  {adam_losses[i]:>10.6f}")

final_sgd = sgd_losses[-1]
final_adam = adam_losses[-1]
print(f"\nFinal loss — SGD: {final_sgd:.6f} | Adam: {final_adam:.6f}")
if final_adam < final_sgd:
    print("-> Adam converged to a lower loss (faster/better convergence).")
else:
    print("-> SGD achieved lower or comparable loss on this run.")
