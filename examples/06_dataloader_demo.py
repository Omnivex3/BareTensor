"""Demonstrate DataLoader usage with BareTensor on binary classification.

Trains a Linear(10, 1) model over 5 epochs using minibatch SGD via the
DataLoader utility, printing loss per epoch.
"""

import numpy as np
from baretensor import Tensor, Linear, SGD, TensorDataset, DataLoader


def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))


# =============================================
# Generate synthetic binary classification data
# =============================================
X = np.random.randn(1000, 10)
y = (X.sum(axis=1) > 0).astype(np.float32).reshape(-1, 1)

# =============================================
# Dataset and DataLoader
# =============================================
dataset = TensorDataset(X, y)
loader = DataLoader(dataset, batch_size=64, shuffle=True)

# =============================================
# Model and optimizer
# =============================================
model = Linear(10, 1)
optimizer = SGD(model.parameters(), lr=0.01)

print("=== DataLoader Minibatch Training ===")
for epoch in range(5):
    epoch_loss = 0.0
    num_batches = 0
    for batch_X, batch_y in loader:
        logits = model(Tensor(batch_X))
        probs = sigmoid(logits.data)

        # Manual BCE loss
        loss_val = -np.mean(
            batch_y * np.log(probs + 1e-8)
            + (1 - batch_y) * np.log(1 - probs + 1e-8)
        )

        # Backward (gradient computation)
        grad = (probs - batch_y) / len(batch_y)

        model.weight.grad = batch_X.T @ grad
        model.bias.grad = grad.sum(axis=0)

        optimizer.step()
        optimizer.zero_grad()

        epoch_loss += loss_val
        num_batches += 1

    print(f"Epoch {epoch}: loss={epoch_loss / num_batches:.6f}")

print("Done.")
