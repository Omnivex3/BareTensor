"""
MNIST Classifier using BareTensor framework.

Demonstrates:
  - DataLoader-based mini-batch training
  - Adam optimizer
  - MLP with ReLU activations
  - cross_entropy_loss for multi-class classification
  - Full train/test evaluation
"""
import time
import numpy as np
from sklearn.datasets import fetch_openml

from baretensor.tensor import Tensor
from baretensor.nn import Module, Linear, cross_entropy_loss
from baretensor.optim import Adam
from baretensor.data import TensorDataset, DataLoader


# =========================================================================
# 1. Model Definition
# =========================================================================

class MNISTClassifier(Module):
    """3-layer MLP: 784 -> 256 -> 128 -> 10."""

    def __init__(self):
        super().__init__()
        self.fc1 = Linear(784, 256)
        self.fc2 = Linear(256, 128)
        self.fc3 = Linear(128, 10)

    def forward(self, x):
        x = self.fc1(x).relu()
        x = self.fc2(x).relu()
        x = self.fc3(x)
        return x


    def parameters(self):
        """Recursively collect all trainable parameters."""
        params = []
        for name in sorted(vars(self)):
            attr = getattr(self, name)
            if isinstance(attr, Tensor) and attr.requires_grad:
                params.append(attr)
            elif isinstance(attr, Module):
                params.extend(attr.parameters())
        return params


# =========================================================================
# 2. Data Loading
# =========================================================================

print("Loading MNIST dataset via sklearn...")
t0 = time.time()
mnist = fetch_openml('mnist_784', version=1, as_frame=False, parser='liac-arff')
X_all = mnist.data.astype(np.float32) / 255.0
y_all = mnist.target.astype(np.int64)
print(f"Loaded {len(X_all)} samples in {time.time()-t0:.1f}s")
print(f"  Data shape:  {X_all.shape}")
print(f"  Label range: {y_all.min()} – {y_all.max()}")

# Train / test split (first 60k train, last 10k test)
X_train, X_test = X_all[:60000], X_all[60000:]
y_train, y_test = y_all[:60000], y_all[60000:]

train_dataset = TensorDataset(X_train, y_train)
test_dataset  = TensorDataset(X_test, y_test)
train_loader  = DataLoader(train_dataset, batch_size=64, shuffle=True)
test_loader   = DataLoader(test_dataset, batch_size=64)


# =========================================================================
# 3. Instantiate Model & Optimizer
# =========================================================================

model = MNISTClassifier()
optimizer = Adam(model.parameters(), lr=0.001)

# Count total parameters
total_params = sum(p.data.size for p in model.parameters())
print(f"\nModel: MNISTClassifier (3-layer MLP)")
print(f"Total parameters: {total_params:,}")


# =========================================================================
# 4. Training Loop
# =========================================================================

EPOCHS = 5

print(f"\n{'='*60}")
print(f"Training for {EPOCHS} epochs (Adam, lr=0.001, batch=64)")
print(f"{'='*60}")

train_start = time.time()

for epoch in range(EPOCHS):
    epoch_loss = 0.0
    num_batches = 0

    for batch_X, batch_y in train_loader:
        # Forward
        logits = model(Tensor(batch_X))
        loss = cross_entropy_loss(logits, batch_y)

        # Backward
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

        epoch_loss += float(loss.data)
        num_batches += 1

    avg_loss = epoch_loss / num_batches
    elapsed = time.time() - train_start
    print(f"Epoch {epoch+1:2d}/{EPOCHS}  |  loss={avg_loss:.6f}  "
          f"({elapsed:.1f}s elapsed)")


# =========================================================================
# 5. Evaluation on Test Set
# =========================================================================

print(f"\n{'='*60}")
print("Evaluating on test set...")
print(f"{'='*60}")

correct = 0
total = 0
eval_start = time.time()

for batch_X, batch_y in test_loader:
    logits = model(Tensor(batch_X))
    predictions = np.argmax(logits.data, axis=1)
    correct += int(np.sum(predictions == batch_y))
    total += len(batch_y)

accuracy = 100.0 * correct / total
eval_time = time.time() - eval_start

print(f"\n{'='*60}")
print(f"TEST ACCURACY: {accuracy:.2f}%  ({correct}/{total} correct)")
print(f"Evaluation time: {eval_time:.2f}s")
print(f"Total training + eval time: {time.time()-train_start:.1f}s")
print(f"{'='*60}")
