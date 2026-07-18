"""
MNIST with Deep MLP + BatchNorm + Dropout on BareTensor.
Aims to beat the plain 3-layer MLP baseline of 98.12%.
"""
import time
import numpy as np
from baretensor import Tensor, Linear, Module, Adam
from baretensor.nn import BatchNorm1d, Dropout, cross_entropy_loss
from baretensor import TensorDataset, DataLoader


class DeepMNIST(Module):
    """4-layer MLP with BatchNorm and Dropout."""

    def __init__(self):
        super().__init__()
        self.fc1 = Linear(784, 512)
        self.bn1 = BatchNorm1d(512)
        self.drop1 = Dropout(0.3)
        self.fc2 = Linear(512, 256)
        self.bn2 = BatchNorm1d(256)
        self.drop2 = Dropout(0.3)
        self.fc3 = Linear(256, 128)
        self.bn3 = BatchNorm1d(128)
        self.drop3 = Dropout(0.3)
        self.fc4 = Linear(128, 10)

    def forward(self, x):
        x = self.drop1(self.bn1(self.fc1(x).relu()))
        x = self.drop2(self.bn2(self.fc2(x).relu()))
        x = self.drop3(self.bn3(self.fc3(x).relu()))
        return self.fc4(x)


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


# Load MNIST
from sklearn.datasets import fetch_openml
print("Loading MNIST...")
t0 = time.time()
mnist = fetch_openml('mnist_784', version=1, as_frame=False, parser='liac-arff')
X_all = mnist.data.astype(np.float32) / 255.0
y_all = mnist.target.astype(np.int64)
print(f"Loaded {len(X_all)} samples in {time.time()-t0:.1f}s")

# Split
X_train, X_test = X_all[:60000], X_all[60000:]
y_train, y_test = y_all[:60000], y_all[60000:]

train_dataset = TensorDataset(X_train, y_train)
test_dataset = TensorDataset(X_test, y_test)
train_loader = DataLoader(train_dataset, batch_size=128, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=256)

model = DeepMNIST()
optimizer = Adam(model.parameters(), lr=0.001)

total_params = sum(p.data.size for p in model.parameters())
print(f"\nModel: DeepMNIST (4-layer MLP + BatchNorm + Dropout)")
print(f"Total parameters: {total_params:,}")

EPOCHS = 10
print(f"\nTraining {EPOCHS} epochs (Adam, lr=0.001, batch=128)")
print("=" * 60)

train_start = time.time()
for epoch in range(EPOCHS):
    model.train()  # <- sets training mode on ALL layers
    epoch_loss = 0.0
    num_batches = 0

    for batch_X, batch_y in train_loader:
        logits = model(Tensor(batch_X))
        loss = cross_entropy_loss(logits, batch_y)
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
        epoch_loss += float(loss.data)
        num_batches += 1

    avg_loss = epoch_loss / num_batches
    elapsed = time.time() - train_start

    # Evaluate at end of each epoch
    model.eval()  # <- disable dropout, use running stats
    correct = 0
    total = 0
    for batch_X, batch_y in test_loader:
        logits = model(Tensor(batch_X))
        predictions = np.argmax(logits.data, axis=1)
        correct += np.sum(predictions == batch_y)
        total += len(batch_y)

    accuracy = 100.0 * correct / total
    print(f"Epoch {epoch:2d}: loss={avg_loss:.6f}  acc={accuracy:.2f}%  ({elapsed:.1f}s)")

# Final
print("=" * 60)
print(f"FINAL TEST ACCURACY: {accuracy:.2f}% ({correct}/{total} correct)")
print(f"Total time: {time.time()-train_start:.1f}s")
