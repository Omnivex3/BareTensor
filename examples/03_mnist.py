import numpy as np
from sklearn.datasets import fetch_openml
from baretensor.tensor import Tensor
from baretensor.nn import Module, Linear, cross_entropy_loss
from baretensor.optim import SGD

class MLP(Module):
    def __init__(self):
        super().__init__()
        self.fc1 = Linear(784, 128)
        self.fc2 = Linear(128, 10)

    def forward(self, x):
        hidden = self.fc1(x).relu()
        logits = self.fc2(hidden)
        return logits

    def parameters(self):
        return self.fc1.parameters() + self.fc2.parameters()

def get_accuracy(logits_data, targets):
    predictions = np.argmax(logits_data, axis=1)
    return np.mean(predictions == targets)

# ==========================================
# 1. Data Loading
# ==========================================
print("Fetching MNIST dataset (this takes a moment)...")
mnist = fetch_openml('mnist_784', version=1, as_frame=False, parser='liac-arff')
X = mnist.data.astype(np.float32) / 255.0
Y = mnist.target.astype(np.int64)

X_train, X_test = X[:60000], X[60000:]
Y_train, Y_test = Y[:60000], Y[60000:]
print(f"Training data shape: {X_train.shape}")

# ==========================================
# 2. Model & Hyperparameters
# ==========================================
np.random.seed(42)
model = MLP()
optimizer = SGD(model.parameters(), lr=0.1)

batch_size = 128
epochs = 10

# ==========================================
# 3. Training Loop
# ==========================================
print("\nStarting Training...")
for epoch in range(epochs):
    indices = np.random.permutation(len(X_train))
    epoch_loss = 0
    epoch_acc = 0
    num_batches = len(X_train) // batch_size

    for i in range(0, len(X_train), batch_size):
        batch_indices = indices[i:i + batch_size]
        if len(batch_indices) < batch_size:
            continue  # Skip incomplete final batch

        X_batch = Tensor(X_train[batch_indices])
        Y_batch = Y_train[batch_indices]

        # Forward Pass
        logits = model(X_batch)

        # Loss
        loss = cross_entropy_loss(logits, Y_batch)

        # Backward & Optimize
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

        epoch_loss += loss.data
        epoch_acc += get_accuracy(logits.data, Y_batch)

    # Evaluate on Test Set
    hidden_test = model.fc1(Tensor(X_test)).relu()
    logits_test = model.fc2(hidden_test)
    test_acc = get_accuracy(logits_test.data, Y_test)

    print(f"Epoch {epoch+1:2d}/{epochs} | "
          f"Train Loss: {epoch_loss/num_batches:.4f} | "
          f"Train Acc: {(epoch_acc/num_batches)*100:.2f}% | "
          f"Test Acc: {test_acc*100:.2f}%")

# ==========================================
# 4. Save & Load Verification
# ==========================================
print("\nTraining complete.")

# Save the trained model
model.save("mnist_mlp_weights")

# Create a fresh untrained model and load
inference_model = MLP()
inference_model.load("mnist_mlp_weights.npz")

# Verify loaded model matches
hidden_loaded = inference_model.fc1(Tensor(X_test)).relu()
logits_loaded = inference_model.fc2(hidden_loaded)
loaded_acc = get_accuracy(logits_loaded.data, Y_test)
print(f"Loaded Model Test Accuracy: {loaded_acc * 100:.2f}%")
