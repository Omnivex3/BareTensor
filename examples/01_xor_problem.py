import numpy as np
from baretensor import Tensor, SGD

def mse_loss(y_pred, y_true):
    diff = y_pred - y_true
    sq = diff * diff
    out = Tensor(np.mean(sq.data), parents=(sq,), requires_grad=True)

    def _backward():
        if sq.requires_grad:
            sq.grad += out.grad * (1.0 / sq.data.size)

    out._backward = _backward
    return out

# ==========================================
# XOR PROBLEM (Data & Architecture)
# ==========================================

# Inputs (4 samples, 2 features)
X = Tensor([[0, 0], [0, 1], [1, 0], [1, 1]])

# Targets (1 if inputs are different, 0 if same)
Y = Tensor([[0], [1], [1], [0]])

# Neural Network Architecture: 2 -> 8 -> 1
np.random.seed(42)  # For reproducibility

# Layer 1 (Hidden Layer)
W1 = Tensor(np.random.randn(2, 8) * 0.5, requires_grad=True)
b1 = Tensor(np.zeros((8,)), requires_grad=True)

# Layer 2 (Output Layer)
W2 = Tensor(np.random.randn(8, 1) * 0.5, requires_grad=True)
b2 = Tensor(np.zeros((1,)), requires_grad=True)

parameters = [W1, b1, W2, b2]
optimizer = SGD(parameters, lr=0.05)
epochs = 1000

# ==========================================
# TRAINING LOOP
# ==========================================

print("Starting XOR training...")
for epoch in range(epochs):
    # 1. Forward Pass
    hidden = (X @ W1 + b1).relu()
    y_pred = hidden @ W2 + b2

    # 2. Loss Calculation
    loss = mse_loss(y_pred, Y)

    # 3. Backward Pass
    loss.backward()

    # 4. SGD Update & Zero Gradients
    optimizer.step()
    optimizer.zero_grad()

    # Print progress
    if (epoch + 1) % 100 == 0:
        print(f"Epoch {epoch + 1:4d} | Loss: {loss.data:.4f}")

# ==========================================
# FINAL PREDICTIONS
# ==========================================

print("\n--- Final XOR Predictions ---")
hidden_final = (X @ W1 + b1).relu()
predictions = hidden_final @ W2 + b2

for i in range(4):
    in_val = X.data[i].astype(int)
    true_val = Y.data[i][0].astype(int)
    pred_val = predictions.data[i][0]
    print(
        f"Input: {in_val} | Target: {true_val} | Prediction: {pred_val:.4f} -> {round(pred_val)}"
    )
