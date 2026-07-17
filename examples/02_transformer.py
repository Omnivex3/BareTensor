import numpy as np
from baretensor import Tensor, TransformerEncoderBlock, SGD

# ==========================================
# Transformer Encoder Verification Script
# ==========================================

# 1. Setup: A sequence of 5 tokens, embedded into 64 dimensions
np.random.seed(42)
seq_length = 5
d_model = 64
X = Tensor(np.random.randn(seq_length, d_model), requires_grad=True)
Y_target = Tensor(np.random.randn(seq_length, d_model))  # Target output

# 2. Initialize the Transformer Block
# d_ff is traditionally 4x the d_model (64 * 4 = 256)
encoder = TransformerEncoderBlock(d_model=d_model, num_heads=8, d_ff=256)
optimizer = SGD(encoder.parameters(), lr=0.01)

print("--- Initializing Transformer Encoder Block ---")
print(f"Sequence length: {seq_length}")
print(f"Embedding dimension: {d_model}")
print(f"Total parameter tensors: {len(encoder.parameters())}")

# 3. Optimize the Transformer Block for 5 steps
print("\n--- Training Loop Simulation (5 Steps) ---")
for step in range(5):
    # Forward Pass
    output = encoder.forward(X)

    # Simple MSE Loss: mean((output - Y_target)^2)
    diff = output - Y_target
    sq = diff * diff
    loss = Tensor(np.mean(sq.data), parents=(sq,), requires_grad=True)

    def _mse_backward():
        if sq.requires_grad:
            sq.grad += loss.grad * (1.0 / sq.data.size)
    loss._backward = _mse_backward

    # Backward Pass
    loss.backward()

    # Optimize and Zero
    optimizer.step()
    optimizer.zero_grad()

    print(f"Step {step + 1} | Loss: {loss.data:.6f}")

# Final validation printouts
print("\n--- Gradient Shapes Verification ---")
print("Input gradient shape:", X.grad.shape if X.grad is not None else "None")
print("MHA Head 0 Q Weight shape:", encoder.mha.W_q[0].grad.shape if encoder.mha.W_q[0].grad is not None else "None")
print("FFN Weight 1 gradient shape:", encoder.W_f1.grad.shape if encoder.W_f1.grad is not None else "None")
