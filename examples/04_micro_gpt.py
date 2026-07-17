import numpy as np
from baretensor import Tensor, Module, Embedding, TransformerEncoderBlock, Linear, SGD, cross_entropy_loss

# ==========================================
# 1. Dataset Preparation
# ==========================================
text = """To be, or not to be, that is the question:
Whether 'tis nobler in the mind to suffer
The slings and arrows of outrageous fortune,
Or to take arms against a sea of troubles
And by opposing end them."""

# Create char-to-id and id-to-char mappings
chars = sorted(list(set(text)))
vocab_size = len(chars)
char2id = {ch: i for i, ch in enumerate(chars)}
id2char = {i: ch for i, ch in enumerate(chars)}

# Encode the text into integer token IDs
data = np.array([char2id[ch] for ch in text], dtype=np.int64)

print("--- Micro-GPT Character-Level Language Model ---")
print(f"Text length: {len(text)} characters")
print(f"Vocabulary size: {vocab_size} unique characters")

# Helper to generate training batches
def get_batch(data, seq_len, batch_size):
    # Select random starting offsets
    offsets = np.random.randint(0, len(data) - seq_len, size=batch_size)
    x_batch = np.stack([data[o : o + seq_len] for o in offsets])
    y_batch = np.stack([data[o + 1 : o + seq_len + 1] for o in offsets])
    return x_batch, y_batch

# ==========================================
# 2. Micro-GPT Architecture
# ==========================================
class MicroGPT(Module):
    def __init__(self, vocab_size, d_model, num_heads, d_ff, max_seq_len):
        self.max_seq_len = max_seq_len
        # Token and Positional Embeddings
        self.tok_emb = Embedding(vocab_size, d_model)
        self.pos_emb = Embedding(max_seq_len, d_model)
        # Single Decoder (Transformer) Block
        self.transformer = TransformerEncoderBlock(d_model, num_heads, d_ff)
        # Output Linear Head
        self.lm_head = Linear(d_model, vocab_size)

    def forward(self, idx):
        # idx shape: (batch_size, seq_len)
        batch_size, seq_len = idx.shape
        assert seq_len <= self.max_seq_len, f"Sequence length {seq_len} exceeds max {self.max_seq_len}"

        # 1. Token & Positional embeddings
        x_tok = self.tok_emb(idx)  # (batch_size, seq_len, d_model)
        pos_ids = np.arange(seq_len)
        x_pos = self.pos_emb(pos_ids)  # (seq_len, d_model)
        x = x_tok + x_pos  # Broadcasts across batch dimension automatically!

        # 2. Create Causal Mask
        # Upper triangle is -inf, rest is 0
        mask_data = np.zeros((seq_len, seq_len), dtype=np.float32)
        mask_data[np.triu_indices(seq_len, k=1)] = -np.inf
        mask = Tensor(mask_data, requires_grad=False)

        # 3. Transformer Block
        x = self.transformer.forward(x, mask=mask)

        # 4. Project back to Vocab size logits
        logits = self.lm_head(x)  # (batch_size, seq_len, vocab_size)
        return logits

    def parameters(self):
        return (
            self.tok_emb.parameters()
            + self.pos_emb.parameters()
            + self.transformer.parameters()
            + self.lm_head.parameters()
        )

# ==========================================
# 3. Training Settings
# ==========================================
np.random.seed(42)
seq_len = 16
batch_size = 8
d_model = 64
num_heads = 4
d_ff = 128
epochs = 600
learning_rate = 0.02

model = MicroGPT(vocab_size, d_model, num_heads, d_ff, seq_len)
optimizer = SGD(model.parameters(), lr=learning_rate)

print("\nTraining the model to memorize the Shakespeare quote...")
for epoch in range(epochs):
    # 1. Get batch
    x_np, y_np = get_batch(data, seq_len, batch_size)
    x_tensor = idx = x_np # Let's keep it as numpy array for custom embedding lookup

    # 2. Forward
    logits = model.forward(x_tensor)  # (batch_size, seq_len, vocab_size)

    # 3. Flatten for Cross Entropy
    # Shape: (batch_size * seq_len, vocab_size)
    logits_flat = logits.reshape((batch_size * seq_len, vocab_size))
    y_flat = y_np.reshape(-1)

    # 4. Compute Loss
    loss = cross_entropy_loss(logits_flat, y_flat)

    # 5. Backward & Step
    loss.backward()
    optimizer.step()
    optimizer.zero_grad()

    if (epoch + 1) % 100 == 0:
        print(f"Epoch {epoch + 1:4d} | Loss: {loss.data:.4f}")

# ==========================================
# 4. Auto-Regressive Text Generation
# ==========================================
def generate(model, prompt, max_new_tokens=60):
    print(f"\nPrompt: '{prompt}'")
    # Encode prompt characters
    idx_list = [char2id[ch] for ch in prompt]
    generated_text = prompt

    for _ in range(max_new_tokens):
        # Keep sequence within max length by slicing from the right
        seq_idx = idx_list[-seq_len:]
        x_in = np.array([seq_idx], dtype=np.int64)  # Batch size 1: (1, seq_len_current)

        # Forward pass (inference only, no grads needed)
        logits = model.forward(x_in)  # (1, seq_len_current, vocab_size)
        
        # Extract last token logits
        last_token_logits = logits.data[0, -1, :]  # (vocab_size,)
        
        # Softmax to get probabilities
        exp_logits = np.exp(last_token_logits - np.max(last_token_logits))
        probs = exp_logits / np.sum(exp_logits)
        
        # Sample next token
        next_char_id = np.random.choice(vocab_size, p=probs)
        generated_text += id2char[next_char_id]
        idx_list.append(next_char_id)

    print(f"Generated output:\n{generated_text}")

# Generate text with two different prompts
generate(model, "To be, or not ", max_new_tokens=80)
generate(model, "Whether 'tis ", max_new_tokens=80)
