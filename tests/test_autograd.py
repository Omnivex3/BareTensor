import torch
import numpy as np
from baretensor.tensor import Tensor
from baretensor.nn import TransformerEncoderBlock, layer_norm, scaled_dot_product_attention

def test_linear_relu_autograd():
    # 1. Generate random starting data
    np.random.seed(42)
    x_data = np.random.randn(32, 64).astype(np.float32)  # Batch of 32, 64 features
    w_data = np.random.randn(64, 16).astype(np.float32)  # 64 in, 16 out
    b_data = np.random.randn(16).astype(np.float32)      # Bias vector

    # ==========================================
    # 2. PyTorch Forward and Backward
    # ==========================================
    pt_x = torch.tensor(x_data, requires_grad=True)
    pt_w = torch.tensor(w_data, requires_grad=True)
    pt_b = torch.tensor(b_data, requires_grad=True)

    pt_out = torch.relu(pt_x @ pt_w + pt_b)
    pt_loss = pt_out.sum()
    pt_loss.backward()

    # ==========================================
    # 3. BareTensor Forward and Backward
    # ==========================================
    bt_x = Tensor(x_data, requires_grad=True)
    bt_w = Tensor(w_data, requires_grad=True)
    bt_b = Tensor(b_data, requires_grad=True)

    bt_out = (bt_x @ bt_w + bt_b).relu()
    bt_loss = bt_out.sum()
    bt_loss.backward()

    # ==========================================
    # 4. Verification
    # ==========================================
    np.testing.assert_allclose(bt_out.data, pt_out.detach().numpy(), atol=1e-5)
    np.testing.assert_allclose(bt_x.grad, pt_x.grad.numpy(), atol=1e-5)
    np.testing.assert_allclose(bt_w.grad, pt_w.grad.numpy(), atol=1e-5)
    np.testing.assert_allclose(bt_b.grad, pt_b.grad.numpy(), atol=1e-5)

def test_layer_norm_autograd():
    np.random.seed(42)
    x_data = np.random.randn(5, 10).astype(np.float32)
    gamma_data = np.random.randn(10).astype(np.float32)
    beta_data = np.random.randn(10).astype(np.float32)

    # PyTorch
    pt_x = torch.tensor(x_data, requires_grad=True)
    pt_gamma = torch.tensor(gamma_data, requires_grad=True)
    pt_beta = torch.tensor(beta_data, requires_grad=True)
    pt_out = torch.nn.functional.layer_norm(pt_x, (10,), pt_gamma, pt_beta, eps=1e-5)
    pt_loss = pt_out.sum()
    pt_loss.backward()

    # BareTensor
    bt_x = Tensor(x_data, requires_grad=True)
    bt_gamma = Tensor(gamma_data, requires_grad=True)
    bt_beta = Tensor(beta_data, requires_grad=True)
    bt_out = layer_norm(bt_x, bt_gamma, bt_beta, eps=1e-5)
    bt_loss = bt_out.sum()
    bt_loss.backward()

    # Verify
    np.testing.assert_allclose(bt_out.data, pt_out.detach().numpy(), atol=1e-5)
    np.testing.assert_allclose(bt_x.grad, pt_x.grad.numpy(), atol=1e-4)
    np.testing.assert_allclose(bt_gamma.grad, pt_gamma.grad.numpy(), atol=1e-4)
    np.testing.assert_allclose(bt_beta.grad, pt_beta.grad.numpy(), atol=1e-4)
