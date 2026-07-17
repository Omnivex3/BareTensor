import numpy as np
from baretensor.tensor import Tensor

def test_tensor_shapes_and_grad():
    x = Tensor([[1.0, 2.0], [3.0, 4.0]], requires_grad=True)
    y = Tensor([[2.0, 0.0], [1.0, 2.0]], requires_grad=True)

    z = x * y
    assert z.data.shape == (2, 2)
    assert np.allclose(z.data, [[2.0, 0.0], [3.0, 8.0]])

    z_sum = z.sum()
    z_sum.backward()

    assert x.grad.shape == (2, 2)
    assert y.grad.shape == (2, 2)
    assert np.allclose(x.grad, y.data)
    assert np.allclose(y.grad, x.data)

def test_unbroadcast():
    # Test broadcasting addition of shape (1, 3) to (4, 3)
    np.random.seed(42)
    a_data = np.random.randn(4, 3).astype(np.float32)
    b_data = np.random.randn(1, 3).astype(np.float32)

    a = Tensor(a_data, requires_grad=True)
    b = Tensor(b_data, requires_grad=True)

    c = a + b
    c_sum = c.sum()
    c_sum.backward()

    # The gradient for b should be summed across all 4 rows
    assert b.grad.shape == (1, 3)
    assert np.allclose(b.grad, [[4.0, 4.0, 4.0]])
    assert np.allclose(a.grad, np.ones((4, 3)))
