import numpy as np

class Tensor:
    def __init__(self, data, requires_grad=False, parents=(), dtype=None):
        if dtype is not None:
            self.data = np.array(data, dtype=dtype)
        else:
            self.data = np.array(data)
        # If requires_grad is True, ensure float type for gradient computation
        if requires_grad and self.data.dtype.kind != 'f':
            self.data = self.data.astype(np.float32)
        elif requires_grad and self.data.dtype != np.float32:
            self.data = self.data.astype(np.float32)
        self.requires_grad = requires_grad
        self.parents = parents
        self.grad = np.zeros_like(self.data) if requires_grad else None
        self._backward = lambda: None

    def _unbroadcast(self, grad, target_shape):
        """Collapses a gradient back to the target_shape by summing over broadcasted dims."""
        ndims_added = len(grad.shape) - len(target_shape)
        for _ in range(ndims_added):
            grad = grad.sum(axis=0)
        for i, dim in enumerate(target_shape):
            if dim == 1 and grad.shape[i] > 1:
                grad = grad.sum(axis=i, keepdims=True)
        return grad

    def __add__(self, other):
        other = other if isinstance(other, Tensor) else Tensor(other)
        out = Tensor(self.data + other.data, parents=(self, other), requires_grad=True)

        def _backward():
            if self.requires_grad:
                self.grad += self._unbroadcast(out.grad, self.data.shape)
            if other.requires_grad:
                other.grad += self._unbroadcast(out.grad, other.data.shape)

        out._backward = _backward
        return out

    def __radd__(self, other):
        return self.__add__(other)

    def __sub__(self, other):
        other = other if isinstance(other, Tensor) else Tensor(other)
        out = Tensor(self.data - other.data, parents=(self, other), requires_grad=True)

        def _backward():
            if self.requires_grad:
                self.grad += self._unbroadcast(out.grad, self.data.shape)
            if other.requires_grad:
                other.grad -= self._unbroadcast(out.grad, other.data.shape)

        out._backward = _backward
        return out

    def __rsub__(self, other):
        other = other if isinstance(other, Tensor) else Tensor(other)
        return other.__sub__(self)

    def __mul__(self, other):
        other = other if isinstance(other, Tensor) else Tensor(other)
        out = Tensor(self.data * other.data, parents=(self, other), requires_grad=True)

        def _backward():
            if self.requires_grad:
                self.grad += self._unbroadcast(out.grad * other.data, self.data.shape)
            if other.requires_grad:
                other.grad += self._unbroadcast(out.grad * self.data, other.data.shape)

        out._backward = _backward
        return out

    def __rmul__(self, other):
        return self.__mul__(other)

    def __matmul__(self, other):
        other = other if isinstance(other, Tensor) else Tensor(other)
        out = Tensor(self.data @ other.data, parents=(self, other), requires_grad=True)

        def _backward():
            if self.requires_grad:
                other_t = other.data.swapaxes(-1, -2)
                self.grad += self._unbroadcast(out.grad @ other_t, self.data.shape)
            if other.requires_grad:
                self_t = self.data.swapaxes(-1, -2)
                other.grad += self._unbroadcast(self_t @ out.grad, other.data.shape)

        out._backward = _backward
        return out

    def __rmatmul__(self, other):
        other = other if isinstance(other, Tensor) else Tensor(other)
        return other.__matmul__(self)

    def __neg__(self):
        out = Tensor(-self.data, parents=(self,), requires_grad=True)

        def _backward():
            if self.requires_grad:
                self.grad += -out.grad

        out._backward = _backward
        return out

    def __truediv__(self, other):
        other = other if isinstance(other, Tensor) else Tensor(other)
        out = Tensor(self.data / other.data, parents=(self, other), requires_grad=True)

        def _backward():
            if self.requires_grad:
                self.grad += self._unbroadcast(out.grad / other.data, self.data.shape)
            if other.requires_grad:
                other.grad += self._unbroadcast(-out.grad * self.data / (other.data ** 2), other.data.shape)

        out._backward = _backward
        return out

    def __rtruediv__(self, other):
        other = other if isinstance(other, Tensor) else Tensor(other)
        return other.__truediv__(self)

    def relu(self):
        out = Tensor(np.maximum(0, self.data), parents=(self,), requires_grad=True)

        def _backward():
            if self.requires_grad:
                self.grad += out.grad * (self.data > 0).astype(np.float32)

        out._backward = _backward
        return out

    def sigmoid(self):
        s = 1.0 / (1.0 + np.exp(-self.data))
        out = Tensor(s, parents=(self,), requires_grad=True)

        def _backward():
            if self.requires_grad:
                self.grad += out.grad * s * (1.0 - s)

        out._backward = _backward
        return out

    def tanh(self):
        t = np.tanh(self.data)
        out = Tensor(t, parents=(self,), requires_grad=True)

        def _backward():
            if self.requires_grad:
                self.grad += out.grad * (1.0 - t ** 2)

        out._backward = _backward
        return out

    def gelu(self):
        c = np.sqrt(2.0 / np.pi)
        inner = c * (self.data + 0.044715 * self.data ** 3)
        tanh_inner = np.tanh(inner)
        out_data = 0.5 * self.data * (1.0 + tanh_inner)
        out = Tensor(out_data, parents=(self,), requires_grad=True)

        def _backward():
            if self.requires_grad:
                sech2 = 1.0 - tanh_inner ** 2
                d_inner = c * (1.0 + 3.0 * 0.044715 * self.data ** 2)
                d_gelu = 0.5 * (1.0 + tanh_inner) + 0.5 * self.data * sech2 * d_inner
                self.grad += out.grad * d_gelu

        out._backward = _backward
        return out

    def transpose(self, axes=None):
        if axes is not None:
            ndim = self.data.ndim
            axes = tuple(ax if ax >= 0 else ndim + ax for ax in axes)
            out_data = self.data.transpose(axes)
        else:
            out_data = self.data.T

        out = Tensor(out_data, parents=(self,), requires_grad=True)

        def _backward():
            if self.requires_grad:
                if axes is not None:
                    inv_axes = np.argsort(axes)
                    self.grad += out.grad.transpose(inv_axes)
                else:
                    self.grad += out.grad.T

        out._backward = _backward
        return out

    def softmax(self, axis=-1):
        shift_x = self.data - np.max(self.data, axis=axis, keepdims=True)
        exps = np.exp(shift_x)
        probs = exps / np.sum(exps, axis=axis, keepdims=True)
        out = Tensor(probs, parents=(self,), requires_grad=True)

        def _backward():
            if self.requires_grad:
                sum_gy = np.sum(out.grad * probs, axis=axis, keepdims=True)
                self.grad += probs * (out.grad - sum_gy)

        out._backward = _backward
        return out

    def embedding(self, indices):
        out = Tensor(self.data[indices], parents=(self,), requires_grad=True)

        def _backward():
            if self.requires_grad:
                np.add.at(self.grad, indices, out.grad)

        out._backward = _backward
        return out

    def reshape(self, shape):
        out = Tensor(self.data.reshape(shape), parents=(self,), requires_grad=True)

        def _backward():
            if self.requires_grad:
                self.grad += out.grad.reshape(self.data.shape)

        out._backward = _backward
        return out


    def exp(self):
        e = np.exp(self.data)
        out = Tensor(e, parents=(self,), requires_grad=True)

        def _backward():
            if self.requires_grad:
                self.grad += out.grad * e

        out._backward = _backward
        return out

    def log(self, eps=1e-7):
        safe = np.maximum(self.data, eps)
        out_data = np.log(safe)
        out = Tensor(out_data, parents=(self,), requires_grad=True)

        def _backward():
            if self.requires_grad:
                self.grad += out.grad / safe

        out._backward = _backward
        return out

    def sum(self):
        out = Tensor(np.sum(self.data), parents=(self,), requires_grad=True)

        def _backward():
            if self.requires_grad:
                self.grad += out.grad * np.ones_like(self.data)

        out._backward = _backward
        return out

    def backward(self):
        topo = []
        visited = set()

        def build_topo(v):
            if v not in visited:
                visited.add(v)
                for parent in v.parents:
                    build_topo(parent)
                topo.append(v)

        build_topo(self)
        self.grad = np.ones_like(self.data)
        for node in reversed(topo):
            node._backward()

    def __repr__(self):
        return f"Tensor({self.data.tolist()}, requires_grad={self.requires_grad})"
