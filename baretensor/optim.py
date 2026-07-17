import numpy as np

class SGD:
    def __init__(self, parameters, lr=0.01):
        self.parameters = parameters
        self.lr = lr

    def step(self):
        for p in self.parameters:
            if p.requires_grad and p.grad is not None:
                p.data -= self.lr * p.grad

    def zero_grad(self):
        for p in self.parameters:
            if p.requires_grad and p.grad is not None:
                p.grad = np.zeros_like(p.grad)

class Adam:
    """Adam optimizer with bias correction and optional weight decay."""

    def __init__(self, parameters, lr=0.001, betas=(0.9, 0.999), eps=1e-8, weight_decay=0.0):
        self.parameters = parameters
        self.lr = lr
        self.beta1, self.beta2 = betas
        self.eps = eps
        self.weight_decay = weight_decay
        self.t = 0
        self.state = {id(p): {} for p in self.parameters}

    def step(self):
        self.t += 1
        for p in self.parameters:
            if p.requires_grad and p.grad is not None:
                grad = p.grad
                if self.weight_decay != 0.0:
                    grad = grad + self.weight_decay * p.data
                pid = id(p)
                s = self.state[pid]
                if 'm' not in s:
                    s['m'] = np.zeros_like(p.data)
                    s['v'] = np.zeros_like(p.data)
                s['m'] = self.beta1 * s['m'] + (1 - self.beta1) * grad
                s['v'] = self.beta2 * s['v'] + (1 - self.beta2) * (grad ** 2)
                m_hat = s['m'] / (1 - self.beta1 ** self.t)
                v_hat = s['v'] / (1 - self.beta2 ** self.t)
                p.data -= self.lr * m_hat / (np.sqrt(v_hat) + self.eps)

    def zero_grad(self):
        for p in self.parameters:
            if p.requires_grad and p.grad is not None:
                p.grad = np.zeros_like(p.grad)
