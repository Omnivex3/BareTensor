import numpy as np

class SGD:
    def __init__(self, parameters, lr=0.01):
        self.parameters = list(parameters)
        self.lr = lr

    def step(self):
        for p in self.parameters:
            if p.requires_grad and p.grad is not None:
                p.data -= self.lr * p.grad

    def zero_grad(self):
        for p in self.parameters:
            if p.requires_grad and p.grad is not None:
                p.grad = np.zeros_like(p.grad)

    def state_dict(self):
        return {'lr': self.lr}

    def load_state_dict(self, sd):
        self.lr = sd['lr']

class Adam:
    """Adam optimizer with bias correction and optional weight decay."""

    def __init__(self, parameters, lr=0.001, betas=(0.9, 0.999), eps=1e-8, weight_decay=0.0):
        self.parameters = list(parameters)
        self.lr = lr
        self.beta1, self.beta2 = betas
        self.eps = eps
        self.weight_decay = weight_decay
        self.t = 0
        self.state = {i: {} for i in range(len(self.parameters))}

    def step(self):
        self.t += 1
        for i, p in enumerate(self.parameters):
            if p.requires_grad and p.grad is not None:
                grad = p.grad
                if self.weight_decay != 0.0:
                    grad = grad + self.weight_decay * p.data
                s = self.state[i]
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

    def state_dict(self):
        return {
            'lr': self.lr,
            'betas': (self.beta1, self.beta2),
            'eps': self.eps,
            'weight_decay': self.weight_decay,
            't': self.t,
            'state': {str(i): {'m': s.get('m', np.array(0)).copy(),
                               'v': s.get('v', np.array(0)).copy()}
                      for i, s in self.state.items()},
        }

    def load_state_dict(self, sd):
        self.lr = sd['lr']
        self.beta1, self.beta2 = sd['betas']
        self.eps = sd['eps']
        self.weight_decay = sd['weight_decay']
        self.t = sd['t']
        self.state = {int(k): {'m': v['m'].copy(), 'v': v['v'].copy()}
                      for k, v in sd['state'].items()}


class AdamW:
    """AdamW optimizer with decoupled weight decay (Loshchilov & Hutter, 2019).

    Unlike Adam with L2 regularization, AdamW applies weight decay directly
    to the parameters rather than mixing it into the gradient. This decoupling
    is critical for modern training recipes.
    """

    def __init__(self, parameters, lr=0.001, betas=(0.9, 0.999), eps=1e-8, weight_decay=0.01):
        self.parameters = list(parameters)
        self.lr = lr
        self.beta1, self.beta2 = betas
        self.eps = eps
        self.weight_decay = weight_decay
        self.t = 0
        self.state = {i: {} for i in range(len(self.parameters))}

    def step(self):
        self.t += 1
        for i, p in enumerate(self.parameters):
            if p.requires_grad and p.grad is not None:
                grad = p.grad
                s = self.state[i]
                if 'm' not in s:
                    s['m'] = np.zeros_like(p.data)
                    s['v'] = np.zeros_like(p.data)
                s['m'] = self.beta1 * s['m'] + (1 - self.beta1) * grad
                s['v'] = self.beta2 * s['v'] + (1 - self.beta2) * (grad ** 2)
                m_hat = s['m'] / (1 - self.beta1 ** self.t)
                v_hat = s['v'] / (1 - self.beta2 ** self.t)
                # Decoupled weight decay (applied to original params before Adam step)
                if self.weight_decay != 0.0:
                    p.data *= (1.0 - self.lr * self.weight_decay)
                # Adam update
                p.data -= self.lr * m_hat / (np.sqrt(v_hat) + self.eps)

    def zero_grad(self):
        for p in self.parameters:
            if p.requires_grad and p.grad is not None:
                p.grad = np.zeros_like(p.grad)

    def state_dict(self):
        return {
            'lr': self.lr,
            'betas': (self.beta1, self.beta2),
            'eps': self.eps,
            'weight_decay': self.weight_decay,
            't': self.t,
            'state': {str(i): {'m': s.get('m', np.array(0)).copy(),
                               'v': s.get('v', np.array(0)).copy()}
                      for i, s in self.state.items()},
        }

    def load_state_dict(self, sd):
        self.lr = sd['lr']
        self.beta1, self.beta2 = sd['betas']
        self.eps = sd['eps']
        self.weight_decay = sd['weight_decay']
        self.t = sd['t']
        self.state = {int(k): {'m': v['m'].copy(), 'v': v['v'].copy()}
                      for k, v in sd['state'].items()}


def clip_grad_norm_(parameters, max_norm):
    """Clips the global gradient norm of an iterable of parameters in-place.

    Computes the total L2 norm over all parameters and scales each gradient
    if the total exceeds *max_norm*. Returns the total norm before clipping.

    Args:
        parameters: Iterable of Tensor with requires_grad=True.
        max_norm: Maximum allowed global gradient norm (positive float).

    Returns:
        Total L2 norm before clipping (float).
    """
    total_norm_sq = 0.0
    for p in parameters:
        if p.requires_grad and p.grad is not None:
            total_norm_sq += np.sum(p.grad ** 2)
    total_norm = float(np.sqrt(total_norm_sq))
    if total_norm > max_norm:
        scale = max_norm / total_norm
        for p in parameters:
            if p.requires_grad and p.grad is not None:
                p.grad *= scale
    return total_norm

class StepLR:
    """Decays the learning rate by *gamma* every *step_size* epochs.

    Args:
        optimizer: SGD, Adam, or AdamW instance.
        step_size: Number of epochs between LR drops.
        gamma: Multiplicative decay factor (default 0.1).
    """

    def __init__(self, optimizer, step_size, gamma=0.1):
        self.optimizer = optimizer
        self.step_size = step_size
        self.gamma = gamma
        self.base_lr = optimizer.lr
        self.last_epoch = -1
        self.step()

    def step(self):
        self.last_epoch += 1
        self.optimizer.lr = self.base_lr * (self.gamma ** (self.last_epoch // self.step_size))

    def get_lr(self):
        return self.optimizer.lr


class CosineAnnealingLR:
    """Cosine annealing learning rate scheduler (Loshchilov & Hutter, 2017).

    Args:
        optimizer: SGD, Adam, or AdamW instance.
        T_max: Number of epochs for a full half-cycle.
        eta_min: Minimum learning rate (default 0).
    """

    def __init__(self, optimizer, T_max, eta_min=0.0):
        self.optimizer = optimizer
        self.T_max = T_max
        self.eta_min = eta_min
        self.base_lr = optimizer.lr
        self.last_epoch = -1
        self.step()

    def step(self):
        self.last_epoch += 1
        self.optimizer.lr = self.eta_min + 0.5 * (self.base_lr - self.eta_min) * \
            (1.0 + np.cos(np.pi * self.last_epoch / self.T_max))

    def get_lr(self):
        return self.optimizer.lr
