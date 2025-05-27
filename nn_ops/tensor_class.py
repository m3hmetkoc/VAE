import numpy as np
import torch.nn as nn 

class Tensor:
    """
    A minimal autograd engine supporting numpy-based operations,
    automatic differentiation, broadcasting, and vectorized ops.
    """
    def __init__(self, data, requires_grad=True):
        if not isinstance(data, np.ndarray):
            data = np.array(data, dtype=np.float32)
        self.data = data
        self.grad = np.zeros_like(self.data) if requires_grad else None
        self.requires_grad = requires_grad
        self._prev = []
        self._op = ''
        self._backward = lambda: None

    def __repr__(self):
        return f"Tensor(data={self.data}, grad={self.grad})"

    # --- Basic ops ---
    def __add__(self, other):
        other = other if isinstance(other, Tensor) else Tensor(other, requires_grad=False)
        out = Tensor(self.data + other.data)
        out._prev = [self, other]
        out._op = 'add'
        def _backward():
            if self.requires_grad:
                grad_self = Tensor._unbroadcast(out.grad, self.data.shape)
                self.grad += grad_self
            if other.requires_grad:
                grad_other = Tensor._unbroadcast(out.grad, other.data.shape)
                other.grad += grad_other
        out._backward = _backward
        return out

    def __mul__(self, other):
        other = other if isinstance(other, Tensor) else Tensor(other, requires_grad=False)
        out = Tensor(self.data * other.data)
        out._prev = [self, other]
        out._op = 'mul'
        def _backward():
            if self.requires_grad:
                grad_self = Tensor._unbroadcast(other.data * out.grad, self.data.shape)
                self.grad += grad_self
            if other.requires_grad:
                grad_other = Tensor._unbroadcast(self.data * out.grad, other.data.shape)
                other.grad += grad_other
        out._backward = _backward
        return out

    def matmul(self, other):
        assert isinstance(other, Tensor)
        out = Tensor(self.data.dot(other.data))
        out._prev = [self, other]
        out._op = 'matmul'
        def _backward():
            if self.requires_grad:
                self.grad += out.grad.dot(other.data.T)
            if other.requires_grad:
                other.grad += self.data.T.dot(out.grad)
        out._backward = _backward
        return out

    # --- Reductions and shape ---
    def sum(self, axis=None, keepdims=False):
        out = Tensor(self.data.sum(axis=axis, keepdims=keepdims))
        out._prev = [self]
        out._op = 'sum'
        def _backward():
            grad = out.grad
            shape = self.data.shape
            if not keepdims and axis is not None:
                grad = grad.reshape([1 if i in (axis if isinstance(axis, tuple) else (axis,)) else dim for i, dim in enumerate(shape)])
            grad = np.broadcast_to(grad, shape)
            self.grad += grad
        out._backward = _backward
        return out

    def reshape(self, *shape):
        out = Tensor(self.data.reshape(*shape))
        out._prev = [self]
        out._op = 'reshape'
        def _backward():
            self.grad += out.grad.reshape(self.data.shape)
        out._backward = _backward
        return out

    def clip(self, min_val, max_val):
        out_data = np.clip(self.data, min_val, max_val)
        out = Tensor(out_data, requires_grad=self.requires_grad)

        if self.requires_grad:
            def _backward():
                grad_mask = (self.data >= min_val) & (self.data <= max_val)
                self.grad += grad_mask * out.grad
            out._backward = _backward
            out._prev = {self}
        
        return out

    # --- Activations and elementwise ---
    def log(self):
        out = Tensor(np.log(self.data))
        out._prev = [self]
        out._op = 'log'
        def _backward(): self.grad += (1 / self.data) * out.grad
        out._backward = _backward
        return out

    def exp(self):
        out = Tensor(np.exp(self.data))
        out._prev = [self]
        out._op = 'exp'
        def _backward(): self.grad += out.data * out.grad
        out._backward = _backward
        return out

    def sigmoid(self):
        out = Tensor(1 / (1 + np.exp(-self.data)))
        out._prev = [self]
        out._op = 'sigmoid'
        def _backward(): self.grad += (out.data * (1 - out.data)) * out.grad
        out._backward = _backward
        return out

    def relu(self):
        out = Tensor(np.where(self.data > 0, self.data, 0))
        out._prev = [self]
        out._op = 'relu'
        def _backward(): self.grad += (self.data > 0).astype(np.float32) * out.grad
        out._backward = _backward
        return out

    def __neg__(self): return self * -1
    def __sub__(self, other): return self + (other * -1)
    def __truediv__(self, other): return self * other**-1
    def __pow__(self, power):
        assert isinstance(power, (int, float)), "only supports int/float powers"
        out = Tensor(self.data ** power)
        out._prev = [self]
        out._op = f'pow{power}'
        def _backward(): self.grad += (power * (self.data ** (power-1))) * out.grad
        out._backward = _backward
        return out
    def __rmul__(self, other): return self * other 
    def __radd__(self, other): return self + other 
    def __rsub__(self, other): return other + (self * -1)

    @staticmethod
    def _unbroadcast(grad, shape):
        grad_shape = grad.shape
        if grad_shape == shape: return grad
        # sum extra dims
        while len(grad_shape) > len(shape):
            grad = grad.sum(axis=0)
            grad_shape = grad.shape
        # sum broadcast dims
        for i, dim in enumerate(shape):
            if dim == 1 and grad_shape[i] > 1:
                grad = grad.sum(axis=i, keepdims=True)
        return grad

    def backward(self):
        topo, visited = [], set()
        def build(v):
            if v not in visited:
                visited.add(v)
                for child in v._prev: build(child)
                topo.append(v)
        build(self)
        self.grad = np.ones_like(self.data)
        for node in reversed(topo): node._backward()

# --- VAE-specific helpers (to be used in your model definitions/layers) ---

def reparameterize(mu: Tensor, logvar: Tensor) -> Tensor:
    """
    Applies the reparameterization trick:
      z = mu + std * eps,  eps ~ N(0,1)
    Keep this in your models forward pass (not inside Tensor class).
    """
    std = (logvar * 0.5).exp()
    eps = Tensor(np.random.randn(*mu.data.shape), requires_grad=False)
    return mu + std * eps


def binary_cross_entropy(recon_x: Tensor, x: Tensor, eps=1e-7) -> Tensor:
    """
    BCE per batch:  -[x*log(recon_x) + (1-x)*log(1-recon_x)].sum() / batch_size
    """
    recon_x_clamped = recon_x.clip(eps, 1 - eps)
    term1 = x * recon_x_clamped.log()
    term2 = (1-x) * (1-recon_x_clamped).log()
    return (-(term1 + term2)).sum() / recon_x.data.shape[0]


def kl_divergence(mu: Tensor, logvar: Tensor) -> Tensor:
    """
    FIXED: KL divergence between N(mu, var) and N(0,1):
    KL = 0.5 * sum(mu^2 + exp(logvar) - logvar - 1) / batch_size
    
    Original formula was: -0.5 * sum(1 + logvar - mu^2 - exp(logvar))
    Which is equivalent to: 0.5 * sum(mu^2 + exp(logvar) - logvar - 1)
    """
    # Correct implementation
    kld = (-0.5 * (1 + logvar - mu**2 - logvar.exp())).sum()
    return kld / mu.data.shape[0]