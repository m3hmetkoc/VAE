import numpy as np

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
    
    def concatenate_tensors(self, other):
        """
        Properly concatenate tensors while maintaining gradient flow
        """
        # Concatenate the data
        concatenated_data = np.concatenate([self.data, other.data], axis=1)
        
        # Create new tensor with proper gradient tracking
        concatenated = Tensor(concatenated_data, requires_grad=self.requires_grad or other.requires_grad)
        
        # Set up gradient computation
        if self.requires_grad or other.requires_grad:
            concatenated._prev = [self, other]
            concatenated._op = 'concatenate'
            
            def _backward():
                if self.requires_grad:
                    self.grad += concatenated.grad[:, :self.data.shape[1]]
                if other.requires_grad:
                    other.grad += concatenated.grad[:, other.data.shape[1]:]
            
            concatenated._backward = _backward
        
        return concatenated

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
    
    def softmax(self):
        shifted_data = self.data - np.max(self.data, axis=1, keepdims=True)
        exp_data = np.exp(shifted_data)
        softmax_data = exp_data / np.sum(exp_data, axis=1, keepdims=True)
        out = Tensor(softmax_data)
        out._prev = [self] 
        out._op = 'softmax' 
        def _backward():
            if self.requires_grad:
                batch_size = self.data.shape[0]
                n_classes = self.data.shape[1]
                dx = np.zeros_like(self.data)
                for i in range(batch_size):
                    s = softmax_data[i]
                    for k in range(n_classes):
                        for j in range(n_classes):
                            if k == j:
                                dx[i, k] += s[k] * (1 - s[k]) * out.grad[i, k]
                            else:
                                dx[i, k] += -s[k] * s[j] * out.grad[i, j]
                self.grad += dx
        out._backward = _backward
        return out

    def __pow__(self, power):
        assert isinstance(power, (int, float)), "only supports int/float powers"
        out = Tensor(self.data ** power)
        out._prev = [self]
        out._op = f'pow{power}'
        def _backward(): self.grad += (power * (self.data ** (power-1))) * out.grad
        out._backward = _backward
        return out
    
    def __neg__(self): return self * -1
    def __sub__(self, other): return self + (other * -1)
    def __truediv__(self, other): return self * other**-1
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

    def conv2d(self, weight: "Tensor", bias: "Tensor" = None, stride: int = 1, padding: int = 0) -> "Tensor":
        """
        2-D convolution (cross-correlation) using a naïve implementation that works
        with the autograd engine. The expected tensor shapes are:
          • input  : (N, C_in, H, W)
          • weight : (C_out, C_in, K_h, K_w)
          • bias   : (C_out,) or (1, C_out, 1, 1)  (optional)

        Args:
            weight (Tensor): convolution kernels/filters.
            bias   (Tensor, optional): bias term.
            stride (int): stride for both H and W dimensions.
            padding (int): implicit zero-padding on both H and W dimensions.

        Returns:
            Tensor: Output of the convolution, shape (N, C_out, H_out, W_out)
        """
        # Ensure tensors
        assert isinstance(weight, Tensor), "weight must be a Tensor"
        if bias is not None and not isinstance(bias, Tensor):
            bias = Tensor(bias, requires_grad=False)

        N, C_in, H, W = self.data.shape
        C_out, C_in_w, K_h, K_w = weight.data.shape
        assert C_in == C_in_w, "Input channels mismatch between input and weight"

        # Output dimensions
        H_out = (H + 2 * padding - K_h) // stride + 1
        W_out = (W + 2 * padding - K_w) // stride + 1

        # Pad input if necessary (only spatial dims)
        if padding > 0:
            padded = np.pad(
                self.data, ((0, 0), (0, 0), (padding, padding), (padding, padding)), mode="constant"
            )
        else:
            padded = self.data
        # Prepare output container
        out_data = np.zeros((N, C_out, H_out, W_out), dtype=self.data.dtype)

        # Forward pass (naïve loops – can be optimized later)
        for n in range(N):
            for c_out in range(C_out):
                for i in range(H_out):
                    for j in range(W_out):
                        h_start = i * stride
                        w_start = j * stride
                        region = padded[n, :, h_start : h_start + K_h, w_start : w_start + K_w]
                        out_data[n, c_out, i, j] = np.sum(region * weight.data[c_out])
                        if bias is not None:
                            # Bias broadcasting – works for (C_out,) or compatible shapes
                            out_data[n, c_out, i, j] += bias.data.flatten()[c_out]

        # Determine if gradients are required for the output
        requires_grad = self.requires_grad or weight.requires_grad or (bias.requires_grad if bias is not None else False)
        out = Tensor(out_data, requires_grad=requires_grad)
        out._prev = [self, weight] + ([bias] if bias is not None else [])
        out._op = "conv2d"

        # Backward closure – captures the context of this forward pass
        if requires_grad:
            def _backward():
                dout = out.grad  # (N, C_out, H_out, W_out)

                # Gradient w.r.t. bias
                if bias is not None and bias.requires_grad:
                    # Sum over N, H_out, W_out for each output channel
                    if bias.grad is None:
                        bias.grad = np.zeros_like(bias.data)
                    bias.grad += dout.sum(axis=(0, 2, 3)).reshape(bias.data.shape)

                # Gradient w.r.t. weight
                if weight.requires_grad:
                    if weight.grad is None:
                        weight.grad = np.zeros_like(weight.data)
                    dw = np.zeros_like(weight.data)
                    for n in range(N):
                        for c_out in range(C_out):
                            for i in range(H_out):
                                for j in range(W_out):
                                    h_start = i * stride
                                    w_start = j * stride
                                    region = padded[n, :, h_start : h_start + K_h, w_start : w_start + K_w]
                                    dw[c_out] += region * dout[n, c_out, i, j]
                    weight.grad += dw

                # Gradient w.r.t. input
                if self.requires_grad:
                    if self.grad is None:
                        self.grad = np.zeros_like(self.data)
                    dx_padded = np.zeros_like(padded)
                    for n in range(N):
                        for c_out in range(C_out):
                            for i in range(H_out):
                                for j in range(W_out):
                                    h_start = i * stride
                                    w_start = j * stride
                                    dx_padded[n, :, h_start : h_start + K_h, w_start : w_start + K_w] += (
                                        weight.data[c_out] * dout[n, c_out, i, j]
                                    )
                    # Remove padding
                    if padding > 0:
                        self.grad += dx_padded[:, :, padding : padding + H, padding : padding + W]
                    else:
                        self.grad += dx_padded

            out._backward = _backward

        return out

# --- VAE-specific helpers (to be used in model definitions/layers) 
# --- implemented with operations of the Tensor class to keep track of the gradients

def reparameterize(mu: Tensor, logvar: Tensor) -> Tensor:
    """
    Applies the reparameterization trick:
      z = mu + std * eps,  eps ~ N(0,1)
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
    Original formula: -0.5 * sum(1 + logvar - mu^2 - exp(logvar))
    """
    kld = (-0.5 * (1 + logvar - mu**2 - logvar.exp())).sum()
    return kld / mu.data.shape[0]

def cross_entropy_loss(y_pred: Tensor, y_true: Tensor, eps = 1e-12) -> Tensor:
    """
    Cross Entropy loss: -(y_true)*log(y_pred)].sum()
    """
    y_pred_clamped = y_pred.clip(eps, 1. - eps)
    loss = -(y_true * y_pred_clamped.log()).sum()
    batch_size = y_pred.data.shape[0]
    return loss / batch_size