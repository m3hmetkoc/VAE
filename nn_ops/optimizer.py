import numpy as np

class Adamax:
    """
    Adamax optimizer implementation.
    
    Adamax is a variant of Adam based on the infinity norm.
    It's particularly well-suited for sparse gradients and can be more stable
    than Adam in some scenarios.
    
    Args:
        parameters: List of parameters to optimize
        lr: Learning rate (default: 0.001)
        beta1: Coefficient for computing running averages of gradient (default: 0.9)
        beta2: Coefficient for computing running averages of squared gradient (default: 0.999)
        eps: Term added to denominator for numerical stability (default: 1e-8)
    """
    def __init__(self, parameters, lr=0.001, beta1=0.9, beta2=0.999, eps=1e-8):
        self.parameters = parameters
        self.lr = lr
        self.beta1 = beta1
        self.beta2 = beta2
        self.eps = eps
        self.t = 0  # Time step
        
        # Initialize momentum and infinity norm
        self.m = [np.zeros_like(p.data) for p in parameters]  # First moment
        self.u = [np.zeros_like(p.data) for p in parameters]  # Infinity norm

    def step(self):
        """Perform one optimization step"""
        self.t += 1
        
        for i, p in enumerate(self.parameters):
            if p.grad is None or np.allclose(p.grad, 0):
                continue
                
            g = p.grad
            
            # Update biased first moment estimate
            self.m[i] = self.beta1 * self.m[i] + (1 - self.beta1) * g
            
            # Update the exponentially weighted infinity norm
            self.u[i] = np.maximum(self.beta2 * self.u[i], np.abs(g))
            
            # Compute bias-corrected first moment estimate
            m_hat = self.m[i] / (1 - self.beta1 ** self.t)
            
            # Update parameters with small epsilon for numerical stability
            denominator = self.u[i] + self.eps
            update = self.lr * m_hat / denominator
            
            # Check for valid updates to prevent NaN propagation
            if np.isfinite(update).all():
                p.data -= update
            else:
                print(f"Warning: Non-finite update detected for parameter {i}, skipping update")

    def zero_grad(self):
        """Zero out gradients for all parameters"""
        for p in self.parameters:
            if p.grad is not None:
                p.grad = np.zeros_like(p.data)

    def get_state(self):
        """Get optimizer state for saving/loading"""
        return {
            'lr': self.lr,
            'beta1': self.beta1,
            'beta2': self.beta2,
            'eps': self.eps,
            't': self.t,
            'm': [m.copy() for m in self.m],
            'u': [u.copy() for u in self.u]
        }
    
    def load_state(self, state):
        """Load optimizer state"""
        self.lr = state['lr']
        self.beta1 = state['beta1']
        self.beta2 = state['beta2']
        self.eps = state['eps']
        self.t = state['t']
        self.m = [m.copy() for m in state['m']]
        self.u = [u.copy() for u in state['u']]


class Adam:
    """
    Adam optimizer implementation for comparison/alternative.
    
    Args:
        parameters: List of parameters to optimize
        lr: Learning rate (default: 0.001)
        beta1: Coefficient for computing running averages of gradient (default: 0.9)
        beta2: Coefficient for computing running averages of squared gradient (default: 0.999)
        eps: Term added to denominator for numerical stability (default: 1e-8)
    """
    def __init__(self, parameters, lr=0.001, beta1=0.9, beta2=0.999, eps=1e-8):
        self.parameters = parameters
        self.lr = lr
        self.beta1 = beta1
        self.beta2 = beta2
        self.eps = eps
        self.t = 0
        
        # Initialize momentum and squared gradient
        self.m = [np.zeros_like(p.data) for p in parameters]  # First moment
        self.v = [np.zeros_like(p.data) for p in parameters]  # Second moment

    def step(self):
        """Perform one optimization step"""
        self.t += 1
        
        for i, p in enumerate(self.parameters):
            if p.grad is None or np.allclose(p.grad, 0):
                continue
                
            g = p.grad
            
            # Update biased first moment estimate
            self.m[i] = self.beta1 * self.m[i] + (1 - self.beta1) * g
            
            # Update biased second raw moment estimate
            self.v[i] = self.beta2 * self.v[i] + (1 - self.beta2) * g**2
            
            # Compute bias-corrected first moment estimate
            m_hat = self.m[i] / (1 - self.beta1 ** self.t)
            
            # Compute bias-corrected second raw moment estimate
            v_hat = self.v[i] / (1 - self.beta2 ** self.t)
            
            # Update parameters
            denominator = np.sqrt(v_hat) + self.eps
            update = self.lr * m_hat / denominator
            
            # Check for valid updates
            if np.isfinite(update).all():
                p.data -= update
            else:
                print(f"Warning: Non-finite update detected for parameter {i}, skipping update")

    def zero_grad(self):
        """Zero out gradients for all parameters"""
        for p in self.parameters:
            if p.grad is not None:
                p.grad = np.zeros_like(p.data)

    def get_state(self):
        """Get optimizer state for saving/loading"""
        return {
            'lr': self.lr,
            'beta1': self.beta1,
            'beta2': self.beta2,
            'eps': self.eps,
            't': self.t,
            'm': [m.copy() for m in self.m],
            'v': [v.copy() for v in self.v]
        }
    
    def load_state(self, state):
        """Load optimizer state"""
        self.lr = state['lr']
        self.beta1 = state['beta1']
        self.beta2 = state['beta2']
        self.eps = state['eps']
        self.t = state['t']
        self.m = [m.copy() for m in state['m']]
        self.v = [v.copy() for v in state['v']]


class SGD:
    """
    Stochastic Gradient Descent optimizer with optional momentum.
    
    Args:
        parameters: List of parameters to optimize
        lr: Learning rate (default: 0.01)
        momentum: Momentum factor (default: 0)
        weight_decay: Weight decay (L2 penalty) (default: 0)
    """
    def __init__(self, parameters, lr=0.01, momentum=0, weight_decay=0):
        self.parameters = parameters
        self.lr = lr
        self.momentum = momentum
        self.weight_decay = weight_decay
        
        # Initialize momentum buffer
        self.velocity = [np.zeros_like(p.data) for p in parameters]

    def step(self):
        """Perform one optimization step"""
        for i, p in enumerate(self.parameters):
            if p.grad is None:
                continue
                
            g = p.grad
            
            # Add weight decay
            if self.weight_decay != 0:
                g = g + self.weight_decay * p.data
            
            # Update velocity
            self.velocity[i] = self.momentum * self.velocity[i] + g
            
            # Update parameters
            update = self.lr * self.velocity[i]
            
            # Check for valid updates
            if np.isfinite(update).all():
                p.data -= update
            else:
                print(f"Warning: Non-finite update detected for parameter {i}, skipping update")

    def zero_grad(self):
        """Zero out gradients for all parameters"""
        for p in self.parameters:
            if p.grad is not None:
                p.grad = np.zeros_like(p.data)

    def get_state(self):
        """Get optimizer state for saving/loading"""
        return {
            'lr': self.lr,
            'momentum': self.momentum,
            'weight_decay': self.weight_decay,
            'velocity': [v.copy() for v in self.velocity]
        }
    
    def load_state(self, state):
        """Load optimizer state"""
        self.lr = state['lr']
        self.momentum = state['momentum']
        self.weight_decay = state['weight_decay']
        self.velocity = [v.copy() for v in state['velocity']]


def get_optimizer(name, parameters, **kwargs):
    """
    Factory function to create optimizers by name.
    
    Args:
        name: Name of the optimizer ('adamax', 'adam', 'sgd')
        parameters: List of parameters to optimize
        **kwargs: Optimizer-specific arguments
        
    Returns:
        Optimizer instance
    """
    name = name.lower()
    
    if name == 'adamax':
        return Adamax(parameters, **kwargs)
    elif name == 'adam':
        return Adam(parameters, **kwargs)
    elif name == 'sgd':
        return SGD(parameters, **kwargs)
    else:
        raise ValueError(f"Unknown optimizer: {name}. Available: 'adamax', 'adam', 'sgd'")