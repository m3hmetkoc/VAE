import numpy as np 
from tensor_class import Tensor, reparameterize

class Layer:
    def __init__(self, nin, nout, activation=None, dropout_rate=0.0, init_method='he'):
        """
        Enhanced layer with initialization options and dropout
        """
        # Initialize weights based on chosen method
        if init_method == 'he':
            self.W = Tensor(self.he_initialization((nin, nout)), requires_grad=True)
        elif init_method == 'xavier':
            self.W = Tensor(self.xavier_initialization((nin, nout)), requires_grad=True)
        else:  # Default uniform initialization
            self.W = Tensor(np.random.uniform(-1, 1, (nin, nout)), requires_grad=True)
            
        self.b = Tensor(np.zeros((1, nout)), requires_grad=True)  # Initialize bias to zeros
        self.activation = activation
        self.dropout = Dropout(dropout_rate) if dropout_rate > 0 else None
        self.training = True

    def he_initialization(self, shape):
        """
        He initialization for weights
        """
        fan_in = shape[0]
        std = np.sqrt(2.0 / fan_in)
        return np.random.normal(0, std, shape)

    def xavier_initialization(self, shape):
        """
        Xavier/Glorot initialization for weights
        """
        fan_in, fan_out = shape
        limit = np.sqrt(6 / (fan_in + fan_out))
        return np.random.uniform(-limit, limit, shape)

    def __call__(self, x):
        """
        Forward pass with dropout
        """
        z = x.matmul(self.W) + self.b

        # Apply activation
        if self.activation == "relu":
            out = z.relu()
        elif self.activation == "sigmoid":
            out = z.sigmoid()
        elif self.activation == "softmax":
            out = z.softmax()
        else:
            out = z
        # Apply dropout if present
        if self.dropout is not None:
            self.dropout.training = self.training
            out = self.dropout(out)

        return out

    def train(self):
        self.training = True

    def eval(self):
        self.training = False

    def parameters(self):
        """
        Collect parameters (weights and biases) of the layer.
        """
        return [self.W, self.b]
    
class Dropout(Layer):
    def __init__(self, p=0.5):
        """
        Dropout layer
        :param p: dropout probability (probability of setting a value to 0)
        """
        self.p = p
        self.mask = None
        self.training = True

    def __call__(self, x):
        if not self.training:  # During evaluation
            return x
        
        # Generate dropout mask
        self.mask = np.random.binomial(1, 1-self.p, size=x.data.shape) / (1-self.p)
        output = Tensor(x.data * self.mask, _children=(x,), _op="dropout")
        
        def _backward():
            if x.requires_grad:
                x.grad += output.grad * self.mask
                
        output._backward = _backward
        return output

    def parameters(self):
        return []

class NN:
    def __init__(self, nin, nouts, activations, dropout_rates=None, init_method='he'):
        """
        Enhanced Neural Network with dropout and initialization options
        """
        if dropout_rates is None:
            dropout_rates = [0.0] * len(nouts)
        
        assert len(nouts) == len(activations) == len(dropout_rates), \
            "Number of outputs, activations, and dropout rates must match"
        self.layers = [
            Layer(
                nin=nin if i == 0 else nouts[i - 1],
                nout=nouts[i],
                activation=activations[i],
                dropout_rate=dropout_rates[i],
                init_method=init_method
            )
            for i in range(len(nouts)) # FOR LOOP HERE, DO NOT PANIC. 
        ]

    def __call__(self, x):
        """
        Forward pass through all layers.
        :param x: Input matrix (batch_size, nin).
        :return: Output matrix after passing through all layers.
        """
        for layer in self.layers:
            x = layer(x)
        return x

    def parameters(self):
        """
        Collect parameters from all layers.
        """
        params = []
        for layer in self.layers:
            params.extend(layer.parameters())
        return params
    
    def train(self):
        """Set model to training mode"""
        for layer in self.layers:
            layer.train()

    def eval(self):
        """Set model to evaluation mode"""
        for layer in self.layers:
            layer.eval()


class Encoder:
    def __init__(self, input_dim, hidden_dim, latent_dim):
        self.fc1 = Layer(input_dim, hidden_dim, activation='relu', init_method='he')
        self.fc_mu = Layer(hidden_dim, latent_dim, activation=None)
        self.fc_logvar = Layer(hidden_dim, latent_dim, activation=None)

    def __call__(self, x):
        h = self.fc1(x)
        mu = self.fc_mu(h)
        logvar = self.fc_logvar(h)
        return mu, logvar
    
    def train(self): 
        self.fc1.train()
        self.fc_mu.train()
        self.fc_logvar.train()

    def eval(self): 
        self.fc1.eval()
        self.fc_mu.eval()
        self.fc_logvar.eval()

    def collect_params(self):
        return self.fc1.parameters() + self.fc_mu.parameters() + self.fc_logvar.parameters()
    

class Decoder:
    def __init__(self, latent_dim, hidden_dim, output_dim):
        self.fc1 = Layer(latent_dim, hidden_dim, activation='relu', init_method='he')
        self.fc2 = Layer(hidden_dim, output_dim, activation='sigmoid')

    def __call__(self, z):
        h = self.fc1(z)
        return self.fc2(h)

    def train(self): # Added
        self.fc1.train()
        self.fc2.train()

    def eval(self): # Added
        self.fc1.eval()
        self.fc2.eval()

    def collect_params(self):
        return self.fc1.parameters() + self.fc2.parameters()


class VAE:
    def __init__(self, input_dim, hidden_dim, latent_dim):
        self.encoder = Encoder(input_dim, hidden_dim, latent_dim)
        self.decoder = Decoder(latent_dim, hidden_dim, input_dim) #output_dim is input_dim for reconstruction

    def forward(self, x): # This is the main forward call
        mu, logvar = self.encoder(x)
        z = reparameterize(mu, logvar) # reparameterize trick to let gradients flow through the latent space. 
        reconstructed = self.decoder(z)
        return reconstructed, mu, logvar
    
    def __call__(self, x): 
        return self.forward(x)

    def parameters(self):
        ps = []
        for p in [self.encoder.collect_params() + self.decoder.collect_params()]:
            ps.extend(p) 
        return ps 

    def train(self): 
        self.encoder.train()
        self.decoder.train()

    def eval(self): 
        self.encoder.eval()
        self.decoder.eval()

