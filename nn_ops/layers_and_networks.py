import numpy as np 
from .tensor_class import Tensor, reparameterize

class Layer:
    def __init__(self, nin, nout, activation=None, dropout_rate=0.0, init_method='he'):
        """
        Enhanced layer with initialization options and dropout
        """
        # Store configuration for saving/loading
        self.nin = nin
        self.nout = nout
        self.activation = activation
        self.dropout_rate = dropout_rate
        self.init_method = init_method
        
        # Initialize weights based on chosen method
        if init_method == 'he':
            self.W = Tensor(self.he_initialization((nin, nout)), requires_grad=True)
        elif init_method == 'xavier':
            self.W = Tensor(self.xavier_initialization((nin, nout)), requires_grad=True)
        else:  # Default uniform initialization
            self.W = Tensor(np.random.uniform(-1, 1, (nin, nout)), requires_grad=True)
            
        self.b = Tensor(np.zeros((1, nout)), requires_grad=True)
        self.dropout = Dropout(dropout_rate) if dropout_rate > 0 else None
        self.training = True

    def he_initialization(self, shape):
        fan_in = shape[0]
        std = np.sqrt(2.0 / fan_in)
        return np.random.normal(0, std, shape)

    def xavier_initialization(self, shape):
        fan_in, fan_out = shape
        limit = np.sqrt(6 / (fan_in + fan_out))
        return np.random.uniform(-limit, limit, shape)

    def __call__(self, x):
        z = x.matmul(self.W) + self.b

        if self.activation == "relu":
            out = z.relu()
        elif self.activation == "sigmoid":
            out = z.sigmoid()
        elif self.activation == "softmax":
            out = z.softmax()
        else:
            out = z
            
        if self.dropout is not None:
            self.dropout.training = self.training
            out = self.dropout(out)

        return out

    def train(self):
        self.training = True

    def eval(self):
        self.training = False

    def parameters(self):
        return [self.W, self.b]
    
    def get_config(self):
        return {
            'nin': self.nin,
            'nout': self.nout,
            'activation': self.activation,
            'dropout_rate': self.dropout_rate,
            'init_method': self.init_method
        }
    
class Dropout(Layer):
    def __init__(self, p=0.5):
        self.p = p
        self.mask = None
        self.training = True

    def __call__(self, x):
        if not self.training:
            return x
        
        self.mask = np.random.binomial(1, 1-self.p, size=x.data.shape) / (1-self.p)
        output = Tensor(x.data * self.mask, _children=(x,), _op="dropout")
        
        def _backward():
            if x.requires_grad:
                x.grad += output.grad * self.mask
                
        output._backward = _backward
        return output

    def parameters(self):
        return []

#Basic neural network implementation
class NN:
    def __init__(self, nin, nouts, activations, dropout_rates=None, init_method='he'):
        if dropout_rates is None:
            dropout_rates = [0.0] * len(nouts)
        
        assert len(nouts) == len(activations) == len(dropout_rates), \
            "Number of outputs, activations, and dropout rates must match"
        
        self.nin = nin
        self.nouts = nouts
        self.activations = activations
        self.dropout_rates = dropout_rates
        self.init_method = init_method
        
        self.layers = [
            Layer(
                nin=nin if i == 0 else nouts[i - 1],
                nout=nouts[i],
                activation=activations[i],
                dropout_rate=dropout_rates[i],
                init_method=init_method
            )
            for i in range(len(nouts))
        ]

    def __call__(self, x):
        for layer in self.layers:
            x = layer(x)
        return x

    def parameters(self):
        params = []
        for layer in self.layers:
            params.extend(layer.parameters())
        return params
    
    def get_config(self):
        return {
            'model_type': 'NN',
            'nin': self.nin,
            'nouts': self.nouts,
            'activations': self.activations,
            'dropout_rates': self.dropout_rates,
            'init_method': self.init_method
        }
    
    def train(self):
        for layer in self.layers:
            layer.train()

    def eval(self):
        for layer in self.layers:
            layer.eval()


class FlexibleEncoder:
    def __init__(self, input_dim, hidden_dims, latent_dim, activations=None, dropout_rates=None, init_method='he'):
        """
        Flexible encoder with configurable layers
        
        Args:
            input_dim: Input dimension
            hidden_dims: List of hidden layer dimensions [hidden1, hidden2, ...]
            latent_dim: Latent space dimension
            activations: List of activations for hidden layers (default: all 'relu')
            dropout_rates: List of dropout rates for hidden layers (default: all 0.0)
            init_method: Weight initialization method
        """
        self.input_dim = input_dim
        self.hidden_dims = hidden_dims if hidden_dims else []
        self.latent_dim = latent_dim
        self.init_method = init_method
        
        # Set default activations if not provided
        if activations is None:
            activations = ['relu'] * len(hidden_dims)
        
        # Set default dropout rates if not provided
        if dropout_rates is None:
            dropout_rates = [0.0] * len(hidden_dims)
            
        self.activations = activations
        self.dropout_rates = dropout_rates
        
        # Build hidden layers
        self.layers = []
        if hidden_dims:
            # First hidden layer
            self.layers.append(
                Layer(input_dim, hidden_dims[0], activations[0], dropout_rates[0], init_method)
            )
            
            # Additional hidden layers
            for i in range(1, len(hidden_dims)):
                self.layers.append(
                    Layer(hidden_dims[i-1], hidden_dims[i], activations[i], dropout_rates[i], init_method)
                )
        
        # Output layers (mu and logvar)
        final_input_dim = hidden_dims[-1] if hidden_dims else input_dim
        self.fc_mu = Layer(final_input_dim, latent_dim, activation=None, init_method=init_method)
        self.fc_logvar = Layer(final_input_dim, latent_dim, activation=None, init_method=init_method)

    def __call__(self, x):
        # Pass through hidden layers
        h = x
        for layer in self.layers:
            h = layer(h)
        
        # Get mu and logvar
        mu = self.fc_mu(h)
        logvar = self.fc_logvar(h)
        return mu, logvar
    
    def train(self): 
        for layer in self.layers:
            layer.train()
        self.fc_mu.train()
        self.fc_logvar.train()

    def eval(self): 
        for layer in self.layers:
            layer.eval()
        self.fc_mu.eval()
        self.fc_logvar.eval()

    def parameters(self):
        params = []
        for layer in self.layers:
            params.extend(layer.parameters())
        params.extend(self.fc_mu.parameters())
        params.extend(self.fc_logvar.parameters())
        return params
    
    def get_config(self):
        return {
            'input_dim': self.input_dim,
            'hidden_dims': self.hidden_dims,
            'latent_dim': self.latent_dim,
            'activations': self.activations,
            'dropout_rates': self.dropout_rates,
            'init_method': self.init_method
        }
    

class FlexibleDecoder:
    def __init__(self, latent_dim, hidden_dims, output_dim, activations=None, dropout_rates=None, init_method='he'):
        """
        Flexible decoder with configurable layers
        
        Args:
            latent_dim: Latent space dimension
            hidden_dims: List of hidden layer dimensions [hidden1, hidden2, ...]
            output_dim: Output dimension
            activations: List of activations for hidden layers + final activation
            dropout_rates: List of dropout rates for hidden layers (default: all 0.0)
            init_method: Weight initialization method
        """
        self.latent_dim = latent_dim
        self.hidden_dims = hidden_dims if hidden_dims else []
        self.output_dim = output_dim
        self.init_method = init_method
        
        # Set default activations if not provided
        if activations is None:
            print("Activation functions has not specified")
            activations = ['relu'] * len(hidden_dims) + ['sigmoid']  # Last layer typically sigmoid for reconstruction
        
        # Set default dropout rates if not provided
        if dropout_rates is None:
            print("Dropout rates has not specified")
            dropout_rates = [0.0] * len(hidden_dims)
            
        self.activations = activations
        self.dropout_rates = dropout_rates
        
        # Build all layers
        self.layers = []
        
        if hidden_dims:
            # First layer: latent_dim -> first hidden
            self.layers.append(
                Layer(latent_dim, hidden_dims[0], activations[0], dropout_rates[0], init_method)
            )
            
            # Hidden layers
            for i in range(1, len(hidden_dims)):
                self.layers.append(
                    Layer(hidden_dims[i-1], hidden_dims[i], activations[i], dropout_rates[i], init_method)
                )
            
            # Final layer: last hidden -> output
            final_activation = activations[len(hidden_dims)] if len(activations) > len(hidden_dims) else 'sigmoid'
            self.layers.append(
                Layer(hidden_dims[-1], output_dim, final_activation, 0.0, init_method)
            )
        else:
            # Direct connection: latent_dim -> output_dim
            final_activation = activations[0] if activations else 'sigmoid'
            self.layers.append(
                Layer(latent_dim, output_dim, final_activation, 0.0, init_method)
            )

    def __call__(self, z):
        h = z
        for layer in self.layers:
            h = layer(h)
        return h

    def train(self):
        for layer in self.layers:
            layer.train()

    def eval(self):
        for layer in self.layers:
            layer.eval()

    def parameters(self):
        params = []
        for layer in self.layers:
            params.extend(layer.parameters())
        return params
    
    def get_config(self):
        return {
            'latent_dim': self.latent_dim,
            'hidden_dims': self.hidden_dims,
            'output_dim': self.output_dim,
            'activations': self.activations,
            'dropout_rates': self.dropout_rates,
            'init_method': self.init_method
        }


class VAE:
    def __init__(self, input_dim, latent_dim, 
                 encoder_hidden_dims: list = None, decoder_hidden_dims: list = None,
                 encoder_activations: list = None, decoder_activations: list = None,
                 encoder_dropout_rates: list = None, decoder_dropout_rates: list = None,
                 init_method='he'):
        """
        Flexible VAE with configurable encoder and decoder architectures
        
        Args:
            input_dim: Input dimension (e.g., 784 for MNIST)
            latent_dim: Latent space dimension (e.g., 20)
            encoder_hidden_dims: List of hidden dimensions for encoder (e.g., [512, 256])
            decoder_hidden_dims: List of hidden dimensions for decoder (e.g., [256, 512])
            encoder_activations: List of activations for encoder layers
            decoder_activations: List of activations for decoder layers + final activation
            encoder_dropout_rates: List of dropout rates for encoder
            decoder_dropout_rates: List of dropout rates for decoder
            init_method: Weight initialization method
            
        Examples:
            # Simple VAE (like original)
            vae = FlexibleVAE(input_dim=784, latent_dim=20, 
                            encoder_hidden_dims=[256], decoder_hidden_dims=[256])
            
            # Deep VAE
            vae = FlexibleVAE(input_dim=784, latent_dim=20,
                            encoder_hidden_dims=[512, 256, 128],
                            decoder_hidden_dims=[128, 256, 512])
            
            # VAE with custom activations and dropout
            vae = FlexibleVAE(input_dim=784, latent_dim=20,
                            encoder_hidden_dims=[512, 256],
                            decoder_hidden_dims=[256, 512],
                            encoder_activations=['relu', 'relu'],
                            decoder_activations=['relu', 'relu', 'sigmoid'],
                            encoder_dropout_rates=[0.2, 0.1],
                            decoder_dropout_rates=[0.1, 0.0])
        """
        self.input_dim = input_dim
        self.latent_dim = latent_dim
        self.encoder_hidden_dims = encoder_hidden_dims if encoder_hidden_dims else [256]
        self.decoder_hidden_dims = decoder_hidden_dims if decoder_hidden_dims else [256]
        self.encoder_activations = encoder_activations
        self.decoder_activations = decoder_activations
        self.encoder_dropout_rates = encoder_dropout_rates
        self.decoder_dropout_rates = decoder_dropout_rates
        self.init_method = init_method
        
        self.encoder = FlexibleEncoder(
            input_dim=input_dim,
            hidden_dims=self.encoder_hidden_dims,
            latent_dim=latent_dim,
            activations=encoder_activations,
            dropout_rates=encoder_dropout_rates,
            init_method=init_method
        )
        
        self.decoder = FlexibleDecoder(
            latent_dim=latent_dim,
            hidden_dims=self.decoder_hidden_dims,
            output_dim=input_dim,
            activations=decoder_activations,
            dropout_rates=decoder_dropout_rates,
            init_method=init_method
        )

    def forward(self, x):
        mu, logvar = self.encoder(x)
        z = reparameterize(mu, logvar)
        reconstructed = self.decoder(z)
        return reconstructed, mu, logvar
    
    def __call__(self, x): 
        return self.forward(x)

    def parameters(self):
        params = []
        params.extend(self.encoder.parameters())
        params.extend(self.decoder.parameters())
        return params

    def get_config(self):
        return {
            'model_type': 'VAE',
            'input_dim': self.input_dim,
            'latent_dim': self.latent_dim,
            'encoder_hidden_dims': self.encoder_hidden_dims,
            'decoder_hidden_dims': self.decoder_hidden_dims,
            'encoder_activations': self.encoder_activations,
            'decoder_activations': self.decoder_activations,
            'encoder_dropout_rates': self.encoder_dropout_rates,
            'decoder_dropout_rates': self.decoder_dropout_rates,
            'init_method': self.init_method
        }

    def train(self): 
        self.encoder.train()
        self.decoder.train()

    def eval(self): 
        self.encoder.eval()
        self.decoder.eval()


# original VAE implementation
class VAE_old:
    def __init__(self, input_dim, hidden_dim, latent_dim, init_method='he'):
        """
        Original VAE implementation (kept for backward compatibility)
        """
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.latent_dim = latent_dim
        self.init_method = init_method
        
        # Use FlexibleVAE internally with single hidden layer
        self.flexible_vae = VAE(
            input_dim=input_dim,
            latent_dim=latent_dim,
            encoder_hidden_dims=[hidden_dim],
            decoder_hidden_dims=[hidden_dim],
            init_method=init_method
        )

    def forward(self, x):
        return self.flexible_vae.forward(x)
    
    def __call__(self, x): 
        return self.forward(x)

    def parameters(self):
        return self.flexible_vae.parameters()

    def get_config(self):
        return {
            'model_type': 'VAE',
            'input_dim': self.input_dim,
            'hidden_dim': self.hidden_dim,
            'latent_dim': self.latent_dim,
            'init_method': self.init_method
        }

    def train(self): 
        self.flexible_vae.train()

    def eval(self): 
        self.flexible_vae.eval()