import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt 
from .data_process import EarlyStopping
from .tensor_class import binary_cross_entropy, kl_divergence, Tensor

class Train:
    def __init__(self, model, train_generator, test_generator, num_epochs, learning_rate, batch_size, early_stopping_patience=5):
        
        self.model = model # This will be your VAE model
        self.train_generator = train_generator
        self.test_generator = test_generator
        self.num_epochs = num_epochs
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.early_stopping_patience = early_stopping_patience

        self.num_train_batches = len(train_generator.data_loader) # Or however you get num batches
        self.num_test_batches = len(test_generator.data_loader)   # Or however you get num batches
        self.early_stopping = EarlyStopping(patience=early_stopping_patience)

        # History will store VAE specific losses
        self.history = {
            'train_loss': [], 'val_loss': [],
            'train_recon_loss': [], 'val_recon_loss': [],
            'train_kld_loss': [], 'val_kld_loss': []
        }

    def lr_schedule(self, epoch):
        return self.learning_rate * (0.1 ** (epoch // 30)) # Or your preferred schedule
    
    def stochastic_gradient_descent(self, model_params, lr):
        for p in model_params:
            if p.requires_grad: # Ensure we only update params that require grad
                p.data -= lr * p.grad 

    def train_one_epoch(self, epoch, optimizer_name="sgd"):
        self.model.train()  # Set VAE to training mode
        model_params = self.model.parameters()

        total_epoch_loss = 0
        total_epoch_recon_loss = 0
        total_epoch_kld_loss = 0
        
        current_lr = self.lr_schedule(epoch)
    
        X_batch, _ = self.train_generator.get_next_batch()
        
        # Forward pass through VAE
        reconstructed_x, mu, logvar = self.model(X_batch)

        # Calculate loss components
        recon_loss = binary_cross_entropy(recon_x = reconstructed_x, x = X_batch)
        kld_loss = kl_divergence(mu, logvar)

        # Calculate total loss with beta weighting
        beta = 1.0
        # This is where the error was happening - need to ensure consistent scalar handling
        # Use scalar multiplication to apply beta
        weighted_kld = kld_loss * beta
        total_loss = recon_loss + weighted_kld

        # Zero gradients
        for p in model_params:
            if p.grad is not None:
                p.grad = np.zeros_like(p.data)
        
        # Backward pass---------------
        total_loss.backward()
        #---------------------
        # Update weights
        if optimizer_name == "sgd":
            self.stochastic_gradient_descent(model_params, current_lr)
        
        # Record losses
        total_epoch_loss += total_loss.data 
        total_epoch_recon_loss += recon_loss.data 
        total_epoch_kld_loss += kld_loss.data 

        avg_epoch_loss = total_epoch_loss / self.num_train_batches
        avg_epoch_recon_loss = total_epoch_recon_loss / self.num_train_batches
        avg_epoch_kld_loss = total_epoch_kld_loss / self.num_train_batches
        
        return avg_epoch_loss, avg_epoch_recon_loss, avg_epoch_kld_loss

    def evaluate(self):
        self.model.eval()  # Set VAE to evaluation mode
        
        total_val_loss = 0
        total_val_recon_loss = 0
        total_val_kld_loss = 0

        X_batch, _ = self.test_generator.get_next_batch()
        
        # Forward pass
        reconstructed_x, mu, logvar = self.model(X_batch)
        
        # Calculate loss components
        recon_loss = binary_cross_entropy(recon_x = reconstructed_x, x = X_batch)
        kld_loss = kl_divergence(mu, logvar)
        
        # Calculate total loss
        beta = 1.0
        weighted_kld = kld_loss * beta
        total_loss = recon_loss + weighted_kld
        
        # Record losses
        total_val_loss += total_loss.data 
        total_val_recon_loss += recon_loss.data 
        total_val_kld_loss += kld_loss.data 

        avg_val_loss = total_val_loss / self.num_test_batches
        avg_val_recon_loss = total_val_recon_loss / self.num_test_batches
        avg_val_kld_loss = total_val_kld_loss / self.num_test_batches

        return avg_val_loss, avg_val_recon_loss, avg_val_kld_loss

    def train(self, optimizer="sgd"):
        # Clear previous history
        self.history = {
            'train_total_loss': [], 'val_total_loss': [],
            'train_recon_loss': [], 'val_recon_loss': [],
            'train_kld_loss': [], 'val_kld_loss': []
        }
        
        #self.optimizer = Adamax()
        
        for epoch in range(self.num_epochs):
            train_loss, train_recon, train_kld = self.train_one_epoch(epoch, optimizer_name=optimizer)
            val_loss, val_recon, val_kld = self.evaluate()

            self.history['train_total_loss'].append(train_loss)
            self.history['train_recon_loss'].append(train_recon)
            self.history['train_kld_loss'].append(train_kld)
            self.history['val_total_loss'].append(val_loss)
            self.history['val_recon_loss'].append(val_recon)
            self.history['val_kld_loss'].append(val_kld)

            print(f'\nEpoch {epoch+1}/{self.num_epochs}:')
            print(f'Train: Total Loss: {train_loss:.4f} (Recon: {train_recon:.4f}, KLD: {train_kld:.4f})')
            print(f'Val:   Total Loss: {val_loss:.4f} (Recon: {val_recon:.4f}, KLD: {val_kld:.4f})')
            print(f'Learning Rate: {self.lr_schedule(epoch):.6f}')
            print(f"Total loss: {train_loss * self.num_train_batches}")
            z = Tensor(np.random.randn(1, 20), requires_grad=False)
            out = self.model.decoder(z)
            out = out.data.reshape(28,28)
            plt.imshow(out, cmap='gray', vmin=0, vmax=1)
            plt.show()

            self.early_stopping(val_loss) # Early stopping based on total validation loss
            if self.early_stopping.should_stop:
                print("Early stopping triggered!")
                break
        
        # After training, you might want to call plot_training_history
        # self.plot_training_history() # Pass self.history

    def plot_training_history(self): # Removed history argument, uses self.history
        """
        Plot training metrics for VAE.
        """
        # Ensure you have matplotlib.pyplot imported as plt
        fig, axes = plt.subplots(1, 3, figsize=(21, 5))
        
        # Plot Total Losses
        axes[0].plot(self.history['train_total_loss'], label='Train Total Loss')
        axes[0].plot(self.history['val_total_loss'], label='Validation Total Loss')
        axes[0].set_title('Total Loss History')
        axes[0].set_xlabel('Epoch')
        axes[0].set_ylabel('Loss')
        axes[0].legend()
        axes[0].grid(True)
        
        # Plot Reconstruction Losses
        axes[1].plot(self.history['train_recon_loss'], label='Train Reconstruction Loss')
        axes[1].plot(self.history['val_recon_loss'], label='Validation Reconstruction Loss')
        axes[1].set_title('Reconstruction Loss History')
        axes[1].set_xlabel('Epoch')
        axes[1].set_ylabel('Loss')
        axes[1].legend()
        axes[1].grid(True)

        # Plot KLD Losses
        axes[2].plot(self.history['train_kld_loss'], label='Train KLD Loss')
        axes[2].plot(self.history['val_kld_loss'], label='Validation KLD Loss')
        axes[2].set_title('KL Divergence History')
        axes[2].set_xlabel('Epoch')
        axes[2].set_ylabel('Loss')
        axes[2].legend()
        axes[2].grid(True)
        
        plt.tight_layout()
        plt.show()


import numpy as np

class Adamax:
    def __init__(self, parameters, lr=0.002, beta1=0.9, beta2=0.999, eps=1e-8):
        self.parameters = parameters
        self.lr = lr
        self.beta1 = beta1
        self.beta2 = beta2
        self.eps = eps
        self.t = 0
        self.m = [np.zeros_like(p.data) for p in parameters]
        self.u = [np.zeros_like(p.data) for p in parameters]

    def step(self):
        self.t += 1
        for i, p in enumerate(self.parameters):
            if p.grad is None:
                continue
            g = p.grad
            self.m[i] = self.beta1 * self.m[i] + (1 - self.beta1) * g
            self.u[i] = np.maximum(self.beta2 * self.u[i], np.abs(g))
            m_hat = self.m[i] / (1 - self.beta1 ** self.t)
            p.data -= self.lr * m_hat / (self.u[i] + self.eps)

    def zero_grad(self):
        for p in self.parameters:
            p.grad = np.zeros_like(p.data)
