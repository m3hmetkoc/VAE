import numpy as np 
import matplotlib.pyplot as plt # Keep if you adapt plotting later
from .data_process import EarlyStopping
from .tensor_class import Tensor, kl_divergence, binary_cross_entropy

class Train:
    def __init__(self, model, train_generator, test_generator, num_epochs, learning_rate, batch_size, early_stopping_patience=5):
        self.model = model
        self.train_generator = train_generator
        self.test_generator = test_generator
        self.num_epochs = num_epochs
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.early_stopping_patience = early_stopping_patience
        self.model_params = self.model.parameters()

        self.num_train_batches = len(train_generator.data_loader)
        print(self.num_train_batches)
        self.num_test_batches = len(test_generator.data_loader)
        print(self.num_test_batches)
        self.early_stopping = EarlyStopping(patience=early_stopping_patience)

        self.history = {
            'train_total_loss': [], 'val_total_loss': [],
            'train_recon_loss': [], 'val_recon_loss': [],
            'train_kld_loss': [], 'val_kld_loss': []
        }

    def lr_schedule(self, epoch):
        return self.learning_rate * (0.95 ** (epoch // 10))  # More gentle decay
    
    def train_one_epoch(self, epoch):
        self.model.train()
        
        total_epoch_loss = 0
        total_epoch_recon_loss = 0
        total_epoch_kld_loss = 0
        
        current_lr = self.lr_schedule(epoch)
        batch_count = 0
        
        # CORRECTED: Process ALL batches in the epoch
        for batch_idx in range(self.num_train_batches):
            # Zero gradients at start of each batch
            self.optimizer.zero_grad()
            
            X_batch, _ = self.train_generator.get_next_batch()
            
            # Forward pass
            reconstructed_x, mu, logvar = self.model(X_batch)
            
            # Calculate losses
            recon_loss = binary_cross_entropy(recon_x=reconstructed_x, x=X_batch)
            kld_loss = kl_divergence(mu, logvar)
            
            # Total loss with beta weighting (start with lower beta for stable training)
            beta = min(1.0, 0.1 + 0.01 * epoch)  # Gradual beta increase
            total_loss = recon_loss + kld_loss * beta
            
            # Backward pass
            total_loss.backward()
            
            # Gradient clipping for stability
            for param in self.model_params:
                if param.grad is not None:
                    param.grad = np.clip(param.grad, -1.0, 1.0)
            
            # Update weights
            self.optimizer.step()
            
            # Accumulate losses
            total_epoch_loss += total_loss.data
            total_epoch_recon_loss += recon_loss.data
            total_epoch_kld_loss += kld_loss.data
            batch_count += 1
            
            # Clear gradients for next iteration
            self.optimizer.zero_grad()
        
        # Calculate averages
        avg_epoch_loss = total_epoch_loss / (batch_count * self.batch_size)
        avg_epoch_recon_loss = total_epoch_recon_loss / batch_count
        avg_epoch_kld_loss = total_epoch_kld_loss / batch_count
        
        return avg_epoch_loss, avg_epoch_recon_loss, avg_epoch_kld_loss

    def evaluate(self):
        self.model.eval()
        
        total_val_loss = 0
        total_val_recon_loss = 0
        total_val_kld_loss = 0
        batch_count = 0
        
        # CORRECTED: Process all validation batches
        for batch_idx in range(self.num_test_batches):
            X_batch, _ = self.test_generator.get_next_batch()
            
            # Forward pass (no gradient computation needed)
            reconstructed_x, mu, logvar = self.model(X_batch)
            
            # Calculate losses
            recon_loss = binary_cross_entropy(recon_x=reconstructed_x, x=X_batch)
            kld_loss = kl_divergence(mu, logvar)
            
            beta = 1.0  # Use full beta for evaluation
            total_loss = recon_loss + kld_loss * beta 
            
            total_val_loss += total_loss.data
            total_val_recon_loss += recon_loss.data
            total_val_kld_loss += kld_loss.data
            batch_count += 1
        
        avg_val_loss = total_val_loss / batch_count
        avg_val_recon_loss = total_val_recon_loss / batch_count
        avg_val_kld_loss = total_val_kld_loss / batch_count
        
        return avg_val_loss, avg_val_recon_loss, avg_val_kld_loss

    def train(self, optimizer="adamax"):
        # Clear history
        self.history = {
            'train_total_loss': [], 'val_total_loss': [],
            'train_recon_loss': [], 'val_recon_loss': [],
            'train_kld_loss': [], 'val_kld_loss': []
        }
        
        # Initialize optimizer with lower learning rate for stability
        self.optimizer = Adamax(self.model_params, self.learning_rate)
        
        for epoch in range(self.num_epochs):
            train_loss, train_recon, train_kld = self.train_one_epoch(epoch)
            val_loss, val_recon, val_kld = self.evaluate()
            
            # Store history
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
            
            # Generate sample during training for monitoring
            if epoch % 5 == 0:
                z = Tensor(np.random.randn(1, 20), requires_grad=False)
                out = self.model.decoder(z)
                out = out.data.reshape(28, 28)
                plt.imshow(out, cmap='gray', vmin=0, vmax=1)
                plt.title(f'Generated Sample - Epoch {epoch+1}')
                plt.show()
            
            # Early stopping
            self.early_stopping(val_loss)
            if self.early_stopping.should_stop:
                print("Early stopping triggered!")
                break

# Corrected Adamax optimizer
class Adamax:
    def __init__(self, parameters, lr=0.001, beta1=0.9, beta2=0.999, eps=1e-8):
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
            if p.grad is None or np.allclose(p.grad, 0):
                continue
                
            g = p.grad
            
            # Update biased first moment estimate
            self.m[i] = self.beta1 * self.m[i] + (1 - self.beta1) * g
            
            # Update the exponentially weighted infinity norm
            self.u[i] = np.maximum(self.beta2 * self.u[i], np.abs(g))
            
            # Compute bias-corrected first moment estimate
            m_hat = self.m[i] / (1 - self.beta1 ** self.t)
            
            # Update parameters
            p.data -= self.lr * m_hat / (self.u[i] + self.eps)

    def zero_grad(self):
        for p in self.parameters:
            if p.grad is not None:
                p.grad = np.zeros_like(p.data)