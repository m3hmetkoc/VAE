import numpy as np 
import matplotlib.pyplot as plt
from tqdm import tqdm
from .data_process import EarlyStopping
from .tensor_class import Tensor, kl_divergence, binary_cross_entropy

class Train:
    def __init__(self, model, latent_dim, train_generator, test_generator, num_epochs, learning_rate, batch_size, train_cvae, early_stopping_patience=5):
        self.model = model
        self.latent_dimension_size = latent_dim
        self.train_generator = train_generator
        self.test_generator = test_generator
        self.num_epochs = num_epochs
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.train_cvae = train_cvae
        self.model_params = self.model.parameters()
        self.num_test_batches = len(test_generator.data_loader)
        self.num_train_batches = len(train_generator.data_loader)

        self.early_stopping_patience = early_stopping_patience
        self.early_stopping = EarlyStopping(patience=early_stopping_patience)

        self.history = {
            'train_total_loss': [], 'val_total_loss': [],
            'train_recon_loss': [], 'val_recon_loss': [],
            'train_kld_loss': [], 'val_kld_loss': []
        }

    def lr_schedule(self, epoch):
        """Learning rate schedule with gentle decay"""
        return self.learning_rate * (0.95 ** (epoch // 10))
    
    def train_one_epoch(self, epoch):
        """Train for one epoch with progress bar"""
        self.model.train()
        
        total_epoch_loss, total_epoch_recon_loss, total_epoch_kld_loss = 0, 0, 0
        
        # Progress bar for training batches
        train_pbar = tqdm(range(self.num_train_batches), 
                         desc=f'Epoch {epoch+1}/{self.num_epochs} [Train]',
                         leave=False)
        
        for batch_idx in train_pbar:
            # Zero gradients at start of each batch
            self.optimizer.zero_grad()
            
            try:
                X_batch, labels = self.train_generator.get_next_batch()
                if self.train_cvae:
                # Forward pass
                    X_concat = Tensor(np.concatenate([X_batch.data, labels.data], axis=1), requires_grad=False)
                    reconstructed_x, mu, logvar = self.model.forward(X_concat, labels) 
                else:
                    reconstructed_x, mu, logvar = self.model.forward(X_batch, labels) 
                # Losses
                recon_loss = binary_cross_entropy(recon_x=reconstructed_x, x=X_batch)
                kld_loss = kl_divergence(mu, logvar)
                
                # Total loss with beta weighting (gradual beta increase for stable training)
                beta = min(1.0, 0.1 + 0.01 * epoch)
                total_loss = recon_loss + kld_loss * beta
                
                # Check for NaN or infinite values
                if not np.isfinite(total_loss.data):
                    print(f"Warning: Non-finite loss detected at batch {batch_idx}")
                    continue
                
                # Backward pass
                total_loss.backward()
                
                # Optional gradient clipping for stability (uncomment if needed)
                # self._clip_gradients(clip_value=1.0)
                
                # Update weights
                self.optimizer.step()
                
                # Accumulate losses
                total_epoch_loss += total_loss.data
                total_epoch_recon_loss += recon_loss.data
                total_epoch_kld_loss += kld_loss.data
                
                # Update progress bar with current loss
                train_pbar.set_postfix({
                    'Loss': f'{total_loss.data:.4f}',
                    'Recon': f'{recon_loss.data:.4f}',
                    'KLD': f'{kld_loss.data:.4f}',
                    'Beta': f'{beta:.3f}'
                })
                
            except Exception as e:
                print(f"Error in batch {batch_idx}: {e}")
                continue
        
        # Calculate averages (per sample, not per batch)
        num_samples = self.num_train_batches 
        avg_epoch_loss = total_epoch_loss / num_samples
        avg_epoch_recon_loss = total_epoch_recon_loss / num_samples
        avg_epoch_kld_loss = total_epoch_kld_loss / num_samples
        
        return avg_epoch_loss, avg_epoch_recon_loss, avg_epoch_kld_loss

    def evaluate(self):
        """Evaluate model on validation set with progress bar"""
        self.model.eval()
        
        total_val_loss, total_val_recon_loss, total_val_kld_loss = 0, 0, 0
        
        # Progress bar for validation batches
        val_pbar = tqdm(range(self.num_test_batches), 
                       desc='Validation', 
                       leave=False)
        
        for batch_idx in val_pbar:
            try:
                X_batch, labels = self.train_generator.get_next_batch()
                if self.train_cvae:
                # Forward pass
                    X_concat = Tensor(np.concatenate([X_batch.data, labels.data], axis=1), requires_grad=False)
                    reconstructed_x, mu, logvar = self.model.forward(X_concat, labels) 
                else:
                    reconstructed_x, mu, logvar = self.model.forward(X_batch, labels) 
                # Losses
                recon_loss = binary_cross_entropy(recon_x=reconstructed_x, x=X_batch)
                kld_loss = kl_divergence(mu, logvar)
                
                beta = 1.0  # Use full beta for evaluation
                total_loss = recon_loss + kld_loss * beta 
                
                # Check for valid loss values
                if np.isfinite(total_loss.data):
                    total_val_loss += total_loss.data
                    total_val_recon_loss += recon_loss.data
                    total_val_kld_loss += kld_loss.data
                
                # Update progress bar
                val_pbar.set_postfix({
                    'Loss': f'{total_loss.data:.4f}',
                    'Recon': f'{recon_loss.data:.4f}',
                    'KLD': f'{kld_loss.data:.4f}'
                })
                
            except Exception as e:
                print(f"Error in validation batch {batch_idx}: {e}")
                continue
        
        # Calculate averages (per sample, not per batch)
        num_samples = self.num_test_batches
        avg_val_loss = total_val_loss / num_samples
        avg_val_recon_loss = total_val_recon_loss / num_samples
        avg_val_kld_loss = total_val_kld_loss / num_samples
        
        return avg_val_loss, avg_val_recon_loss, avg_val_kld_loss


    def _clip_gradients(self, clip_value=1.0):
        """Gradient clipping for stability"""
        for param in self.model_params:
            if param.grad is not None:
                param.grad = np.clip(param.grad, -clip_value, clip_value)

    def _generate_sample_image(self, epoch, save_path="images"):
        """Generate and save a sample image during training"""
        try:
            import os
            os.makedirs(save_path, exist_ok=True)
            if self.train_cvae:
                z = Tensor(data = np.concatenate([np.random.randn(1, self.latent_dimension_size), np.array([[0., 0., 0., 0., 0., 0., 0., 1., 0., 0.]])], axis=1), requires_grad=False)
            else:
                z = Tensor(np.random.randn(1, self.latent_dimension_size), requires_grad=False)
            out = self.model.decoder(z)
            out_data = out.data.reshape(28, 28)
            
            # Ensure values are in valid range for image saving
            out_data = np.clip(out_data, 0, 1)
            
            plt.imsave(f"{save_path}/generated_image_epoch_{epoch}.png", 
                      out_data, cmap="gray")
        except Exception as e:
            print(f"Could not save sample image: {e}")

    def train(self, optimizer="adamax"):
        """Main training loop with progress tracking"""
        print(f"Starting training for {self.num_epochs} epochs...")
        print(f"Training batches: {self.num_train_batches}, Validation batches: {self.num_test_batches}")
        print(f"Batch size: {self.batch_size}")
        
        # Clear history
        self.history = {
            'train_total_loss': [], 'val_total_loss': [],
            'train_recon_loss': [], 'val_recon_loss': [],
            'train_kld_loss': [], 'val_kld_loss': []
        }
        
        # Initialize optimizer
        self.optimizer = Adamax(self.model_params, lr=self.learning_rate)
        
        # Main training loop with epoch progress bar
        epoch_pbar = tqdm(range(self.num_epochs), desc="Training Progress")
        
        for epoch in epoch_pbar:
            # Update learning rate
            current_lr = self.lr_schedule(epoch)
            self.optimizer.lr = current_lr
            
            train_loss, train_recon, train_kld = self.train_one_epoch(epoch)
            
            # Evaluate on validation set
            val_loss, val_recon, val_kld = self.evaluate()
            
            # Store history
            self.history['train_total_loss'].append(train_loss)
            self.history['train_recon_loss'].append(train_recon)
            self.history['train_kld_loss'].append(train_kld)
            self.history['val_total_loss'].append(val_loss)
            self.history['val_recon_loss'].append(val_recon)
            self.history['val_kld_loss'].append(val_kld)
            
            # Update epoch progress bar
            epoch_pbar.set_postfix({
                'Train Loss': f'{train_loss:.4f}',
                'Val Loss': f'{val_loss:.4f}',
                'LR': f'{current_lr:.6f}'
            })
            
            # Print detailed metrics every few epochs
            if epoch % 5 == 0 or epoch == self.num_epochs - 1:
                print(f'\nEpoch {epoch+1}/{self.num_epochs}:')
                print(f'Train: Total Loss: {train_loss:.4f} (Recon: {train_recon:.4f}, KLD: {train_kld:.4f})')
                print(f'Val:   Total Loss: {val_loss:.4f} (Recon: {val_recon:.4f}, KLD: {val_kld:.4f})')
                print(f'Learning Rate: {current_lr:.6f}')
            
            # Generate sample during training for monitoring
            if epoch % 10 == 0:
                self._generate_sample_image(epoch)
            
            # Early stopping check
            self.early_stopping(val_loss)
            if self.early_stopping.should_stop:
                print(f"\nEarly stopping triggered at epoch {epoch+1}!")
                break
        
        print("\nTraining completed!")
        return self.history

    def plot_training_history(self):
        """Plot training history"""
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        
        # Total loss
        axes[0].plot(self.history['train_total_loss'], label='Train', alpha=0.7)
        axes[0].plot(self.history['val_total_loss'], label='Validation', alpha=0.7)
        axes[0].set_title('Total Loss')
        axes[0].set_xlabel('Epoch')
        axes[0].set_ylabel('Loss')
        axes[0].legend()
        axes[0].grid(True, alpha=0.3)
        
        # Reconstruction loss
        axes[1].plot(self.history['train_recon_loss'], label='Train', alpha=0.7)
        axes[1].plot(self.history['val_recon_loss'], label='Validation', alpha=0.7)
        axes[1].set_title('Reconstruction Loss')
        axes[1].set_xlabel('Epoch')
        axes[1].set_ylabel('Loss')
        axes[1].legend()
        axes[1].grid(True, alpha=0.3)
        
        # KLD loss
        axes[2].plot(self.history['train_kld_loss'], label='Train', alpha=0.7)
        axes[2].plot(self.history['val_kld_loss'], label='Validation', alpha=0.7)
        axes[2].set_title('KL Divergence Loss')
        axes[2].set_xlabel('Epoch')
        axes[2].set_ylabel('Loss')
        axes[2].legend()
        axes[2].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.show()


class Adamax:
    """Adamax optimizer implementation"""
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
            
            # Check for valid updates
            if np.isfinite(update).all():
                p.data -= update

    def zero_grad(self):
        """Zero out gradients"""
        for p in self.parameters:
            if p.grad is not None:
                p.grad = np.zeros_like(p.data)