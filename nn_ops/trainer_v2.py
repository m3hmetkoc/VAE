import numpy as np 
from tqdm import tqdm
import matplotlib.pyplot as plt
from .data_process import EarlyStopping
from .optimizer import Adamax, SGD, Adam, get_optimizer
from .tensor_class import Tensor, kl_divergence, binary_cross_entropy, cross_entropy_loss

class BaseTrainer:
    """Base trainer class with common functionality"""
    def __init__(self, model, train_generator, test_generator, num_epochs, 
                 learning_rate, batch_size, early_stopping_patience=5):
        self.model = model
        self.train_generator = train_generator
        self.test_generator = test_generator
        self.num_epochs = num_epochs
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.model_params = self.model.parameters()
        self.num_test_batches = len(test_generator.data_loader)
        self.num_train_batches = len(train_generator.data_loader)
        
        self.early_stopping_patience = early_stopping_patience
        self.early_stopping = EarlyStopping(patience=early_stopping_patience)
        
        self.history = {}
        self.optimizer = None
        
    def lr_schedule(self, epoch):
        """Learning rate schedule with gentle decay"""
        return self.learning_rate * (0.95 ** (epoch // 10))
    
    def _clip_gradients(self, clip_value=1.0):
        """Gradient clipping for stability"""
        for param in self.model_params:
            if param.grad is not None:
                param.grad = np.clip(param.grad, -clip_value, clip_value)
    
    def process_batch(self, X_batch, labels, is_training=True):
        """Abstract method to be implemented by subclasses"""
        raise NotImplementedError("Subclasses must implement process_batch")
    
    def train_and_evaluate_epoch(self, epoch):
        """Combined training and evaluation for one epoch"""
        # Training phase
        self.model.train()
        train_metrics = self._run_epoch(epoch, is_training=True)
        
        # Evaluation phase
        self.model.eval()
        val_metrics = self._run_epoch(epoch, is_training=False)
        
        return train_metrics, val_metrics
    
    def _run_epoch(self, epoch, is_training=True):
        """Run one epoch of training or evaluation"""
        generator = self.train_generator if is_training else self.test_generator
        num_batches = self.num_train_batches if is_training else self.num_test_batches
        phase = "Train" if is_training else "Val"
        
        # Initialize metrics
        metrics = self._initialize_metrics()
        
        # Progress bar
        pbar = tqdm(range(num_batches), 
                   desc=f'Epoch {epoch+1}/{self.num_epochs} [{phase}]',
                   leave=False)
        
        for batch_idx in pbar:
            if is_training:
                self.optimizer.zero_grad()
            try:
                X_batch, labels = generator.get_next_batch()
                batch_metrics = self.process_batch(X_batch, labels, is_training, epoch)

                #Check for valid metrics
                # if not all(np.isfinite(v.data) for v in batch_metrics.values()):
                #     print(f"Warning: Non-finite metrics detected at batch {batch_idx}")
                #     continue

                if is_training:
                    # Backward pass and update
                    batch_metrics['total_loss'].backward()
                    self.optimizer.step()
                
               # Accumulate metrics
                for key, value in batch_metrics.items():
                    if hasattr(value, 'data'):
                        metrics[key] += value.data

                # Update progress bar
                pbar.set_postfix(self._format_progress_metrics(batch_metrics))
            except Exception as e:
                print(f"Error in {phase.lower()} batch {batch_idx}: {e}")
                continue
        
        # Calculate averages
        for key in metrics:
            metrics[key] /= num_batches
            
        return metrics
    
    def _initialize_metrics(self):
        """Initialize metrics dictionary - to be overridden by subclasses"""
        return {}
    
    def _format_progress_metrics(self, metrics):
        """Format metrics for progress bar display"""
        return {}
    
    def train(self, optimizer="adam"):
        """Main training loop"""
        print(f"Starting training for {self.num_epochs} epochs...")
        print(f"Training batches: {self.num_train_batches}, Validation batches: {self.num_test_batches}")
        print(f"Batch size: {self.batch_size}")
        
        # Clear history
        self.history = {key: [] for key in self._get_history_keys()}
        
        # Initialize optimizer
        self.optimizer = get_optimizer(parameters=self.model_params, lr=self.learning_rate, name=optimizer)
        
        # Main training loop
        epoch_pbar = tqdm(range(self.num_epochs), desc="Training Progress")
        
        for epoch in epoch_pbar:
            # Update learning rate
            current_lr = self.lr_schedule(epoch)
            self.optimizer.lr = current_lr
            
            # Train and evaluate
            train_metrics, val_metrics = self.train_and_evaluate_epoch(epoch)
            
            # Store history
            # Update epoch progress bar
            self._update_history(train_metrics, val_metrics)
            epoch_pbar.set_postfix(self._format_epoch_metrics(train_metrics, val_metrics, current_lr))
            # Print detailed metrics
            if epoch % 5 == 0 or epoch == self.num_epochs - 1:
                self._print_epoch_summary(epoch, train_metrics, val_metrics, current_lr)
            
            # Generate samples for monitoring (if applicable)
            if hasattr(self, '_generate_sample') and epoch % 5 == 0:
                self._generate_sample(epoch)
            
            # Early stopping check
            early_stop_metric = self._get_early_stop_metric(val_metrics)
            self.early_stopping(early_stop_metric)
            if self.early_stopping.should_stop:
                print(f"\nEarly stopping triggered at epoch {epoch+1}!")
                break
        
        print("\nTraining completed!")
        return self.history
    
    def _get_history_keys(self):
        """Get keys for history tracking - to be overridden"""
        return []
    
    def _update_history(self, train_metrics, val_metrics):
        """Update training history - to be overridden"""
        pass
    
    def _format_epoch_metrics(self, train_metrics, val_metrics, lr):
        """Format metrics for epoch progress bar"""
        return {}
    
    def _print_epoch_summary(self, epoch, train_metrics, val_metrics, lr):
        """Print detailed epoch summary"""
        pass
    
    def _get_early_stop_metric(self, val_metrics):
        """Get metric for early stopping"""
        return val_metrics.get('total_loss', float('inf'))


class VAETrainer(BaseTrainer):
    """Trainer for VAE and CVAE models"""
    def __init__(self, model, latent_dim, train_generator, test_generator, 
                 num_epochs, learning_rate, batch_size, train_cvae=False, 
                 early_stopping_patience=5):
        super().__init__(model, train_generator, test_generator, num_epochs, 
                        learning_rate, batch_size, early_stopping_patience)
        self.latent_dimension_size = latent_dim
        self.train_cvae = train_cvae
    
    def process_batch(self, X_batch, labels, is_training=True, epoch=0):
        """Process one batch for VAE/CVAE training"""
        # Prepare input based on model type
        if self.train_cvae:
            X_concat = Tensor(np.concatenate([X_batch.data, labels.data], axis=1), requires_grad=False)
            reconstructed_x, mu, logvar = self.model.forward(X_concat, labels)
        else:
            reconstructed_x, mu, logvar = self.model.forward(X_batch, labels)
        
        # Calculate losses
        recon_loss = binary_cross_entropy(recon_x=reconstructed_x, x=X_batch)
        kld_loss = kl_divergence(mu, logvar)
        
        # Beta scheduling for training stability
        if is_training:
            beta = min(1.0, 0.1 + 0.01 * epoch)
        else:
            beta = 1.0
        
        total_loss = recon_loss + kld_loss * beta
    
        return {
            'total_loss': total_loss,
            'recon_loss': recon_loss,
            'kld_loss': kld_loss,
            'beta': beta
        }
    
    def _initialize_metrics(self):
        return {
            'total_loss': 0.0,
            'recon_loss': 0.0,
            'kld_loss': 0.0,
            'beta': 0.0
        }
    
    def _format_progress_metrics(self, metrics):
        return {
            'Loss': f'{metrics["total_loss"].data:.4f}',
            'Recon': f'{metrics["recon_loss"].data:.4f}',
            'KLD': f'{metrics["kld_loss"].data:.4f}',
            'Beta': f'{metrics['beta']:.3f}'
        }
    
    def _get_history_keys(self):
        return [
            'train_total_loss', 'val_total_loss',
            'train_recon_loss', 'val_recon_loss',
            'train_kld_loss', 'val_kld_loss'
        ]
    
    def _update_history(self, train_metrics, val_metrics):
        self.history['train_total_loss'].append(train_metrics['total_loss'])
        self.history['train_recon_loss'].append(train_metrics['recon_loss'])
        self.history['train_kld_loss'].append(train_metrics['kld_loss'])
        self.history['val_total_loss'].append(val_metrics['total_loss'])
        self.history['val_recon_loss'].append(val_metrics['recon_loss'])
        self.history['val_kld_loss'].append(val_metrics['kld_loss'])
    
    def _format_epoch_metrics(self, train_metrics, val_metrics, lr):
        return {
            'Train Loss': f'{train_metrics["total_loss"]:.4f}',
            'Val Loss': f'{val_metrics["total_loss"]:.4f}',
            'LR': f'{lr:.6f}'
        }
    
    def _print_epoch_summary(self, epoch, train_metrics, val_metrics, lr):
        print(f'\nEpoch {epoch+1}/{self.num_epochs}:')
        print(f'Train: Total Loss: {train_metrics["total_loss"]:.4f} '
              f'(Recon: {train_metrics["recon_loss"]:.4f}, KLD: {train_metrics["kld_loss"]:.4f})')
        print(f'Val:   Total Loss: {val_metrics["total_loss"]:.4f} '
              f'(Recon: {val_metrics["recon_loss"]:.4f}, KLD: {val_metrics["kld_loss"]:.4f})')
        print(f'Learning Rate: {lr:.6f}')
    
    def _generate_sample(self, epoch, save_path="generated_imgs_on_training"):
        """Generate and save a sample image during training"""
        try:
            import os
            os.makedirs(save_path, exist_ok=True)
            if self.train_cvae:
                z = Tensor(data=np.concatenate([
                    np.random.randn(1, self.latent_dimension_size), 
                    np.array([[0., 0., 0., 0., 0., 0., 0., 1., 0., 0.]])
                ], axis=1), requires_grad=False)
            else:
                z = Tensor(np.random.randn(1, self.latent_dimension_size), requires_grad=False)
            
            out = self.model.decoder(z)
            out_data = out.data.reshape(28, 28)
            out_data = np.clip(out_data, 0, 1)
            
            plt.imsave(f"{save_path}/generated_image_epoch_{epoch}.png", 
                      out_data, cmap="gray")
        except Exception as e:
            print(f"Could not save sample image: {e}")
    
    def plot_training_history(self):
        """Plot VAE training history"""
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


class NNTrainer(BaseTrainer):
    """Trainer for standard Neural Network classification"""
    def __init__(self, model, train_generator, test_generator, num_epochs, 
                 learning_rate, batch_size, early_stopping_patience=5):
        super().__init__(model, train_generator, test_generator, num_epochs, 
                        learning_rate, batch_size, early_stopping_patience)
    
    def process_batch(self, X_batch, labels, is_training=True, epoch=0):
        """Process one batch for NN classification training"""
        # Forward pass
        predictions = self.model.forward(X_batch)
        
        # Calculate loss and accuracy
        loss = cross_entropy_loss(predictions, labels)
        accuracy = self._calculate_accuracy(predictions, labels)
        return {
            'total_loss': loss,
            'accuracy_value': accuracy
        }
    
    def _calculate_accuracy(self, predictions, labels):
        """Calculate classification accuracy"""
        # Apply softmax if not already applied
            
        pred_classes = np.argmax(predictions.data, axis=1)
        true_classes = np.argmax(labels.data, axis=1)
        accuracy = np.mean(pred_classes == true_classes)
        return Tensor(accuracy, requires_grad=False)
    
    def _initialize_metrics(self):
        return {
            'total_loss': 0.0,
            'accuracy_value': 0.0
        }
    
    def _format_progress_metrics(self, metrics):
        return {
            'Loss': f'{metrics["total_loss"].data:.4f}',
            'Acc': f'{metrics["accuracy_value"].data:.3f}'
        }
    
    def _get_history_keys(self):
        return [
            'train_loss', 'val_loss',
            'train_accuracy', 'val_accuracy'
        ]
    
    def _update_history(self, train_metrics, val_metrics):
        self.history['train_loss'].append(train_metrics['total_loss'])
        self.history['train_accuracy'].append(train_metrics['accuracy_value'])
        self.history['val_loss'].append(val_metrics['total_loss'])
        self.history['val_accuracy'].append(val_metrics['accuracy_value'])
    
    def _format_epoch_metrics(self, train_metrics, val_metrics, lr):
        return {
            'Train Loss': f'{train_metrics["total_loss"]:.4f}',
            'Train Acc': f'{train_metrics["accuracy_value"]:.3f}',
            'Val Loss': f'{val_metrics["total_loss"]:.4f}',
            'Val Acc': f'{val_metrics["accuracy_value"]:.3f}',
            'LR': f'{lr:.6f}'
        }
    
    def _print_epoch_summary(self, epoch, train_metrics, val_metrics, lr):
        print(f'\nEpoch {epoch+1}/{self.num_epochs}:')
        print(f'Train: Loss: {train_metrics["total_loss"]:.4f}, Accuracy: {train_metrics["accuracy_value"]:.3f}')
        print(f'Val:   Loss: {val_metrics["total_loss"]:.4f}, Accuracy: {val_metrics["accuracy_value"]:.3f}')
        print(f'Learning Rate: {lr:.6f}')
    
    def plot_training_history(self):
        """Plot NN training history"""
        fig, axes = plt.subplots(1, 2, figsize=(12, 5))
        
        # Loss
        axes[0].plot(self.history['train_loss'], label='Train', alpha=0.7)
        axes[0].plot(self.history['val_loss'], label='Validation', alpha=0.7)
        axes[0].set_title('Loss')
        axes[0].set_xlabel('Epoch')
        axes[0].set_ylabel('Loss')
        axes[0].legend()
        axes[0].grid(True, alpha=0.3)
        
        # Accuracy
        axes[1].plot(self.history['train_accuracy'], label='Train', alpha=0.7)
        axes[1].plot(self.history['val_accuracy'], label='Validation', alpha=0.7)
        axes[1].set_title('Accuracy')
        axes[1].set_xlabel('Epoch')
        axes[1].set_ylabel('Accuracy')
        axes[1].legend()
        axes[1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.show()


def create_trainer(model_type, model, train_generator, test_generator, 
                   num_epochs, learning_rate, batch_size, **kwargs):
    """Factory function to create appropriate trainer"""
    if model_type.upper() == 'NN':
        return NNTrainer(
            model=model,
            train_generator=train_generator,
            test_generator=test_generator,
            num_epochs=num_epochs,
            learning_rate=learning_rate,
            batch_size=batch_size,
            early_stopping_patience=kwargs.get('early_stopping_patience', 5)
        )
    elif model_type.upper() in ['VAE', 'CVAE']:
        return VAETrainer(
            model=model,
            latent_dim=kwargs.get('latent_dim', 20),
            train_generator=train_generator,
            test_generator=test_generator,
            num_epochs=num_epochs,
            learning_rate=learning_rate,
            batch_size=batch_size,
            train_cvae=kwargs.get('train_cvae', False),
            early_stopping_patience=kwargs.get('early_stopping_patience', 5)
        )
    else:
        raise ValueError(f"Unsupported model type: {model_type}")