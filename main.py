from nn_ops import MNISTBatchGenerator, VAE, NN, ModelSaver, load_dataset, create_trainer
import time
import argparse
import json

def create_model_from_config(config, cvae):
    """
    Create a model instance based on configuration dictionary.
    
    Args:
        config: Dictionary containing model configuration
        cvae: Boolean indicating if training CVAE
        
    Returns:
        Model instance (VAE or NN)
    """
    model_type = config.get('model_type')
    
    if model_type == 'VAE':
        return VAE(
            input_dim=config['input_dim'],
            latent_dim=config['latent_dim'],
            encoder_hidden_dims=config.get('encoder_hidden_dims', [256]),
            decoder_hidden_dims=config.get('decoder_hidden_dims', [256]),
            encoder_activations=config.get('encoder_activations'),
            decoder_activations=config.get('decoder_activations'),
            encoder_dropout_rates=config.get('encoder_dropout_rates'),
            decoder_dropout_rates=config.get('decoder_dropout_rates'),
            init_method=config.get('init_method', 'he'),
            cvae=cvae
        )

    elif model_type == 'NN':
        return NN(
            nin=config['nin'],
            nouts=config['nouts'],
            activations=config['activations'],
            dropout_rates=config.get('dropout_rates', [0.0] * len(config['nouts'])),
            init_method=config.get('init_method', 'he')
        )
    else:
        raise ValueError(f"Unsupported model type: {model_type}")

def load_config_from_file(config_path):
    """
    Load model configuration from JSON file.
    
    Args:
        config_path: Path to the configuration JSON file
        
    Returns:
        Dictionary containing model configuration
    """
    try:
        with open(config_path, 'r') as f:
            return json.load(f)
    except FileNotFoundError:
        raise FileNotFoundError(f"Configuration file not found: {config_path}")
    except json.JSONDecodeError:
        raise ValueError(f"Invalid JSON format in configuration file: {config_path}")

def get_default_vae_config():
    """Return default VAE configuration for MNIST."""
    return {
        "model_type": "VAE",
        "input_dim": 784,
        "latent_dim": 20,
        "encoder_hidden_dims": [256],
        "decoder_hidden_dims": [256],
        "init_method": "he"
    }

def get_default_nn_config():
    """Return default NN configuration for MNIST classification."""
    return {
        "model_type": "NN",
        "nin": 784,
        "nouts": [128, 64, 10],
        "activations": ["relu", "relu", "softmax"],
        "dropout_rates": [0.2, 0.3, 0.0],
        "init_method": "he"
    }

def adjust_config_for_dataset(model_config, dataset_name):
    """
    Adjust model configuration based on dataset characteristics.
    
    Args:
        model_config: Dictionary containing model configuration
        dataset_name: Name of the dataset ('mnist', 'fashion_mnist', etc.)
        
    Returns:
        Updated model configuration
    """
    # Dataset-specific adjustments
    dataset_configs = {
        'mnist': {'input_dim': 784, 'nin': 784, 'num_classes': 10},
        'fashion_mnist': {'input_dim': 784, 'nin': 784, 'num_classes': 10},
        # Add more datasets as needed
    }
    
    if dataset_name.lower() in dataset_configs:
        dataset_config = dataset_configs[dataset_name.lower()]
        
        # Update input dimensions
        if 'input_dim' in model_config:
            model_config['input_dim'] = dataset_config['input_dim']
        if 'nin' in model_config:
            model_config['nin'] = dataset_config['nin']
            
        # Update output dimensions for NN models
        if model_config.get('model_type') == 'NN' and 'nouts' in model_config:
            # Keep the architecture but adjust the final layer
            model_config['nouts'][-1] = dataset_config['num_classes']
    
    return model_config

def train_model(model_config, training_config, model_name=None):
    """
    Train a model with given configurations.
    
    Args:
        model_config: Dictionary containing model architecture configuration
        training_config: Dictionary containing training parameters
        model_name: Optional name for saving the model
        
    Returns:
        Tuple of (trained_model, trainer, training_time, saved_model_path)
    """
    print("="*60)
    print("TRAINING CONFIGURATION")
    print("="*60)
    
    # Determine model type for display
    model_type = model_config.get('model_type')
    if model_type == 'VAE' and training_config['cvae']:
        display_type = 'CVAE'
    else:
        display_type = model_type
    
    print(f"Model Type: {display_type}")
    print(f"Dataset: {training_config['dataset'].upper()}")
    print(f"Epochs: {training_config['num_epochs']}")
    print(f"Batch Size: {training_config['batch_size']}")
    print(f"Learning Rate: {training_config['learning_rate']}")
    print("="*60)
    
    # Load data
    print(f"Loading {training_config['dataset']} dataset...")
    train_loader, test_loader = load_dataset(
        batch_size=training_config['batch_size'], 
        dataset_name=training_config['dataset']
    )
    train_generator = MNISTBatchGenerator(train_loader)
    test_generator = MNISTBatchGenerator(test_loader)
    print(f"Successfully loaded the {training_config['dataset']} dataset.")
    
    # Adjust model config for dataset
    model_config = adjust_config_for_dataset(model_config, training_config['dataset'])
    
    # Create model
    print("Creating model...")
    model = create_model_from_config(model_config, cvae=training_config['cvae'])
    print(f"Model created: {type(model).__name__}")
    
    # Create appropriate trainer
    print("Initializing trainer...")
    trainer_kwargs = {
        'early_stopping_patience': training_config.get('early_stopping_patience', 15)
    }
    
    # Add model-specific parameters
    if model_type == 'VAE':
        trainer_kwargs.update({
            'latent_dim': model_config['latent_dim'],
            'train_cvae': training_config['cvae']
        })
    
    trainer = create_trainer(
        model_type=display_type,
        model=model,
        train_generator=train_generator,
        test_generator=test_generator,
        num_epochs=training_config['num_epochs'],
        learning_rate=training_config['learning_rate'],
        batch_size=training_config['batch_size'],
        **trainer_kwargs
    )
    
    # Train the model
    print("Starting training...")
    start_time = time.time()
    history = trainer.train()
    end_time = time.time()
    training_time = end_time - start_time
    
    print(f"Training completed in {training_time:.2f} seconds")
    
    # Save model
    print("Saving model...")
    saver = ModelSaver(
        model=trainer.model,
        model_name=model_name,
        cvae=training_config['cvae'],
        include_history={
            "epochs": training_config['num_epochs'],
            "batch_size": training_config['batch_size'],
            "learning_rate": training_config['learning_rate'],
            "training_time": training_time,
            "dataset": training_config['dataset'],
            "model_type": display_type
        }
    )
    saved_model_path = saver.save_model()
    
    return trainer.model, trainer, training_time, saved_model_path

def main():
    """Main training function with command line interface."""
    parser = argparse.ArgumentParser(description='Train neural network models')
    parser.add_argument('--config', type=str, help='Path to model configuration JSON file')
    parser.add_argument('--model-type', type=str, choices=['VAE', 'NN'], 
                       default='VAE', help='Type of model to train (if no config file)')
    parser.add_argument('--dataset', type=str, default='mnist', 
                       choices=['mnist', 'fashion_mnist'],
                       help='Dataset to use for training')
    parser.add_argument('--epochs', type=int, default=50, help='Number of training epochs')
    parser.add_argument('--batch-size', type=int, default=64, help='Batch size for training')
    parser.add_argument('--learning-rate', type=float, default=0.001, help='Learning rate')
    parser.add_argument('--model-name', type=str, help='Name for saving the trained model')
    parser.add_argument('--list-models', action='store_true', help='List all saved models')
    parser.add_argument('--cvae', action='store_true', help='Train a Conditional VAE (only for VAE model type)')
    parser.add_argument('--early-stopping-patience', type=int, default=15, 
                       help='Early stopping patience (number of epochs)')
    
    args = parser.parse_args()
    
    # List saved models if requested
    if args.list_models:
        print("="*60)
        print("SAVED MODELS")
        print("="*60)
        models = ModelSaver.list_saved_models()
        if models:
            for i, model_info in enumerate(models, 1):
                print(f"{i}. {model_info['name']}")
                print(f"   Type: {model_info['type']}")
                print(f"   Saved: {model_info['saved_date']}")
                print(f"   Path: {model_info['path']}")
                print()
        else:
            print("No saved models found.")
        return
    
    # Validate CVAE argument
    if args.cvae and args.model_type != 'VAE':
        print("Warning: --cvae flag is only applicable for VAE model type. Ignoring...")
        args.cvae = False
    
    # Load or create model configuration
    if args.config:
        print(f"Loading configuration from: {args.config}")
        model_config = load_config_from_file(args.config)
        # Override model type from config if specified in args
        if hasattr(args, 'model_type') and args.model_type:
            model_config['model_type'] = args.model_type
    else:
        print(f"Using default {args.model_type} configuration")
        if args.model_type == 'VAE':
            model_config = get_default_vae_config()
        elif args.model_type == 'NN':
            model_config = get_default_nn_config()
        else:
            raise ValueError(f"Unsupported model type: {args.model_type}")
    
    # Training configuration
    training_config = {
        'num_epochs': args.epochs,
        'batch_size': args.batch_size,
        'learning_rate': args.learning_rate,
        'dataset': args.dataset,
        'cvae': args.cvae,
        'early_stopping_patience': args.early_stopping_patience
    }
    
    # Validate configuration
    if model_config.get('model_type') == 'NN' and args.cvae:
        print("Warning: CVAE training is not applicable for NN models. Setting cvae=False")
        training_config['cvae'] = False
    
    # Train the model
    try:
        trained_model, trainer, training_time, saved_path = train_model(
            model_config, training_config, args.model_name
        )
        
        print("="*60)
        print("TRAINING SUMMARY")
        print("="*60)
        print(f"Model successfully trained and saved!")
        print(f"Training time: {training_time:.2f} seconds")
        print(f"Model saved to: {saved_path}")
        print(f"Model type: {model_config.get('model_type')}")
        print(f"Dataset: {training_config['dataset']}")
        
        # Show final metrics
        if hasattr(trainer, 'history') and trainer.history:
            if 'val_total_loss' in trainer.history:
                final_val_loss = trainer.history['val_total_loss'][-1]
                print(f"Final validation loss: {final_val_loss:.4f}")
            if 'val_accuracy' in trainer.history:
                final_val_acc = trainer.history['val_accuracy'][-1]
                print(f"Final validation accuracy: {final_val_acc:.3f}")
        
        # Additional info for VAE models
        if model_config.get('model_type') == 'VAE':
            print(f"\nTo generate digits with this VAE model, run:")
            print(f"python generate_digits.py --model-path '{saved_path}'")
        
        # Plot training history if requested
        plot_choice = input("\nWould you like to plot the training history? (y/n): ").lower()
        if plot_choice in ['y', 'yes']:
            trainer.plot_training_history()
        
    except KeyboardInterrupt:
        print("\nTraining interrupted by user.")
        return 1
    except Exception as e:
        print(f"Error during training: {str(e)}")
        import traceback
        traceback.print_exc()
        return 1
    
    return 0

if __name__ == "__main__":
    exit(main())