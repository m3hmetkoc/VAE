from nn_ops import ModelSaver, Tensor
import matplotlib.pyplot as plt
import numpy as np
import argparse
import os

def generate_digits(vae_model, num_samples, latent_dim):
    """
    Generate new digits using the VAE's decoder.

    Args:
        vae_model: The trained VAE model
        num_samples: Number of digits to generate
        latent_dim: The dimensionality of the latent space

    Returns:
        NumPy array of generated image data
    """
    # Sample from standard normal distribution
    z_samples_np = np.random.randn(num_samples, latent_dim).astype(np.float32)
    z_samples_tensor = Tensor(z_samples_np, requires_grad=False)
    
    # Generate images using decoder
    vae_model.eval()  # Set to evaluation mode
    generated_images_tensor = vae_model.decoder(z_samples_tensor)
    generated_images_data = generated_images_tensor.data
    
    return generated_images_data

def plot_generated_digits(images_data, num_samples, image_shape=(28, 28), 
                         title="Generated Digits", save_path=None):
    """
    Plot the generated digits in a grid.

    Args:
        images_data: NumPy array of generated image data
        num_samples: Number of digits that were generated
        image_shape: Tuple, the shape of a single image (default: (28, 28) for MNIST)
        title: The title for the plot
        save_path: Optional path to save the plot image
    """
    # Determine grid size (try to make it roughly square)
    cols = int(np.ceil(np.sqrt(num_samples)))
    rows = int(np.ceil(num_samples / cols))

    fig, axes = plt.subplots(rows, cols, figsize=(cols * 1.5, rows * 1.5))
    
    # Handle single subplot case
    if rows == 1 and cols == 1:
        axes = [axes]
    elif rows == 1 or cols == 1:
        axes = axes.flatten()
    else:
        axes = axes.flatten()

    for i in range(rows * cols):
        if i < num_samples and i < len(images_data):
            # Reshape and display the image
            img = images_data[i].reshape(image_shape)
            # Ensure values are in [0, 1] range for proper display
            img = np.clip(img, 0, 1)
            axes[i].imshow(img, cmap='gray', vmin=0, vmax=1)
            axes[i].set_title(f'Sample {i+1}', fontsize=8)
        axes[i].axis('off')  # Hide axes for all subplots

    plt.suptitle(title, fontsize=16)
    plt.tight_layout(rect=[0, 0, 1, 0.96])  # Make space for suptitle
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Generated digits saved to: {save_path}")
    
    plt.show()

def interpolate_in_latent_space(vae_model, latent_dim, num_steps=10, save_path=None):
    """
    Generate images by interpolating between two random points in latent space.
    
    Args:
        vae_model: The trained VAE model
        latent_dim: Dimensionality of the latent space
        num_steps: Number of interpolation steps
        save_path: Optional path to save the interpolation image
    """
    # Generate two random latent vectors
    z1 = np.random.randn(1, latent_dim).astype(np.float32)
    z2 = np.random.randn(1, latent_dim).astype(np.float32)
    
    # Create interpolation steps
    alphas = np.linspace(0, 1, num_steps)
    interpolated_images = []
    
    vae_model.eval()
    for alpha in alphas:
        # Linear interpolation in latent space
        z_interp = (1 - alpha) * z1 + alpha * z2
        z_tensor = Tensor(z_interp, requires_grad=False)
        
        # Generate image
        generated_img = vae_model.decoder(z_tensor)
        interpolated_images.append(generated_img.data[0])
    
    # Plot interpolation
    fig, axes = plt.subplots(1, num_steps, figsize=(num_steps * 1.5, 2))
    if num_steps == 1:
        axes = [axes]
    
    for i, img_data in enumerate(interpolated_images):
        img = img_data.reshape(28, 28)
        img = np.clip(img, 0, 1)
        axes[i].imshow(img, cmap='gray', vmin=0, vmax=1)
        axes[i].set_title(f'α={alphas[i]:.2f}', fontsize=8)
        axes[i].axis('off')
    
    plt.suptitle('Latent Space Interpolation', fontsize=14)
    plt.tight_layout(rect=[0, 0, 1, 0.9])
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Interpolation saved to: {save_path}")
    
    plt.show()

def generate_specific_samples(vae_model, latent_dim, manual_latent_vector=None):
    """
    Generate a single digit from a specific or random latent vector.
    
    Args:
        vae_model: The trained VAE model
        latent_dim: Dimensionality of the latent space
        manual_latent_vector: Optional specific latent vector to use
    """
    if manual_latent_vector is not None:
        if len(manual_latent_vector) != latent_dim:
            print(f"Warning: Manual vector length ({len(manual_latent_vector)}) "
                  f"doesn't match latent_dim ({latent_dim}). Using random vector.")
            z = np.random.randn(1, latent_dim).astype(np.float32)
        else:
            z = np.array([manual_latent_vector], dtype=np.float32)
    else:
        z = np.random.randn(1, latent_dim).astype(np.float32)
    
    vae_model.eval()
    z_tensor = Tensor(z, requires_grad=False)
    generated_img = vae_model.decoder(z_tensor)
    img_data = generated_img.data[0].reshape(28, 28)
    img_data = np.clip(img_data, 0, 1)
    
    plt.figure(figsize=(4, 4))
    plt.imshow(img_data, cmap='gray', vmin=0, vmax=1)
    plt.title('Generated Digit')
    plt.axis('off')
    plt.show()
    
    print("Latent vector used:")
    print(z[0])

def main():
    """Main function with command line interface for digit generation."""
    parser = argparse.ArgumentParser(description='Generate digits using trained VAE models')
    parser.add_argument('--model-path', type=str, required=True,
                       help='Path to the saved VAE model directory')
    parser.add_argument('--num-samples', type=int, default=25,
                       help='Number of digits to generate (default: 25)')
    parser.add_argument('--save-images', type=str,
                       help='Path to save generated images (optional)')
    parser.add_argument('--interpolate', action='store_true',
                       help='Generate latent space interpolation')
    parser.add_argument('--interp-steps', type=int, default=10,
                       help='Number of interpolation steps (default: 10)')
    parser.add_argument('--single', action='store_true',
                       help='Generate a single digit with random latent vector')
    parser.add_argument('--list-models', action='store_true',
                       help='List all saved models and exit')
    
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
                
                # Show VAE models specifically
                if model_info['type'] in ['VAE']:
                    arch = model_info['architecture']
                    if model_info['type'] == 'VAE':
                        print(f"   Latent dim: {arch.get('latent_dim')}")
                        print(f"   Encoder layers: {arch.get('encoder_hidden_dims')}")
                        print(f"   Decoder layers: {arch.get('decoder_hidden_dims')}")

                    print()
        return 0
    
    # Validate model path
    if not os.path.exists(args.model_path):
        print(f"Error: Model path does not exist: {args.model_path}")
        return 1
    
    try:
        # Load the model
        print(f"Loading model from: {args.model_path}")
        loaded_model, history = ModelSaver.load_model(args.model_path)
        
        # Check if it's a VAE model
        model_type = type(loaded_model).__name__
        if model_type not in ['VAE']:
            print(f"Error: This script is for VAE models only. Loaded model type: {model_type}")
            return 1
        
        # Get model configuration
        config = loaded_model.get_config()
        latent_dim = config['latent_dim']
        
        print(f"Model loaded successfully!")
        print(f"Model type: {model_type}")
        print(f"Latent dimension: {latent_dim}")
        if history:
            print(f"Training epochs: {history.get('epochs', 'Unknown')}")
            print(f"Training time: {history.get('training_time', 'Unknown')} seconds")
        print()
        
        # Generate based on user choice
        if args.single:
            print("Generating single digit...")
            generate_specific_samples(loaded_model, latent_dim)
            
        elif args.interpolate:
            print(f"Generating latent space interpolation with {args.interp_steps} steps...")
            save_path = None
            if args.save_images:
                save_path = args.save_images.replace('.png', '_interpolation.png')
            interpolate_in_latent_space(loaded_model, latent_dim, args.interp_steps, save_path)
            
        else:
            print(f"Generating {args.num_samples} digits...")
            generated_images = generate_digits(loaded_model, args.num_samples, latent_dim)
            
            plot_generated_digits(
                generated_images, 
                args.num_samples,
                title=f"Generated Digits (Latent dim: {latent_dim})",
                save_path=args.save_images
            )
        
        print("Generation completed successfully!")
        
    except Exception as e:
        print(f"Error: {str(e)}")
        return 1
    
    return 0

if __name__ == "__main__":
    exit(main())