from nn_ops import ModelSaver, Tensor
import matplotlib.pyplot as plt
import numpy as np
import argparse
import os

def generate_digits(vae_model, num_samples, latent_dim):
    """
    Generate new digits using a standard VAE's decoder.

    Args:
        vae_model: The trained VAE model.
        num_samples: Number of digits to generate.
        latent_dim: The dimensionality of the latent space.

    Returns:
        NumPy array of generated image data.
    """
    # Sample from standard normal distribution
    z_samples_np = np.random.randn(num_samples, latent_dim).astype(np.float32)
    z_samples_tensor = Tensor(z_samples_np, requires_grad=False)
    
    # Generate images using decoder
    vae_model.eval()  # Set to evaluation mode
    generated_images_tensor = vae_model.decoder(z_samples_tensor)
    
    return generated_images_tensor.data

def generate_specific_digits_with_label(cvae_model, label, num_samples, latent_dim, num_classes=10):
    """
    Generate multiple images of a specific digit using a CVAE.

    Args:
        cvae_model: The trained Conditional VAE model.
        label (int): The digit label (e.g., 9) to generate.
        num_samples (int): Number of different digits to generate for the given label.
        latent_dim (int): The dimensionality of the latent space.
        num_classes (int): The number of possible classes (e.g., 10 for MNIST).

    Returns:
        NumPy array of generated image data.
    """
    # 1. Sample random latent vectors from a standard normal distribution
    z_samples_np = np.random.randn(num_samples, latent_dim).astype(np.float32)

    # 2. Create one-hot encoded label vectors
    # This vector will be repeated for each sample.
    label_one_hot = np.zeros((1, num_classes), dtype=np.float32)
    label_one_hot[0, label] = 1
    labels_np = np.tile(label_one_hot, (num_samples, 1))

    # 3. Concatenate latent vectors and one-hot encoded labels
    # This combined vector is the input for the CVAE decoder.
    combined_input_np = np.hstack([z_samples_np, labels_np])
    combined_input_tensor = Tensor(combined_input_np, requires_grad=False)
    
    # 4. Generate images using the decoder
    cvae_model.eval()  # Set to evaluation mode
    generated_images_tensor = cvae_model.decoder(combined_input_tensor)
    
    return generated_images_tensor.data


def plot_generated_digits(images_data, num_samples, image_shape=(28, 28), 
                         title="Generated Digits", save_path=None):
    """
    Plot the generated digits in a grid.

    Args:
        images_data: NumPy array of generated image data.
        num_samples: Number of digits that were generated.
        image_shape: Tuple, the shape of a single image (default: (28, 28) for MNIST).
        title: The title for the plot.
        save_path: Optional path to save the plot image.
    """
    cols = int(np.ceil(np.sqrt(num_samples)))
    rows = int(np.ceil(num_samples / cols))

    fig, axes = plt.subplots(rows, cols, figsize=(cols * 1.5, rows * 1.5))
    
    if rows * cols == 1:
        axes = [axes]
    else:
        axes = axes.flatten()

    for i in range(rows * cols):
        if i < num_samples:
            img = images_data[i].reshape(image_shape)
            img = np.clip(img, 0, 1) # Ensure values are in [0, 1] range
            axes[i].imshow(img, cmap='gray', vmin=0, vmax=1)
        axes[i].axis('off')

    plt.suptitle(title, fontsize=16)
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Generated digits saved to: {save_path}")
    
    plt.show()

def interpolate_in_latent_space(vae_model, latent_dim, num_steps=10, save_path=None):
    """
    Generate images by interpolating between two random points in latent space.
    
    Args:
        vae_model: The trained VAE model.
        latent_dim: Dimensionality of the latent space.
        num_steps: Number of interpolation steps.
        save_path: Optional path to save the interpolation image.
    """
    z1 = np.random.randn(1, latent_dim).astype(np.float32)
    z2 = np.random.randn(1, latent_dim).astype(np.float32)
    
    alphas = np.linspace(0, 1, num_steps)
    interpolated_images = []
    
    vae_model.eval()
    for alpha in alphas:
        z_interp = (1 - alpha) * z1 + alpha * z2
        z_tensor = Tensor(z_interp, requires_grad=False)
        generated_img = vae_model.decoder(z_tensor)
        interpolated_images.append(generated_img.data[0])
    
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

def main():
    """Main function with command line interface for digit generation."""
    parser = argparse.ArgumentParser(description='Generate digits using trained VAE/CVAE models')
    parser.add_argument('--model-path', type=str, required=True,
                       help='Path to the saved VAE/CVAE model directory')
    parser.add_argument('--num-samples', type=int, default=16,
                       help='Number of digits to generate (default: 16)')
    parser.add_argument('--label', type=int,
                       help='(CVAE only) Specify a digit (0-9) to generate. Overrides other modes.')
    parser.add_argument('--save-images', type=str,
                       help='Path to save the generated image grid (optional)')
    parser.add_argument('--interpolate', action='store_true',
                       help='(VAE only) Generate latent space interpolation')
    parser.add_argument('--interp-steps', type=int, default=10,
                       help='Number of interpolation steps (default: 10)')
    parser.add_argument('--list-models', action='store_true',
                       help='List all saved models and exit')
    
    args = parser.parse_args()
    
    if args.list_models:
        # (Your list_models logic remains unchanged)
        return 0

    if not os.path.exists(args.model_path):
        print(f"Error: Model path does not exist: {args.model_path}")
        return 1
    
    try:
        print(f"Loading model from: {args.model_path}")
        loaded_model, history = ModelSaver.load_model(args.model_path)
        
        model_type = loaded_model.model_type
        if model_type not in ['VAE', 'CVAE']:
            print(f"Error: This script supports VAE or CVAE models. Loaded model type: {model_type}")
            return 1

        config = loaded_model.get_config()
        latent_dim = config['latent_dim']
        
        print(f"Model loaded successfully! Type: {loaded_model.model_type}, Latent dim: {latent_dim}\n")

        # --- Generation Logic ---

        # Priority 1: Conditional generation if --label is specified
        if args.label is not None:
            if loaded_model.model_type != 'CVAE':
                print("Error: The --label argument requires a CVAE model. The loaded model is a VAE.")
                return 1
            if not (0 <= args.label <= 9):
                 print(f"Error: --label must be an integer between 0 and 9, but got {args.label}.")
                 return 1
            
            # num_classes = config.get('num_classes')
            # if num_classes is None:
            #     print("Error: CVAE model config is missing 'num_classes'. Cannot generate with label.")
            #     return 1

            print(f"Generating {args.num_samples} samples of digit '{args.label}'...")
            generated_images = generate_specific_digits_with_label(
                loaded_model, args.label, args.num_samples, latent_dim, num_classes=10
            )
            plot_generated_digits(
                generated_images,
                args.num_samples,
                title=f"Generated '{args.label}' (n={args.num_samples})",
                save_path=args.save_images
            )
            
        # Priority 2: VAE interpolation
        elif args.interpolate:
            if model_type == 'CVAE':
                print("Warning: Latent space interpolation is designed for standard VAEs, not CVAEs.")
            print(f"Generating latent space interpolation with {args.interp_steps} steps...")
            save_path = None
            if args.save_images:
                save_path = args.save_images.replace('.png', '_interpolation.png')
            interpolate_in_latent_space(loaded_model, latent_dim, args.interp_steps, save_path)
            
        # Default: Unconditional generation (for VAE)
        else:
            if model_type == 'VAE':
                print("Warning: Running unconditional generation on a CVAE model.")
                print("Results may be unpredictable. Use --label <digit> for proper generation.")
            print(f"Generating {args.num_samples} random digits...")
            generated_images = generate_digits(loaded_model, args.num_samples, latent_dim)
            plot_generated_digits(
                generated_images, 
                args.num_samples,
                title=f"Unconditionally Generated Digits (Latent dim: {latent_dim})",
                save_path=args.save_images
            )
        
        print("\nGeneration completed successfully!")
        
    except Exception as e:
        print(f"\nAn error occurred: {str(e)}")
        return 1
    
    return 0

if __name__ == "__main__":
    exit(main())