from nn_ops import MNISTBatchGenerator, VAE, Tensor, ModelSaver, Train, load_mnist_data
import time 
import numpy as np
import matplotlib.pyplot as plt

# In main.py
def generate_digits(vae_model, num_samples, latent_dim):
    """
    Generates new digits using the VAE's decoder.

    Args:
        vae_model: The trained VAE model.
        num_samples: Number of digits to generate.
        latent_dim: The dimensionality of the latent space.

    Returns:
        NumPy array of generated image data.
    """
    z_samples_np = np.random.randn(num_samples, latent_dim).astype(np.float32)
    z_samples_value = Tensor(z_samples_np, requires_grad=False)
    generated_images_value = vae_model.decoder(z_samples_value)
    generated_images_data = generated_images_value.data

    return generated_images_data


def plot_generated_digits(images_data, num_samples, image_shape=(28, 28), title="Generated Digits"):
    """
    Plots the generated digits in a grid.

    Args:
        images_data: NumPy array of generated image data from generate_digits.
        num_samples: Number of digits that were generated.
        image_shape: Tuple, the shape of a single image (e.g., (28, 28) for MNIST).
        title: The title for the plot.
    """
    # Determine grid size (try to make it squarish)
    cols = int(np.ceil(np.sqrt(num_samples)))
    rows = int(np.ceil(num_samples / cols))

    fig, axes = plt.subplots(rows, cols, figsize=(cols * 1.5, rows * 1.5))
    axes = axes.flatten() # Flatten in case of single row/column

    for i in range(num_samples):
        if i < len(images_data):
            img = images_data[i].reshape(image_shape)
            axes[i].imshow(img, cmap='gray') # MNIST digits are grayscale
            axes[i].axis('off') # Hide axes ticks
        else:
            axes[i].axis('off') # Hide axes for empty subplots

    plt.suptitle(title, fontsize=16)
    plt.tight_layout(rect=[0, 0, 1, 0.96]) # Adjust layout to make space for suptitle
    plt.show()

# Usage example:
if __name__ == "__main__":
    start = time.time() 
    # Training loop
    num_epochs = 50
    learning_rate = 0.001
    batch_size = 64
    
    # Load data
    train_loader, test_loader = load_mnist_data(batch_size)
    train_generator = MNISTBatchGenerator(train_loader)
    test_generator = MNISTBatchGenerator(test_loader)
    input_dim, hidden_dim, latent_dim = 784, 256, 20
    # Create model (assuming your MLP class is defined)
    model = VAE(
        input_dim=input_dim,  # 28x28 pixels
        hidden_dim=hidden_dim,  # Example architecture
        latent_dim=latent_dim
    )

    start = time.time() 
    
    trainer = Train(
        model=model,
        train_generator=train_generator,
        test_generator=test_generator,
        num_epochs=num_epochs,
        learning_rate=learning_rate,
        batch_size=batch_size
    )

    trainer.train()
    end = time.time() 
    
    trainer.plot_training_history(trainer.history)

    model_saver = ModelSaver(model=model, model_name="mnist_classifier_v1", include_history = trainer.history)
    
    
    print(f"The training time for number of epochs: {num_epochs} and batch size: {batch_size} is: {(end-start):.2f}")
    print("Generating digits from the trained VAE...")

    trained_vae_model = trainer.model # Get the trained model from the trainer
    trained_vae_model.eval()          # Set to evaluation mode
    num_digits_to_generate = 25       # How many digits to generate and plot

    # Generate new digit data
    generated_image_data = generate_digits(
        vae_model=trained_vae_model,
        num_samples=num_digits_to_generate,
        latent_dim=latent_dim
    )
    # Plot the generated digits
    # The image_shape should match your input data, e.g., (28, 28) for flattened MNIST
    plot_generated_digits(
        images_data=generated_image_data,
        num_samples=num_digits_to_generate,
        image_shape=(int(np.sqrt(input_dim)), int(np.sqrt(input_dim))) # e.g., (28,28)
    )
    print(f"{num_digits_to_generate} digits generated and plotted.")
