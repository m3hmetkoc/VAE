import matplotlib.pyplot as plt 
import numpy as np 
import argparse
import os 
import sys 

current_script_dir = os.path.dirname(os.path.abspath(__file__))

# Go up one level to reach project_root/
project_root = os.path.join(current_script_dir, '..')

# Add project_root to sys.path
sys.path.append(project_root)

from nn_ops import load_random_test_samples, Tensor, ModelSaver

def test_classifier(nn_model, num_samples, dataset_name, save_path=None):
    """
    Test a trained NN classifier model on test data and display results.
    
    Args:
        nn_model: The trained NN classifier model.
        num_samples: Number of test samples to evaluate.
        model_path: Path to the model directory (used to determine dataset).
        save_path: Optional path to save the results visualization.
    
    Returns:
        Dictionary containing test results (accuracy, predictions, etc.)
    """
    # Determine dataset based on model path or config
    # Look for dataset indicators in the model path

    
    print(f"Loading {dataset_name} test dataset...")
    
    # Load the test dataset
# Load random test samples
    try:
        test_images, test_labels = load_random_test_samples(
            dataset_name=dataset_name, 
            num_samples=num_samples
        )
    except Exception as e:
        print(f"Error loading dataset: {e}")
        return None
    
    print(f"Testing classifier on {test_images.data.shape[0]} samples...")
    
    # Set model to evaluation mode and make predictions
    nn_model.eval()
    predictions_tensor = nn_model.forward(test_images)
    predictions = predictions_tensor.data
    
    # Convert predictions to class labels (argmax)
    predicted_labels = np.argmax(predictions, axis=1)
    true_labels = np.argmax(test_labels.data, axis=1)
    
    # Calculate accuracy
    accuracy = np.mean(predicted_labels == true_labels)
    
    # Calculate per-class statistics
    unique_classes = np.unique(true_labels)
    class_accuracies = {}
    for class_idx in unique_classes:
        class_mask = true_labels == class_idx
        if np.sum(class_mask) > 0:
            class_acc = np.mean(predicted_labels[class_mask] == true_labels[class_mask])
            class_accuracies[class_idx] = class_acc
    
    # Create visualization
    plot_classification_results(
        test_images.data, true_labels, predicted_labels, 
        accuracy, dataset_name, save_path
    )
    
    # Print results
    print(f"\n=== Classification Results ===")
    print(f"Dataset: {dataset_name.upper()}")
    print(f"Total samples tested: {len(true_labels)}")
    print(f"Overall Accuracy: {accuracy:.4f} ({accuracy*100:.2f}%)")
    print(f"\nPer-class accuracies:")
    for class_idx in sorted(class_accuracies.keys()):
        class_count = np.sum(true_labels == class_idx)
        print(f"  Class {class_idx}: {class_accuracies[class_idx]:.4f} "
              f"({class_accuracies[class_idx]*100:.2f}%) - {class_count} samples")
    
    return {
        'accuracy': accuracy,
        'predicted_labels': predicted_labels,
        'true_labels': true_labels,
        'class_accuracies': class_accuracies,
        'dataset': dataset_name,
        'num_samples': len(true_labels)
    }

def plot_classification_results(images_data, true_labels, predicted_labels, 
                              accuracy, dataset_name, save_path=None):
    """
    Plot test images with true and predicted labels.
    
    Args:
        images_data: NumPy array of test image data (flattened).
        true_labels: Array of true class labels.
        predicted_labels: Array of predicted class labels.
        accuracy: Overall accuracy score.
        dataset_name: Name of the dataset being tested.
        save_path: Optional path to save the plot.
    """
    num_samples = len(true_labels)
    cols = min(8, num_samples)  # Max 8 columns
    rows = min(4, int(np.ceil(num_samples / cols)))  # Max 4 rows
    display_samples = min(num_samples, rows * cols)
    
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 1.5, rows * 2))
    
    if rows == 1 and cols == 1:
        axes = [axes]
    elif rows == 1 or cols == 1:
        axes = axes.flatten()
    else:
        axes = axes.flatten()
    
    for i in range(rows * cols):
        if i < display_samples:
            # Reshape image back to 28x28
            img = images_data[i].reshape(28, 28)
            img = np.clip(img, 0, 1)
            
            # Determine if prediction is correct
            is_correct = true_labels[i] == predicted_labels[i]
            color = 'green' if is_correct else 'red'
            
            axes[i].imshow(img, cmap='gray')
            axes[i].set_title(f'True:{true_labels[i]} Pred:{predicted_labels[i]}', 
                            color=color, fontsize=10)
        axes[i].axis('off')
    
    plt.suptitle(f'{dataset_name.upper()} Classification Results\n'
                f'Accuracy: {accuracy:.4f} ({accuracy*100:.2f}%)', 
                fontsize=14, fontweight='bold')
    plt.tight_layout(rect=[0, 0, 1, 0.92])
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Classification results saved to: {save_path}")
    
    plt.show()

def predict_digit(nn_model, canvas_data):
    """
    Predict a digit from canvas data using a trained NN classifier.
    
    Args:
        nn_model: The trained NN classifier model.
        canvas_data: NumPy array of pixel values (784 elements, 0-255 range) or 2D array (28x28).
    
    Returns:
        Dictionary containing prediction results:
        - predicted_digit: The predicted digit (0-9)
        - confidence: Confidence score for the prediction
        - probabilities: Array of probabilities for each digit (0-9)
    """
    # Ensure canvas_data is in the right format
    if isinstance(canvas_data, list):
        canvas_data = np.array(canvas_data, dtype=np.float32)
    
    # If 2D array, flatten it
    if canvas_data.ndim == 2:
        canvas_data = canvas_data.flatten()
    
    # Ensure we have exactly 784 pixels
    if canvas_data.shape[0] != 784:
        raise ValueError(f"Expected 784 pixels, got {canvas_data.shape[0]}")
    
    # Normalize pixel values to [0, 1] range
    canvas_data = canvas_data.astype(np.float32) / 255.0
    
    # Reshape for model input (add batch dimension)
    input_tensor = Tensor(canvas_data.reshape(1, 784), requires_grad=False)
    
    # Set model to evaluation mode and make prediction
    nn_model.eval()
    predictions_tensor = nn_model.forward(input_tensor)
    probabilities = predictions_tensor.data[0]  # Remove batch dimension
    
    # Get predicted digit and confidence
    predicted_digit = int(np.argmax(probabilities))
    confidence = float(probabilities[predicted_digit])
    
    return {
        'predicted_digit': predicted_digit,
        'confidence': confidence,
        'probabilities': probabilities.tolist()
    }

def main():
    """Main function with command line interface for digit generation and classification testing."""
    parser = argparse.ArgumentParser(description='Generate digits using trained VAE/CVAE models or test NN classifiers')
    parser.add_argument('--model-path', type=str, required=True,
                       help='Path to the saved VAE/CVAE/NN model directory')
    parser.add_argument('--num-samples', type=int, default=16,
                       help='Number of digits to generate or test samples for classification (default: 16)')
    parser.add_argument('--save-images', type=str,
                       help='Path to save the generated image grid or classification results (optional)')
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
        if model_type is not "NN":
            print(f"Error: This script supports VAE, CVAE, or NN models. Loaded model type: {model_type}")
            return 1

        config = loaded_model.get_config()
        
        print(f"Model loaded successfully! Type: {loaded_model.model_type}")
        
        # --- Model Type Specific Logic ---
        
        if model_type == 'NN':
            # Classification testing mode
            print(f"NN Classifier detected. Running classification test...")
            print(f"Model config: Input: {config['nin']}, Output: {config['nouts']}")
            
            results = test_classifier(loaded_model, args.num_samples, history['dataset'], args.save_images)
            if results is None:
                print("Classification test failed.")
                return 1
        
        print("\nOperation completed successfully!")
        
    except Exception as e:
        print(f"\nAn error occurred: {str(e)}")
        return 1
    
    return 0

if __name__ == "__main__":
    exit(main())