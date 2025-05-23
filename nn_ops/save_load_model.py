import numpy as np 
import os 
import datetime 
import json 
import pickle
from .layers_and_networks import VAE 

def convert_to_json_serializable(item):
    """
    Recursively convert numpy types to JSON serializable native Python types.
    """
    if isinstance(item, np.ndarray):
        return item.tolist()
    if isinstance(item, (np.float32, np.float64, np.float16)): # Handle various numpy float types
        return float(item)
    if isinstance(item, (np.int32, np.int64, np.int8, np.int16)): # Handle various numpy int types
        return int(item)
    if isinstance(item, np.bool_): # Handle numpy bool type
        return bool(item)
    if isinstance(item, dict):
        return {key: convert_to_json_serializable(value) for key, value in item.items()}
    if isinstance(item, list):
        return [convert_to_json_serializable(element) for element in item]
    return item

class ModelSaver:
    def __init__(self, model, model_name, include_history, base_path='saved_models'):
        """
        Initialize ModelSaver with a base directory for saved models
        """
        self.base_path = base_path
        self.model = model
        self.model_name = model_name
        self.include_history = include_history
        self.base_path = base_path 

        os.makedirs(base_path, exist_ok=True)
        self.save_model()
    
    def save_model(self):
        """
        Save the model, its architecture, and training history
        
        Args:
            model: The trained MLP model
            model_name: Optional custom name for the model
            include_history: Optional training history dictionary
        """
        model_name_to_use = self.model_name
        if model_name_to_use is None:
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            model_name_to_use = f'model_{timestamp}'
            # If self.model_name was None, you might want to store the generated name
            # self.model_name = model_name_to_use # Optional: update instance attribute

        # Create directory for this model
        model_dir = os.path.join(self.base_path, model_name_to_use)
        os.makedirs(model_dir, exist_ok=True)

        # Prepare model architecture for saving
        architecture = {
            'nin': self.model.layers[0].weights.data.shape[0], # Assuming weights.data is a numpy array
            'nouts': [layer.weights.data.shape[1] for layer in self.model.layers],
            'activations': [str(layer.activation) for layer in self.model.layers], # Ensure activations are serializable (e.g., strings)
            'dropout_rates': [layer.dropout.p if layer.dropout is not None else 0.0
                            for layer in self.model.layers]
        }

        # Convert architecture dictionary to be JSON serializable
        serializable_architecture = convert_to_json_serializable(architecture)
        with open(os.path.join(model_dir, 'architecture.json'), 'w') as f:
            json.dump(serializable_architecture, f, indent=4) # Added indent for readability

        # Save weights and biases
        # Assuming layer.weights.data and layer.bias.data are numpy arrays, pickle handles them fine.
        weights_dict = {
            f'layer_{i}': {
                'weights': layer.weights.data,
                'bias': layer.bias.data
            }
            for i, layer in enumerate(self.model.layers)
        }

        with open(os.path.join(model_dir, 'weights.pkl'), 'wb') as f:
            pickle.dump(weights_dict, f)

        # Save training history if provided
        if self.include_history:
            # Convert the entire history dictionary to be JSON serializable
            serializable_history = convert_to_json_serializable(self.include_history)
            with open(os.path.join(model_dir, 'history.json'), 'w') as f:
                json.dump(serializable_history, f, indent=4) # Added indent for readability

        print(f"Model saved successfully to {model_dir}")
        return model_dir

    def load_model(self, model_path):
        """
        Load a saved model
        
        Args:
            model_path: Path to the saved model directory
        Returns:
            loaded_model: The loaded MLP model
            history: Training history if it was saved, None otherwise
        """
        # Load architecture
        with open(os.path.join(model_path, 'architecture.json'), 'r') as f:
            architecture = json.load(f)
        
        input_dim, hidden_dim, latent_dim = 784, 256, 20
        # Create new model with saved architecture
        model = VAE(
        input_dim=input_dim,  # 28x28 pixels
        hidden_dim=hidden_dim,  # Example architecture
        latent_dim=latent_dim
    )

        
        # Load weights and biases
        with open(os.path.join(model_path, 'weights.pkl'), 'rb') as f:
            weights_dict = pickle.load(f)
        
        # Set weights and biases
        for i, layer in enumerate(model.layers):
            layer_weights = weights_dict[f'layer_{i}']
            layer.weights.data = layer_weights['weights']
            layer.bias.data = layer_weights['bias']
        
        # Load history if it exists
        history = None
        history_path = os.path.join(model_path, 'history.json')
        if os.path.exists(history_path):
            with open(history_path, 'r') as f:
                history = json.load(f)
        
        return model, history

# Helper function to get list of saved models
    def list_saved_models(self, base_path='saved_models'):
        """
        List all saved models with their details
        """
        if not os.path.exists(base_path):
            print("No saved models found.")
            return []
        
        models = []
        for model_name in os.listdir(base_path):
            model_path = os.path.join(base_path, model_name)
            if os.path.isdir(model_path):
                # Load architecture to get model details
                try:
                    with open(os.path.join(model_path, 'architecture.json'), 'r') as f:
                        architecture = json.load(f)
                    models.append({
                        'name': model_name,
                        'path': model_path,
                        'architecture': architecture,
                        'saved_date': datetime.fromtimestamp(os.path.getctime(model_path))
                    })
                except:
                    continue
        
        return models
