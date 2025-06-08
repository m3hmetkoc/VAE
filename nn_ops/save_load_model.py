import numpy as np
import os
import datetime
import json
import pickle 
from .layers_and_networks import VAE, NN, VAE_old 


def convert_to_json_serializable(item):
    """
    Recursively convert numpy types to JSON serializable native Python types.
    """
    if isinstance(item, np.ndarray):
        return item.tolist()
    if isinstance(item, (np.float32, np.float64, np.float16)):
        return float(item)
    if isinstance(item, (np.int32, np.int64, np.int8, np.int16)):
        return int(item)
    if isinstance(item, np.bool_):
        return bool(item)
    if isinstance(item, dict):
        return {key: convert_to_json_serializable(value) for key, value in item.items()}
    if isinstance(item, list):
        return [convert_to_json_serializable(element) for element in item]
    return item

class ModelSaver:
    def __init__(self, model, model_name=None, include_history=None, base_path='/saved_models'):
        """
        Initialize ModelSaver with a base directory for saved models
        """
        self.model = model
        self.model_name = model_name
        self.include_history = include_history
        self.base_path = base_path

        os.makedirs(self.base_path, exist_ok=True)

    def save_model(self):
        """
        Save the model, its architecture, and training history.
        Uses the model's get_config() method to save architecture information.
        """
        model_name_to_use = self.model_name
        if model_name_to_use is None:
            timestamp = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
            model_name_to_use = f'model_{timestamp}'

        # Create unique directory with timestamp to avoid conflicts
        timestamp_suffix = datetime.datetime.now().strftime('%H:%M:%S')
        model_dir = os.path.join(self.base_path, f"{model_name_to_use}_{timestamp_suffix}")
        os.makedirs(model_dir, exist_ok=True)
    
        # Get architecture configuration from model
        try:
            architecture = self.model.get_config()
        except AttributeError:
            raise ValueError(f"Model type {type(self.model)} does not support get_config() method")

        # Save architecture
        serializable_architecture = convert_to_json_serializable(architecture)
        with open(os.path.join(model_dir, 'architecture.json'), 'w') as f:
            json.dump(serializable_architecture, f, indent=4)

        # Save weights
        weights_data_list = [param.data for param in self.model.parameters()]
        with open(os.path.join(model_dir, 'weights.pkl'), 'wb') as f:
            pickle.dump(weights_data_list, f)

        # Save training history if provided
        if self.include_history:
            serializable_history = convert_to_json_serializable(self.include_history)
            with open(os.path.join(model_dir, 'history.json'), 'w') as f:
                json.dump(serializable_history, f, indent=4)

        print(f"Model saved successfully to {model_dir}")
        return model_dir

    @staticmethod
    def load_model(model_path):
        """
        Load a saved model.
        Uses the architecture.json to reconstruct the model with correct configuration.
        """
        # Load architecture
        arch_path = os.path.join(model_path, 'architecture.json')
        if not os.path.exists(arch_path):
            raise FileNotFoundError(f"Architecture file not found: {arch_path}")
        with open(arch_path, 'r') as f:
            architecture = json.load(f)

        # Load weights
        weights_path = os.path.join(model_path, 'weights.pkl')
        if not os.path.exists(weights_path):
            raise FileNotFoundError(f"Weights file not found: {weights_path}")
        with open(weights_path, 'rb') as f:
            loaded_weights_data_list = pickle.load(f)

        # Get model type
        model_type = architecture.get('model_type')
        if not model_type:
            raise ValueError("Model type not specified in architecture.json.")

        # Reconstruct model based on type
        loaded_model = None
        if model_type == 'VAE_old':
            loaded_model = VAE_old(
                input_dim=architecture['input_dim'],
                hidden_dim=architecture['hidden_dim'],
                latent_dim=architecture['latent_dim'],
                init_method=architecture.get('init_method', 'he')
            )
        elif model_type == 'VAE':
            loaded_model = VAE(
                input_dim=architecture['input_dim'],
                latent_dim=architecture['latent_dim'],
                encoder_hidden_dims=architecture.get('encoder_hidden_dims', [256]),
                decoder_hidden_dims=architecture.get('decoder_hidden_dims', [256]),
                encoder_activations=architecture.get('encoder_activations'),
                decoder_activations=architecture.get('decoder_activations'),
                encoder_dropout_rates=architecture.get('encoder_dropout_rates'),
                decoder_dropout_rates=architecture.get('decoder_dropout_rates'),
                init_method=architecture.get('init_method', 'he')
            )
        elif model_type == 'NN':
            loaded_model = NN(
                nin=architecture['nin'],
                nouts=architecture['nouts'],
                activations=architecture['activations'],
                dropout_rates=architecture.get('dropout_rates', [0.0] * len(architecture['nouts'])),
                init_method=architecture.get('init_method', 'he')
            )
        else:
            raise ValueError(f"Unknown model type in architecture.json: {model_type}")

        # Load weights into model parameters
        model_params_tensors = loaded_model.parameters()
        if len(loaded_weights_data_list) != len(model_params_tensors):
            raise ValueError(
                f"Mismatch in the number of saved weight arrays ({len(loaded_weights_data_list)}) "
                f"and model parameters ({len(model_params_tensors)}). "
                "The model architecture may have changed or the weights file is corrupted."
            )

        # Assign loaded weights to model parameters
        for param_tensor, loaded_data in zip(model_params_tensors, loaded_weights_data_list):
            if param_tensor.data.shape != loaded_data.shape:
                raise ValueError(
                    f"Shape mismatch for a parameter: model expects {param_tensor.data.shape}, "
                    f"saved data has {loaded_data.shape}. Model architecture may be inconsistent."
                )
            param_tensor.data = loaded_data.copy()  # Use copy to avoid reference issues

        # Load training history if available
        history = None
        history_path = os.path.join(model_path, 'history.json')
        if os.path.exists(history_path):
            with open(history_path, 'r') as f:
                history = json.load(f)
        
        print(f"Model loaded successfully from {model_path}")
        return loaded_model, history

    @staticmethod
    def list_saved_models(base_path='../saved_models'):
        """
        List all saved models with their details.
        """
        if not os.path.exists(base_path):
            print("No saved models directory found.")
            return []
        
        models_info = []
        for model_name in os.listdir(base_path):
            model_path = os.path.join(base_path, model_name)
            if os.path.isdir(model_path):
                try:
                    arch_path = os.path.join(model_path, 'architecture.json')
                    if os.path.exists(arch_path):
                        with open(arch_path, 'r') as f:
                            architecture_content = json.load(f)
                        
                        models_info.append({
                            'name': model_name,
                            'path': model_path,
                            'type': architecture_content.get('model_type', 'Unknown'),
                            'architecture': architecture_content,
                            'saved_date': datetime.datetime.fromtimestamp(os.path.getctime(model_path))
                        })
                except Exception as e:
                    print(f"Could not read model {model_name} metadata: {e}")
                    continue
        
        # Sort by saved date (newest first)
        models_info.sort(key=lambda x: x['saved_date'], reverse=True)
        
        if not models_info:
            print("No saved models found in the directory.")
        return models_info
