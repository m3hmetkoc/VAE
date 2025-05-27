import numpy as np
import os
import datetime
import json
import pickle 
from data_process import MNISTBatchGenerator, load_mnist_data
from train_ops import Train
from layers_and_networks import VAE, NN 


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
    def __init__(self, model, model_name=None, include_history=None, base_path='../saved_models'):
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
        Weights are saved as a flat list of numpy arrays obtained from model.parameters().
        """
        model_name_to_use = self.model_name
        if model_name_to_use is None:
            timestamp = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
            model_name_to_use = f'model_{timestamp}'

        model_dir = os.path.join(self.base_path, model_name_to_use)
        os.makedirs(f"{model_dir}_{datetime.datetime.now().strftime('%H:%M')}", exist_ok=True)
    
        architecture = {}
        if isinstance(self.model, VAE): #
            architecture['model_type'] = 'VAE'
            architecture['input_dim'] = self.model.encoder.fc1.W.data.shape[0] #
            architecture['hidden_dim'] = self.model.encoder.fc1.W.data.shape[1] #
            architecture['latent_dim'] = self.model.encoder.fc_mu.W.data.shape[1] #
        elif isinstance(self.model, NN): #
            architecture['model_type'] = 'NN'
            if not self.model.layers: #
                raise ValueError("Cannot save an NN model with no layers.")
            architecture['nin'] = self.model.layers[0].W.data.shape[0] #
            architecture['nouts'] = [layer.W.data.shape[1] for layer in self.model.layers] #
            architecture['activations'] = [str(layer.activation) for layer in self.model.layers] #
            architecture['dropout_rates'] = [layer.dropout.p if layer.dropout is not None else 0.0
                                           for layer in self.model.layers] #
        else:
            raise TypeError(f"Unsupported model type for saving: {type(self.model)}")

        serializable_architecture = convert_to_json_serializable(architecture)
        with open(os.path.join(model_dir, 'architecture.json'), 'w') as f:
            json.dump(serializable_architecture, f, indent=4)

        weights_data_list = [param.data for param in self.model.parameters()] #

        with open(os.path.join(model_dir, 'weights.pkl'), 'wb') as f:
            pickle.dump(weights_data_list, f)

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
        Weights are loaded by iterating through model.parameters() and assigning the saved numpy arrays.
        """
        arch_path = os.path.join(model_path, 'architecture.json')
        if not os.path.exists(arch_path):
            raise FileNotFoundError(f"Architecture file not found: {arch_path}")
        with open(arch_path, 'r') as f:
            architecture = json.load(f)

        weights_path = os.path.join(model_path, 'weights.pkl')
        if not os.path.exists(weights_path):
            raise FileNotFoundError(f"Weights file not found: {weights_path}")
        with open(weights_path, 'rb') as f:
            loaded_weights_data_list = pickle.load(f)

        model_type = architecture.get('model_type')
        if not model_type:
            raise ValueError("Model type not specified in architecture.json.")

        loaded_model = None
        if model_type == 'VAE':
            loaded_model = VAE(
                input_dim=architecture['input_dim'],
                hidden_dim=architecture['hidden_dim'],
                latent_dim=architecture['latent_dim']
            )
        elif model_type == 'NN':
            nin = architecture['nin'] #
            nouts = architecture['nouts'] #
            activations = architecture['activations'] #
            dropout_rates = architecture.get('dropout_rates', [0.0] * len(nouts)) #
            # init_method = architecture.get('init_method', 'he') # Default if not saved

            loaded_model = NN(
                nin=nin,
                nouts=nouts,
                activations=activations,
                dropout_rates=dropout_rates
                # init_method=init_method # Pass if NN constructor uses it
            ) #
        else:
            raise ValueError(f"Unknown model type in architecture.json: {model_type}")

        # Load weights using model.parameters()
        # This relies on model.parameters() returning tensors in a consistent order.
        model_params_tensors = loaded_model.parameters() #
        if len(loaded_weights_data_list) != len(model_params_tensors):
            raise ValueError(
                f"Mismatch in the number of saved weight arrays ({len(loaded_weights_data_list)}) "
                f"and model parameters ({len(model_params_tensors)}). "
                "The model architecture may have changed or the weights file is corrupted."
            )

        for param_tensor, loaded_data in zip(model_params_tensors, loaded_weights_data_list):
            if param_tensor.data.shape != loaded_data.shape:
                raise ValueError(
                    f"Shape mismatch for a parameter: model expects {param_tensor.data.shape}, "
                    f"saved data has {loaded_data.shape}. Model architecture may be inconsistent."
                )
            param_tensor.data = loaded_data

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
        
        if not models_info:
            print("No saved models found in the directory.")
        return models_info

# Example Usage (assuming layers_and_networks.py and tensor_class.py are accessible)
if __name__ == '__main__':
    # Create a dummy VAE model for testing
    print("Testing VAE save and load...")
    try:
        # Simulate some "training" by assigning random data to weights
        num_epochs = 3
        learning_rate = 0.001
        batch_size = 64
        
        # Load data
        train_loader, test_loader = load_mnist_data(batch_size)
        train_generator = MNISTBatchGenerator(train_loader)
        test_generator = MNISTBatchGenerator(test_loader)
        input_dim, hidden_dim, latent_dim = 784, 256, 20
        # Create model (assuming your MLP class is defined)
        vae_model = VAE(
            input_dim=input_dim,  # 28x28 pixels
            hidden_dim=hidden_dim,  # Example architecture
            latent_dim=latent_dim
        )
        
        trainer = Train(
            model=vae_model,
            train_generator=train_generator,
            test_generator=test_generator,
            num_epochs=num_epochs,
            learning_rate=learning_rate,
            batch_size=batch_size,
            early_stopping_patience=15
        )

        trainer.train()

        saver_vae = ModelSaver(model=vae_model, model_name="my_test_vae_simplified", include_history={"epoch": 10, "loss": 0.123})
        saved_vae_path = saver_vae.save_model()

        loaded_vae, history_vae = ModelSaver.load_model(saved_vae_path)
        print(f"VAE model loaded. History: {history_vae}")
        
        # Verification: Check if weights are loaded by comparing one parameter
        original_vae_param_data = vae_model.parameters()[0].data #
        loaded_vae_param_data = loaded_vae.parameters()[0].data #
        np.testing.assert_array_almost_equal(original_vae_param_data, loaded_vae_param_data)
        print("VAE weights verified successfully (first parameter).")

    except Exception as e:
        print(f"Error during VAE test: {e}")
        import traceback
        traceback.print_exc()

    print("\nTesting NN save and load...")
    try:
        nn_model = NN(nin=100, nouts=[64, 32], activations=['relu', 'sigmoid'], dropout_rates=[0.1, 0.0]) #
        # Simulate some "training"
        for param in nn_model.parameters(): #
            param.data = np.random.rand(*param.data.shape).astype(np.float32)
            
        saver_nn = ModelSaver(model=nn_model, model_name="my_test_nn_simplified", include_history={"steps": 1000, "accuracy": 0.95})
        saved_nn_path = saver_nn.save_model()

        loaded_nn, history_nn = ModelSaver.load_model(saved_nn_path)
        print(f"NN model loaded. History: {history_nn}")

        # Verification: Check if weights are loaded by comparing one parameter
        original_nn_param_data = nn_model.parameters()[0].data #
        loaded_nn_param_data = loaded_nn.parameters()[0].data #
        np.testing.assert_array_almost_equal(original_nn_param_data, loaded_nn_param_data)
        print("NN weights verified successfully (first parameter).")


    except Exception as e:
        print(f"Error during NN test: {e}")
        import traceback
        traceback.print_exc()

    print("\nListing saved models:")
    all_models = ModelSaver.list_saved_models()
    for model_info in all_models:
        print(f" - Name: {model_info['name']}, Type: {model_info['type']}, Path: {model_info['path']}, Saved: {model_info['saved_date']}")