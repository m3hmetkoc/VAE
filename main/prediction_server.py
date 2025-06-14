"""
Flask server for MNIST digit prediction.
Loads the trained model and provides API endpoints for real-time predictions.
"""
from flask import Flask, request, jsonify
from flask_cors import CORS
import numpy as np
import argparse
import os
import sys

# Add VAE directory to path
current_script_dir = os.path.dirname(os.path.abspath(__file__))

# Go up one level to reach project_root/
project_root = os.path.join(current_script_dir, '..')

# Add project_root to sys.path
sys.path.append(project_root)

from nn_ops import ModelSaver, Tensor
from predict_digits import predict_digit
from generate_images import generate_image, generate_specific_images_with_label
from utils import numpy_to_base64

app = Flask(__name__)
CORS(app)  # Enable CORS for React frontend

# Global model variable
model = None
# Cache for loaded generator models
generator_models_cache = {}

def load_model(model_path):
    """Load the trained MNIST classifier model."""
    global model
    
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model not found at {model_path}")
    
    try:
        loaded_model, history = ModelSaver.load_model(model_path)
        print(f"Model loaded successfully!")
        print(f"Model type: {loaded_model.model_type}")
        print(f"Training dataset: {history.get('dataset', 'Unknown')}")
        return loaded_model
    except Exception as e:
        print(f"Error loading model: {e}")
        raise

@app.route('/models', methods=['GET'])
def list_models():
    """List all available VAE/CVAE models."""
    try:
        models = ModelSaver.list_saved_models()
        # Filter for only generative models
        gen_models = [m for m in models if m['type'] in ['VAE', 'CVAE']]
        return jsonify(gen_models)
    except Exception as e:
        print(f"Error listing models: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/generate', methods=['POST'])
def generate():
    """Generate digits using a selected VAE/CVAE model."""
    try:
        data = request.get_json()
        model_path = data.get('model_path')
        num_samples = data.get('num_samples', 9)
        label = data.get('label')  # Can be None for unconditional or VAE

        if not model_path:
            return jsonify({'error': 'model_path is required'}), 400

        # Use a simple cache for generator models
        if model_path in generator_models_cache:
            gen_model = generator_models_cache[model_path]
        else:
            gen_model, _ = ModelSaver.load_model(model_path)
            generator_models_cache[model_path] = gen_model
        
        config = gen_model.get_config()
        latent_dim = config['latent_dim']
        model_type = config.get('model_type')

        if model_type not in ['VAE', 'CVAE']:
            return jsonify({'error': 'Selected model is not a generative model (VAE/CVAE)'}), 400

        images_data = None
        if label is not None and model_type == 'CVAE':
            images_data = generate_specific_images_with_label(
                gen_model, int(label), num_samples, latent_dim, num_classes=10
            )
        else:
            images_data = generate_image(gen_model, num_samples, latent_dim)

        base64_images = [numpy_to_base64(img.reshape(28, 28)) for img in images_data]
        
        return jsonify({'images': base64_images})

    except Exception as e:
        import traceback
        traceback.print_exc()
        print(f"Generation error: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/health', methods=['GET'])
def health_check():
    """Health check endpoint."""
    return jsonify({
        'status': 'healthy',
        'model_loaded': model is not None
    })

@app.route('/predict', methods=['POST'])
def predict():
    """
    Predict digit from canvas data.
    
    Expected JSON payload:
    {
        "canvas_data": [pixel_values_array_of_784_elements]
    }
    
    Returns:
    {
        "predicted_digit": int,
        "confidence": float,
        "probabilities": [float_array_of_10_elements]
    }
    """
    try:
        if model is None:
            return jsonify({'error': 'Model not loaded'}), 500
        
        # Get canvas data from request
        data = request.get_json()
        if not data or 'canvas_data' not in data:
            return jsonify({'error': 'canvas_data is required'}), 400
        
        canvas_data = data['canvas_data']
        
        # Validate canvas data
        if not isinstance(canvas_data, list):
            return jsonify({'error': 'canvas_data must be an array'}), 400
        
        if len(canvas_data) != 784:
            return jsonify({'error': f'canvas_data must have 784 elements, got {len(canvas_data)}'}), 400
        
        # Make prediction
        result = predict_digit(model, canvas_data)
        
        return jsonify(result)
        
    except Exception as e:
        print(f"Prediction error: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/model-info', methods=['GET'])
def model_info():
    """Get information about the loaded model."""
    try:
        if model is None:
            return jsonify({'error': 'Model not loaded'}), 500
        
        config = model.get_config()
        return jsonify({
            'model_type': config.get('model_type'),
            'input_size': config.get('nin'),
            'output_size': len(config.get('nouts', [])),
            'architecture': config.get('nouts'),
            'activations': config.get('activations')
        })
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    try:
        parser = argparse.ArgumentParser(description='Train neural network models')
        parser.add_argument('--model-path', type=str, required=True,
                       help='Path to the saved VAE/CVAE/NN model directory')
        args = parser.parse_args() 
        model = load_model(args.model_path)
        print("Starting Flask server...")
        app.run(host='127.0.0.1', port=5000, debug=True)
    except Exception as e:
        print(f"Failed to start server: {e}")
        sys.exit(1) 