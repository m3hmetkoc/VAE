# Neural Network Training and VAE Digit Generation

A comprehensive framework for training various neural network architectures (VAE, NN, VAE_old) and generating MNIST digits using trained Variational Autoencoders (VAEs).

## 🚀 Features

- **Universal Training Pipeline**: Train VAE, NN, and VAE_old models with a single script
- **Flexible Architecture Configuration**: Define custom model architectures via JSON files
- **VAE Digit Generation**: Generate new MNIST-style digits using trained VAE models
- **Latent Space Exploration**: Interpolate between points in latent space
- **Model Management**: Save, load, and list trained models with metadata
- **Command-Line Interface**: Easy-to-use CLI for all operations

## 📁 Project Structure

```
project/
├── main.py                 # Universal model training script
├── generate_digits.py      # VAE digit generation script
├── model_configs.json      # Example model configurations
├── save_load_model.py      # Model saving/loading utilities
├── nn_ops/                 # Neural network operations module
│   ├── __init__.py
│   ├── layers_and_networks.py
│   ├── data_process.py
│   └── train_ops.py
└── saved_models/           # Directory for saved models (auto-created)
```

## 🛠️ Installation

1. **Clone the repository**:
```bash
git clone <repository-url>
cd neural-network-training
```

2. **Install dependencies**:
```bash
pip install numpy matplotlib torch torchvision
```

3. **Ensure your `nn_ops` module is properly set up** with all required components:
   - `VAE`, `NN`, `VAE_old` classes
   - `MNISTBatchGenerator`, `Train`, `ModelSaver`
   - `load_mnist_data` function

## 🎯 Quick Start

### Training Your First VAE

```bash
# Train a default VAE with 50 epochs
python main.py --model-type VAE --epochs 50 --batch-size 64

# Train with custom parameters
python main.py --model-type VAE --epochs 100 --learning-rate 0.001 --model-name my_vae
```

### Generating Digits

```bash
# First, list your trained models
python generate_digits.py --list-models

# Generate 25 digits using a trained VAE
python generate_digits.py --model-path "saved_models/my_vae_12:34:56" --num-samples 25
```

## 📚 Detailed Usage Guide

### 1. Training Models

The `main.py` script supports multiple ways to train models:

#### Command Line Arguments

```bash
python main.py [OPTIONS]
```

**Available Options:**
- `--config PATH`: Path to JSON configuration file
- `--model-type {VAE,NN,VAE_old}`: Model type (default: VAE)
- `--epochs INT`: Number of training epochs (default: 50)
- `--batch-size INT`: Training batch size (default: 64)  
- `--learning-rate FLOAT`: Learning rate (default: 0.001)
- `--model-name STR`: Custom name for saved model
- `--list-models`: List all saved models

#### Training Examples

**Basic VAE Training:**
```bash
python main.py --model-type VAE --epochs 50 --batch-size 64
```

**Advanced VAE with Custom Parameters:**
```bash
python main.py --model-type VAE --epochs 100 --batch-size 128 --learning-rate 0.0005 --model-name advanced_vae
```

**Neural Network Classifier:**
```bash
python main.py --model-type NN --epochs 30 --learning-rate 0.01 --model-name mnist_classifier
```

**Using Configuration Files:**
```bash
python main.py --config model_configs.json --epochs 75 --model-name deep_vae
```

### 2. Model Configuration Files

Create custom architectures using JSON configuration files:

#### VAE Configuration Example:
```json
{
  "model_type": "VAE",
  "input_dim": 784,
  "latent_dim": 32,
  "encoder_hidden_dims": [512, 256, 128],
  "decoder_hidden_dims": [128, 256, 512],
  "encoder_activations": ["relu", "relu", "relu"],
  "decoder_activations": ["relu", "relu", "sigmoid"],
  "encoder_dropout_rates": [0.2, 0.3, 0.2],
  "decoder_dropout_rates": [0.2, 0.3, 0.0],
  "init_method": "he"
}
```

#### NN Configuration Example:
```json
{
  "model_type": "NN",
  "nin": 784,
  "nouts": [256, 128, 64, 10],
  "activations": ["relu", "relu", "relu", "softmax"],
  "dropout_rates": [0.2, 0.3, 0.4, 0.0],
  "init_method": "he"
}
```

### 3. Generating Digits with VAE

The `generate_digits.py` script provides multiple generation modes:

#### Command Line Arguments

```bash
python generate_digits.py [OPTIONS]
```

**Available Options:**
- `--model-path PATH`: Path to saved VAE model (required)
- `--num-samples INT`: Number of digits to generate (default: 25)
- `--save-images PATH`: Save generated images to file
- `--interpolate`: Generate latent space interpolation
- `--interp-steps INT`: Number of interpolation steps (default: 10)
- `--single`: Generate single digit with random latent vector
- `--list-models`: List all saved models

#### Generation Examples

**Basic Digit Generation:**
```bash
python generate_digits.py --model-path "saved_models/vae_trained_12:34:56" --num-samples 25
```

**Generate and Save Images:**
```bash
python generate_digits.py --model-path "saved_models/vae_trained_12:34:56" --num-samples 100 --save-images my_digits.png
```

**Latent Space Interpolation:**
```bash
python generate_digits.py --model-path "saved_models/vae_trained_12:34:56" --interpolate --interp-steps 15
```

**Single Digit Generation:**
```bash
python generate_digits.py --model-path "saved_models/vae_trained_12:34:56" --single
```

## 📊 Model Management

### Listing Saved Models:
```bash
python main.py --list-models
# or
python generate_digits.py --list-models
```

### Model Directory Structure:
```
saved_models/
└── vae_trained_12:34:56/
    ├── architecture.json    # Model configuration
    ├── weights.pkl         # Model parameters
    └── history.json        # Training metadata
```

### Loading Models Programmatically:
```python
from nn_ops import ModelSaver

# Load a saved model
model, history = ModelSaver.load_model("saved_models/vae_trained_12:34:56")
print(f"Model trained for {history['epochs']} epochs")
```

## 🎨 Generation Examples

### Batch Generation Workflow:
```bash
# 1. Train a VAE
python main.py --model-type VAE --epochs 100 --model-name production_vae

# 2. List models to get the exact path
python generate_digits.py --list-models

# 3. Generate digits
python generate_digits.py --model-path "saved_models/production_vae_XX:XX:XX" --num-samples 50 --save-images results.png

# 4. Explore latent space
python generate_digits.py --model-path "saved_models/production_vae_XX:XX:XX" --interpolate --interp-steps 20
```

## 🐛 Troubleshooting

### Common Issues:

1. **"Model type not supported" error**:
   - Make sure your model is a VAE when using `generate_digits.py`
   - Check that `model_type` in configuration is correct

2. **"Model path does not exist"**:
   - Use `--list-models` to see available models
   - Copy the exact path from the list

3. **Memory issues**:
   - Reduce batch size (`--batch-size 32`)
   - Reduce model size in configuration

For issues and questions:
mail: kocmehmet3366@gmail.com

Mehmet Koç 

---

**Happy Training and Generating! 🎨✨**