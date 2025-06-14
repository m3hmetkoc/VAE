# A Deep Learning Framework for VAEs with an Interactive Web UI

This project is a comprehensive deep learning framework built entirely from scratch in Python using NumPy. It focuses on generative modeling with Variational Autoencoders (VAEs) and Conditional VAEs (CVAEs), and features a custom-built computational engine that handles automatic differentiation and backpropagation.

To make the models accessible and fun to use, the project includes a full-stack web application with a React frontend and a Flask backend. This UI allows users to draw digits and see a trained classifier predict them in real-time, as well as generate novel images of digits and clothing using the trained VAEs and CVAEs.

This project was developed as a senior computer engineering project to gain a deep, foundational understanding of how neural networks and generative models work.

## ✨ Key Features

-   **Custom Autograd Engine:** A NumPy-based `Tensor` class that dynamically builds a computational graph and performs backpropagation.
-   **From-Scratch Models:** Implementation of VAE, CVAE, and MLP classifiers.
-   **Interactive Web Application:** A React + Flask web app for interacting with the models.
-   **Real-time Digit Recognition:** Draw a digit on an HTML canvas and have a trained model predict it.
-   **Conditional Image Generation:** Select a trained model (VAE or CVAE) and generate new images of MNIST digits or Fashion-MNIST clothing items.
-   **Flexible Configuration:** Define model architectures using JSON files for easy experimentation.

## 📁 Project Structure

```
.
├── data_and_models/    # Directory for datasets and saved models
├── digit-guess-web/    # React.js frontend application
├── images_report/      # Supporting images for the project report
├── main/               # Main Python scripts (Flask server, training, generation)
├── nn_ops/             # The core "from-scratch" deep learning engine
├── readme.md
└── requirements.txt
```

## 🚀 Getting Started

To run the web application, you need to start both the Flask backend server and the React frontend client.

### Prerequisites

-   Python 3.8+
-   Node.js and npm

### 1. Backend Setup

First, set up and run the Flask server which provides the API for the frontend.

```bash
# 1. Navigate to the project root directory
cd /path/to/your/project

# 2. Create and activate a virtual environment (recommended)
python -m venv venv
source venv/bin/activate  # On Windows, use `venv\Scripts\activate`

# 3. Install Python dependencies
pip install -r requirements.txt

# 4. Start the Flask server
#    You must provide a path to a trained classifier model.
#    A pre-trained model might be in `data_and_models/saved_models/`
python main/prediction_server.py --model-path "path/to/your/classifier_model"
```

The backend server will now be running on `http://127.0.0.1:5000`.

### 2. Frontend Setup

In a **new terminal**, set up and run the React application.

```bash
# 1. Navigate to the frontend directory
cd /path/to/your/project/digit-guess-web

# 2. Install Node.js dependencies
npm install

# 3. Start the React development server
npm start
```

The web application will automatically open in your browser at `http://localhost:3000`. You can now interact with the digit predictor and image generation features.

## Advanced Usage: Command-Line Tools

For users who want to bypass the web interface, the project includes command-line scripts for training, generation, and evaluation. These scripts are located in the `main/` directory.

### Training Models (`main/main.py`)

The `main.py` script is the entry point for training VAE, CVAE, and standard Neural Network (NN) classifiers.

**Key Arguments:**

*   `--model-type`: Specify the model architecture. Choices: `VAE`, `NN`.
*   `--cvae`: If specified with `--model-type VAE`, a Conditional VAE will be trained.
*   `--dataset`: Choose the dataset for training. Choices: `mnist`, `fashion_mnist`.
*   `--epochs`: Number of training epochs.
*   `--batch-size`: Training batch size.
*   `--learning-rate`: Learning rate for the optimizer.
*   `--model-name`: A custom name for the saved model directory.

**Training Examples:**

*   **Train a standard VAE on MNIST:**
    ```bash
    python main/main.py --model-type VAE --dataset mnist --epochs 50 --model-name MyVAE_MNIST
    ```

*   **Train a Conditional VAE (CVAE) on Fashion-MNIST:**
    ```bash
    python main/main.py --model-type VAE --cvae --dataset fashion_mnist --epochs 100 --model-name MyCVAE_Fashion
    ```

*   **Train a Neural Network Classifier on MNIST:**
    ```bash
    python main/main.py --model-type NN --dataset mnist --epochs 30 --model-name MyNN_Classifier
    ```

### Generating Images (`main/generate_images.py`)

This script uses a trained VAE or CVAE to generate new images.

**Key Arguments:**

*   `--model-path`: **(Required)** Path to the saved model directory (e.g., `models/MyCVAE_Fashion_...`).
*   `--num-samples`: Number of images to generate.
*   `--label`: **(CVAE Only)** Generate images of a specific class (e.g., `--label 7`).
*   `--interpolate`: **(VAE Only)** Generate a smooth transition between two random points in the latent space.
*   `--save-images`: Optional path to save the output image (e.g., `generated_images.png`).

**Generation Examples:**

*   **Generate random images with a VAE:**
    ```bash
    python main/generate_images.py --model-path models/MyVAE_MNIST_... --num-samples 16
    ```

*   **Generate specific digits (e.g., '8') with a CVAE:**
    ```bash
    python main/generate_images.py --model-path models/MyCVAE_Fashion_... --label 8 --num-samples 10
    ```

*   **Create a latent space interpolation with a VAE:**
    ```bash
    python main/generate_images.py --model-path models/MyVAE_MNIST_... --interpolate --interp-steps 12
    ```

### Predicting Digits (`main/predict_digits.py`)

This script is used to evaluate the performance of a trained **NN classifier** on the test set. For interactive single-digit prediction, please use the web application.

**Key Arguments:**

*   `--model-path`: **(Required)** Path to the saved NN classifier model directory (e.g., `models/MyNN_Classifier_...`).
*   `--num-samples`: Number of random test images to evaluate.
*   `--save-images`: Optional path to save the results visualization.

**Prediction Example:**

*   **Test an NN classifier on 20 random samples from its dataset:**
    ```bash
    python main/predict_digits.py --model-path models/MyNN_Classifier_... --num-samples 20
    ```

This will display and plot the predictions versus the true labels and print the overall accuracy.

## Project Structure

```
.
├── data_and_models/    # Directory for datasets and saved models
├── digit-guess-web/    # React.js frontend application
├── images_report/      # Supporting images for the project report
├── main/               # Main Python scripts (Flask server, training, generation)
├── nn_ops/             # The core "from-scratch" deep learning engine
├── final_project_report.md
├── readme.md
└── requirements.txt
```