import os
import torch
from torchvision import datasets, transforms
from .tensor_class import Tensor

# Dictionary to configure dataset-specific parameters
DATASET_CONFIG = {
    'mnist': {
        'class': datasets.MNIST,
        'dir': 'MNIST'
    },
    'fashion_mnist': {
        'class': datasets.FashionMNIST,
        'dir': 'FashionMNIST'
    }
}

def load_dataset(dataset_name='mnist', batch_size=64, data_path='../data_and_models/data'):
    """
    Load and preprocess a specified dataset (MNIST or FashionMNIST) using PyTorch.

    This function dynamically selects the dataset based on the 'dataset_name'
    parameter. It correctly checks if the data already exists in the specified
    path ('./data/MNIST' or './data/FashionMNIST') to avoid re-downloading.

    Args:
        dataset_name (str): The name of the dataset to load.
                            Accepts 'mnist' or 'fashion_mnist'.
        batch_size (int): The number of samples per batch in the DataLoader.
        data_path (str): The root directory where the dataset is or will be stored.

    Returns:
        A tuple containing the training and testing torch.utils.data.DataLoader objects.
    
    Raises:
        ValueError: If an unsupported dataset_name is provided.
    """
    dataset_name = dataset_name.lower()
    if dataset_name not in DATASET_CONFIG:
        raise ValueError(f"Unknown dataset '{dataset_name}'. Please use 'mnist' or 'fashion_mnist'.")

    # Select the correct dataset class and directory name from the config
    config = DATASET_CONFIG[dataset_name]
    dataset_class = config['class']
    dataset_dir = config['dir']
    
    # Define the transformation pipeline
    transform = transforms.ToTensor()  # Normalize to [0, 1]
    
    # Check if the dataset exists by verifying the raw data files
    # This prevents re-downloading if data is already present
    dataset_folder_path = os.path.join(data_path, dataset_dir, 'raw')
    train_exists = os.path.exists(os.path.join(dataset_folder_path, 'train-images-idx3-ubyte'))
    test_exists = os.path.exists(os.path.join(dataset_folder_path, 't10k-images-idx3-ubyte'))

    download_flag = not (train_exists and test_exists)

    # Load training data
    train_dataset = dataset_class(
        root=data_path, 
        train=True, 
        transform=transform, 
        download=download_flag
    )

    # Load test data
    test_dataset = dataset_class(
        root=data_path, 
        train=False, 
        transform=transform, 
        download=download_flag
    )
    
    # Create data loaders
    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True
    )

    test_loader = torch.utils.data.DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False
    )
    return train_loader, test_loader

def load_random_test_samples(dataset_name='mnist', num_samples=16, data_path='./data'):
    """
    Load a random subset of test samples from the specified dataset.
    
    Args:
        dataset_name (str): The name of the dataset to load ('mnist' or 'fashion_mnist').
        num_samples (int): Number of random samples to return.
        data_path (str): The root directory where the dataset is stored.
    
    Returns:
        tuple: (images_tensor, labels_tensor) where:
            - images_tensor: Tensor object with flattened image data (num_samples, 784)
            - labels_tensor: Tensor object with one-hot encoded labels (num_samples, 10)
    """
    # Load the full test dataset
    _, test_loader = load_dataset(dataset_name=dataset_name, batch_size=128, data_path=data_path)
    print("???")
    # Get all test data
    all_images, all_labels = next(iter(test_loader))
    
    # Randomly select indices
    total_samples = all_images.shape[0]
    if num_samples > total_samples:
        print(f"Warning: Requested {num_samples} samples, but only {total_samples} available. Using all samples.")
        num_samples = total_samples
    
    random_indices = torch.randperm(total_samples)[:num_samples]
    
    # Select random samples
    selected_images = all_images[random_indices]
    selected_labels = all_labels[random_indices]
    
    # Reshape images to (batch_size, 784) and convert labels to one-hot
    selected_images = selected_images.view(selected_images.shape[0], -1)
    labels_one_hot = torch.zeros(selected_labels.size(0), 10)
    labels_one_hot.scatter_(1, selected_labels.unsqueeze(1), 1)
    
    # Convert to Tensor objects
    return Tensor(selected_images.numpy(), requires_grad=False), Tensor(labels_one_hot.numpy(), requires_grad=False)


class MNISTBatchGenerator:
    def __init__(self, data_loader):
        self.data_loader = data_loader
        self.iterator = iter(data_loader)

    def get_next_batch(self):
        """
        Get next batch and convert it directly into Tensor objects.
        """
        try:
            images, labels = next(self.iterator)
        except StopIteration:
            self.iterator = iter(self.data_loader)
            images, labels = next(self.iterator)

        # Reshape images to (batch_size, 784)
        images = images.view(images.shape[0], -1)  # in PyTorch Tensor format

        # Convert labels to one-hot encoding
        labels_one_hot = torch.zeros(labels.size(0), 10)
        labels_one_hot.scatter_(1, labels.unsqueeze(1), 1)

        return Tensor(images.numpy(), requires_grad=False), Tensor(labels_one_hot.numpy(), requires_grad=False)


class EarlyStopping:
    def __init__(self, patience=5, min_delta=0.001):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.best_loss = None
        self.should_stop = False

    def __call__(self, val_loss):
        if self.best_loss is None:
            self.best_loss = val_loss
        elif val_loss > self.best_loss - self.min_delta:
            self.counter += 1
            if self.counter >= self.patience:
                self.should_stop = True
        else:
            self.best_loss = val_loss
            self.counter = 0