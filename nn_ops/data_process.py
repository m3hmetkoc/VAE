import os
import torch
from torchvision import datasets, transforms
from .tensor_class import Tensor 

def load_mnist_data(batch_size=64, data_path='./data'):
    """
    Load and preprocess the MNIST dataset using PyTorch, ensuring that it loads from 
    a specified path if the dataset already exists, otherwise it downloads it.
    """
    # Define the transformation pipeline
    transform = transforms.ToTensor()  # Normalize to [0, 1]
    
    # Check if the dataset exists (by verifying both train and test folders)
    train_exists = os.path.exists(os.path.join(data_path, 'MNIST', 'raw', 'train-images-idx3-ubyte'))
    test_exists = os.path.exists(os.path.join(data_path, 'MNIST', 'raw', 't10k-images-idx3-ubyte'))

    # Load dataset without downloading if it already exists
    download_flag = not (train_exists and test_exists)

    # Load training data
    train_dataset = datasets.MNIST(
        root=data_path, 
        train=True, 
        transform=transform, 
        download=download_flag
    )

    # Load test data
    test_dataset = datasets.MNIST(
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

        # Convert tensors to Value objects
        images_value = Tensor(images.numpy(), requires_grad=False)
        labels_value = Tensor(labels_one_hot.numpy(), requires_grad=False)

        return images_value, labels_value


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
