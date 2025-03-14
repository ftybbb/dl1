import os
import pickle
import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset, random_split
import torchvision.transforms as transforms
from torchvision.transforms import AutoAugment, AutoAugmentPolicy

def unpickle(file):
    """Load the CIFAR-10 pickled data"""
    with open(file, 'rb') as fo:
        dict = pickle.load(fo, encoding='bytes')
    return dict

class RandomCutout:
    def __init__(self, min_size=16, max_size=32): 
        self.min_size = min_size
        self.max_size = max_size

    def __call__(self, img):
        h, w = img.shape[1:3]
        size = np.random.randint(self.min_size, self.max_size+1)
        
        mask = np.ones((h, w), np.float32)
        y = np.random.randint(h)
        x = np.random.randint(w)

        y1 = np.clip(y - size // 2, 0, h)
        y2 = np.clip(y + size // 2, 0, h)
        x1 = np.clip(x - size // 2, 0, w)
        x2 = np.clip(x + size // 2, 0, w)

        mask[y1:y2, x1:x2] = 0.
        mask = torch.from_numpy(mask).expand_as(img)
        img *= mask
        return img
    
def get_transforms():
    # Training data preprocessing with augmentation
    train_transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),  
        transforms.RandomRotation(10),
        transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1, hue=0.1),
        AutoAugment(AutoAugmentPolicy.CIFAR10),
        transforms.ToTensor(),
        RandomCutout(8, 16),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
    ])
    
    # Validation and test data preprocessing (no augmentation)
    test_transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
    ])
    
    return train_transform, test_transform
    
    
def setup_data(data_dir, batch_size=128, val_split=0.1, return_dataset=False):
    """
    Load and prepare CIFAR-10 data
    Args:
        data_dir: Directory where CIFAR-10 data is located
        batch_size: Batch size for dataloaders
        val_split: Proportion of training data to use for validation
    Returns:
        train_loader, val_loader, test_loader, classes
    """
    # Create data directory if it doesn't exist
    os.makedirs(data_dir, exist_ok=True)
    
    train_transform, test_transform = get_transforms()
    
    # Load training data
    x_train = []
    y_train = []
    
    for i in range(1, 6):
        batch_file = os.path.join(data_dir, f'data_batch_{i}')
        batch_data = unpickle(batch_file)
        
        batch_images = batch_data[b'data']
        batch_labels = batch_data[b'labels']
        
        # Reshape images to (channels, height, width)
        batch_images = batch_images.reshape(-1, 3, 32, 32)
        
        x_train.append(batch_images)
        y_train.extend(batch_labels)
    
    x_train = np.concatenate(x_train, axis=0)
    y_train = np.array(y_train)
    
    # Load meta data to get class names
    meta_file = os.path.join(data_dir, 'batches.meta')
    meta_data = unpickle(meta_file)
    classes = [label.decode('utf-8') for label in meta_data[b'label_names']]
    
    # Create PyTorch tensors
    x_train_tensor = torch.from_numpy(x_train).float()
    y_train_tensor = torch.from_numpy(y_train).long()
    
    # Create dataset
    train_dataset = TensorDataset(x_train_tensor, y_train_tensor)
    
    # Split into train and validation
    train_size = int((1 - val_split) * len(train_dataset))
    val_size = len(train_dataset) - train_size
    
    # Use random_split to create train and validation datasets
    train_dataset, val_dataset = random_split(
        train_dataset, [train_size, val_size],
        generator=torch.Generator().manual_seed(42)
    )
    
    # Create train and validation dataloaders with respective transforms
    train_loader = DataLoader(
        TransformDataset(train_dataset, train_transform),
        batch_size=batch_size, 
        shuffle=True, 
        num_workers=4, 
        pin_memory=True
    )
    
    val_loader = DataLoader(
        TransformDataset(val_dataset, test_transform),
        batch_size=batch_size, 
        shuffle=False, 
        num_workers=4, 
        pin_memory=True
    )
    
    # Process the custom test dataset
    custom_test_dir = os.path.join(os.path.dirname(data_dir), 'custom_test_data')
    if os.path.exists(custom_test_dir):
        # Load custom test data (modify as per the actual format)
        test_data = unpickle(os.path.join(custom_test_dir, 'test_data'))
        test_images = test_data[b'data'].reshape(-1, 3, 32, 32)
        test_ids = test_data[b'ids'] if b'ids' in test_data else np.arange(len(test_images))
        
        test_images_tensor = torch.from_numpy(test_images).float()
        test_ids_tensor = torch.from_numpy(np.array(test_ids))
        
        test_dataset = TensorDataset(test_images_tensor, test_ids_tensor)
        test_loader = DataLoader(
            TransformDataset(test_dataset, test_transform),
            batch_size=batch_size,
            shuffle=False,
            num_workers=4,
            pin_memory=True
        )
    else:
        print("Warning: Custom test directory not found. Creating dummy test loader.")
        test_loader = None
    
    if return_dataset:
        return train_loader, val_loader, test_loader, classes, train_dataset, val_dataset
    else:
        return train_loader, val_loader, test_loader, classes


def setup_data_test(data_dir, batch_size=128, val_split=0.1):
    """
    Load and prepare CIFAR-10 data
    Args:
        data_dir: Directory where CIFAR-10 data is located
        batch_size: Batch size for dataloaders
        val_split: Proportion of training data to use for validation
    Returns:
        train_loader, val_loader, test_loader, classes
    """
    # Process the custom test dataset
    custom_test_dir = data_dir
    if os.path.exists(custom_test_dir):
        # Load custom test data (modify as per the actual format)
        test_data = unpickle(custom_test_dir)
        print(test_data)
        test_images = test_data[b'data']
        test_images = test_images.transpose(0, 3, 1, 2)
        test_ids = test_data[b'ids'] if b'ids' in test_data else np.arange(len(test_images))
        try:
            test_labels = test_data[b'labels']
        except:
            test_labels = None
        
        test_images_tensor = torch.from_numpy(test_images).float()
        test_ids_tensor = torch.from_numpy(np.array(test_ids))
        
        test_dataset = TensorDataset(test_images_tensor, test_ids_tensor)
        test_transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
        ])
        test_loader = DataLoader(
            TransformDataset(test_dataset, test_transform),
            batch_size=batch_size,
            shuffle=False,
            num_workers=1,
            pin_memory=True
        )
    else:
        print("Warning: Custom test directory not found. Creating dummy test loader.")
        test_loader = None
    
    return test_loader, test_labels


def setup_data_test_cifar(data_dir, batch_size=128, val_split=0.1):
    """
    Load and prepare CIFAR-10 data
    Args:
        data_dir: Directory where CIFAR-10 data is located
        batch_size: Batch size for dataloaders
        val_split: Proportion of training data to use for validation
    Returns:
        train_loader, val_loader, test_loader, classes
    """
    # Process the custom test dataset
    custom_test_dir = data_dir
    if os.path.exists(custom_test_dir):
        # Load custom test data (modify as per the actual format)
        test_data = unpickle(data_dir)
        test_images = test_data[b'data'].reshape(-1, 3, 32, 32)
        test_ids = test_data[b'ids'] if b'ids' in test_data else np.arange(len(test_images))
        try:
            test_labels = test_data[b'labels']
        except:
            test_labels = None
        
        test_images_tensor = torch.from_numpy(test_images).float()
        test_ids_tensor = torch.from_numpy(np.array(test_ids))
        
        test_dataset = TensorDataset(test_images_tensor, test_ids_tensor)
        test_transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
        ])
        test_loader = DataLoader(
            TransformDataset(test_dataset, test_transform),
            batch_size=batch_size,
            shuffle=False,
            num_workers=1,
            pin_memory=True
        )
    else:
        print("Warning: Custom test directory not found. Creating dummy test loader.")
        test_loader = None
    
    return test_loader, test_labels


class TransformDataset:
    """Dataset wrapper to apply transformations"""
    def __init__(self, dataset, transform=None):
        self.dataset = dataset
        self.transform = transform
        
    def __len__(self):
        return len(self.dataset)
        
    def __getitem__(self, idx):
        data, target = self.dataset[idx]
        if self.transform:
            data = self.transform(data)
        return data, target