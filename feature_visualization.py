import os
import numpy as np
import torch
import matplotlib.pyplot as plt
from PIL import Image
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, Subset

# Import your project modules
from data_setup import setup_data
from model_improved_v2 import improved_resnet_cifar, BasicBlock, BottleneckBlock

# Create output directory
OUTPUT_DIR = './visualization_results'
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Create subdirectories
FEATURE_MAPS_DIR = os.path.join(OUTPUT_DIR, 'feature_maps')
ACTIVATION_DIR = os.path.join(OUTPUT_DIR, 'activation_patterns')
FILTERS_DIR = os.path.join(OUTPUT_DIR, 'filters')

os.makedirs(FEATURE_MAPS_DIR, exist_ok=True)
os.makedirs(ACTIVATION_DIR, exist_ok=True)
os.makedirs(FILTERS_DIR, exist_ok=True)

class FeatureExtractor:
    """Class to extract and visualize features from intermediate layers"""
    def __init__(self, model):
        self.model = model
        self.model.eval()
        self.features = {}
        self.hooks = []
        
    def register_hooks(self):
        """Register forward hooks on the model layers"""
        # Clear existing hooks
        self.remove_hooks()
        
        # Define hook function
        def hook_fn(name):
            def hook(module, input, output):
                self.features[name] = output.detach()
            return hook
        
        # Register hooks for layers of interest
        # Conv1
        self.hooks.append(self.model.conv1.register_forward_hook(hook_fn('conv1')))
        
        # Layer1 (first residual block)
        self.hooks.append(self.model.layer1[0].conv1.register_forward_hook(hook_fn('layer1.0.conv1')))
        self.hooks.append(self.model.layer1[0].conv2.register_forward_hook(hook_fn('layer1.0.conv2')))
        
        # Layer2 (downsampling block)
        self.hooks.append(self.model.layer2[0].conv1.register_forward_hook(hook_fn('layer2.0.conv1')))
        self.hooks.append(self.model.layer2[0].conv2.register_forward_hook(hook_fn('layer2.0.conv2')))
        
        # Layer3 (final block)
        self.hooks.append(self.model.layer3[0].conv1.register_forward_hook(hook_fn('layer3.0.conv1')))
        self.hooks.append(self.model.layer3[0].conv2.register_forward_hook(hook_fn('layer3.0.conv2')))
        
    def remove_hooks(self):
        """Remove all hooks"""
        for hook in self.hooks:
            hook.remove()
        self.hooks = []
        
    def get_features(self, input_tensor):
        """Forward pass to get features"""
        self.features = {}
        _ = self.model(input_tensor)
        return self.features
    
    def visualize_features(self, features, layer_name, num_filters=16, figsize=(15, 8)):
        """Visualize feature maps from a specific layer"""
        # Get feature maps
        feature_maps = features[layer_name]
        
        # Use the first image in the batch
        feature_maps = feature_maps[0]
        
        # Limit the number of filters to display
        num_filters = min(num_filters, feature_maps.shape[0])
        
        # Create figure
        fig, axes = plt.subplots(nrows=4, ncols=num_filters//4, figsize=figsize)
        
        # Flatten axes for easier indexing
        if num_filters > 4:
            axes = axes.flatten()
        
        # Plot feature maps
        for i in range(num_filters):
            if num_filters <= 4:
                ax = axes[i]
            else:
                ax = axes[i]
            
            # Get feature map
            feature_map = feature_maps[i].cpu().numpy()
            
            # Plot
            im = ax.imshow(feature_map, cmap='viridis')
            ax.set_title(f'Filter {i}')
            ax.axis('off')
        
        # Add colorbar
        plt.colorbar(im, ax=axes.tolist(), shrink=0.8)
        
        # Set title
        plt.suptitle(f'Feature Maps - {layer_name}', fontsize=16)
        plt.tight_layout()
        plt.savefig(os.path.join(FEATURE_MAPS_DIR, f'{layer_name.replace(".", "_")}_feature_maps.png'))
        plt.close()

def visualize_model_architecture(model):
    """Create a text representation of the model architecture"""
    # Print model summary to console
    print(model)
    
    # Save model architecture to file
    with open(os.path.join(OUTPUT_DIR, 'model_architecture.txt'), 'w') as f:
        f.write(str(model))
    
    # Try to create a more visual representation
    try:
        from torchviz import make_dot
        
        # Create a dummy input
        device = next(model.parameters()).device
        dummy_input = torch.randn(1, 3, 32, 32).to(device)
        
        # Make dot graph
        dot = make_dot(model(dummy_input), params=dict(model.named_parameters()))
        
        # Save graph
        dot.format = 'png'
        dot.render(os.path.join(OUTPUT_DIR, 'model_architecture'))
        print(f"Model architecture visualization created as '{os.path.join(OUTPUT_DIR, 'model_architecture.png')}'")
    except ImportError:
        print("torchviz not available. Install it with: pip install torchviz")
        print("Only text representation of model architecture created.")

def visualize_sample_images(val_loader, class_names, num_samples=10):
    """Visualize sample images from the validation set"""
    # Get a batch
    images, labels = next(iter(val_loader))
    
    # Limit to num_samples
    images = images[:num_samples]
    labels = labels[:num_samples]
    
    # De-normalize the images for display
    mean = torch.tensor([0.4914, 0.4822, 0.4465]).view(3, 1, 1)
    std = torch.tensor([0.2023, 0.1994, 0.2010]).view(3, 1, 1)
    
    # Create figure
    fig, axes = plt.subplots(1, num_samples, figsize=(3*num_samples, 3))
    
    for i, (img, label) in enumerate(zip(images, labels)):
        # De-normalize
        img = img.clone()
        img = img * std + mean
        img = img.permute(1, 2, 0).numpy()
        img = np.clip(img, 0, 1)
        
        # Plot
        axes[i].imshow(img)
        axes[i].set_title(f'{class_names[label]}')
        axes[i].axis('off')
    
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, 'sample_validation_images.png'))
    plt.close()

def visualize_activation_patterns(model, val_loader, class_names, device):
    """Visualize activation patterns for different classes"""
    # Get sample images from each class
    class_images = {}
    
    with torch.no_grad():
        for images, labels in val_loader:
            for img, label in zip(images, labels):
                label_idx = label.item()
                if label_idx not in class_images:
                    class_images[label_idx] = img
                
                if len(class_images) == len(class_names):
                    break
            
            if len(class_images) == len(class_names):
                break
    
    # Create feature extractor
    extractor = FeatureExtractor(model)
    extractor.register_hooks()
    
    # For each class, visualize activations in key layers
    layers_to_visualize = ['conv1', 'layer1.0.conv2', 'layer2.0.conv2', 'layer3.0.conv2']
    
    for class_idx, img in class_images.items():
        class_name = class_names[class_idx]
        
        # Create a directory for this class
        class_dir = os.path.join(ACTIVATION_DIR, class_name)
        os.makedirs(class_dir, exist_ok=True)
        
        # Forward pass to get features
        img_batch = img.unsqueeze(0).to(device)
        features = extractor.get_features(img_batch)
        
        # De-normalize for display
        mean = torch.tensor([0.4914, 0.4822, 0.4465]).view(3, 1, 1)
        std = torch.tensor([0.2023, 0.1994, 0.2010]).view(3, 1, 1)
        img_display = img.clone() * std + mean
        img_display = img_display.permute(1, 2, 0).numpy()
        img_display = np.clip(img_display, 0, 1)
        
        # Save the original image
        plt.figure(figsize=(4, 4))
        plt.imshow(img_display)
        plt.title(f'Class: {class_name}')
        plt.axis('off')
        plt.savefig(os.path.join(class_dir, 'original.png'))
        plt.close()
        
        # Visualize features for each layer
        for layer in layers_to_visualize:
            extractor.visualize_features(features, layer, num_filters=16, figsize=(15, 8))
            
            # Copy the file to the class directory
            source_file = os.path.join(FEATURE_MAPS_DIR, f'{layer.replace(".", "_")}_feature_maps.png')
            target_file = os.path.join(class_dir, f'{layer.replace(".", "_")}_feature_maps.png')
            
            # Make a copy of the file instead of moving it
            plt.figure(figsize=(15, 8))
            plt.imshow(plt.imread(source_file))
            plt.axis('off')
            plt.savefig(target_file)
            plt.close()
    
    # Clean up
    extractor.remove_hooks()

def visualize_filters(model):
    """Visualize the filters (weights) in the convolutional layers"""
    # Dictionary of layers to visualize
    layers = {
        'First Conv Layer': model.conv1.weight.data,
        'Layer1 First Block': model.layer1[0].conv1.weight.data,
        'Layer2 First Block': model.layer2[0].conv1.weight.data,
        'Layer3 First Block': model.layer3[0].conv1.weight.data
    }
    
    # Visualize each layer
    for name, weights in layers.items():
        # Get weights
        weights = weights.cpu().clone()
        
        # Determine number of filters to display
        num_filters = min(16, weights.shape[0])
        
        # Create figure
        plt.figure(figsize=(12, 6))
        
        for i in range(num_filters):
            # Get a single filter
            if name == 'First Conv Layer':
                # For the first layer, we can visualize the RGB channels
                plt.subplot(2, num_filters//2, i+1)
                
                # Normalize for better visualization
                filt = weights[i]
                min_val = filt.min()
                max_val = filt.max()
                filt = (filt - min_val) / (max_val - min_val)
                
                # Reorder dimensions for display
                filt = filt.permute(1, 2, 0).numpy()
                
                plt.imshow(filt)
                plt.title(f'Filter {i}')
                plt.axis('off')
            else:
                # For deeper layers, take the mean across input channels for visualization
                plt.subplot(2, num_filters//2, i+1)
                plt.imshow(torch.mean(weights[i], dim=0).numpy(), cmap='viridis')
                plt.title(f'Filter {i}')
                plt.axis('off')
        
        plt.suptitle(f'Filters in {name}', fontsize=16)
        plt.tight_layout()
        plt.savefig(os.path.join(FILTERS_DIR, f'{name.replace(" ", "_").lower()}_filters.png'))
        plt.close()

def plot_model_size_params(model):
    """Plot model size and parameter distribution by layer"""
    # Get parameter count by layer
    layer_params = {}
    
    for name, param in model.named_parameters():
        # Get base layer name
        base_name = name.split('.')[0]
        
        # Initialize if not exists
        if base_name not in layer_params:
            layer_params[base_name] = 0
        
        # Add parameter count
        layer_params[base_name] += param.numel()
    
    # Create bar chart
    plt.figure(figsize=(10, 6))
    bars = plt.bar(layer_params.keys(), layer_params.values())
    
    # Add parameter count labels
    for bar in bars:
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height + 1,
                f'{height:,}', ha='center', va='bottom', rotation=45)
    
    plt.xlabel('Layer')
    plt.ylabel('Number of Parameters')
    plt.title('Parameter Distribution by Layer')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, 'parameter_distribution.png'))
    plt.close()
    
    # Print total parameters
    total_params = sum(layer_params.values())
    print(f"Total parameters: {total_params:,}")
    
    # Save to file
    with open(os.path.join(OUTPUT_DIR, 'model_parameters.txt'), 'w') as f:
        f.write(f"Total parameters: {total_params:,}\n\n")
        f.write("Parameter distribution by layer:\n")
        for layer, params in layer_params.items():
            f.write(f"{layer}: {params:,} ({params/total_params*100:.2f}%)\n")

def main():
    # Configuration
    data_dir = 'deep-learning-spring-2025-project-1/cifar-10-python/cifar-10-batches-py'  
    model_path = 'models/best_model.pth'  
    model_size = 'medium'
    batch_size = 128  
    activation = 'relu'
    dropout = True
    
    print(f"Output directory: {os.path.abspath(OUTPUT_DIR)}")
    
    # Setup data
    train_loader, val_loader, _, class_names = setup_data(data_dir, batch_size=batch_size)
    
    # Load model
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = improved_resnet_cifar(num_classes=len(class_names), model_size=model_size, 
                                 activation=activation, dropout=dropout)
    checkpoint = torch.load(model_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model = model.to(device)
    model.eval()
    
    # Visualize model architecture
    print("Visualizing model architecture...")
    visualize_model_architecture(model)
    
    # Plot model size and parameters
    print("Plotting model parameters...")
    plot_model_size_params(model)
    
    # Visualize sample images
    print("Visualizing sample images...")
    visualize_sample_images(val_loader, class_names)
    
    # Visualize filters
    print("Visualizing filters...")
    visualize_filters(model)
    
    # Visualize activation patterns for different classes
    print("Visualizing activation patterns for each class...")
    visualize_activation_patterns(model, val_loader, class_names, device)
    
    print(f"Feature visualization complete! Check the directory {OUTPUT_DIR} for results.")

if __name__ == "__main__":
    main()