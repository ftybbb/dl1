import os
import numpy as np
import pandas as pd
import torch
import math
import random
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report
from torchvision import transforms
from torch.utils.data import DataLoader, TensorDataset

from data_setup import setup_data, get_transforms
from model_improved_v2 import improved_resnet_cifar

OUTPUT_DIR = './visualization_results'
os.makedirs(OUTPUT_DIR, exist_ok=True)

def load_model(model_path, model_size='medium', num_classes=10, activation='relu', dropout=True):
    """Load the trained model"""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = improved_resnet_cifar(num_classes=num_classes, model_size=model_size, 
                                 activation=activation, dropout=dropout)
    checkpoint = torch.load(model_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model = model.to(device)
    model.eval()
    return model, device

def get_predictions(model, val_loader, device):
    """Get model predictions and true labels from validation data"""
    all_preds = []
    all_targets = []
    all_probs = []
    
    with torch.no_grad():
        for inputs, targets in val_loader:
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs)
            probs = torch.nn.functional.softmax(outputs, dim=1)
            _, preds = torch.max(outputs, 1)
            
            all_preds.extend(preds.cpu().numpy())
            all_targets.extend(targets.cpu().numpy())
            all_probs.extend(probs.cpu().numpy())
    
    return np.array(all_preds), np.array(all_targets), np.array(all_probs)

def plot_confusion_matrix(y_true, y_pred, class_names):
    """Plot and save confusion matrix"""
    plt.figure(figsize=(10, 8))
    cm = confusion_matrix(y_true, y_pred)
    
    # Normalize the confusion matrix
    cm_norm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    
    # Create heatmap
    sns.heatmap(cm_norm, annot=True, cmap='Blues', fmt='.2f', 
                xticklabels=class_names, yticklabels=class_names)
    
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title('Confusion Matrix')
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, 'confusion_matrix.png'))
    plt.close()
    
    # Also save raw confusion matrix counts
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, cmap='Blues', fmt='d', 
                xticklabels=class_names, yticklabels=class_names)
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title('Confusion Matrix (Counts)')
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, 'confusion_matrix_counts.png'))
    plt.close()
    
    return cm, cm_norm

def plot_class_accuracy(cm_norm, class_names):
    """Plot and save class-wise accuracy"""
    accuracies = np.diag(cm_norm)
    
    plt.figure(figsize=(12, 6))
    bars = plt.bar(class_names, accuracies * 100)
    
    # Add percentage labels on top of bars
    for bar in bars:
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height + 1,
                f'{height:.1f}%', ha='center', va='bottom')
    
    plt.xlabel('Class')
    plt.ylabel('Accuracy (%)')
    plt.title('Class-wise Prediction Accuracy')
    plt.ylim(0, 100)
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, 'class_accuracy.png'))
    plt.close()

def plot_misclassification_examples(model, val_loader, dataset, device, class_names, max_examples=100):
    """Plot all misclassified images in a single figure using original images"""
    misclassified_indices = []
    misclassified_labels = []
    misclassified_preds = []
    
    # Track the indices of images in the validation dataset
    current_idx = 0
    
    with torch.no_grad():
        for images, labels in val_loader:
            batch_size = images.size(0)
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, preds = torch.max(outputs, 1)
            
            # Find misclassified examples in this batch
            incorrect_mask = preds != labels
            
            if incorrect_mask.sum() > 0:
                # Get the actual dataset indices for misclassified examples
                batch_indices = torch.arange(current_idx, current_idx + batch_size)
                misclassified_batch_indices = batch_indices[incorrect_mask.cpu()]
                
                misclassified_indices.extend(misclassified_batch_indices.tolist())
                misclassified_labels.append(labels[incorrect_mask].cpu())
                misclassified_preds.append(preds[incorrect_mask].cpu())
                
            current_idx += batch_size
            
            # Break once we have enough examples
            if len(misclassified_indices) >= max_examples:
                break
                
    # Concatenate the label and prediction tensors
    if misclassified_labels:
        misclassified_labels = torch.cat(misclassified_labels)
        misclassified_preds = torch.cat(misclassified_preds)
        
        # Limit to max_examples
        n = min(max_examples, len(misclassified_indices))
        misclassified_indices = misclassified_indices[:n]
        misclassified_labels = misclassified_labels[:n]
        misclassified_preds = misclassified_preds[:n]
        
        # Set up the grid dimensions
        grid_size = math.ceil(math.sqrt(n))
        fig, axes = plt.subplots(grid_size, grid_size, figsize=(grid_size*3, grid_size*3))
        
        # Flatten axes for easier indexing if more than one row/column
        if grid_size > 1:
            axes = axes.flatten()
            
        for i in range(n):
            ax = axes[i] if grid_size > 1 else axes
            
            # Get the original image from the dataset
            original_img, _ = dataset[misclassified_indices[i]]
            
            # Convert PIL image to numpy array if needed
            if hasattr(original_img, 'permute'):  # If it's already a tensor
                img = original_img.permute(1, 2, 0).numpy()  # CHW -> HWC
                # If normalized, denormalize it
                if img.max() <= 1.0:  # Check if already in [0,1] range
                    pass  # Already in good range
                else:
                    img = img / 255.0  # Normalize to [0,1]
            else:  # If it's a PIL image
                img = np.array(original_img) / 255.0  # Convert to numpy and normalize
            
            true_label = misclassified_labels[i].item()
            pred_label = misclassified_preds[i].item()
            
            ax.imshow(img)
            ax.set_title(f'True: {class_names[true_label]}\nPred: {class_names[pred_label]}')
            ax.axis('off')
            
        # Turn off axes for unused subplots
        for i in range(n, grid_size * grid_size):
            if grid_size > 1:
                axes[i].axis('off')
                
        plt.tight_layout()
        plt.savefig(os.path.join(OUTPUT_DIR, 'all_original_misclassified_examples.png'))
        plt.close()
        
        # Create a report about these misclassifications
        misclass_counts = {}
        for true_label, pred_label in zip(misclassified_labels, misclassified_preds):
            true_class = class_names[true_label]
            pred_class = class_names[pred_label]
            pair = (true_class, pred_class)
            if pair not in misclass_counts:
                misclass_counts[pair] = 0
            misclass_counts[pair] += 1
            
        # Sort by frequency
        sorted_misclass = sorted(misclass_counts.items(), key=lambda x: x[1], reverse=True)
        
        # Write report
        with open(os.path.join(OUTPUT_DIR, 'misclassification_report.txt'), 'w') as f:
            f.write("Most Common Misclassifications:\n\n")
            for (true_class, pred_class), count in sorted_misclass:
                f.write(f"True: {true_class}, Predicted: {pred_class} - {count} instances\n")

def plot_confidence_distribution(all_probs, all_preds, all_targets, class_names):
    """Plot confidence distribution for correct and incorrect predictions"""
    # Get the confidence (probability) for the predicted class
    confidences = np.array([all_probs[i, pred] for i, pred in enumerate(all_preds)])
    
    # Separate confidences for correct and incorrect predictions
    correct_mask = all_preds == all_targets
    correct_conf = confidences[correct_mask]
    incorrect_conf = confidences[~correct_mask]
    
    plt.figure(figsize=(10, 6))
    
    # Plot histograms
    plt.hist(correct_conf, bins=20, alpha=0.5, label='Correct Predictions', density=True)
    plt.hist(incorrect_conf, bins=20, alpha=0.5, label='Incorrect Predictions', density=True)
    
    plt.xlabel('Confidence (Probability)')
    plt.ylabel('Density')
    plt.title('Confidence Distribution for Correct vs Incorrect Predictions')
    plt.legend()
    plt.grid(alpha=0.3)
    plt.savefig(os.path.join(OUTPUT_DIR, 'confidence_distribution.png'))
    plt.close()
    
    # Also plot class-wise confidence
    plt.figure(figsize=(12, 8))
    
    for i, class_name in enumerate(class_names):
        class_mask = all_targets == i
        class_correct_mask = (all_targets == i) & (all_preds == i)
        class_incorrect_mask = (all_targets == i) & (all_preds != i)
        
        if np.sum(class_correct_mask) > 0:
            correct_conf_class = confidences[class_correct_mask]
            sns.kdeplot(correct_conf_class, label=f'{class_name} (Correct)', alpha=0.7)
        
        if np.sum(class_incorrect_mask) > 0:
            incorrect_conf_class = confidences[class_incorrect_mask]
            sns.kdeplot(incorrect_conf_class, label=f'{class_name} (Incorrect)', alpha=0.7, linestyle='--')
    
    plt.xlabel('Confidence (Probability)')
    plt.ylabel('Density')
    plt.title('Class-wise Confidence Distribution')
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, 'class_wise_confidence.png'))
    plt.close()
    
def plot_augmentation_effects(train_dataset, num_examples=5, num_augmentations=4):
    """
    Visualize the effect of data augmentation on training images.
    """
    # Get the train transform
    train_transform, _ = get_transforms()
    
    # Set the random seed for reproducibility
    random.seed(42)
    torch.manual_seed(42)
    
    # Randomly select images from training dataset
    indices = random.sample(range(len(train_dataset)), num_examples)
    
    # Create figure
    fig, axes = plt.subplots(num_examples, num_augmentations + 1, figsize=(3 * (num_augmentations + 1), 3 * num_examples))
    
    # For each selected image
    for i, idx in enumerate(indices):
        # Get original image and label
        img, label = train_dataset[idx]
        
        # If img is a tensor, convert to numpy array for display
        if isinstance(img, torch.Tensor):
            # De-normalize if needed
            if img.max() <= 1.0:  # Normalized image
                mean = torch.tensor([0.4914, 0.4822, 0.4465]).view(3, 1, 1)
                std = torch.tensor([0.2023, 0.1994, 0.2010]).view(3, 1, 1)
                img_display = img * std + mean
            else:  # Not normalized
                img_display = img / 255.0
                
            # Convert to HWC format for matplotlib
            img_display = img_display.permute(1, 2, 0).numpy()
            img_display = np.clip(img_display, 0, 1)
        else:
            # Assuming it's already in the right format (could be a PIL image)
            img_display = np.array(img) / 255.0
        
        # Plot original image
        axes[i, 0].imshow(img_display)
        axes[i, 0].set_title(f"Original")
        axes[i, 0].axis('off')
        
        # Apply different augmentations to the same image
        for j in range(num_augmentations):
            # Reset seed for each augmentation to ensure variation
            random.seed(42 + j)
            torch.manual_seed(42 + j)
            
            # Apply transform to get augmented image
            augmented = train_transform(img.cpu() if isinstance(img, torch.Tensor) else img)
            
            # Convert to numpy array for display
            if isinstance(augmented, torch.Tensor):
                # De-normalize
                mean = torch.tensor([0.4914, 0.4822, 0.4465]).view(3, 1, 1)
                std = torch.tensor([0.2023, 0.1994, 0.2010]).view(3, 1, 1)
                aug_display = augmented * std + mean
                aug_display = aug_display.permute(1, 2, 0).numpy()
                aug_display = np.clip(aug_display, 0, 1)
            else:
                aug_display = np.array(augmented) / 255.0
            
            # Plot augmented image
            axes[i, j+1].imshow(aug_display)
            axes[i, j+1].set_title(f"Augmentation {j+1}")
            axes[i, j+1].axis('off')
    
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, 'augmentation_examples.png'))
    plt.close()
    print(f"Augmentation visualization saved to {os.path.join(OUTPUT_DIR, 'augmentation_examples.png')}")

def main():
    # Configuration
    data_dir = 'deep-learning-spring-2025-project-1/cifar-10-python/cifar-10-batches-py' 
    model_path = 'models/best_model.pth'
    model_size = 'medium'
    batch_size = 128
    activation = 'relu'
    dropout = True
    
    # Setup data
    train_loader, val_loader, _, class_names, train_dataset, val_dataset = setup_data(data_dir, batch_size=batch_size, return_dataset=True)
    
    # Load model
    model, device = load_model(model_path, model_size, len(class_names), activation, dropout)
    
    # Get predictions
    all_preds, all_targets, all_probs = get_predictions(model, val_loader, device)
    
    cm, cm_norm = plot_confusion_matrix(all_targets, all_preds, class_names)
    plot_class_accuracy(cm_norm, class_names)
    plot_misclassification_examples(model, val_loader, val_dataset, device, class_names)
    plot_confidence_distribution(all_probs, all_preds, all_targets, class_names)
    plot_augmentation_effects(train_dataset, 5, 4)
    
    report = classification_report(all_targets, all_preds, target_names=class_names)
    print("Classification Report:")
    print(report)
    
    with open(os.path.join(OUTPUT_DIR,'classification_report.txt'), 'w') as f:
        f.write(report)
    
    print("Visualization complete! Check the current directory for output images.")

if __name__ == "__main__":
    main()