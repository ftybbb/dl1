import os
import time
import argparse
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR, OneCycleLR, StepLR
from torchsummary import summary
import torchvision
from data_setup import unpickle
import matplotlib.pyplot as plt


from data_setup import setup_data, setup_data_test, setup_data_test_cifar
from model import resnet_cifar
from model_improved import improved_resnet_cifar

def train_epoch(model, train_loader, criterion, optimizer, device):
    """Train for one epoch"""
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0
    
    for batch_idx, (inputs, targets) in enumerate(train_loader):
        inputs, targets = inputs.to(device), targets.to(device)
        
        # Zero the parameter gradients
        optimizer.zero_grad()
        
        # Forward pass
        outputs = model(inputs)
        loss = criterion(outputs, targets)
        
        # Backward pass and optimize
        loss.backward()
        optimizer.step()
        
        # Statistics
        running_loss += loss.item()
        _, predicted = outputs.max(1)
        total += targets.size(0)
        correct += predicted.eq(targets).sum().item()
        
        # Print progress every 50 batches
        if (batch_idx + 1) % 50 == 0 or (batch_idx + 1) == len(train_loader):
            print(f'Train Batch: {batch_idx+1}/{len(train_loader)} | '
                  f'Loss: {running_loss/(batch_idx+1):.3f} | '
                  f'Acc: {100.*correct/total:.2f}% ({correct}/{total})')
    
    return running_loss / len(train_loader), correct / total


def validate(model, val_loader, criterion, device):
    """Validate the model"""
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0
    
    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(val_loader):
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            
            running_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()
    
    print(f'Validation | Loss: {running_loss/len(val_loader):.3f} | '
          f'Acc: {100.*correct/total:.2f}% ({correct}/{total})')
    
    return running_loss / len(val_loader), correct / total


def generate_predictions(model, test_loader, device):
    """Generate predictions for test data"""
    model.eval()
    predictions = []
    ids = []
    
    with torch.no_grad():
        for inputs, batch_ids in test_loader:
            inputs = inputs.to(device)
            outputs = model(inputs)
            _, predicted = outputs.max(1)
            
            # Convert predictions to numpy and append to list
            batch_preds = predicted.cpu().numpy()
            batch_ids = batch_ids.cpu().numpy()
            
            predictions.extend(batch_preds)
            ids.extend(batch_ids)
    
    return np.array(ids), np.array(predictions)


def main():
    parser = argparse.ArgumentParser(description='Modified ResNet CIFAR-10 Training')
    parser.add_argument('--data-dir', default='/scratch/tf2387/deep-learning-spring-2025-project-1/cifar-10-batches-py',
                        help='path to CIFAR-10 data')
    parser.add_argument('--output-dir', default='/home/tf2387/7123pj_zzp/outputs',
                        help='path to save outputs')
    parser.add_argument('--batch-size', type=int, default=128, help='batch size')
    parser.add_argument('--epochs', type=int, default=150, help='number of epochs')
    parser.add_argument('--lr', type=float, default=0.1, help='learning rate')
    parser.add_argument('--model-size', default='medium', choices=['small', 'medium', 'large'],
                        help='model size')
    parser.add_argument('--optimizer', default='sgd', choices=['sgd', 'adam'],
                        help='optimizer')
    parser.add_argument('--scheduler', default='cosine', choices=['cosine', 'onecycle', 'step'],
                        help='learning rate scheduler')
    parser.add_argument('--weight-decay', type=float, default=5e-4, help='weight decay')
    parser.add_argument('--device', default='cuda', help='device to use')
    parser.add_argument('--task', default='train', choices=['train', 'test'],)
    parser.add_argument('--activation', default='mish', choices=['relu', 'mish', 'gelu'],)
    parser.add_argument('--dropout', action='store_true', help='apply dropout')
    parser.add_argument('--test-data', default='', help='test data file')

    
    args = parser.parse_args()
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Set device
    device = torch.device(args.device if torch.cuda.is_available() and args.device == 'cuda' else 'cpu')
    print(f"Using device: {device}")
    
    # Setup data loaders
    print("Setting up data...")
    train_loader, val_loader, test_loader, classes = setup_data(
        args.data_dir, batch_size=args.batch_size
    )
    print(f"Classes: {classes}")
    
    # Create model
    print(f"Creating {args.model_size} model...")
    model = improved_resnet_cifar(num_classes=len(classes), model_size=args.model_size, activation=args.activation, dropout=args.dropout)
    model = model.to(device)
    
    # Print model summary
    print(summary(model, (3, 32, 32)))
    
    # Loss function and optimizer
    criterion = nn.CrossEntropyLoss()
    
    if args.optimizer == 'sgd':
        optimizer = optim.SGD(model.parameters(), lr=args.lr, 
                              momentum=0.9, weight_decay=args.weight_decay)
    else:
        optimizer = optim.Adam(model.parameters(), lr=args.lr,
                               weight_decay=args.weight_decay)
    
    # Learning rate scheduler
    if args.scheduler == 'cosine':
        scheduler = CosineAnnealingLR(optimizer, T_max=args.epochs)
    elif args.scheduler == 'onecycle':
        scheduler = OneCycleLR(optimizer, max_lr=args.lr, 
                               epochs=args.epochs, steps_per_epoch=len(train_loader))
    elif args.scheduler == 'step':
        scheduler = StepLR(optimizer, step_size=30, gamma=0.1)
    else:
        scheduler = None
    
    # Training loop
    best_acc = 0.0
    best_epoch = 0
    train_losses, train_accs = [], []
    val_losses, val_accs = [], []
    
    print(f"Starting training for {args.epochs} epochs...")
    start_time = time.time()
    
    if args.task == 'train':
        for epoch in range(args.epochs):
            print(f"\nEpoch {epoch+1}/{args.epochs}")
            
            # Train
            train_loss, train_acc = train_epoch(model, train_loader, criterion, optimizer, device)
            train_losses.append(train_loss)
            train_accs.append(train_acc)
            
            # Validate
            val_loss, val_acc = validate(model, val_loader, criterion, device)
            val_losses.append(val_loss)
            val_accs.append(val_acc)
            
            # Update learning rate
            if scheduler is not None:
                if args.scheduler == 'cosine':
                    scheduler.step()
                # For OneCycleLR, step is called in the training loop
            
            # Save best model
            if val_acc > best_acc:
                print(f"New best model with accuracy: {val_acc:.4f}")
                best_acc = val_acc
                best_epoch = epoch
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'accuracy': val_acc,
                }, os.path.join(args.output_dir, 'best_model.pth'))
        
        training_time = time.time() - start_time
        print(f"\nTraining completed in {training_time:.2f} seconds")
        print(f"Best validation accuracy: {best_acc:.4f} at epoch {best_epoch+1}")
    
    # Load best model for testing
    checkpoint = torch.load(os.path.join(args.output_dir, 'best_model.pth'))
    model.load_state_dict(checkpoint['model_state_dict'])

    if args.task == 'test':
        model.eval()
        test_loader, test_labels = setup_data_test_cifar(args.test_data, batch_size=args.batch_size)
        ids, predictions = generate_predictions(model, test_loader, device)
        d = unpickle(args.test_data)
        # print(d.keys())
        # dd = d[b'data']
        # img = dd[5]
        # print(img.shape)
        # # Save the first image and its prediction

        # img_path = os.path.join(args.output_dir, 'sample_image.png')
        # plt.imsave(img_path, img)
        # print(f"Sample image saved to {img_path}")
        # print(f"Prediction for the sample image: {predictions[5]}")
        # Write predictions to CSV
        output_file = os.path.join(args.output_dir, 'test_predictions.csv')
        df = pd.DataFrame({'ID': ids, 'Labels': predictions})
        df.to_csv(output_file, index=False)
        print(f"Test predictions saved to {output_file}")
        print(ids, predictions)
        predictions = np.array(predictions)
        test_labels = np.array(test_labels)
        print((predictions-test_labels)/test_labels.shape[0])
        assert 1==2
    
    # Generate predictions on test set
    if test_loader:
        print("Generating predictions on test set...")
        ids, predictions = generate_predictions(model, test_loader, device)
        
        # Create submission file
        submission = pd.DataFrame({
            'ID': ids,
            'Label': predictions
        })
        submission.to_csv(os.path.join(args.output_dir, 'submission.csv'), index=False)
        print(f"Submission file saved to {os.path.join(args.output_dir, 'submission.csv')}")
    
    # Save training history
    history = pd.DataFrame({
        'epoch': range(1, args.epochs + 1),
        'train_loss': train_losses,
        'train_acc': train_accs,
        'val_loss': val_losses,
        'val_acc': val_accs
    })
    history.to_csv(os.path.join(args.output_dir, 'training_history.csv'), index=False)
    print(f"Training history saved to {os.path.join(args.output_dir, 'training_history.csv')}")


if __name__ == '__main__':
    main()