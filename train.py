import os
import time
import argparse
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR, OneCycleLR, StepLR, LambdaLR, SequentialLR, CosineAnnealingWarmRestarts
from torchsummary import summary
import torchvision
from data_setup import unpickle
import matplotlib.pyplot as plt
import json
from torch_ema import ExponentialMovingAverage


from data_setup import setup_data, setup_data_test, setup_data_test_cifar
from model import resnet_cifar
from model_improved_v2 import improved_resnet_cifar

# import wandb

def rand_bbox(size, lam):
    W = size[2]
    H = size[3]
    cut_rat = np.sqrt(1. - lam)
    cut_w = int(W * cut_rat)
    cut_h = int(H * cut_rat)

    cx = np.random.randint(W)
    cy = np.random.randint(H)

    bbx1 = np.clip(cx - cut_w // 2, 0, W)
    bby1 = np.clip(cy - cut_h // 2, 0, H)
    bbx2 = np.clip(cx + cut_w // 2, 0, W)
    bby2 = np.clip(cy + cut_h // 2, 0, H)

    return bbx1, bby1, bbx2, bby2


def cutmix_data(x, y, alpha=1.0):
    lam = np.random.beta(alpha, alpha)
    rand_index = torch.randperm(x.size()[0]).to(x.device)

    y_a = y
    y_b = y[rand_index]

    bbx1, bby1, bbx2, bby2 = rand_bbox(x.size(), lam)
    x[:, :, bbx1:bbx2, bby1:bby2] = x[rand_index, :, bbx1:bbx2, bby1:bby2]

    lam = 1 - ((bbx2 - bbx1) * (bby2 - bby1) / (x.size()[-1] * x.size()[-2]))

    return x, y_a, y_b, lam


def train_epoch(model, train_loader, criterion, optimizer, device, ema=None, cutmix_prob=0.5):
    """Train for one epoch"""
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0
    
    for batch_idx, (inputs, targets) in enumerate(train_loader):
        inputs, targets = inputs.to(device), targets.to(device)
        
        # Zero the parameter gradients
        optimizer.zero_grad()
        
        random_number = np.random.rand()

        if cutmix_prob > 0 and random_number < cutmix_prob:
            inputs, targets_a, targets_b, lam = cutmix_data(inputs, targets)
            outputs = model(inputs)
            loss = criterion(outputs, targets_a) * lam + criterion(outputs, targets_b) * (1 - lam)
        else:
            outputs = model(inputs)
            loss = criterion(outputs, targets)
        
        # # Forward pass
        # outputs = model(inputs)
        # loss = criterion(outputs, targets)
        
        # Backward pass and optimize
        loss.backward()
        optimizer.step()
        if ema is not None:
            ema.update()
        
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
    parser.add_argument('--data-dir', default='dataset/cifar-10-python/cifar-10-batches-py',
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
    parser.add_argument('--scheduler', default='cosine', choices=['cosine', 'onecycle', 'step', 'cosine-restart'],
                        help='learning rate scheduler')
    parser.add_argument('--weight-decay', type=float, default=5e-4, help='weight decay')
    parser.add_argument('--device', default='cuda', help='device to use')
    parser.add_argument('--task', default='train', choices=['train', 'test','submit'],)
    parser.add_argument('--activation', default='mish', type=str)
    parser.add_argument('--dropout', action='store_true', help='apply dropout')
    parser.add_argument('--test-data', default='dataset/cifar-10-python/cifar-10-batches-py/test_batch', help='test data file')
    parser.add_argument('--cutmix', action='store_true', help='apply cutmix')
    args = parser.parse_args()
    import time
    cur_time = time.strftime("%Y-%m-%d_%H-%M-%S", time.localtime())
    # Create output directory
    

    
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
    criterion = nn.CrossEntropyLoss(label_smoothing=0.1) # add label smoothing
    
    if args.optimizer == 'sgd':
        optimizer = optim.SGD(model.parameters(), lr=args.lr, 
                              momentum=0.9, weight_decay=args.weight_decay)
    else:
        optimizer = optim.Adam(model.parameters(), lr=args.lr,
                               weight_decay=args.weight_decay)
    
    warmup_epochs = 5
    # Learning rate scheduler
    if args.scheduler == 'cosine':
        warmup_scheduler = LambdaLR(optimizer, lr_lambda=lambda epoch: (epoch+1) / warmup_epochs)
        cos_scheduler = CosineAnnealingLR(optimizer, T_max=600 - warmup_epochs, eta_min=1e-5)
        # cos_scheduler = CosineAnnealingWarmRestarts(optimizer, T_0=restart_epochs, T_mult=2)
        scheduler = SequentialLR(optimizer, [
            warmup_scheduler,
            cos_scheduler
        ], milestones=[warmup_epochs])
    elif args.scheduler == 'cosine-restart':
        restart_epochs = 100
        warmup_scheduler = LambdaLR(optimizer, lr_lambda=lambda epoch: (epoch+1) / warmup_epochs)
        cos_scheduler = CosineAnnealingWarmRestarts(optimizer, T_0=restart_epochs, T_mult=2)
        scheduler = SequentialLR(optimizer, [
            warmup_scheduler,
            cos_scheduler
        ], milestones=[warmup_epochs])
    elif args.scheduler == 'onecycle':
        scheduler = OneCycleLR(optimizer, max_lr=args.lr, 
                               epochs=args.epochs, steps_per_epoch=len(train_loader))
    elif args.scheduler == 'step':
        scheduler = StepLR(optimizer, step_size=30, gamma=0.1)
    else:
        scheduler = None
    
    # Training loop
    best_acc = 0.0
    best_ema_acc = 0.0
    best_epoch = 0
    train_losses, train_accs = [], []
    val_losses, val_accs = [], []
    ema_val_losses, ema_val_accs = [], []
    print(f"Starting training for {args.epochs} epochs...")
    start_time = time.time()
    
    cutmix_prob = 0.0
    
    if args.task == 'train':
        args.output_dir = args.output_dir + '_' + cur_time
        os.makedirs(args.output_dir, exist_ok=True)
        run_name = os.path.basename(args.output_dir)
        #save the args
        with open(os.path.join(args.output_dir, 'args.json'), 'w') as f:
            json.dump(args.__dict__, f)
        # wandb.init(project='deep-learning-spring-2025-project-1', name=run_name, config=args)
        ema_start_epoch = 0
        ema = ExponentialMovingAverage(model.parameters(), decay=0.99)
        for epoch in range(args.epochs):
            print(f"\nEpoch {epoch+1}/{args.epochs}")
            
                
            if epoch == 100:
                ema_start_epoch = 0
                cutmix_prob = 0.3
            elif epoch == 300:
                ema_start_epoch = 0
                cutmix_prob = 0.5
                
            if not args.cutmix:
                cutmix_prob = 0.0

            # Train
            if ema_start_epoch < 50:
                train_loss, train_acc = train_epoch(model, train_loader, criterion, optimizer, device)
            else:
                train_loss, train_acc = train_epoch(model, train_loader, criterion, optimizer, device, ema=ema, cutmix_prob=cutmix_prob)
            train_losses.append(train_loss)
            train_accs.append(train_acc)
            
            # Validate
            print(f"Validating at epoch {epoch+1}...")
            val_loss, val_acc = validate(model, val_loader, criterion, device)
            val_losses.append(val_loss)
            val_accs.append(val_acc)
            print(f"EMA validation at epoch {epoch+1}...")
            ema_val_loss, ema_val_acc = 0.0, 0.0
            if ema_start_epoch >= 100:
                with ema.average_parameters():
                    ema_val_loss, ema_val_acc = validate(model, val_loader, criterion, device)
            ema_val_losses.append(ema_val_loss)
            ema_val_accs.append(ema_val_acc)
            
            # wandb.log({
            #     'train_loss': train_loss,
            #     'train_acc': train_acc,
            #     'val_loss': val_loss,
            #     'val_acc': val_acc
            # })
            
            # Update learning rate
            if scheduler is not None:
                if args.scheduler == 'cosine':
                    scheduler.step()
                # For OneCycleLR, step is called in the training loop
            ema_start_epoch += 1
            # Save best model
            save_model = False
            if val_acc > best_acc:
                print(f"New best model with accuracy: {val_acc:.4f}")
                best_acc = val_acc
                best_epoch = epoch
                save_model = True
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'accuracy': val_acc,
                }, os.path.join(args.output_dir, 'best_model.pth'))
            # Save best ema model
            if ema_val_acc > best_ema_acc:
                print(f"New best ema model with accuracy: {ema_val_acc:.4f}")
                best_ema_acc = ema_val_acc
                best_ema_epoch = epoch
                with ema.average_parameters():
                    torch.save({
                        'epoch': epoch, 
                        'model_state_dict': model.state_dict(),
                        'optimizer_state_dict': optimizer.state_dict(),
                        'accuracy': ema_val_acc,
                    }, os.path.join(args.output_dir, 'best_ema_model.pth'))
                    save_model = True
            if save_model:
                # Save training history
                history = pd.DataFrame({
                    'epoch': range(1, epoch+1),
                    'train_loss': train_losses[:epoch],
                    'train_acc': train_accs[:epoch],
                    'val_loss': val_losses[:epoch],
                    'val_acc': val_accs[:epoch]
                })
                history.to_csv(os.path.join(args.output_dir, 'training_history.csv'), index=False)
                print(f"Training history saved to {os.path.join(args.output_dir, 'training_history.csv')}")
        
        training_time = time.time() - start_time
        print(f"\nTraining completed in {training_time:.2f} seconds")
        print(f"Best validation accuracy: {best_acc:.4f} at epoch {best_epoch+1}")
    
    # Load best model for testing
    # checkpoint = torch.load(os.path.join(args.output_dir, 'best_model.pth'))
    checkpoint = torch.load(os.path.join(args.output_dir, 'best_ema_model.pth'))
    model.load_state_dict(checkpoint['model_state_dict'])

    if args.task == 'test':
        model.eval()
        test_loader, test_labels = setup_data_test_cifar(args.test_data, batch_size=args.batch_size)
        ids, predictions = generate_predictions(model, test_loader, device)
        d = unpickle(args.test_data)
        
        output_file = os.path.join(args.output_dir, 'test_predictions.csv')
        df = pd.DataFrame({'ID': ids, 'Labels': predictions})
        df.to_csv(output_file, index=False)
        if test_labels:
        # print(f"Test predictions saved to {output_file}")
        # print(ids, predictions)
            predictions = np.array(predictions)
            test_labels = np.array(test_labels)
            print(np.sum(predictions == test_labels) / len(test_labels))
        return
    
    if args.task == 'submit':
        args.test_data = 'dataset/cifar_test_nolabel.pkl'
        model.eval()
        test_loader, test_labels = setup_data_test(args.test_data, batch_size=args.batch_size)
        ids, predictions = generate_predictions(model, test_loader, device)
        d = unpickle(args.test_data)
        
        output_file = os.path.join(args.output_dir, 'test_predictions.csv')
        df = pd.DataFrame({'ID': ids, 'Labels': predictions})
        df.to_csv(output_file, index=False)
        return
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