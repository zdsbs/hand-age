#!/usr/bin/env python3
"""
Train a regression model to predict age from dorsal hand images.
Uses pre-trained ResNet50 as backbone.
"""

import json
import numpy as np
from pathlib import Path
from PIL import Image
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, models
from tqdm import tqdm
import random
import os

# Set seeds for reproducibility
def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

class HandAgeDataset(Dataset):
    """Dataset for hand images with age labels"""

    def __init__(self, json_path, img_dir="Hands", transform=None, augment=False):
        with open(json_path, 'r') as f:
            self.data = json.load(f)
        self.img_dir = Path(img_dir)
        self.transform = transform
        self.augment = augment

        # Create augmentation transforms
        if augment:
            self.aug_transform = transforms.Compose([
                transforms.RandomRotation(10),
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.1),
                transforms.RandomAffine(degrees=0, translate=(0.1, 0.1)),
            ])

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]
        img_path = self.img_dir / item['image']

        # Load image
        image = Image.open(img_path).convert('RGB')

        # Apply augmentation if training
        if self.augment and self.transform:
            image = self.aug_transform(image)

        # Apply main transform
        if self.transform:
            image = self.transform(image)

        # Age as float32 for regression
        age = torch.tensor(item['age'], dtype=torch.float32)

        return image, age, item['image']

class AgeRegressionModel(nn.Module):
    """ResNet-based regression model for age prediction"""

    def __init__(self, backbone='resnet50', pretrained=True):
        super().__init__()

        # Load pre-trained backbone
        if backbone == 'resnet50':
            self.backbone = models.resnet50(pretrained=pretrained)
            num_features = self.backbone.fc.in_features
            # Replace classifier with regression head
            self.backbone.fc = nn.Sequential(
                nn.Linear(num_features, 256),
                nn.ReLU(),
                nn.Dropout(0.3),
                nn.Linear(256, 64),
                nn.ReLU(),
                nn.Dropout(0.2),
                nn.Linear(64, 1)  # Single output for age
            )
        else:
            raise ValueError(f"Unknown backbone: {backbone}")

    def forward(self, x):
        return self.backbone(x).squeeze()

def train_epoch(model, dataloader, criterion, optimizer, device):
    """Train for one epoch"""
    model.train()
    total_loss = 0
    total_mae = 0
    num_batches = 0

    pbar = tqdm(dataloader, desc="Training")
    for images, ages, _ in pbar:
        images = images.to(device)
        ages = ages.to(device)

        # Forward pass
        outputs = model(images)
        loss = criterion(outputs, ages)

        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # Calculate MAE
        mae = torch.abs(outputs - ages).mean()

        total_loss += loss.item()
        total_mae += mae.item()
        num_batches += 1

        pbar.set_postfix({'loss': f'{loss.item():.3f}', 'mae': f'{mae.item():.2f}'})

    return total_loss / num_batches, total_mae / num_batches

def validate(model, dataloader, criterion, device):
    """Validate the model"""
    model.eval()
    total_loss = 0
    total_mae = 0
    all_preds = []
    all_targets = []
    num_batches = 0

    with torch.no_grad():
        pbar = tqdm(dataloader, desc="Validating")
        for images, ages, _ in pbar:
            images = images.to(device)
            ages = ages.to(device)

            outputs = model(images)
            loss = criterion(outputs, ages)
            mae = torch.abs(outputs - ages).mean()

            total_loss += loss.item()
            total_mae += mae.item()
            num_batches += 1

            all_preds.extend(outputs.cpu().numpy())
            all_targets.extend(ages.cpu().numpy())

            pbar.set_postfix({'loss': f'{loss.item():.3f}', 'mae': f'{mae.item():.2f}'})

    # Calculate additional metrics
    all_preds = np.array(all_preds)
    all_targets = np.array(all_targets)

    within_5 = np.mean(np.abs(all_preds - all_targets) <= 5) * 100
    within_10 = np.mean(np.abs(all_preds - all_targets) <= 10) * 100

    return {
        'loss': total_loss / num_batches,
        'mae': total_mae / num_batches,
        'within_5': within_5,
        'within_10': within_10
    }

def main():
    set_seed(42)

    # Configuration
    config = {
        'batch_size': 32,
        'learning_rate': 1e-4,
        'num_epochs': 50,
        'device': 'cuda' if torch.cuda.is_available() else 'cpu',
        'backbone': 'resnet50',
        'img_size': 224,
        'patience': 10  # Early stopping patience
    }

    print(f"Using device: {config['device']}")

    # Data transforms
    train_transform = transforms.Compose([
        transforms.Resize((config['img_size'], config['img_size'])),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                           std=[0.229, 0.224, 0.225])
    ])

    val_transform = transforms.Compose([
        transforms.Resize((config['img_size'], config['img_size'])),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                           std=[0.229, 0.224, 0.225])
    ])

    # Create datasets
    train_dataset = HandAgeDataset(
        'dataset_splits/train.json',
        transform=train_transform,
        augment=True
    )
    val_dataset = HandAgeDataset(
        'dataset_splits/val.json',
        transform=val_transform,
        augment=False
    )

    # Create dataloaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=config['batch_size'],
        shuffle=True,
        num_workers=4
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=config['batch_size'],
        shuffle=False,
        num_workers=4
    )

    print(f"Train samples: {len(train_dataset)}")
    print(f"Val samples: {len(val_dataset)}")

    # Create model
    model = AgeRegressionModel(backbone=config['backbone'])
    model = model.to(config['device'])

    # Loss and optimizer
    criterion = nn.MSELoss()
    optimizer = optim.AdamW(model.parameters(), lr=config['learning_rate'])
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=5
    )

    # Training loop
    best_val_mae = float('inf')
    patience_counter = 0
    history = {'train_loss': [], 'train_mae': [], 'val_loss': [], 'val_mae': []}

    for epoch in range(config['num_epochs']):
        print(f"\n{'='*60}")
        print(f"Epoch {epoch+1}/{config['num_epochs']}")
        print(f"{'='*60}")

        # Train
        train_loss, train_mae = train_epoch(
            model, train_loader, criterion, optimizer, config['device']
        )

        # Validate
        val_metrics = validate(model, val_loader, criterion, config['device'])

        # Update learning rate
        scheduler.step(val_metrics['mae'])

        # Save history
        history['train_loss'].append(train_loss)
        history['train_mae'].append(train_mae)
        history['val_loss'].append(val_metrics['loss'])
        history['val_mae'].append(val_metrics['mae'])

        print(f"\nTrain Loss: {train_loss:.4f}, Train MAE: {train_mae:.2f}")
        print(f"Val Loss: {val_metrics['loss']:.4f}, Val MAE: {val_metrics['mae']:.2f}")
        print(f"Within 5 years: {val_metrics['within_5']:.1f}%")
        print(f"Within 10 years: {val_metrics['within_10']:.1f}%")

        # Save best model
        if val_metrics['mae'] < best_val_mae:
            best_val_mae = val_metrics['mae']
            patience_counter = 0
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_mae': val_metrics['mae'],
                'config': config
            }, 'best_model.pth')
            print(f"✓ Saved best model (MAE: {best_val_mae:.2f})")
        else:
            patience_counter += 1
            if patience_counter >= config['patience']:
                print(f"\nEarly stopping triggered after {epoch+1} epochs")
                break

    # Save training history
    with open('training_history.json', 'w') as f:
        json.dump(history, f, indent=2)

    print(f"\n{'='*60}")
    print(f"Training complete! Best validation MAE: {best_val_mae:.2f} years")
    print(f"Model saved to best_model.pth")

if __name__ == "__main__":
    main()