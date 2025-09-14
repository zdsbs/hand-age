#!/usr/bin/env python3
"""
Multi-model training script - easily try different architectures.
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
from torchvision.models import ResNet50_Weights, EfficientNet_B0_Weights, ConvNeXt_Tiny_Weights
from tqdm import tqdm
import random
import os
import sys

# Import dataset class from original
from train_model import HandAgeDataset, set_seed

class MultiModelRegression(nn.Module):
    """Flexible model class supporting multiple architectures"""

    def __init__(self, architecture='resnet50', pretrained=True):
        super().__init__()
        self.architecture = architecture.lower()

        if self.architecture == 'resnet18':
            weights = models.ResNet18_Weights.DEFAULT if pretrained else None
            self.model = models.resnet18(weights=weights)
            num_features = self.model.fc.in_features
            self.model.fc = nn.Sequential(
                nn.Linear(num_features, 256),
                nn.ReLU(),
                nn.Dropout(0.3),
                nn.Linear(256, 1)
            )

        elif self.architecture == 'resnet50':
            weights = ResNet50_Weights.DEFAULT if pretrained else None
            self.model = models.resnet50(weights=weights)
            num_features = self.model.fc.in_features
            self.model.fc = nn.Sequential(
                nn.Linear(num_features, 256),
                nn.ReLU(),
                nn.Dropout(0.3),
                nn.Linear(256, 64),
                nn.ReLU(),
                nn.Dropout(0.2),
                nn.Linear(64, 1)
            )

        elif self.architecture == 'efficientnet_b0':
            weights = EfficientNet_B0_Weights.DEFAULT if pretrained else None
            self.model = models.efficientnet_b0(weights=weights)
            num_features = self.model.classifier[1].in_features
            self.model.classifier = nn.Sequential(
                nn.Dropout(0.2),
                nn.Linear(num_features, 256),
                nn.ReLU(),
                nn.Dropout(0.2),
                nn.Linear(256, 1)
            )

        elif self.architecture == 'efficientnet_b1':
            weights = models.EfficientNet_B1_Weights.DEFAULT if pretrained else None
            self.model = models.efficientnet_b1(weights=weights)
            num_features = self.model.classifier[1].in_features
            self.model.classifier = nn.Sequential(
                nn.Dropout(0.2),
                nn.Linear(num_features, 256),
                nn.ReLU(),
                nn.Dropout(0.2),
                nn.Linear(256, 1)
            )

        elif self.architecture == 'convnext_tiny':
            weights = ConvNeXt_Tiny_Weights.DEFAULT if pretrained else None
            self.model = models.convnext_tiny(weights=weights)
            num_features = self.model.classifier[2].in_features
            self.model.classifier[2] = nn.Linear(num_features, 1)

        elif self.architecture == 'mobilenet_v3':
            weights = models.MobileNet_V3_Small_Weights.DEFAULT if pretrained else None
            self.model = models.mobilenet_v3_small(weights=weights)
            num_features = self.model.classifier[3].in_features
            self.model.classifier[3] = nn.Linear(num_features, 1)

        elif self.architecture == 'densenet121':
            weights = models.DenseNet121_Weights.DEFAULT if pretrained else None
            self.model = models.densenet121(weights=weights)
            num_features = self.model.classifier.in_features
            self.model.classifier = nn.Sequential(
                nn.Linear(num_features, 256),
                nn.ReLU(),
                nn.Dropout(0.3),
                nn.Linear(256, 1)
            )

        else:
            raise ValueError(f"Unknown architecture: {architecture}")

        print(f"Initialized {architecture} for regression")

    def forward(self, x):
        return self.model(x).squeeze()

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

        outputs = model(images)
        loss = criterion(outputs, ages)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

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
    # Parse architecture from command line
    if len(sys.argv) < 2:
        print("Usage: python train_multimodel.py <architecture>")
        print("\nAvailable architectures:")
        print("  resnet18      - Lightweight, fast (your friend's choice)")
        print("  resnet50      - Deeper, more capacity (your original)")
        print("  efficientnet_b0 - Best efficiency/accuracy tradeoff")
        print("  efficientnet_b1 - Slightly larger EfficientNet")
        print("  convnext_tiny - Modern CNN architecture (2022)")
        print("  mobilenet_v3  - Fast and lightweight")
        print("  densenet121   - Dense connections")
        sys.exit(1)

    architecture = sys.argv[1].lower()
    set_seed(42)

    # Device
    if torch.backends.mps.is_available():
        device = 'mps'
    elif torch.cuda.is_available():
        device = 'cuda'
    else:
        device = 'cpu'

    # Configuration
    config = {
        'architecture': architecture,
        'batch_size': 128,  # Adjust based on model size
        'learning_rate': 1e-4,
        'num_epochs': 50,
        'device': device,
        'img_size': 224,
        'patience': 10
    }

    # Adjust batch size for larger models
    if architecture in ['convnext_tiny', 'efficientnet_b1']:
        config['batch_size'] = 64

    print(f"Training {architecture} on {device}")
    print(f"Batch size: {config['batch_size']}")

    # Data transforms
    train_transform = transforms.Compose([
        transforms.Resize((config['img_size'], config['img_size'])),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    val_transform = transforms.Compose([
        transforms.Resize((config['img_size'], config['img_size'])),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    # Datasets - using FULL dataset
    train_dataset = HandAgeDataset(
        'full_dataset_splits/train.json',
        transform=train_transform,
        augment=True
    )
    val_dataset = HandAgeDataset(
        'full_dataset_splits/val.json',
        transform=val_transform,
        augment=False
    )

    # Dataloaders
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

    # Model
    model = MultiModelRegression(architecture=architecture)
    model = model.to(device)

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
            model, train_loader, criterion, optimizer, device
        )

        # Validate
        val_metrics = validate(model, val_loader, criterion, device)

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

            save_path = f'{architecture}_model.pth'
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_mae': val_metrics['mae'],
                'config': config,
                'architecture': architecture
            }, save_path)
            print(f"✓ Saved best {architecture} model (MAE: {best_val_mae:.2f}) to {save_path}")
        else:
            patience_counter += 1
            if patience_counter >= config['patience']:
                print(f"\nEarly stopping triggered after {epoch+1} epochs")
                break

    # Save history
    with open(f'{architecture}_history.json', 'w') as f:
        json.dump(history, f, indent=2)

    print(f"\n{'='*60}")
    print(f"Training complete for {architecture}!")
    print(f"Best validation MAE: {best_val_mae:.2f} years")
    print(f"Model saved to {architecture}_model.pth")

if __name__ == "__main__":
    main()