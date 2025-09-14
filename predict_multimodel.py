#!/usr/bin/env python3
"""
Universal predictor for all model architectures.
Automatically detects model type and loads appropriately.
"""

import torch
import numpy as np
from PIL import Image
from pathlib import Path
from torchvision import transforms
import sys
import json

# Import architectures
from train_multimodel import MultiModelRegression

def detect_model_architecture(checkpoint):
    """Detect which architecture was used"""

    # Check if architecture is stored in checkpoint
    if isinstance(checkpoint, dict):
        if 'architecture' in checkpoint:
            return checkpoint['architecture']

        # Try to infer from config
        if 'config' in checkpoint and 'architecture' in checkpoint['config']:
            return checkpoint['config']['architecture']

        # Check state dict keys
        state_dict = checkpoint.get('model_state_dict', checkpoint)
    else:
        state_dict = checkpoint

    # Infer from layer names
    keys = list(state_dict.keys())

    # Check for model wrapper patterns
    if any('model.features' in k for k in keys):
        if any('model.features.8' in k for k in keys):
            return 'efficientnet_b1'
        else:
            return 'efficientnet_b0'
    elif any('model.classifier' in k for k in keys) and any('model.features.7.0.block' in k for k in keys):
        return 'convnext_tiny'
    elif any('model.features.17' in k for k in keys):
        return 'mobilenet_v3'
    elif any('model.features.denseblock' in k for k in keys):
        return 'densenet121'
    elif any('model.layer4.2' in k for k in keys):
        return 'resnet50'
    elif any('model.layer4.1' in k for k in keys) and not any('model.layer4.2' in k for k in keys):
        return 'resnet18'

    # Default fallback
    return 'resnet50'

def load_multimodel(checkpoint_path='best_model.pth', device=None):
    """Load any model architecture"""

    # Auto-detect device
    if device is None:
        if torch.backends.mps.is_available():
            device = 'mps'
        elif torch.cuda.is_available():
            device = 'cuda'
        else:
            device = 'cpu'

    print(f"Loading model on device: {device}")
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)

    # Detect architecture
    architecture = detect_model_architecture(checkpoint)
    print(f"Detected architecture: {architecture}")

    # Create model
    model = MultiModelRegression(architecture=architecture, pretrained=False)

    # Load weights
    if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
        state_dict = checkpoint['model_state_dict']

        # Handle old model format (backbone. -> model.)
        if any(k.startswith('backbone.') for k in state_dict.keys()):
            new_state_dict = {}
            for k, v in state_dict.items():
                if k.startswith('backbone.'):
                    new_k = k.replace('backbone.', 'model.')
                    new_state_dict[new_k] = v
                else:
                    new_state_dict[k] = v
            model.load_state_dict(new_state_dict)
        else:
            model.load_state_dict(state_dict)

        config = checkpoint.get('config', {})
    else:
        # Try loading as complete model state
        try:
            model.load_state_dict(checkpoint)
            config = {}
        except:
            # Handle old model format (backbone. -> model.)
            if any(k.startswith('backbone.') for k in checkpoint.keys()):
                # Convert old format to new format
                new_checkpoint = {}
                for k, v in checkpoint.items():
                    if k.startswith('backbone.'):
                        new_k = k.replace('backbone.', 'model.')
                        new_checkpoint[new_k] = v
                    else:
                        new_checkpoint[k] = v
                model.load_state_dict(new_checkpoint)
                config = {}
            else:
                # Might be a different format, try the model's internal model
                model.model.load_state_dict(checkpoint)
                config = {}

    model = model.to(device)
    model.eval()

    # Update config
    if not config:
        config = {'img_size': 224, 'device': device, 'architecture': architecture}
    else:
        config = config.copy()
        config['device'] = device
        config['architecture'] = architecture

    return model, config

def predict_single_image(model, image_path, config):
    """Predict age for a single image"""
    device = config.get('device', 'cpu')
    img_size = config.get('img_size', 224)

    # Image transform
    transform = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                           std=[0.229, 0.224, 0.225])
    ])

    # Load and preprocess
    image = Image.open(image_path).convert('RGB')
    image_tensor = transform(image).unsqueeze(0).to(device)

    # Predict
    with torch.no_grad():
        age_pred = model(image_tensor).item()

    return age_pred

def compare_all_models(image_path, model_dir="."):
    """Test image on all available models"""

    results = {}

    # Find all model files
    model_patterns = [
        'resnet18_model.pth',
        'resnet50_model.pth',
        'efficientnet_b0_model.pth',
        'efficientnet_b1_model.pth',
        'convnext_tiny_model.pth',
        'mobilenet_v3_model.pth',
        'densenet121_model.pth',
        'best_model.pth',
        'compromise_model.pth',
        'hand_age_model.pth'
    ]

    print(f"\nTesting {image_path} on all available models:\n")
    print(f"{'Model':<25} {'Prediction':<15} {'Architecture':<20}")
    print("-" * 60)

    for model_file in model_patterns:
        model_path = Path(model_dir) / model_file
        if model_path.exists():
            try:
                model, config = load_multimodel(str(model_path))
                age_pred = predict_single_image(model, image_path, config)
                arch = config.get('architecture', 'unknown')

                results[model_file] = {
                    'prediction': age_pred,
                    'architecture': arch
                }

                print(f"{model_file:<25} {age_pred:<15.1f} {arch:<20}")

            except Exception as e:
                print(f"{model_file:<25} {'Error':<15} {str(e)[:20]}")

    # Find best and worst
    if results:
        predictions = [(k, v['prediction']) for k, v in results.items()]
        best = min(predictions, key=lambda x: abs(x[1] - 30))  # Assuming ~30 is reasonable
        worst_young = min(predictions, key=lambda x: x[1])
        worst_old = max(predictions, key=lambda x: x[1])

        print("\n" + "=" * 60)
        print(f"Youngest prediction: {worst_young[0]} = {worst_young[1]:.1f} years")
        print(f"Oldest prediction: {worst_old[0]} = {worst_old[1]:.1f} years")
        print(f"Range: {worst_old[1] - worst_young[1]:.1f} years")

    return results

def main():
    if len(sys.argv) < 2:
        print("Usage:")
        print("  python predict_multimodel.py <image_path> [model.pth]     # Single prediction")
        print("  python predict_multimodel.py <image_path> --compare        # Compare all models")
        print("\nExamples:")
        print("  python predict_multimodel.py IMG_6782.jpeg efficientnet_b0_model.pth")
        print("  python predict_multimodel.py IMG_6782.jpeg --compare")
        sys.exit(1)

    image_path = sys.argv[1]

    if not Path(image_path).exists():
        print(f"Error: Image not found: {image_path}")
        sys.exit(1)

    if len(sys.argv) > 2 and sys.argv[2] == '--compare':
        # Compare all models
        compare_all_models(image_path)
    else:
        # Single model prediction
        model_path = sys.argv[2] if len(sys.argv) > 2 else 'best_model.pth'

        if not Path(model_path).exists():
            print(f"Error: Model not found: {model_path}")
            sys.exit(1)

        print(f"Loading {model_path}...")
        model, config = load_multimodel(model_path)

        age_pred = predict_single_image(model, image_path, config)

        print(f"\nPredicted age: {age_pred:.1f} years")
        print(f"Architecture: {config.get('architecture', 'unknown')}")

if __name__ == "__main__":
    main()