#!/usr/bin/env python3
"""
Inference script for hand age prediction.
Can predict on single images or test set.
"""

import json
import torch
import numpy as np
from PIL import Image
from pathlib import Path
from torchvision import transforms
import sys

# Import model class from training script
from train_model import AgeRegressionModel

def load_model(checkpoint_path='best_model.pth', device='cpu'):
    """Load trained model from checkpoint"""
    checkpoint = torch.load(checkpoint_path, map_location=device)

    model = AgeRegressionModel(backbone='resnet50')
    model.load_state_dict(checkpoint['model_state_dict'])
    model = model.to(device)
    model.eval()

    return model, checkpoint['config']

def predict_single_image(model, image_path, config):
    """Predict age for a single image"""
    device = config.get('device', 'cpu')

    # Image transform
    transform = transforms.Compose([
        transforms.Resize((config['img_size'], config['img_size'])),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                           std=[0.229, 0.224, 0.225])
    ])

    # Load and preprocess image
    image = Image.open(image_path).convert('RGB')
    image_tensor = transform(image).unsqueeze(0).to(device)

    # Predict
    with torch.no_grad():
        age_pred = model(image_tensor).item()

    return age_pred

def evaluate_test_set(model, config, test_json='dataset_splits/test.json', img_dir='Hands'):
    """Evaluate model on test set"""
    device = config.get('device', 'cpu')

    # Load test data
    with open(test_json, 'r') as f:
        test_data = json.load(f)

    # Image transform
    transform = transforms.Compose([
        transforms.Resize((config['img_size'], config['img_size'])),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                           std=[0.229, 0.224, 0.225])
    ])

    predictions = []
    errors = []

    print(f"Evaluating {len(test_data)} test images...")

    for item in test_data:
        img_path = Path(img_dir) / item['image']
        true_age = item['age']

        # Predict
        pred_age = predict_single_image(model, img_path, config)
        error = abs(pred_age - true_age)

        predictions.append({
            'image': item['image'],
            'person_id': item['person_id'],
            'true_age': true_age,
            'predicted_age': round(pred_age, 1),
            'error': round(error, 1),
            'gender': item.get('gender'),
            'aspect': item.get('aspect')
        })
        errors.append(error)

    # Calculate metrics
    errors = np.array(errors)
    mae = errors.mean()
    within_5 = (errors <= 5).mean() * 100
    within_10 = (errors <= 10).mean() * 100

    # Sort by error (worst predictions first)
    predictions.sort(key=lambda x: x['error'], reverse=True)

    return {
        'predictions': predictions,
        'metrics': {
            'mae': round(mae, 2),
            'within_5_years': round(within_5, 1),
            'within_10_years': round(within_10, 1),
            'median_error': round(np.median(errors), 2),
            'max_error': round(errors.max(), 1),
            'min_error': round(errors.min(), 1)
        }
    }

def main():
    # Parse arguments
    if len(sys.argv) < 2:
        print("Usage:")
        print("  python predict.py <image_path>           # Predict single image")
        print("  python predict.py --test                 # Evaluate test set")
        print("  python predict.py --verification         # Check verification samples")
        sys.exit(1)

    # Device
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")

    # Load model
    print("Loading model...")
    model, config = load_model('best_model.pth', device)
    config['device'] = device

    if sys.argv[1] == '--test':
        # Evaluate full test set
        results = evaluate_test_set(model, config)

        print("\n" + "="*60)
        print("TEST SET RESULTS")
        print("="*60)
        print(f"MAE: {results['metrics']['mae']} years")
        print(f"Within 5 years: {results['metrics']['within_5_years']}%")
        print(f"Within 10 years: {results['metrics']['within_10_years']}%")
        print(f"Median error: {results['metrics']['median_error']} years")
        print(f"Max error: {results['metrics']['max_error']} years")

        # Show worst predictions
        print("\nWorst 10 predictions:")
        for pred in results['predictions'][:10]:
            print(f"  {pred['image']}: True={pred['true_age']}, Pred={pred['predicted_age']}, Error={pred['error']}")

        # Save results
        with open('test_results.json', 'w') as f:
            json.dump(results, f, indent=2)
        print("\nFull results saved to test_results.json")

    elif sys.argv[1] == '--verification':
        # Check verification samples
        with open('dataset_splits/test_verification_samples.json', 'r') as f:
            samples = json.load(f)

        print("\nVERIFICATION SAMPLES:")
        print("-" * 60)

        for sample in samples:
            img_path = Path('Hands') / sample['image']
            pred_age = predict_single_image(model, img_path, config)
            error = abs(pred_age - sample['age'])

            print(f"\nImage: {sample['image']}")
            print(f"  Person ID: {sample['person_id']}")
            print(f"  True age: {sample['age']}")
            print(f"  Predicted age: {pred_age:.1f}")
            print(f"  Error: {error:.1f} years")
            print(f"  Gender: {sample['gender']}, Aspect: {sample['aspect']}")

    else:
        # Single image prediction
        image_path = sys.argv[1]
        if not Path(image_path).exists():
            print(f"Error: Image not found: {image_path}")
            sys.exit(1)

        pred_age = predict_single_image(model, image_path, config)
        print(f"\nPredicted age: {pred_age:.1f} years")

if __name__ == "__main__":
    main()