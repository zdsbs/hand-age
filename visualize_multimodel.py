#!/usr/bin/env python3
"""
Universal Grad-CAM visualizer for all model architectures.
"""

import torch
import torch.nn.functional as F
import numpy as np
import cv2
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from PIL import Image
from pathlib import Path
import sys

# Import model loading
from predict_multimodel import load_multimodel, predict_single_image

class UniversalGradCAM:
    """Grad-CAM that works with any architecture"""

    def __init__(self, model, architecture):
        self.model = model
        self.model.eval()
        self.architecture = architecture.lower()

        # Find the appropriate target layer for each architecture
        self.target_layer = self._find_target_layer()

        if self.target_layer is None:
            raise ValueError(f"Could not find target layer for {architecture}")

        # Storage
        self.gradients = None
        self.activations = None

        # Register hooks
        self.target_layer.register_forward_hook(self.save_activation)
        self.target_layer.register_full_backward_hook(self.save_gradient)

    def _find_target_layer(self):
        """Find the best layer for Grad-CAM based on architecture"""

        if 'resnet' in self.architecture:
            # ResNet: use last bottleneck
            if hasattr(self.model, 'model'):
                return self.model.model.layer4[-1]
            else:
                return self.model.layer4[-1]

        elif 'efficientnet' in self.architecture:
            # EfficientNet: use last conv layer in features
            return self.model.model.features[-1]

        elif 'convnext' in self.architecture:
            # ConvNeXt: use last stage
            return self.model.model.features[-1][-1]

        elif 'mobilenet' in self.architecture:
            # MobileNet: use last conv layer
            return self.model.model.features[-1][0]

        elif 'densenet' in self.architecture:
            # DenseNet: use last dense block
            return self.model.model.features.denseblock4

        else:
            # Try to find any layer4 or last conv layer
            for name, module in self.model.named_modules():
                if 'layer4' in name and not any(c in name for c in ['.', '0', '1', '2']):
                    return module

        return None

    def save_activation(self, module, input, output):
        self.activations = output

    def save_gradient(self, module, grad_input, grad_output):
        self.gradients = grad_output[0]

    def generate_cam(self, input_image):
        """Generate CAM for the input"""

        # Forward pass
        output = self.model(input_image)

        # Clear gradients
        self.model.zero_grad()

        # Backward pass
        output.backward()

        # Get gradients and activations
        gradients = self.gradients[0]
        activations = self.activations[0]

        # Global Average Pooling
        weights = torch.mean(gradients, dim=(1, 2))

        # Weighted combination
        cam = torch.zeros(activations.shape[1:], dtype=torch.float32, device=activations.device)
        for i, w in enumerate(weights):
            cam += w * activations[i]

        # ReLU
        cam = F.relu(cam)

        # Normalize
        if cam.max() > 0:
            cam = cam / cam.max()

        return cam.detach().cpu().numpy()

def visualize_multimodel(image_path, model_path='best_model.pth', save_path=None):
    """Create Grad-CAM visualization for any model"""

    # Load model
    device = 'mps' if torch.backends.mps.is_available() else 'cpu'
    print(f"Loading model...")
    model, config = load_multimodel(model_path, device)
    architecture = config.get('architecture', 'unknown')

    # Load and preprocess image
    from torchvision import transforms

    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                           std=[0.229, 0.224, 0.225])
    ])

    original = Image.open(image_path).convert('RGB')
    original_resized = original.resize((224, 224))
    input_tensor = transform(original).unsqueeze(0).to(device)

    # Predict
    with torch.no_grad():
        predicted_age = model(input_tensor).item()

    print(f"Architecture: {architecture}")
    print(f"Predicted age: {predicted_age:.1f} years")

    # Generate Grad-CAM
    print("Generating Grad-CAM...")
    try:
        gradcam = UniversalGradCAM(model, architecture)
        cam = gradcam.generate_cam(input_tensor)
    except Exception as e:
        print(f"Warning: Could not generate Grad-CAM for {architecture}: {e}")
        cam = np.ones((7, 7)) * 0.5  # Fallback

    # Resize CAM
    cam_resized = cv2.resize(cam, (224, 224))

    # Create heatmap
    heatmap = cm.jet(cam_resized)[:, :, :3]
    heatmap = (heatmap * 255).astype(np.uint8)

    # Overlay
    original_np = np.array(original_resized)
    overlay = cv2.addWeighted(original_np, 0.4, heatmap, 0.6, 0)

    # Create figure
    fig, axes = plt.subplots(2, 2, figsize=(12, 12))

    # Original
    axes[0, 0].imshow(original_np)
    axes[0, 0].set_title(f'Original Hand Image')
    axes[0, 0].axis('off')

    # Heatmap
    axes[0, 1].imshow(cam, cmap='jet')
    axes[0, 1].set_title(f'Grad-CAM Heatmap ({architecture})')
    axes[0, 1].axis('off')

    # Overlay
    axes[1, 0].imshow(overlay)
    axes[1, 0].set_title(f'Overlay - Predicted: {predicted_age:.1f} years')
    axes[1, 0].axis('off')

    # Info
    axes[1, 1].text(0.1, 0.9, f'Model: {Path(model_path).stem}', fontsize=12, fontweight='bold')
    axes[1, 1].text(0.1, 0.8, f'Architecture: {architecture}', fontsize=11)
    axes[1, 1].text(0.1, 0.7, f'Predicted Age: {predicted_age:.1f} years', fontsize=11)

    axes[1, 1].text(0.1, 0.5, 'Interpretation:', fontsize=11, fontweight='bold')
    axes[1, 1].text(0.1, 0.4, '• Red/Yellow: High importance', fontsize=10)
    axes[1, 1].text(0.1, 0.35, '• Blue/Dark: Low importance', fontsize=10)

    axes[1, 1].text(0.1, 0.2, 'Focus Areas:', fontsize=11, fontweight='bold')
    axes[1, 1].text(0.1, 0.1, '• Knuckles, wrinkles, skin texture', fontsize=10)
    axes[1, 1].text(0.1, 0.05, '• Veins, joints, hand contours', fontsize=10)

    axes[1, 1].set_xlim(0, 1)
    axes[1, 1].set_ylim(0, 1)
    axes[1, 1].axis('off')

    plt.tight_layout()

    # Save
    if save_path is None:
        image_name = Path(image_path).stem
        model_name = Path(model_path).stem
        save_path = f"gradcam_{image_name}_{model_name}.png"

    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"Visualization saved to {save_path}")
    plt.close()

    return predicted_age, cam, overlay

def compare_all_visualizations(image_path):
    """Generate Grad-CAM for all available models"""

    model_patterns = [
        'resnet18_model.pth',
        'resnet50_model.pth',
        'efficientnet_b0_model.pth',
        'convnext_tiny_model.pth',
        'mobilenet_v3_model.pth',
        'best_model.pth',
        'hand_age_model.pth'
    ]

    results = []

    for model_file in model_patterns:
        if Path(model_file).exists():
            print(f"\nProcessing {model_file}...")
            try:
                age, cam, overlay = visualize_multimodel(image_path, model_file)
                results.append((model_file, age))
            except Exception as e:
                print(f"Error with {model_file}: {e}")

    if results:
        print("\n" + "="*60)
        print("Summary of all models:")
        for model, age in sorted(results, key=lambda x: x[1]):
            print(f"  {model}: {age:.1f} years")

def main():
    if len(sys.argv) < 2:
        print("Usage:")
        print("  python visualize_multimodel.py <image_path> [model.pth]")
        print("  python visualize_multimodel.py <image_path> --compare")
        print("\nExamples:")
        print("  python visualize_multimodel.py IMG_6782.jpeg efficientnet_b0_model.pth")
        print("  python visualize_multimodel.py IMG_6782.jpeg --compare")
        sys.exit(1)

    image_path = sys.argv[1]

    if len(sys.argv) > 2 and sys.argv[2] == '--compare':
        compare_all_visualizations(image_path)
    else:
        model_path = sys.argv[2] if len(sys.argv) > 2 else 'best_model.pth'
        visualize_multimodel(image_path, model_path)

if __name__ == "__main__":
    main()