#!/usr/bin/env python3
"""
Generate Grad-CAM visualizations to understand what hand features
the model focuses on for age prediction.
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

# Import model
from train_model import AgeRegressionModel
from predict import load_model

class GradCAM:
    """Generate Grad-CAM heatmaps for age prediction model"""

    def __init__(self, model, target_layer_name='backbone.layer4'):
        self.model = model
        self.model.eval()

        # Find the target layer
        self.target_layer = None
        for name, module in self.model.named_modules():
            if name == target_layer_name:
                self.target_layer = module
                break

        if self.target_layer is None:
            raise ValueError(f"Target layer {target_layer_name} not found")

        # Storage for gradients and activations
        self.gradients = None
        self.activations = None

        # Register hooks
        self.target_layer.register_forward_hook(self.save_activation)
        self.target_layer.register_full_backward_hook(self.save_gradient)

    def save_activation(self, module, input, output):
        """Save forward pass activation"""
        self.activations = output

    def save_gradient(self, module, grad_input, grad_output):
        """Save backward pass gradient"""
        self.gradients = grad_output[0]

    def generate_cam(self, input_image, predicted_age):
        """Generate Class Activation Map"""

        # Forward pass
        output = self.model(input_image)

        # Clear gradients
        self.model.zero_grad()

        # Backward pass
        # For regression, we use the output directly as the "class score"
        output.backward()

        # Get gradients and activations
        gradients = self.gradients[0]  # Remove batch dimension
        activations = self.activations[0]  # Remove batch dimension

        # Global Average Pooling of gradients
        weights = torch.mean(gradients, dim=(1, 2))

        # Weighted combination of activation maps
        cam = torch.zeros(activations.shape[1:], dtype=torch.float32, device=activations.device)
        for i, w in enumerate(weights):
            cam += w * activations[i]

        # Apply ReLU (only positive contributions)
        cam = F.relu(cam)

        # Normalize to [0, 1]
        if cam.max() > 0:
            cam = cam / cam.max()

        return cam.detach().cpu().numpy()

def load_and_preprocess_image(image_path, img_size=224):
    """Load and preprocess image for model input"""
    from torchvision import transforms

    # Same preprocessing as training
    transform = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                           std=[0.229, 0.224, 0.225])
    ])

    # Load original image for visualization
    original = Image.open(image_path).convert('RGB')
    original_resized = original.resize((img_size, img_size))

    # Preprocess for model
    input_tensor = transform(original).unsqueeze(0)  # Add batch dimension

    return input_tensor, np.array(original_resized)

def create_heatmap_overlay(original_img, cam, alpha=0.6):
    """Create heatmap overlay on original image"""

    # Resize CAM to match image size
    h, w = original_img.shape[:2]
    cam_resized = cv2.resize(cam, (w, h))

    # Convert to heatmap
    heatmap = cm.jet(cam_resized)[:, :, :3]  # Remove alpha channel
    heatmap = (heatmap * 255).astype(np.uint8)

    # Overlay on original image
    overlay = cv2.addWeighted(original_img, 1-alpha, heatmap, alpha, 0)

    return overlay, heatmap

def visualize_age_prediction(image_path, model_path='best_model.pth', save_path=None):
    """Create comprehensive visualization of age prediction"""

    # Load model
    device = 'mps' if torch.backends.mps.is_available() else 'cpu'
    print(f"Loading model on {device}...")
    model, config = load_model(model_path)

    # Load and preprocess image
    input_tensor, original_img = load_and_preprocess_image(image_path)
    input_tensor = input_tensor.to(device)

    # Make prediction
    with torch.no_grad():
        predicted_age = model(input_tensor).item()

    print(f"Predicted age: {predicted_age:.1f} years")

    # Generate Grad-CAM
    print("Generating Grad-CAM...")
    gradcam = GradCAM(model)
    cam = gradcam.generate_cam(input_tensor, predicted_age)

    # Create visualizations
    overlay, heatmap = create_heatmap_overlay(original_img, cam, alpha=0.6)

    # Create figure with subplots
    fig, axes = plt.subplots(2, 2, figsize=(12, 12))

    # Original image
    axes[0, 0].imshow(original_img)
    axes[0, 0].set_title(f'Original Hand Image')
    axes[0, 0].axis('off')

    # Heatmap only
    axes[0, 1].imshow(cam, cmap='jet')
    axes[0, 1].set_title(f'Grad-CAM Heatmap')
    axes[0, 1].axis('off')

    # Overlay
    axes[1, 0].imshow(overlay)
    axes[1, 0].set_title(f'Grad-CAM Overlay\nPredicted Age: {predicted_age:.1f} years')
    axes[1, 0].axis('off')

    # Feature importance explanation (using simple text instead of emojis)
    axes[1, 1].text(0.1, 0.8, 'Grad-CAM Interpretation:', fontsize=14, fontweight='bold')
    axes[1, 1].text(0.1, 0.7, '• Red areas: Features that increase predicted age', fontsize=12)
    axes[1, 1].text(0.1, 0.6, '• Blue areas: Features that decrease predicted age', fontsize=12)
    axes[1, 1].text(0.1, 0.5, '• Bright areas: High importance for prediction', fontsize=12)
    axes[1, 1].text(0.1, 0.4, '• Dark areas: Low importance for prediction', fontsize=12)

    axes[1, 1].text(0.1, 0.2, f'Model focuses on:', fontsize=12, fontweight='bold')
    axes[1, 1].text(0.1, 0.1, '• Hand creases and wrinkles\n• Knuckle definition\n• Skin texture\n• Vein visibility', fontsize=10)
    axes[1, 1].set_xlim(0, 1)
    axes[1, 1].set_ylim(0, 1)
    axes[1, 1].axis('off')

    plt.tight_layout()

    # Auto-generate save path if not provided
    if save_path is None:
        image_name = Path(image_path).stem
        save_path = f"gradcam_{image_name}.png"

    # Always save to disk
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"Visualization saved to {save_path}")
    plt.close()  # Close to free memory

    return predicted_age, cam, overlay

def batch_visualize_hands(hand_dir="Hands", model_path='best_model.pth', output_dir="gradcam_outputs", num_samples=5):
    """Generate Grad-CAM for multiple hand images"""

    Path(output_dir).mkdir(exist_ok=True)

    # Get random sample of hand images
    hand_images = list(Path(hand_dir).glob("*.jpg"))[:num_samples]

    print(f"Generating Grad-CAM visualizations for {len(hand_images)} images...")

    for i, img_path in enumerate(hand_images):
        print(f"\nProcessing {img_path.name} ({i+1}/{len(hand_images)})...")

        output_path = Path(output_dir) / f"gradcam_{img_path.stem}.png"

        try:
            visualize_age_prediction(
                image_path=str(img_path),
                model_path=model_path,
                save_path=str(output_path)
            )
        except Exception as e:
            print(f"Error processing {img_path.name}: {e}")
            continue

    print(f"\nGrad-CAM visualizations saved to {output_dir}/")

def main():
    if len(sys.argv) < 2:
        print("Usage:")
        print("  python visualize_gradcam.py <image_path>                    # Single image")
        print("  python visualize_gradcam.py --batch                        # Sample from training data")
        print("  python visualize_gradcam.py --batch --model balanced_model.pth  # Use specific model")
        sys.exit(1)

    if sys.argv[1] == '--batch':
        # Batch mode
        model_path = 'best_model.pth'
        if len(sys.argv) > 2 and sys.argv[2] == '--model':
            model_path = sys.argv[3]

        batch_visualize_hands(model_path=model_path)

    else:
        # Single image mode
        image_path = sys.argv[1]
        model_path = 'best_model.pth' if len(sys.argv) <= 2 else sys.argv[2]

        if not Path(image_path).exists():
            print(f"Image not found: {image_path}")
            sys.exit(1)

        print(f"Generating Grad-CAM for {image_path}...")
        visualize_age_prediction(image_path, model_path)

if __name__ == "__main__":
    main()