"""
Explainable AI (XAI) using Grad-CAM for Facial Expression Recognition

This script visualizes which facial features (eyes, mouth, eyebrows) the
Custom 4-Block CNN focused on when predicting an emotion.

Usage:
    python explain.py --image path/to/face.jpg --model best_model_epochs30_...pth
"""

import os
import argparse
import numpy as np
from PIL import Image

import cv2
import torch
import torch.nn.functional as F
from torchvision import transforms

from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget

from dataset import EMOTION_DICT
from model import CustomCNN


def load_and_preprocess_image(image_path, img_size=112):
    """
    Loads an image from disk and prepares:
    1. rgb_img: float32 NumPy array in range [0, 1] for Grad-CAM overlay.
    2. input_tensor: Normalized PyTorch tensor of shape [1, 3, 112, 112].
    """
    if not os.path.exists(image_path):
        raise FileNotFoundError(f"Image not found at: '{image_path}'")

    # Load and resize image using PIL
    pil_img = Image.open(image_path).convert("RGB")
    pil_img = pil_img.resize((img_size, img_size))

    # 1. Prepare RGB image array for visualization (values in [0, 1])
    rgb_img = np.float32(pil_img) / 255.0

    # 2. Prepare PyTorch input tensor with ImageNet normalization
    preprocess = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    ])
    input_tensor = preprocess(pil_img).unsqueeze(0)  # Shape: [1, 3, 112, 112]

    return rgb_img, input_tensor


def apply_gradcam_overlay(rgb_img, cam_mask, threshold=0.15, alpha=0.6, colormap=cv2.COLORMAP_JET):
    """
    Overlays Grad-CAM heatmap onto rgb_img with thresholded alpha transparency.
    Regions with activation below threshold keep the original face's natural colors
    instead of being washed out with solid blue.
    """
    # 1. Apply OpenCV colormap to CAM mask (normalized in [0, 1])
    heatmap = cv2.applyColorMap(np.uint8(255 * cam_mask), colormap)
    heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)
    heatmap = np.float32(heatmap) / 255.0

    # 2. Dynamic alpha mask: 0 below threshold, transitions smoothly to alpha
    mask_weight = np.clip((cam_mask - threshold) / (1.0 - threshold + 1e-7), 0.0, 1.0)
    mask_weight = np.expand_dims(mask_weight, axis=2) * alpha

    # 3. Alpha blend: original face in inactive regions, heatmap in active hotspots
    overlay = (1.0 - mask_weight) * rgb_img + mask_weight * heatmap
    overlay = np.clip(overlay, 0.0, 1.0)
    return np.uint8(255 * overlay)


def explain_prediction(model, input_tensor, rgb_img, output_path="gradcam_result.png"):
    """
    Applies Grad-CAM to visualize model attention on the input image.
    Saves a side-by-side comparison: [Original Face | Grad-CAM Overlay].
    """
    model.eval()

    # 1. Run forward pass to get predicted emotion and confidence
    with torch.no_grad():
        logits = model(input_tensor)
        probs = F.softmax(logits, dim=1)
        pred_idx = torch.argmax(probs, dim=1).item()
        confidence = probs[0, pred_idx].item() * 100

    predicted_emotion = EMOTION_DICT.get(pred_idx, f"Class {pred_idx}")

    print("-" * 65)
    print("PREDICTION SUMMARY:")
    print(f"  * Predicted Emotion : {predicted_emotion.upper()}")
    print(f"  * Confidence Score  : {confidence:.2f}%")
    print(f"  * Target Layer      : model.block4 (Output of Block 4: 7x7x512)")
    print("-" * 65)

    # 2. Point to the completed Block 4 (post-ReLU, post-pooling 7x7x512 feature map)
    target_layers = [model.block4]

    # 3. Initialize Grad-CAM
    cam = GradCAM(model=model, target_layers=target_layers)

    # 4. Generate the heatmap for the predicted emotion
    targets = [ClassifierOutputTarget(pred_idx)]
    grayscale_cam = cam(input_tensor=input_tensor, targets=targets)
    cam_mask = grayscale_cam[0, :]

    # 5. Overlay heatmap with thresholded alpha mask (retaining natural skin colors)
    overlay = apply_gradcam_overlay(rgb_img, cam_mask, threshold=0.15, alpha=0.6)

    # 6. Create side-by-side comparison: [Original Face | Grad-CAM Overlay]
    orig_uint8 = np.uint8(255 * rgb_img)
    side_by_side = np.hstack([orig_uint8, overlay])

    # 7. Save the visualization
    result_img = Image.fromarray(side_by_side)
    result_img.save(output_path)
    print(f"[✓] Side-by-side Grad-CAM visualization saved to: '{output_path}'")
    print(f"    (Left: Original Face | Right: Heatmap Overlay with Natural Skin Colors)")
    print("-" * 65)


def parse_arguments():
    parser = argparse.ArgumentParser(
        description="Generate Grad-CAM Explainable AI visualization for a face image"
    )
    parser.add_argument(
        "--image",
        type=str,
        required=True,
        help="Path to the input face image (e.g. face.jpg)"
    )
    parser.add_argument(
        "--model",
        type=str,
        required=True,
        help="Path to the trained .pth checkpoint file"
    )
    parser.add_argument(
        "--output",
        type=str,
        default="gradcam_result.png",
        help="Path to save the output heatmap image (default: 'gradcam_result.png')"
    )
    return parser.parse_args()


def main():
    args = parse_arguments()

    # Verify model checkpoint exists
    if not os.path.exists(args.model):
        raise FileNotFoundError(f"Model checkpoint not found: '{args.model}'")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[+] Using device: {device}")
    print(f"[+] Loading model from '{args.model}'...")

    # Initialize model architecture and load weights
    model = CustomCNN(num_classes=7)
    model.load_state_dict(torch.load(args.model, map_location=device))
    model = model.to(device)

    # Load and preprocess input image
    print(f"[+] Processing image: '{args.image}'...")
    rgb_img, input_tensor = load_and_preprocess_image(args.image, img_size=112)
    input_tensor = input_tensor.to(device)

    # Generate and save Grad-CAM explanation
    explain_prediction(
        model=model,
        input_tensor=input_tensor,
        rgb_img=rgb_img,
        output_path=args.output
    )


if __name__ == "__main__":
    main()
