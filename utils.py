import os
import torch
import json
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import torchvision.transforms as transforms
import cv2

def preprocess_image(image_path, target_size=(224, 224)):
    """Preprocess an image for display."""
    image = Image.open(image_path).convert('RGB')
    image = image.resize(target_size)
    return np.array(image)

def display_prediction(image, prediction):
    """Display an image with its prediction."""
    plt.figure(figsize=(8, 6))
    plt.imshow(image)
    plt.axis('off')
    
    if isinstance(prediction, list):
        # Multiple predictions
        title = "\n".join([f"{p['class']}: {p['probability']:.2%}" for p in prediction[:3]])
    else:
        # Single prediction
        title = f"{prediction['class']}: {prediction['probability']:.2%}"
    
    plt.title(title)
    plt.tight_layout()
    plt.show()

def get_device():
    """Get the appropriate device (CPU or CUDA)."""
    return torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def save_json(data, path):
    """Save data to a JSON file."""
    with open(path, 'w') as f:
        json.dump(data, f, indent=4)

def load_json(path):
    """Load data from a JSON file."""
    if os.path.exists(path):
        with open(path, 'r') as f:
            return json.load(f)
    return None

def create_directory_structure():
    """Create the necessary directory structure for the application."""
    directories = [
        'data/initial_dataset',
        'data/feedback',
        'data/model_checkpoints'
    ]
    
    for directory in directories:
        os.makedirs(directory, exist_ok=True)
        
    print("Directory structure created successfully.")

def capture_image_from_webcam():
    """Capture an image from the webcam."""
    # Initialize the webcam
    cap = cv2.VideoCapture(0)
    
    if not cap.isOpened():
        print("Error: Could not open webcam.")
        return None
    
    # Capture frame
    ret, frame = cap.read()
    
    # Release webcam
    cap.release()
    
    if not ret:
        print("Error: Could not capture frame.")
        return None
    
    # Convert from BGR to RGB
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    
    # Convert to PIL Image
    image = Image.fromarray(frame_rgb)
    
    return image