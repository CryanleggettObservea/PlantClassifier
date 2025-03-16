import torch
import torchvision.transforms as transforms
from PIL import Image
import os
import json

class PlantPredictor:
    def __init__(self, model, class_map_path="data/class_map.json", device='cuda'):
        """Initialize the plant predictor."""
        self.model = model
        self.device = device if torch.cuda.is_available() else 'cpu'
        self.model.to(self.device)
        self.model.eval()
        
        # Load class map
        self.class_map = self._load_class_map(class_map_path)
        self.idx_to_class = {v: k for k, v in self.class_map.items()}
        
        # Define inference transforms
        self.transform = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])
    
    def _load_class_map(self, class_map_path):
        """Load class map from file."""
        if os.path.exists(class_map_path):
            with open(class_map_path, 'r') as f:
                return json.load(f)
        return {}
    
    def predict(self, image, top_k=1):
        """Predict the plant class from an image."""
        # Open the image if it's a path
        if isinstance(image, str):
            image = Image.open(image).convert('RGB')
        
        # Transform the image
        if not isinstance(image, torch.Tensor):
            image = self.transform(image)
        
        # Add batch dimension if needed
        if len(image.shape) == 3:
            image = image.unsqueeze(0)
        
        # Move to device
        image = image.to(self.device)
        
        # Get predictions
        with torch.no_grad():
            outputs = self.model(image)
            probabilities = torch.nn.functional.softmax(outputs, dim=1)
            
            # Get top-k predictions
            if top_k == 1:
                _, predicted_idx = torch.max(outputs, 1)
                predicted_idx = predicted_idx.item()
                probability = probabilities[0][predicted_idx].item()
                
                class_name = self.idx_to_class.get(predicted_idx, "Unknown")
                return {
                    'class': class_name,
                    'probability': probability
                }
            else:
                # Get top-k predictions
                probs, indices = torch.topk(probabilities, top_k)
                predictions = []
                
                for i in range(top_k):
                    idx = indices[0][i].item()
                    class_name = self.idx_to_class.get(idx, "Unknown")
                    probability = probs[0][i].item()
                    predictions.append({
                        'class': class_name,
                        'probability': probability
                    })
                
                return predictions