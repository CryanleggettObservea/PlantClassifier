import torch
import torch.nn as nn
import torchvision.models as models

class PlantClassifier(nn.Module):
    def __init__(self, num_classes, use_pretrained=True):
        super(PlantClassifier, self).__init__()
        # Use a pre-trained ResNet model as our base
        self.model = models.resnet18(pretrained=use_pretrained)
        
        # Replace the final fully connected layer
        num_features = self.model.fc.in_features
        self.model.fc = nn.Sequential(
            nn.Linear(num_features, 256),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(256, num_classes)
        )
        
    def forward(self, x):
        return self.model(x)
    
    def save_model(self, path):
        torch.save({
            'model_state_dict': self.state_dict(),
        }, path)
        
    def load_model(self, path):
        checkpoint = torch.load(path)
        self.load_state_dict(checkpoint['model_state_dict'])