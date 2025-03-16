import os
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
import numpy as np

class PlantDataset(Dataset):
    def __init__(self, image_dir, transform=None, class_map=None):
        self.image_dir = image_dir
        self.transform = transform
        self.image_paths = []
        self.labels = []
        self.class_map = class_map or {}
        
        # If no class_map is provided, create one from the directory structure
        if not class_map:
            classes = [d for d in os.listdir(image_dir) if os.path.isdir(os.path.join(image_dir, d))]
            self.class_map = {cls: idx for idx, cls in enumerate(sorted(classes))}
        
        # Load image paths and labels
        for class_name, class_idx in self.class_map.items():
            class_dir = os.path.join(image_dir, class_name)
            if os.path.isdir(class_dir):
                for img_name in os.listdir(class_dir):
                    if img_name.lower().endswith(('.png', '.jpg', '.jpeg')):
                        self.image_paths.append(os.path.join(class_dir, img_name))
                        self.labels.append(class_idx)
    
    def __len__(self):
        return len(self.image_paths)
    
    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        label = self.labels[idx]
        
        # Load and transform image
        image = Image.open(img_path).convert('RGB')
        if self.transform:
            image = self.transform(image)
            
        return image, label

def get_transforms():
    # Define transformations for training and validation
    train_transform = transforms.Compose([
        transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(15),
        transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    
    val_transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    
    return train_transform, val_transform

def create_dataloaders(train_dir, val_dir, batch_size=32, class_map=None):
    train_transform, val_transform = get_transforms()
    
    train_dataset = PlantDataset(train_dir, transform=train_transform, class_map=class_map)
    val_dataset = PlantDataset(val_dir, transform=val_transform, class_map=class_map)
    
    # Create class_map if not provided
    if not class_map:
        class_map = train_dataset.class_map
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=4)
    
    return train_loader, val_loader, class_map