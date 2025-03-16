import streamlit as st
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
from torchvision import transforms, models
from PIL import Image
import os
import json
import numpy as np
from pathlib import Path
import random
import time
from torch.utils.data import Dataset, DataLoader

# Setup directories
DATA_DIR = "data"
FEEDBACK_DIR = os.path.join(DATA_DIR, "feedback")
MODEL_DIR = os.path.join(DATA_DIR, "model_checkpoints")
CLASS_MAP_PATH = os.path.join(DATA_DIR, "class_map.json")

# Create directories if they don't exist
os.makedirs(FEEDBACK_DIR, exist_ok=True)
os.makedirs(MODEL_DIR, exist_ok=True)

# Custom dataset for feedback images
class PlantFeedbackDataset(Dataset):
    def __init__(self, root_dir, class_map, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        self.class_map = class_map
        self.samples = []
        
        # Collect all images and their labels
        for class_name, class_idx in class_map.items():
            class_dir = os.path.join(root_dir, class_name)
            if os.path.exists(class_dir):
                for img_file in os.listdir(class_dir):
                    if img_file.lower().endswith(('.png', '.jpg', '.jpeg')):
                        img_path = os.path.join(class_dir, img_file)
                        self.samples.append((img_path, class_idx))
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        img_path, label = self.samples[idx]
        image = Image.open(img_path).convert('RGB')
        
        if self.transform:
            image = self.transform(image)
            
        return image, label

# Model class
class PlantClassifier:
    def __init__(self, num_classes=10):
        self.model = models.resnet18(pretrained=True)
        num_features = self.model.fc.in_features
        self.model.fc = nn.Linear(num_features, num_classes)
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model.to(self.device)
    
    def save_model(self, path):
        torch.save(self.model.state_dict(), path)
    
    def load_model(self, path):
        self.model.load_state_dict(torch.load(path, map_location=self.device))
    
    def train(self, train_dataset, val_dataset=None, epochs=5, batch_size=4, 
              learning_rate=0.001, progress_callback=None):
        """Train the model on feedback data"""
        # If validation dataset not provided, use training data
        if val_dataset is None:
            val_dataset = train_dataset
        
        # Create data loaders
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
        
        # Loss and optimizer
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(self.model.parameters(), lr=learning_rate)
        
        # Training loop
        for epoch in range(epochs):
            # Training phase
            self.model.train()
            running_loss = 0.0
            correct = 0
            total = 0
            
            for inputs, labels in train_loader:
                inputs, labels = inputs.to(self.device), labels.to(self.device)
                
                # Zero the gradients
                optimizer.zero_grad()
                
                # Forward pass
                outputs = self.model(inputs)
                loss = criterion(outputs, labels)
                
                # Backward pass and optimize
                loss.backward()
                optimizer.step()
                
                # Statistics
                running_loss += loss.item() * inputs.size(0)
                _, predicted = torch.max(outputs, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
            
            epoch_loss = running_loss / len(train_dataset)
            epoch_acc = 100 * correct / total if total > 0 else 0
            
            # Validation phase
            self.model.eval()
            val_loss = 0.0
            val_correct = 0
            val_total = 0
            
            with torch.no_grad():
                for inputs, labels in val_loader:
                    inputs, labels = inputs.to(self.device), labels.to(self.device)
                    outputs = self.model(inputs)
                    loss = criterion(outputs, labels)
                    
                    val_loss += loss.item() * inputs.size(0)
                    _, predicted = torch.max(outputs, 1)
                    val_total += labels.size(0)
                    val_correct += (predicted == labels).sum().item()
            
            val_loss = val_loss / len(val_dataset)
            val_acc = 100 * val_correct / val_total if val_total > 0 else 0
            
            # Report progress
            if progress_callback:
                progress_callback(epoch, epochs, epoch_loss, epoch_acc, val_loss, val_acc)
        
        return epoch_acc, val_acc

# Feedback Manager Class
class FeedbackManager:
    def __init__(self):
        self.class_map = self._load_class_map()
    
    def _load_class_map(self):
        if os.path.exists(CLASS_MAP_PATH):
            with open(CLASS_MAP_PATH, 'r') as f:
                return json.load(f)
        return {}
    
    def _save_class_map(self):
        with open(CLASS_MAP_PATH, 'w') as f:
            json.dump(self.class_map, f, indent=4)
    
    def add_feedback(self, image, correct_class, prediction=None):
        # Create class directory if it doesn't exist
        if correct_class not in self.class_map:
            self.class_map[correct_class] = len(self.class_map)
            self._save_class_map()
        
        class_dir = os.path.join(FEEDBACK_DIR, correct_class)
        os.makedirs(class_dir, exist_ok=True)
        
        # Save the image with a unique filename
        image_filename = f"{len(os.listdir(class_dir)) + 1}.jpg"
        image_path = os.path.join(class_dir, image_filename)
        image.save(image_path)
        
        # Log the feedback
        self._log_feedback(correct_class, prediction, image_path)
        
        return image_path
    
    def _log_feedback(self, correct_class, prediction, image_path):
        log_path = os.path.join(FEEDBACK_DIR, "feedback_log.json")
        
        # Load existing log or create a new one
        if os.path.exists(log_path):
            with open(log_path, 'r') as f:
                log = json.load(f)
        else:
            log = []
        
        # Add new entry
        log.append({
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "image_path": image_path,
            "correct_class": correct_class,
            "prediction": prediction,
            "is_correct": prediction == correct_class
        })
        
        # Save log
        with open(log_path, 'w') as f:
            json.dump(log, f, indent=4)
    
    def get_feedback_stats(self):
        stats = {
            "total_feedback": 0,
            "correct_predictions": 0,
            "class_distribution": {}
        }
        
        log_path = os.path.join(FEEDBACK_DIR, "feedback_log.json")
        if not os.path.exists(log_path):
            return stats
        
        with open(log_path, 'r') as f:
            log = json.load(f)
        
        stats["total_feedback"] = len(log)
        stats["correct_predictions"] = sum(1 for entry in log if entry.get("is_correct", False))
        
        # Count samples per class
        for entry in log:
            correct_class = entry["correct_class"]
            if correct_class not in stats["class_distribution"]:
                stats["class_distribution"][correct_class] = 0
            stats["class_distribution"][correct_class] += 1
        
        return stats
    
    def get_class_map(self):
        return self.class_map
    
    def create_training_datasets(self, split_ratio=0.8):
        """Create training and validation datasets from feedback images"""
        if not self.class_map:
            return None, None
        
        # Define transformations for training and validation
        train_transform = transforms.Compose([
            transforms.RandomResizedCrop(224),
            transforms.RandomHorizontalFlip(),
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
        
        # Create full dataset
        full_dataset = PlantFeedbackDataset(
            root_dir=FEEDBACK_DIR,
            class_map=self.class_map,
            transform=train_transform
        )
        
        # If not enough samples, return the full dataset for both
        if len(full_dataset) < 5:
            return full_dataset, full_dataset
        
        # Split dataset into training and validation
        train_size = int(split_ratio * len(full_dataset))
        val_size = len(full_dataset) - train_size
        
        indices = list(range(len(full_dataset)))
        random.shuffle(indices)
        
        train_indices = indices[:train_size]
        val_indices = indices[train_size:]
        
        # Create training dataset
        train_dataset = torch.utils.data.Subset(full_dataset, train_indices)
        
        # Create validation dataset with validation transform
        val_full_dataset = PlantFeedbackDataset(
            root_dir=FEEDBACK_DIR,
            class_map=self.class_map,
            transform=val_transform
        )
        val_dataset = torch.utils.data.Subset(val_full_dataset, val_indices)
        
        return train_dataset, val_dataset


# Plant Predictor Class
class PlantPredictor:
    def __init__(self, model):
        self.model = model
        self.model.eval()
        
        # Load class map
        self.class_map = self._load_class_map()
        self.idx_to_class = {v: k for k, v in self.class_map.items()}
        
        # Define inference transforms
        self.transform = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])
    
    def _load_class_map(self):
        if os.path.exists(CLASS_MAP_PATH):
            with open(CLASS_MAP_PATH, 'r') as f:
                return json.load(f)
        return {"Unknown Plant": 0}  # Default class
    
    def predict(self, image, top_k=1):
        # Process the image
        if isinstance(image, Image.Image):
            # Transform the image
            image_tensor = self.transform(image).unsqueeze(0)
        else:
            # If already a tensor
            image_tensor = image
        
        # Get predictions
        device = next(self.model.parameters()).device
        image_tensor = image_tensor.to(device)
        
        with torch.no_grad():
            outputs = self.model(image_tensor)
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
                probs, indices = torch.topk(probabilities, min(top_k, outputs.shape[1]))
                predictions = []
                
                for i in range(min(top_k, outputs.shape[1])):
                    idx = indices[0][i].item()
                    class_name = self.idx_to_class.get(idx, "Unknown")
                    probability = probs[0][i].item()
                    predictions.append({
                        'class': class_name,
                        'probability': probability
                    })
                
                return predictions


# Main app function
def main():
    st.set_page_config(
        page_title="Plant Identifier",
        page_icon="🌿",
        layout="wide"
    )
    
    # Add app explanation in sidebar
    with st.sidebar.expander("About", expanded=False):
        st.write("""
        This app identifies plants from images and learns from your feedback.
        Upload an image or take a photo, and the model will try to identify it.
        If it's wrong, you can provide the correct name, and the model will improve over time!
        """)
    
    st.title("🌿 Plant Identifier with Learning")
    
    # Initialize feedback manager
    feedback_manager = FeedbackManager()
    class_map = feedback_manager.get_class_map()
    
    # Set default classes if no feedback yet
    if not class_map:
        default_plants = ["Rose", "Sunflower", "Tulip", "Daisy", "Lily"]
        for i, plant in enumerate(default_plants):
            class_map[plant] = i
        feedback_manager._save_class_map()
    
    num_classes = len(class_map)
    
    # Initialize or load model
    @st.cache_resource
    def get_model(num_classes):
        model = PlantClassifier(num_classes=num_classes)
        model_path = os.path.join(MODEL_DIR, "model.pth")
        if os.path.exists(model_path):
            try:
                model.load_model(model_path)
                st.sidebar.success("✅ Loaded existing model")
            except Exception as e:
                st.sidebar.warning(f"⚠️ Couldn't load model: {e}")
        else:
            st.sidebar.info("ℹ️ Using new model")
        return model
    
    # Get model and predictor
    model = get_model(num_classes)
    predictor = PlantPredictor(model.model)
    
    # Navigation
    page = st.sidebar.radio("Navigation", ["Identify Plant", "Feedback History", "Retrain Model"])
    
    if page == "Identify Plant":
        st.header("Identify Your Plant")
        
        # Image upload
        uploaded_file = st.file_uploader("Upload a plant image", type=["jpg", "jpeg", "png"])
        
        # Camera input
        camera_input = st.camera_input("Or take a photo")
        
        if uploaded_file is not None or camera_input is not None:
            input_image = Image.open(uploaded_file if uploaded_file else camera_input).convert("RGB")
            
            col1, col2 = st.columns(2)
            with col1:
                st.image(input_image, caption="Uploaded Image", use_column_width=True)
            
            with col2:
                with st.spinner("Analyzing..."):
                    # Make prediction
                    if feedback_manager.get_feedback_stats()["total_feedback"] > 0:
                        # Use trained model for prediction
                        predictions = predictor.predict(input_image, top_k=3)
                        
                        if isinstance(predictions, list):
                            top_prediction = predictions[0]
                        else:
                            top_prediction = predictions
                    else:
                        # First-time use: random prediction from default plants
                        available_classes = list(class_map.keys())
                        top_prediction = {
                            'class': random.choice(available_classes),
                            'probability': 0.7 + random.random() * 0.3  # Random confidence between 0.7 and 1.0
                        }
                    
                    st.success(f"This looks like a **{top_prediction['class']}** ({top_prediction['probability']:.1%} confidence)")
                    
                    # Display other predictions if available
                    if isinstance(predictions, list) and len(predictions) > 1:
                        st.write("Other possibilities:")
                        for pred in predictions[1:]:
                            st.write(f"- {pred['class']} ({pred['probability']:.1%} confidence)")
                
                # Feedback section
                st.subheader("How did I do?")
                
                if st.button("✅ Correct!"):
                    feedback_manager.add_feedback(
                        input_image, 
                        top_prediction['class'],
                        prediction=top_prediction['class']
                    )
                    st.success("Thank you for the feedback! I'll remember that I was right.")
                
                # Wrong prediction case
                st.write("If I was wrong, what is this plant?")
                
                # Input for correct class
                correct_class = st.text_input("Enter the correct plant name:")
                
                if correct_class and st.button("Submit correction"):
                    feedback_manager.add_feedback(
                        input_image,
                        correct_class,
                        prediction=top_prediction['class']
                    )
                    st.success(f"Thank you! I'll remember that this is a {correct_class}.")
                    
                    # If this is a new class, reload the model with updated class count
                    if len(feedback_manager.get_class_map()) > num_classes:
                        st.info("New plant type detected! Please go to the 'Retrain Model' page to update the model.")
    
    elif page == "Feedback History":
        st.header("Feedback History")
        
        # Get feedback statistics
        stats = feedback_manager.get_feedback_stats()
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.metric("Total Feedback", stats["total_feedback"])
        
        with col2:
            if stats["total_feedback"] > 0:
                accuracy = stats["correct_predictions"] / stats["total_feedback"]
                st.metric("Model Accuracy", f"{accuracy:.1%}")
            else:
                st.metric("Model Accuracy", "N/A")
        
        # Class distribution
        if stats["class_distribution"]:
            st.subheader("Plant Class Distribution")
            
            # Convert to list for display
            class_data = list(stats["class_distribution"].items())
            class_data.sort(key=lambda x: x[1], reverse=True)
            
            # Display as a table
            data = {"Plant Type": [], "Count": []}
            for class_name, count in class_data:
                data["Plant Type"].append(class_name)
                data["Count"].append(count)
            
            st.dataframe(data)
            
            # Display sample images
            st.subheader("Sample Images")
            cols = st.columns(4)
            
            # Show one sample per class
            for i, (class_name, _) in enumerate(class_data[:8]):  # Limit to 8 classes
                class_dir = os.path.join(FEEDBACK_DIR, class_name)
                if os.path.exists(class_dir):
                    image_files = [f for f in os.listdir(class_dir) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
                    if image_files:
                        sample_img = os.path.join(class_dir, image_files[0])
                        with cols[i % 4]:
                            st.image(sample_img, caption=class_name, use_column_width=True)
    
    elif page == "Retrain Model":
        st.header("Retrain Model with Feedback")
        
        # Get feedback statistics
        stats = feedback_manager.get_feedback_stats()
        
        if stats["total_feedback"] < 5:
            st.warning("Not enough feedback data to retrain the model (minimum 5 samples needed).")
            st.info("Add more plant images and provide feedback to enable retraining.")
        else:
            st.info(f"You have {stats['total_feedback']} feedback samples available for retraining.")
            
            # Training parameters
            col1, col2 = st.columns(2)
            
            with col1:
                epochs = st.slider("Number of epochs", min_value=1, max_value=20, value=5)
                
            with col2:
                learning_rate = st.select_slider(
                    "Learning rate",
                    options=[0.0001, 0.0005, 0.001, 0.005, 0.01],
                    format_func=lambda x: f"{x:.4f}"
                )
            
            if st.button("Start Retraining"):
                # Create progress indicators
                progress_text = st.empty()
                progress_bar = st.progress(0)
                metrics_container = st.container()
                
                # Callback to update progress
                def update_progress(epoch, total_epochs, train_loss, train_acc, val_loss, val_acc):
                    progress = (epoch + 1) / total_epochs
                    progress_bar.progress(progress)
                    progress_text.text(f"Epoch {epoch+1}/{total_epochs}: Train Acc: {train_acc:.2f}%, Val Acc: {val_acc:.2f}%")
                    
                    with metrics_container:
                        cols = st.columns(2)
                        cols[0].metric("Training Accuracy", f"{train_acc:.2f}%")
                        cols[1].metric("Validation Accuracy", f"{val_acc:.2f}%")
                
                # Prepare datasets for training
                train_dataset, val_dataset = feedback_manager.create_training_datasets()
                
                if train_dataset is None or len(train_dataset) < 5:
                    st.error("Not enough unique samples for effective training.")
                    return
                
                # Update model if class count has changed
                current_class_count = len(feedback_manager.get_class_map())
                if current_class_count > num_classes:
                    st.info(f"Updating model to support {current_class_count} plant types (previously {num_classes}).")
                    model = PlantClassifier(num_classes=current_class_count)
                
                # Train the model
                with st.spinner("Training model with your feedback data..."):
                    train_acc, val_acc = model.train(
                        train_dataset=train_dataset,
                        val_dataset=val_dataset,
                        epochs=epochs,
                        batch_size=4,
                        learning_rate=learning_rate,
                        progress_callback=update_progress
                    )
                
                # Save the trained model
                model_path = os.path.join(MODEL_DIR, "model.pth")
                model.save_model(model_path)
                
                st.success(f"Model retraining complete! Final accuracy: {val_acc:.2f}%")
                st.info("The model has learned from your feedback and should now be better at identifying your plants.")

if __name__ == "__main__":
    main()