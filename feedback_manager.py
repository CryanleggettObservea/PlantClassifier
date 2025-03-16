import os
import shutil
import json
from PIL import Image
import uuid

class FeedbackManager:
    def __init__(self, feedback_dir="data/feedback", class_map_path="data/class_map.json"):
        """Initialize the feedback manager."""
        self.feedback_dir = feedback_dir
        self.class_map_path = class_map_path
        
        # Make sure feedback directory exists
        os.makedirs(feedback_dir, exist_ok=True)
        
        # Load or create class map
        self.class_map = self._load_class_map()
    
    def _load_class_map(self):
        """Load class map from file or create a new one."""
        if os.path.exists(self.class_map_path):
            with open(self.class_map_path, 'r') as f:
                return json.load(f)
        return {}
    
    def _save_class_map(self):
        """Save class map to file."""
        with open(self.class_map_path, 'w') as f:
            json.dump(self.class_map, f, indent=4)
    
    def add_feedback(self, image, correct_class, prediction=None):
        """Add a user feedback entry."""
        # Create class directory if it doesn't exist
        if correct_class not in self.class_map:
            self.class_map[correct_class] = len(self.class_map)
            self._save_class_map()
        
        class_dir = os.path.join(self.feedback_dir, correct_class)
        os.makedirs(class_dir, exist_ok=True)
        
        # Generate a unique filename
        image_filename = f"{uuid.uuid4()}.jpg"
        image_path = os.path.join(class_dir, image_filename)
        
        # Save the image
        if isinstance(image, str) and os.path.exists(image):
            # If image is a path to an existing file
            shutil.copy(image, image_path)
        else:
            # If image is a PIL Image or similar
            if not isinstance(image, Image.Image):
                # Convert to PIL Image if needed
                image = Image.fromarray(image)
            image.save(image_path)
        
        # Log the feedback
        self._log_feedback(correct_class, prediction, image_path)
        
        return image_path
    
    def _log_feedback(self, correct_class, prediction, image_path):
        """Log feedback details for future analysis."""
        log_path = os.path.join(self.feedback_dir, "feedback_log.json")
        
        # Load existing log or create a new one
        if os.path.exists(log_path):
            with open(log_path, 'r') as f:
                log = json.load(f)
        else:
            log = []
        
        # Add new entry
        log.append({
            "timestamp": str(uuid.uuid4()),  # Using UUID as timestamp for simplicity
            "image_path": image_path,
            "correct_class": correct_class,
            "prediction": prediction,
            "is_correct": prediction == correct_class
        })
        
        # Save log
        with open(log_path, 'w') as f:
            json.dump(log, f, indent=4)
    
    def get_feedback_stats(self):
        """Get statistics about collected feedback."""
        stats = {
            "total_feedback": 0,
            "correct_predictions": 0,
            "class_distribution": {}
        }
        
        log_path = os.path.join(self.feedback_dir, "feedback_log.json")
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
        """Get the current class map."""
        return self.class_map
    
    def get_feedback_data(self):
        """Get paths to feedback images for retraining."""
        return self.feedback_dir