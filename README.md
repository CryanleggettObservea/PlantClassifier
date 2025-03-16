# Plant Identifier with Learning

A PyTorch-based application that identifies plants from images and improves over time based on user feedback.

## Features

- 🌱 **Plant Identification**: Upload images or take photos to identify plant species
- 🔄 **Continuous Learning**: The model improves as you provide feedback
- 📊 **Performance Tracking**: Monitor how the model's accuracy improves over time
- 🧠 **Adaptive Model**: Automatically handles new plant species added through feedback
- 🖼️ **Image Management**: Organizes your plant images by species

## How It Works

1. **Upload an image** of a plant
2. **Get a prediction** from the model
3. **Provide feedback** on whether the prediction was correct
4. **Retrain the model** to improve future predictions

The application uses transfer learning with a ResNet18 backbone, fine-tuned on your specific plant collection. The more feedback you provide, the more accurate the model becomes for your specific plants.

## Installation

### Prerequisites

- Python 3.7+ 
- PyTorch 1.9+
- Streamlit 1.0+

### Setup

1. Clone the repository:
   ```bash
   git clone https://github.com/yourusername/plant-identifier.git
   cd plant-identifier
   ```

2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

3. Run the application:
   ```bash
   python -m streamlit run plant_identifier.py
   ```

## Usage Guide

### Identifying Plants

1. Navigate to the "Identify Plant" page
2. Upload an image or take a photo with your camera
3. Review the identification results
4. Confirm if the prediction is correct or provide the correct plant name

### Viewing Feedback History

1. Navigate to the "Feedback History" page
2. View statistics about model performance
3. See distribution of plant species in your collection
4. Browse sample images from your feedback data

### Retraining the Model

1. Navigate to the "Retrain Model" page
2. Set training parameters (epochs and learning rate)
3. Click "Start Retraining" to update the model with your feedback
4. Monitor training progress to see how the model improves

## Training Recommendations

For optimal learning performance:

- Provide at least 5-10 images per plant type
- Include diverse angles, lighting conditions, and backgrounds
- Retrain with 5-15 epochs depending on dataset size
- Use a learning rate of 0.001 for balanced learning

## Project Structure

```
plant_identifier/
├── plant_identifier.py     # Main application file
├── requirements.txt        # Dependencies
└── data/                   # Data directory (created on first run)
    ├── feedback/           # User feedback images
    │   ├── Rose/           # Images organized by plant type
    │   ├── Sunflower/
    │   └── feedback_log.json  # Feedback tracking
    ├── model_checkpoints/  # Saved model versions
    └── class_map.json      # Mapping of plant names to indices
```

## How It's Built

- **Frontend**: Streamlit for the user interface
- **Backend**: PyTorch for the deep learning model
- **Model Architecture**: ResNet18 with custom classification head
- **Training**: Transfer learning with Adam optimizer

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgements

- PyTorch team for the deep learning framework
- Streamlit team for the interactive app framework
- ResNet architecture for the pre-trained model backbone
