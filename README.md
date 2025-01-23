# Multimodal Emotion Recognition System

## Overview
A comprehensive emotion recognition system that combines predictions from image, text, and speech modalities to provide robust emotion analysis.

## Features
- Face emotion detection from images
- Text sentiment and emotion classification
- Speech emotion recognition
- Multi-modal emotion prediction with confidence scoring
- Visualization of emotion prediction results

## Requirements
- Python 3.8+
- Dependencies:
  - numpy
  - librosa
  - opencv-python
  - torch
  - keras
  - matplotlib
  - PIL
  - flask

## Installation
```bash
git clone https://github.com/KalyaniWaghchaure/multimodal-emotion-recognition.git
cd multimodal-emotion-recognition
pip install -r requirements.txt
```

## Usage
```python
from emotion_recognizer import MultimodalEmotionRecognizer

# Initialize recognizer with pre-trained models
recognizer = MultimodalEmotionRecognizer(
    image_model=model1, 
    text_model=model2, 
    speech_model=model3,
    # Other required components
)

# Process different modalities
image_pred = recognizer.process_image('image.jpg')
text_pred = recognizer.process_text('I am feeling happy today')
speech_pred = recognizer.process_speech('audio.wav')

# Combine predictions
result = recognizer.combine_predictions([image_pred, text_pred, speech_pred])
```

## Emotion Categories
- Anger
- Disgust
- Fear
- Joy
- Sadness
- Neutral
- Surprise

## Web Application
The system includes a Flask route for multimodal emotion analysis, supporting image, text, and audio uploads.

## Contributing
1. Fork the repository
2. Create a feature branch
3. Commit your changes
4. Push to the branch
5. Create a Pull Request

## Acknowledgements
- Pre-trained models used for emotion recognition
- Libraries and frameworks that made this project possible
