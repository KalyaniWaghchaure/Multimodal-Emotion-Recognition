from flask import Flask, request, render_template
import os
import librosa
import numpy as np
from tensorflow.keras.models import load_model
import pickle
import tensorflow as tf

app = Flask(__name__)

# Load the trained model
model = load_model('speech_emotion_model.h5')

# Load the OneHotEncoder
with open('encoder.pkl', 'rb') as f:
    enc = pickle.load(f)
# Home route
@app.route('/')
def index():
    return render_template('index1.html')


@app.route('/Audio', methods=['GET', 'POST'])
def Audio():
    if request.method == 'POST':
        # Get the uploaded audio file
        audio_file = request.files['audio_file']

        # Save the uploaded audio file
        audio_path = os.path.join('uploads', audio_file.filename)
        audio_file.save(audio_path)

        # Load and preprocess the audio
        sample_audio, sample_rate = librosa.load(audio_path, duration=3, offset=0.5)
        sample_mfcc = np.mean(librosa.feature.mfcc(y=sample_audio, sr=sample_rate, n_mfcc=40).T, axis=0)

        # Reshape to match the input shape expected by the model
        sample_mfcc = np.expand_dims(sample_mfcc, axis=0)  # (1, 40)

        # Print shapes for debugging
        print("Input shape for prediction:", sample_mfcc.shape)

        # Make the prediction
        prediction = model.predict(sample_mfcc)

        # Debugging: Print the prediction output
        print("Prediction output:", prediction)
        print("Prediction shape:", prediction.shape)

        # Find the predicted index and map it to the emotion label
        predicted_emotion_index = np.argmax(prediction, axis=-1)[0]

        # Print encoder categories for debugging
        print("Encoder categories:", enc.categories_)

        # Ensure the index is within the bounds of encoder categories
        if predicted_emotion_index < len(enc.categories_[0]):
            predicted_emotion = enc.categories_[0][predicted_emotion_index]
        else:
            predicted_emotion = "Unknown"  # Fallback in case of index mismatch

        return render_template('audio1.html', predicted_emotion=predicted_emotion)

    return render_template('audio.html')


if __name__ == '__main__':
    app.run(debug=True)

