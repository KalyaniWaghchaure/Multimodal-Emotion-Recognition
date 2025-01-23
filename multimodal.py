from typing import Dict, List, Tuple
import numpy as np
from dataclasses import dataclass
import librosa
import cv2
from PIL import Image
import torch
from keras.preprocessing.sequence import pad_sequences
import base64
import io
import matplotlib.pyplot as plt


@dataclass
class EmotionPrediction:
    emotion: str
    confidence: float
    source: str


class MultimodalEmotionRecognizer:
    def __init__(self, image_model, text_model, speech_model, face_classifier,
                 tokenizer, speech_encoder, mtcnn, feature_extractor):
        self.image_model = image_model
        self.text_model = text_model
        self.speech_model = speech_model
        self.face_classifier = face_classifier
        self.tokenizer = tokenizer
        self.speech_encoder = speech_encoder
        self.mtcnn = mtcnn
        self.feature_extractor = feature_extractor

        # Unified emotion categories
        self.emotion_mapping = {
            'angry': 'anger',
            'disgust': 'disgust',
            'fear': 'fear',
            'happy': 'joy',
            'sad': 'sadness',
            'neutral': 'neutral',
            'surprise': 'surprise'
        }

    def clean_text(self, text: str) -> List[str]:
        # Reuse your existing text cleaning function
        text = re.sub(r"(#[\d\w\.]+)", '', text)
        text = re.sub(r"(@[\d\w\.]+)", '', text)
        return text.split()

    def process_image(self, image_file) -> EmotionPrediction:
        # Read and process image
        frame = cv2.imread(image_file)
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = self.face_classifier.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

        if len(faces) == 0:
            return EmotionPrediction('neutral', 0.0, 'image')

        max_confidence = 0
        final_emotion = 'neutral'

        for (x, y, w, h) in faces:
            roi_gray = gray[y:y + h, x:x + w]
            roi_gray = cv2.resize(roi_gray, (48, 48), interpolation=cv2.INTER_AREA)

            if np.sum([roi_gray]) != 0:
                roi = roi_gray.astype('float') / 255.0
                roi = img_to_array(roi)
                roi = np.expand_dims(roi, axis=0)

                prediction = self.image_model.predict(roi)[0]
                confidence = np.max(prediction)

                if confidence > max_confidence:
                    max_confidence = confidence
                    emotion_idx = np.argmax(prediction)
                    final_emotion = emotion_labels[emotion_idx]

        return EmotionPrediction(
            self.emotion_mapping.get(final_emotion.lower(), final_emotion),
            float(max_confidence),
            'image'
        )

    def process_text(self, text: str) -> EmotionPrediction:
        cleaned_text = [' '.join(self.clean_text(text))]
        sequence = self.tokenizer.texts_to_sequences(cleaned_text)
        padded = pad_sequences(sequence, maxlen=500)

        prediction_probs = self.text_model.predict(padded)[0]
        predicted_class_index = np.argmax(prediction_probs)
        confidence = float(prediction_probs[predicted_class_index])
        predicted_emotion = class_names[predicted_class_index]

        return EmotionPrediction(
            self.emotion_mapping.get(predicted_emotion.lower(), predicted_emotion),
            confidence,
            'text'
        )

    def process_speech(self, audio_path: str) -> EmotionPrediction:
        sample_audio, sample_rate = librosa.load(audio_path, duration=3, offset=0.5)
        sample_mfcc = np.mean(librosa.feature.mfcc(y=sample_audio, sr=sample_rate, n_mfcc=40).T, axis=0)
        sample_mfcc = np.expand_dims(sample_mfcc, axis=0)

        prediction = self.speech_model.predict(sample_mfcc)
        predicted_emotion_index = np.argmax(prediction, axis=-1)[0]
        confidence = float(np.max(prediction))

        if predicted_emotion_index < len(self.speech_encoder.categories_[0]):
            predicted_emotion = self.speech_encoder.categories_[0][predicted_emotion_index]
        else:
            predicted_emotion = "neutral"

        return EmotionPrediction(
            self.emotion_mapping.get(predicted_emotion.lower(), predicted_emotion),
            confidence,
            'speech'
        )

    def combine_predictions(self, predictions: List[EmotionPrediction]) -> Dict:
        # Weights for different modalities (can be adjusted)
        weights = {
            'image': 0.4,
            'text': 0.3,
            'speech': 0.3
        }

        # Initialize emotion scores
        emotion_scores = {}

        # Combine weighted predictions
        for pred in predictions:
            if pred.emotion not in emotion_scores:
                emotion_scores[pred.emotion] = 0
            emotion_scores[pred.emotion] += pred.confidence * weights[pred.source]

        # Get final emotion and confidence
        final_emotion = max(emotion_scores.items(), key=lambda x: x[1])

        return {
            'final_emotion': final_emotion[0],
            'confidence': final_emotion[1],
            'individual_predictions': [
                {'emotion': p.emotion, 'confidence': p.confidence, 'source': p.source}
                for p in predictions
            ]
        }

    def create_visualization(self, result: Dict) -> str:
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))

        # Plot individual predictions
        sources = [p['source'] for p in result['individual_predictions']]
        emotions = [p['emotion'] for p in result['individual_predictions']]
        confidences = [p['confidence'] for p in result['individual_predictions']]

        ax1.bar(sources, confidences)
        ax1.set_title('Confidence by Source')
        ax1.set_ylim(0, 1)

        # Plot final emotion
        ax2.bar(['Final Prediction'], [result['confidence']])
        ax2.set_title(f"Final Emotion: {result['final_emotion']}")
        ax2.set_ylim(0, 1)

        plt.tight_layout()

        # Convert plot to base64 string
        buf = io.BytesIO()
        plt.savefig(buf, format='png')
        buf.seek(0)
        plot_url = base64.b64encode(buf.getvalue()).decode('utf8')
        plt.close()

        return plot_url


# Flask route for multimodal analysis
@app.route('/multimodal', methods=['GET', 'POST'])
def multimodal_analysis():
    if 'user_id' not in session:
        flash('Please log in to use multimodal emotion recognition.', 'warning')
        return redirect(url_for('login'))

    if request.method == 'POST':
        recognizer = MultimodalEmotionRecognizer(
            image_model=classifier,
            text_model=model1,
            speech_model=model,
            face_classifier=face_classifier,
            tokenizer=tokenizer,
            speech_encoder=enc,
            mtcnn=mtcnn,
            feature_extractor=extractor
        )

        predictions = []

        # Process image if provided
        if 'image_file' in request.files:
            image_file = request.files['image_file']
            if image_file and image_file.filename:
                image_path = os.path.join(app.config['UPLOAD_FOLDER'], secure_filename(image_file.filename))
                image_file.save(image_path)
                image_pred = recognizer.process_image(image_path)
                predictions.append(image_pred)
                os.remove(image_path)

        # Process text if provided
        if 'text' in request.form and request.form['text'].strip():
            text_pred = recognizer.process_text(request.form['text'])
            predictions.append(text_pred)

        # Process audio if provided
        if 'audio_file' in request.files:
            audio_file = request.files['audio_file']
            if audio_file and audio_file.filename:
                audio_path = os.path.join(app.config['UPLOAD_FOLDER'], secure_filename(audio_file.filename))
                audio_file.save(audio_path)
                audio_pred = recognizer.process_speech(audio_path)
                predictions.append(audio_pred)
                os.remove(audio_path)

        if not predictions:
            flash('Please provide at least one input (image, text, or audio).', 'error')
            return redirect(request.url)

        # Combine predictions and create visualization
        result = recognizer.combine_predictions(predictions)
        plot_url = recognizer.create_visualization(result)

        # Get content suggestions based on final emotion
        suggestions = get_content_suggestions(result['final_emotion'])

        return render_template(
            'multimodal_result.html',
            result=result,
            plot_url=plot_url,
            suggestions=suggestions,
            logged_in=True
        )

    return render_template('multimodal.html', logged_in=True)