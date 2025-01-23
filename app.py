import os
import librosa
from flask import Flask, render_template, request, redirect, url_for, send_file, session, flash, Response, jsonify
import cv2
import numpy as np
import torch
from facenet_pytorch import MTCNN
from transformers import AutoFeatureExtractor, AutoModelForImageClassification, AutoConfig
from PIL import Image
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
import pandas as pd
import io
import base64
from flask_mysqldb import MySQL
from werkzeug.security import generate_password_hash, check_password_hash
from tensorflow.keras.models import load_model
from keras.preprocessing.image import img_to_array
import time
from werkzeug.utils import secure_filename
import pickle
from flask import Flask, render_template, request
from keras.models import load_model
from keras.preprocessing.sequence import pad_sequences
import numpy as np
import re
import nltk
import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt
import io
import base64
from matplotlib.figure import Figure


app = Flask(__name__, static_url_path='/static')

# MySQL Configuration
app.config['MYSQL_HOST'] = 'localhost'
app.config['MYSQL_USER'] = 'root'
app.config['MYSQL_PASSWORD'] = ''
app.config['MYSQL_DB'] = 'emotiondata'
mysql = MySQL(app)

app.secret_key = 'your_secret_key'

app.config['UPLOAD_FOLDER'] = 'uploads'
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max-limit

# Ensure upload folder exists
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

# Initialize models and configurations
device = torch.device('cpu')
mtcnn = MTCNN(image_size=160, margin=0, min_face_size=20, thresholds=[0.6, 0.7, 0.7], factor=0.709, post_process=True,
              device=device)
emotion_classifier = load_model(r'model.h5')
emotion_labels = ['Angry', 'Disgust', 'Fear', 'Happy', 'Neutral', 'Sad', 'Surprise']


extractor = AutoFeatureExtractor.from_pretrained("trpakov/vit-face-expression")
model = AutoModelForImageClassification.from_pretrained("trpakov/vit-face-expression")
config = AutoConfig.from_pretrained("trpakov/vit-face-expression")
id2label = config.id2label

# Load models for video interview
face_classifier = cv2.CascadeClassifier(r'haarcascade_frontalface_default.xml')
classifier = load_model(r'model.h5')



def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in {'mp4', 'avi', 'mov'}


def detect_face(frame):
    """Detect face in frame using MTCNN"""
    try:
        # Convert frame to RGB if it's not already
        if isinstance(frame, np.ndarray):
            frame = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))

        # Detect face using MTCNN
        boxes, _ = mtcnn.detect(frame)

        if boxes is not None and len(boxes) > 0:
            box = boxes[0]
            x1, y1, x2, y2 = [int(b) for b in box]
            return frame.crop((x1, y1, x2, y2))
        return None
    except Exception as e:
        print(f"Error in detect_face: {str(e)}")
        return None


def predict_emotion(face_image):
    """Predict emotion from face image"""
    try:
        if face_image is None:
            return None

        # Convert PIL Image to numpy array if necessary
        if isinstance(face_image, Image.Image):
            face_image = np.array(face_image)

        # Convert to grayscale
        gray_face = cv2.cvtColor(face_image, cv2.COLOR_RGB2GRAY)

        # Resize to 48x48 pixels (input size for emotion classifier)
        resized_face = cv2.resize(gray_face, (48, 48))

        # Normalize pixel values
        normalized_face = resized_face / 255.0

        # Reshape for model input (batch_size, height, width, channels)
        input_face = normalized_face.reshape(1, 48, 48, 1)

        # Predict emotions
        predictions = emotion_classifier.predict(input_face)

        # Convert predictions to dictionary
        emotion_probs = {emotion_labels[i]: float(pred) for i, pred in enumerate(predictions[0])}
        return emotion_probs

    except Exception as e:
        print(f"Error in predict_emotion: {str(e)}")
        return None


def process_video(video_path):
    try:
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise ValueError("Error opening video file")

        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        all_emotions = []
        frames_processed = 0

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            # Process every 5th frame
            if frames_processed % 5 == 0:
                # Detect face
                face = detect_face(frame)

                if face is not None:
                    # Predict emotions
                    emotions = predict_emotion(face)
                    if emotions is not None:
                        all_emotions.append(emotions)

            frames_processed += 1

        cap.release()

        if not all_emotions:
            raise ValueError("No faces detected in the video")

        # Calculate overall percentages
        df = pd.DataFrame(all_emotions)
        overall_percentages = df.mean() * 100
        dominant_emotion = overall_percentages.idxmax()

        return all_emotions, overall_percentages.to_dict(), dominant_emotion

    except Exception as e:
        print(f"Error in process_video: {str(e)}")
        return [], {}, "Neutral"


def create_emotion_plot(all_emotions):
    try:
        if not all_emotions:
            raise ValueError("No emotion data available for plotting")

        df = pd.DataFrame(all_emotions)
        df = df * 100

        plt.figure(figsize=(15, 8))
        colors = {
            'Angry': 'red',
            'Disgust': 'green',
            'Fear': 'gray',
            'Happy': 'yellow',
            'Neutral': 'purple',
            'Sad': 'blue',
            'Surprise': 'orange'
        }

        for emotion in emotion_labels:
            if emotion in df.columns:
                plt.plot(df[emotion], label=emotion, color=colors[emotion])

        plt.xlabel('Frame Number')
        plt.ylabel('Emotion Probability (%)')
        plt.title('Emotion Probabilities Over Time')
        plt.legend()
        plt.grid(True, alpha=0.3)

        img_buf = io.BytesIO()
        plt.savefig(img_buf, format='png')
        img_buf.seek(0)
        img_str = base64.b64encode(img_buf.getvalue()).decode()
        plt.close()

        return img_str

    except Exception as e:
        print(f"Error in create_emotion_plot: {str(e)}")
        return None


@app.route('/upload', methods=['GET', 'POST'])
def upload_file():
    if 'user_id' not in session:
        flash('Please log in to upload files or start a video interview.', 'warning')
        return redirect(url_for('login'))

    if request.method == 'POST':
        if 'file' not in request.files:
            flash('No file part', 'error')
            return redirect(request.url)

        file = request.files['file']
        if file.filename == '':
            flash('No selected file', 'error')
            return redirect(request.url)

        if file and allowed_file(file.filename):
            try:
                filename = secure_filename(file.filename)
                filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
                file.save(filepath)

                # Process video
                all_emotions, overall_percentages, dominant_emotion = process_video(filepath)

                if not all_emotions:
                    flash('No faces detected in the video or error processing video', 'error')
                    return redirect(request.url)

                # Create plot
                plot_img = create_emotion_plot(all_emotions)

                if plot_img is None:
                    flash('Error generating emotion plot', 'error')
                    return redirect(request.url)

                # Get content suggestions
                suggestions = get_content_suggestions(dominant_emotion)

                return render_template('result.html',
                                       plot_img=plot_img,
                                       overall_percentages=overall_percentages,
                                       dominant_emotion=dominant_emotion,
                                       suggestions=suggestions)

            except Exception as e:
                flash(f'Error processing video: {str(e)}', 'error')
                return redirect(request.url)
            finally:
                # Clean up uploaded file
                if os.path.exists(filepath):
                    os.remove(filepath)

        else:
            flash('Invalid file type. Please upload MP4, AVI, or MOV files only.', 'error')
            return redirect(request.url)

    return render_template('upload.html',logged_in=True)
#Real Time Video analysis
import os
import json
from datetime import datetime
from flask import session, Response, jsonify
import cv2
import time
import numpy as np
from threading import Lock

# Global variables
interview_results_lock = Lock()
interview_results = {}
TEMP_DIR = 'temp_data'  # Create this directory in your project

# Ensure temp directory exists
if not os.path.exists(TEMP_DIR):
    os.makedirs(TEMP_DIR)


def save_emotion_data(user_id, data):
    """Save emotion data to a temporary file"""
    filename = f"{user_id}_{int(time.time())}.json"
    filepath = os.path.join(TEMP_DIR, filename)
    with open(filepath, 'w') as f:
        json.dump(data, f)
    return filename


def load_emotion_data(filename):
    """Load emotion data from temporary file"""
    filepath = os.path.join(TEMP_DIR, filename)
    try:
        with open(filepath, 'r') as f:
            data = json.load(f)
        # Optionally delete the file after reading
        os.remove(filepath)
        return data
    except (FileNotFoundError, json.JSONDecodeError):
        return None


@app.route('/video_interview')
def video_interview():
    if 'user_id' not in session:
        flash('Please log in to start a video interview.', 'warning')
        return redirect(url_for('login'))

    # Reset any existing interview data
    global interview_results
    with interview_results_lock:
        interview_results.clear()

    session.pop('interview_results', None)
    session.pop('emotion_data_file', None)
    session['interview_active'] = True
    session['start_time'] = time.time()
    return render_template('video_interview.html')


@app.route('/check_interview_status')
def check_interview_status():
    """Check if interview is complete and return results if available"""
    global interview_results
    with interview_results_lock:
        if interview_results:
            # Save detailed emotion data to file
            emotion_data_file = save_emotion_data(
                session.get('user_id', 'anonymous'),
                interview_results['emotion_data']
            )

            # Store only essential data in session
            session['interview_results'] = {
                'emotion_counts': interview_results['emotion_counts'],
                'emotion_percentages': interview_results['emotion_percentages'],
                'dominant_emotion': interview_results['dominant_emotion'],
                'total_duration': interview_results['total_duration']
            }
            session['emotion_data_file'] = emotion_data_file

            interview_results.clear()
            return jsonify({'status': 'complete'})
    return jsonify({'status': 'recording'})


@app.route('/video_interview_results')
def video_interview_results():
    if 'interview_results' not in session:
        flash('No interview results available.', 'error')
        return redirect(url_for('video_interview'))

    results = session['interview_results']

    # Load detailed emotion data from file
    if 'emotion_data_file' in session:
        emotion_data = load_emotion_data(session['emotion_data_file'])
        if emotion_data:
            plot_img = create_interview_emotion_plot(emotion_data)
        else:
            plot_img = None
    else:
        plot_img = None

    suggestions = get_content_suggestions(results['dominant_emotion'])

    # Clean up session
    session.pop('emotion_data_file', None)

    return render_template('interview_results.html',
                           plot_img=plot_img,
                           emotion_counts=results['emotion_counts'],
                           emotion_percentages=results['emotion_percentages'],
                           dominant_emotion=results['dominant_emotion'],
                           total_duration=results['total_duration'],
                           suggestions=suggestions)


# The gen_frames function remains the same as in the previous version

# Add cleanup function to delete old temporary files
def cleanup_old_temp_files():
    """Delete temporary emotion data files older than 1 hour"""
    current_time = time.time()
    for filename in os.listdir(TEMP_DIR):
        filepath = os.path.join(TEMP_DIR, filename)
        if os.path.getmtime(filepath) < current_time - 3600:  # 1 hour
            try:
                os.remove(filepath)
            except OSError:
                pass


# Add periodic cleanup to your app initialization
from apscheduler.schedulers.background import BackgroundScheduler

scheduler = BackgroundScheduler()
scheduler.add_job(cleanup_old_temp_files, 'interval', hours=1)
scheduler.start()


@app.route('/video_feed')
def video_feed():
    if 'interview_active' not in session:
        return "Interview not started", 400
    return Response(gen_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')


def gen_frames():
    global interview_results
    camera = cv2.VideoCapture(0)
    start_time = time.time()
    duration = 60  # 60 seconds duration
    emotion_data = []
    emotion_counts = {emotion: 0 for emotion in emotion_labels}

    try:
        while True:
            elapsed_time = time.time() - start_time
            if elapsed_time >= duration:
                break

            success, frame = camera.read()
            if not success:
                break

            # Process the frame
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            faces = face_classifier.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

            current_emotions = []
            for (x, y, w, h) in faces:
                roi_gray = gray[y:y + h, x:x + w]
                roi_gray = cv2.resize(roi_gray, (48, 48), interpolation=cv2.INTER_AREA)

                if np.sum([roi_gray]) != 0:
                    roi = roi_gray.astype('float') / 255.0
                    roi = img_to_array(roi)
                    roi = np.expand_dims(roi, axis=0)

                    prediction = classifier.predict(roi)[0]
                    label = emotion_labels[prediction.argmax()]
                    emotion_counts[label] += 1
                    current_emotions.append(label)

                    cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
                    cv2.putText(frame, f"{label}", (x, y - 10),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

            # Add countdown timer to frame
            remaining_time = int(duration - elapsed_time)
            cv2.putText(frame, f"Time: {remaining_time}s", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

            if current_emotions:
                emotion_data.append({
                    'time': elapsed_time,
                    'emotions': current_emotions
                })

            ret, buffer = cv2.imencode('.jpg', frame)
            frame = buffer.tobytes()

            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

    finally:
        camera.release()

        if emotion_counts:
            total_frames = sum(emotion_counts.values())
            emotion_percentages = {
                emotion: (count / total_frames * 100) if total_frames > 0 else 0
                for emotion, count in emotion_counts.items()
            }

            dominant_emotion = max(emotion_counts.items(), key=lambda x: x[1])[0]

            # Store results in global variable instead of session
            with interview_results_lock:
                interview_results.update({
                    'emotion_data': emotion_data,
                    'emotion_counts': emotion_counts,
                    'emotion_percentages': emotion_percentages,
                    'dominant_emotion': dominant_emotion,
                    'total_duration': elapsed_time
                })


def create_interview_emotion_plot(emotion_data):
    # Create figure
    plt.figure(figsize=(12, 6))

    # Process emotion data
    emotions_over_time = {emotion: [] for emotion in emotion_labels}
    times = []

    for data_point in emotion_data:
        times.append(data_point['time'])
        current_emotions = data_point['emotions']

        for emotion in emotion_labels:
            count = current_emotions.count(emotion)
            emotions_over_time[emotion].append(
                count / len(current_emotions) if current_emotions else 0
            )

    # Plot each emotion
    colors = {
        'Angry': 'red',
        'Disgust': 'green',
        'Fear': 'gray',
        'Happy': 'yellow',
        'Neutral': 'purple',
        'Sad': 'blue',
        'Surprise': 'orange'
    }

    for emotion in emotion_labels:
        plt.plot(times, emotions_over_time[emotion],
                 label=emotion, color=colors[emotion])

    plt.xlabel('Time (seconds)')
    plt.ylabel('Emotion Intensity')
    plt.title('Emotion Analysis During Interview')
    plt.legend()
    plt.grid(True)

    # Convert plot to base64 string
    img_buf = io.BytesIO()
    plt.savefig(img_buf, format='png', bbox_inches='tight')
    img_buf.seek(0)
    img_str = base64.b64encode(img_buf.getvalue()).decode()
    plt.close()

    return img_str

#Emotion Detection in Image
@app.route('/image_emotion', methods=['GET', 'POST'])
def image_emotion():
    if 'user_id' not in session:
        flash('Please log in to use image emotion recognition.', 'warning')
        return redirect(url_for('login'))

    if request.method == 'POST':
        if 'file' not in request.files:
            flash('No file part', 'error')
            return redirect(request.url)

        file = request.files['file']
        if file.filename == '':
            flash('No selected file', 'error')
            return redirect(request.url)

        if file and file.filename.lower().endswith(('.png', '.jpg', '.jpeg')):
            # Save the uploaded file
            filename = os.path.join(app.config['UPLOAD_FOLDER'], secure_filename(file.filename))
            file.save(filename)

            # Read the image
            frame = cv2.imread(filename)
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            faces = face_classifier.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

            if len(faces) == 0:
                os.remove(filename)
                flash('No face detected in the image', 'error')
                return redirect(request.url)

            emotions_detected = []
            emotion_probabilities = {}

            for (x, y, w, h) in faces:
                roi_gray = gray[y:y + h, x:x + w]
                roi_gray = cv2.resize(roi_gray, (48, 48), interpolation=cv2.INTER_AREA)

                if np.sum([roi_gray]) != 0:
                    roi = roi_gray.astype('float') / 255.0
                    roi = img_to_array(roi)
                    roi = np.expand_dims(roi, axis=0)

                    # Make prediction
                    prediction = classifier.predict(roi)[0]

                    # Get emotion label and probabilities
                    label = emotion_labels[prediction.argmax()]
                    emotions_detected.append(label)

                    # Store probabilities for each emotion
                    emotion_probabilities = {
                        emotion: float(prob)
                        for emotion, prob in zip(emotion_labels, prediction)
                    }

                    # Draw rectangle and emotion on image
                    cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
                    cv2.putText(frame, label, (x, y - 10),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

            # Save the annotated image
            output_filename = os.path.join(app.config['UPLOAD_FOLDER'], 'annotated_' + secure_filename(file.filename))
            cv2.imwrite(output_filename, frame)

            # Create plot for emotion probabilities
            plot_img = create_emotion_plot_image(emotion_probabilities)

            # Get the dominant emotion
            dominant_emotion = max(emotion_probabilities.items(), key=lambda x: x[1])[0]

            # Clean up original file
            os.remove(filename)

            # Convert annotated image to base64 for display
            with open(output_filename, 'rb') as img_file:
                img_data = base64.b64encode(img_file.read()).decode('utf-8')

            # Clean up annotated file
            os.remove(output_filename)

            # Get content suggestions based on the dominant emotion
            suggestions = get_content_suggestions(dominant_emotion)

            return render_template('result_image.html',
                                   plot_img=plot_img,
                                   emotions=emotion_probabilities,
                                   annotated_image=img_data,
                                   suggestions=suggestions,
                                   logged_in=True)
        else:
            flash('Invalid file type. Please upload a PNG or JPEG image.', 'error')
            return redirect(request.url)

    return render_template('upload_image.html',logged_in=True)


def create_emotion_plot_image(emotion_probabilities):
    plt.figure(figsize=(10, 6))
    emotions = list(emotion_probabilities.keys())
    probabilities = list(emotion_probabilities.values())

    bars = plt.bar(emotions, probabilities)

    # Color coding for emotions
    colors = ['red', 'green', 'gray', 'yellow', 'purple', 'blue', 'orange']
    for bar, color in zip(bars, colors):
        bar.set_color(color)

    plt.title('Detected Emotions')
    plt.xlabel('Emotions')
    plt.ylabel('Probability')
    plt.ylim(0, 1)
    plt.xticks(rotation=45)

    img_buf = io.BytesIO()
    plt.savefig(img_buf, format='png', bbox_inches='tight')
    img_buf.seek(0)
    img_str = base64.b64encode(img_buf.getvalue()).decode()
    plt.close()

    return img_str


#Text Emotion Analysis
# Load the pre-trained model
model1 = load_model('models/cnn_w2v.h5')

# Mapping of emotion classes
class_names = ['joy', 'fear', 'anger', 'sadness', 'neutral']

# Define the maximum sequence length for input
max_seq_len = 500

def clean_text(data):
    # remove hashtags and @usernames
    data = re.sub(r"(#[\d\w\.]+)", '', data)
    data = re.sub(r"(@[\d\w\.]+)", '', data)

    # tokenization using str.split()
    data = data.split()

    return data

# Load the tokenizer
with open('tokenizer.pkl', 'rb') as f:
    tokenizer = pickle.load(f)

def create_probability_graph(predictions):
    plt.figure(figsize=(8, 4))
    bars = plt.bar(class_names, predictions)

    # Color coding for emotions
    colors = ['#ff9999', '#ff3333', '#3366cc', '#99ccff', '#33cc33']
    for bar, color in zip(bars, colors):
        bar.set_color(color)

    plt.xlabel('Emotions')
    plt.ylabel('Probability')
    plt.title('Prediction Probability')
    plt.ylim(0, 1.0)
    plt.grid(axis='y', linestyle='--', alpha=0.7)

    # Rotate x-axis labels for better readability
    plt.xticks(rotation=45)

    # Adjust layout to prevent label cutoff
    plt.tight_layout()

    # Save plot to a temporary buffer.
    buf = io.BytesIO()
    plt.savefig(buf, format='png', dpi=300, bbox_inches='tight')
    plt.close()

    # Encode the plot to base64 string
    buf.seek(0)
    plot_url = base64.b64encode(buf.getvalue()).decode('utf8')
    return plot_url

@app.route('/text')
def text():
    if 'user_id' not in session:
        flash('Please log in to use multimodal emotion recognition.', 'warning')
        return redirect(url_for('login'))

    return render_template('text.html',logged_in=True)


@app.route('/predict', methods=['POST'])
def predict():
    text = request.form['text']

    # Preprocess the input text
    cleaned_text = [' '.join(clean_text(text))]
    sequence = tokenizer.texts_to_sequences(cleaned_text)
    padded = pad_sequences(sequence, maxlen=max_seq_len)

    # Make the prediction
    prediction_probs = model1.predict(padded)[0]
    predicted_class_index = np.argmax(prediction_probs)
    predicted_emotion = class_names[predicted_class_index]
    confidence = float(prediction_probs[predicted_class_index])

    # Create the probability graph
    plot_url = create_probability_graph(prediction_probs)

    suggestions = get_content_suggestions(predicted_emotion)

    return render_template('text.html',
                           text=text,
                           prediction=predicted_emotion,
                           confidence=confidence,
                           plot_url=plot_url,
                           suggestions=suggestions,
                           logged_in=True
                           )

#Image to Text
import pytesseract
import os

pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'

# Create upload folder if it doesn't exist
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
def create_probability_graph_text_multimodal(predictions):
    """Generate a probability bar graph for text-based predictions"""
    plt.figure(figsize=(8, 4))
    bars = plt.bar(class_names, predictions)

    # Color coding for emotions
    colors = ['#ff9999', '#ff3333', '#3366cc', '#99ccff', '#33cc33']
    for bar, color in zip(bars, colors):
        bar.set_color(color)

    plt.xlabel('Emotions')
    plt.ylabel('Probability')
    plt.title('Prediction Probability')
    plt.ylim(0, 1.0)
    plt.grid(axis='y', linestyle='--', alpha=0.7)

    plt.xticks(rotation=45)
    plt.tight_layout()

    buf = io.BytesIO()
    plt.savefig(buf, format='png', dpi=300, bbox_inches='tight')
    plt.close()

    buf.seek(0)
    plot_url = base64.b64encode(buf.getvalue()).decode('utf8')
    return plot_url

def extract_text_from_image_multimodal(image_path):
    """Extract text from uploaded image using pytesseract for multimodal data"""
    try:
        img = Image.open(image_path)
        text = pytesseract.image_to_string(img)
        return text.strip()
    except Exception as e:
        print(f"Error in text extraction: {str(e)}")
        return None

def predict_emotion_text_multimodal(text):
    """Predict emotion from text in multimodal data"""
    cleaned_text = [' '.join(clean_text(text))]
    sequence = tokenizer.texts_to_sequences(cleaned_text)
    padded = pad_sequences(sequence, maxlen=max_seq_len)

    prediction_probs = model1.predict(padded)[0]
    predicted_class_index = np.argmax(prediction_probs)
    predicted_emotion = class_names[predicted_class_index]
    confidence = float(prediction_probs[predicted_class_index])

    return predicted_emotion, confidence, prediction_probs

@app.route('/text_image')
def text_image():
    if 'user_id' not in session:
        flash('Please log in to use multimodal emotion recognition.', 'warning')
        return redirect(url_for('login'))

    return render_template('text_image.html', logged_in=True)

@app.route('/predict_text_multimodal', methods=['POST'])
def predict_text_multimodal():
    user_text = ""

    if 'image' in request.files:
        image = request.files['image']
        if image.filename != '':
            # Save the uploaded image
            image_path = os.path.join(app.config['UPLOAD_FOLDER'], image.filename)
            image.save(image_path)

            # Extract text from image
            extracted_text = extract_text_from_image_multimodal(image_path)
            if extracted_text:
                user_text = extracted_text

            # Clean up: remove the uploaded image
            os.remove(image_path)
    else:
        user_text = request.form.get('text', '')

    if not user_text:
        return render_template('index.html', error="No text could be extracted from the image")

    # Predict emotion
    predicted_emotion, confidence, prediction_probs = predict_emotion_text_multimodal(user_text)

    # Create the probability graph
    plot_url = create_probability_graph_text_multimodal(prediction_probs)

    suggestions = get_content_suggestions(predicted_emotion)

    return render_template('text_image.html',
                           user_text=user_text,
                           prediction=predicted_emotion,
                           confidence=confidence,
                           plot_url=plot_url,
                           suggestions=suggestions,
                           logged_in=True)

#Speech emotion recognition

# Load the trained model
model = load_model('speech_emotion_model.h5')

# Load the OneHotEncoder
with open('encoder.pkl', 'rb') as f:
    enc = pickle.load(f)



@app.route('/Audio', methods=['GET', 'POST'])
def Audio():
    if 'user_id' not in session:
        flash('Please log in to use multimodal emotion recognition.', 'warning')
        return redirect(url_for('login'))

    if request.method == 'POST':
        # Get the uploaded audio file
        audio_file = request.files['audio_file']

        # Create uploads directory if it doesn't exist
        if not os.path.exists('uploads'):
            os.makedirs('uploads')

        # Save the uploaded audio file
        audio_path = os.path.join('uploads', audio_file.filename)
        audio_file.save(audio_path)

        try:
            # Load and preprocess the audio
            sample_audio, sample_rate = librosa.load(audio_path, duration=3, offset=0.5)
            sample_mfcc = np.mean(librosa.feature.mfcc(y=sample_audio, sr=sample_rate, n_mfcc=40).T, axis=0)
            sample_mfcc = np.expand_dims(sample_mfcc, axis=0)  # (1, 40)

            # Make the prediction
            prediction = model.predict(sample_mfcc)

            # Get the predicted emotion and probabilities
            predicted_emotion_index = np.argmax(prediction, axis=-1)[0]
            emotion_probs = prediction[0] * 100  # Convert to percentages

            # Get emotion labels and ensure the index is valid
            if predicted_emotion_index < len(enc.categories_[0]):
                predicted_emotion = enc.categories_[0][predicted_emotion_index]
                emotion_labels = enc.categories_[0].tolist()
                emotion_values = emotion_probs.tolist()
            else:
                predicted_emotion = "Unknown"
                emotion_labels = []
                emotion_values = []

            # Get suggestions based on the predicted emotion
            suggestions = get_content_suggestions(predicted_emotion)

            # Clean up the uploaded file
            os.remove(audio_path)

            return render_template('audio1.html',
                                   predicted_emotion=predicted_emotion,
                                   suggestions=suggestions,
                                   emotion_probs=True,
                                   emotion_labels=emotion_labels,
                                   emotion_values=emotion_values,
                                   logged_in=True)

        except Exception as e:
            print(f"Error processing audio: {str(e)}")
            return render_template('audio1.html',
                                   error="Error processing audio file. Please try again.",
                                   logged_in=True)

    return render_template('audio1.html', logged_in=True)

import pyaudio
import wave
import threading
import queue
import librosa
import time
import json
from datetime import datetime
# Audio recording parameters
CHUNK = 1024
FORMAT = pyaudio.paFloat32
CHANNELS = 1
RATE = 22050
RECORD_SECONDS = 3


class AudioStreamer:
    def __init__(self, model, encoder):
        self.model = model
        self.encoder = encoder
        self.audio_queue = queue.Queue()
        self.is_recording = False
        self.p = pyaudio.PyAudio()
        self.emotion_history = []

    def start_stream(self):
        self.is_recording = True
        self.stream = self.p.open(
            format=FORMAT,
            channels=CHANNELS,
            rate=RATE,
            input=True,
            frames_per_buffer=CHUNK,
            stream_callback=self._audio_callback
        )
        self.stream.start_stream()

    def stop_stream(self):
        self.is_recording = False
        if hasattr(self, 'stream'):
            self.stream.stop_stream()
            self.stream.close()

    def _audio_callback(self, in_data, frame_count, time_info, status):
        if self.is_recording:
            self.audio_queue.put(np.frombuffer(in_data, dtype=np.float32))
        return (in_data, pyaudio.paContinue)

    def analyze_audio(self):
        while self.is_recording:
            audio_data = []
            start_time = time.time()

            while time.time() - start_time < RECORD_SECONDS and self.is_recording:
                if not self.audio_queue.empty():
                    audio_data.extend(self.audio_queue.get())

            if len(audio_data) > 0:
                # Process audio
                audio_array = np.array(audio_data)
                mfcc = np.mean(librosa.feature.mfcc(
                    y=audio_array,
                    sr=RATE,
                    n_mfcc=40
                ).T, axis=0)

                mfcc = np.expand_dims(mfcc, axis=0)

                # Predict emotion
                prediction = self.model.predict(mfcc, verbose=0)
                predicted_emotion_index = np.argmax(prediction, axis=-1)[0]
                emotion_probs = prediction[0] * 100

                # Get emotion labels
                emotion_labels = self.encoder.categories_[0]
                predicted_emotion = emotion_labels[predicted_emotion_index]

                # Store in history
                self.emotion_history.append({
                    'emotion': predicted_emotion,
                    'timestamp': datetime.now().strftime('%H:%M:%S'),
                    'confidence': float(np.max(emotion_probs))
                })

                # Keep only last 10 predictions
                if len(self.emotion_history) > 10:
                    self.emotion_history.pop(0)

                # Create result
                result = {
                    'emotion': predicted_emotion,
                    'probabilities': {
                        label: float(prob)
                        for label, prob in zip(emotion_labels, emotion_probs)
                    },
                    'history': self.emotion_history
                }

                yield f"data: {json.dumps(result)}\n\n"

            time.sleep(0.1)

@app.route('/real_audio')
def real_audio():
    if 'user_id' not in session:
        flash('Please log in to use multimodal emotion recognition.', 'warning')
        return redirect(url_for('login'))

    return render_template('Audio_real_time.html')


@app.route('/start_recording', methods=['POST'])
def start_recording():
    global audio_streamer
    audio_streamer = AudioStreamer(model, enc)
    audio_streamer.start_stream()
    return jsonify({'status': 'Recording started'})


@app.route('/stop_recording', methods=['POST'])
def stop_recording():
    global audio_streamer
    if audio_streamer:
        audio_streamer.stop_stream()
    return jsonify({'status': 'Recording stopped'})


@app.route('/emotion_stream')
def emotion_stream():
    def generate():
        global audio_streamer
        if audio_streamer:
            yield from audio_streamer.analyze_audio()

    return Response(generate(), mimetype='text/event-stream')
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
        try:
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
            if 'image_file' in request.files and request.files['image_file'].filename:
                image_file = request.files['image_file']
                image_path = os.path.join(app.config['UPLOAD_FOLDER'], secure_filename(image_file.filename))
                image_file.save(image_path)
                try:
                    image_pred = recognizer.process_image(image_path)
                    predictions.append(image_pred)
                finally:
                    if os.path.exists(image_path):
                        os.remove(image_path)

            # Process text if provided
            if 'text' in request.form and request.form['text'].strip():
                text_pred = recognizer.process_text(request.form['text'])
                predictions.append(text_pred)

            # Process audio if provided
            if 'audio_file' in request.files and request.files['audio_file'].filename:
                audio_file = request.files['audio_file']
                audio_path = os.path.join(app.config['UPLOAD_FOLDER'], secure_filename(audio_file.filename))
                audio_file.save(audio_path)
                try:
                    audio_pred = recognizer.process_speech(audio_path)
                    predictions.append(audio_pred)
                finally:
                    if os.path.exists(audio_path):
                        os.remove(audio_path)

            if not predictions:
                flash('Please provide at least one input (image, text, or audio).', 'error')
                return redirect(url_for('multimodal_analysis'))

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

        except Exception as e:
            print(f"Error in multimodal analysis: {str(e)}")
            flash(f'An error occurred during analysis: {str(e)}', 'error')
            return redirect(url_for('multimodal_analysis'))

    # GET request - show the input form
    return render_template('multimodal.html', logged_in=True)
import random

class ContentSuggestionSystem:
    def __init__(self):
        self.content_map = {
            'happy': {
                'websites': [
                    {'name': 'Positive Psychology', 'url': 'https://www.mentalhealthishealth.us/feeling/sad/',
                     'description': 'Learn about the science of happiness'},
                    {'name': 'Happify', 'url': 'https://www.happify.com/',
                     'description': 'Games and activities to boost happiness'},
                    {'name': 'Action for Happiness', 'url': 'https://actionforhappiness.org/',
                     'description': 'Movement for positive social change'}
                ],
                'playlists': [
                    {'name': 'Happy Hits!', 'url': 'https://open.spotify.com/playlist/37i9dQZF1DXdPec7aLTmlC',
                     'platform': 'Spotify', 'description': 'Upbeat pop hits to keep you smiling'},
                    {'name': 'Good Vibes',
                     'url': 'https://music.youtube.com/playlist?list=PLMC9KNkIncKtPzgY-5rmhvj7fax8fdxoj',
                     'platform': 'YouTube Music', 'description': 'Feel-good tunes for a great day'},
                    {'name': 'Happy Folk', 'url': 'https://open.spotify.com/playlist/37i9dQZF1DX9ud5dZ7dU0j',
                     'platform': 'Spotify', 'description': 'Cheerful folk and acoustic tracks'}
                ],
                'books': [
                    {'title': 'The Happiness of Pursuit', 'author': 'Chris Guillebeau',
                     'description': 'Finding the quest that will bring purpose to your life'},
                    {'title': 'Authentic Happiness', 'author': 'Martin Seligman',
                     'description': 'Using the new positive psychology to realize your potential'},
                    {'title': 'The Happiness Advantage', 'author': 'Shawn Achor',
                     'description': 'The seven principles of positive psychology that fuel success and performance at work'}
                ],
                'activities': [
                    'Start a gratitude journal',
                    'Plan a fun outing with friends',
                    'Try a new hobby you have been curious about',
                                                              'Practice random acts of kindness'
                ]
            },
            'joy': {
                'websites': [
                    {'name': 'Positive Psychology', 'url': 'https://www.mentalhealthishealth.us/feeling/sad/',
                     'description': 'Learn about the science of happiness'},
                    {'name': 'Happify', 'url': 'https://www.happify.com/',
                     'description': 'Games and activities to boost happiness'},
                    {'name': 'Action for Happiness', 'url': 'https://actionforhappiness.org/',
                     'description': 'Movement for positive social change'}
                ],
                'playlists': [
                    {'name': 'Happy Hits!', 'url': 'https://open.spotify.com/playlist/37i9dQZF1DXdPec7aLTmlC',
                     'platform': 'Spotify', 'description': 'Upbeat pop hits to keep you smiling'},
                    {'name': 'Good Vibes',
                     'url': 'https://music.youtube.com/playlist?list=PLMC9KNkIncKtPzgY-5rmhvj7fax8fdxoj',
                     'platform': 'YouTube Music', 'description': 'Feel-good tunes for a great day'},
                    {'name': 'Happy Folk', 'url': 'https://open.spotify.com/playlist/37i9dQZF1DX9ud5dZ7dU0j',
                     'platform': 'Spotify', 'description': 'Cheerful folk and acoustic tracks'}
                ],
                'books': [
                    {'title': 'The Happiness of Pursuit', 'author': 'Chris Guillebeau',
                     'description': 'Finding the quest that will bring purpose to your life'},
                    {'title': 'Authentic Happiness', 'author': 'Martin Seligman',
                     'description': 'Using the new positive psychology to realize your potential'},
                    {'title': 'The Happiness Advantage', 'author': 'Shawn Achor',
                     'description': 'The seven principles of positive psychology that fuel success and performance at work'}
                ],
                'activities': [
                    'Start a gratitude journal',
                    'Plan a fun outing with friends',
                    'Try a new hobby you have been curious about',
                    'Practice random acts of kindness'
                ]
            },
            'sad': {
                'websites': [
                    {'name': '7 Cups', 'url': 'https://www.7cups.com/',
                     'description': 'Free emotional support and counseling'},
                    {'name': 'Calm', 'url': 'https://www.calm.com/', 'description': 'Meditation and relaxation app'},
                    {'name': 'Moodfit', 'url': 'https://www.getmoodfit.com/',
                     'description': 'Tools and insights to shape up your mood'}
                ],
                'playlists': [
                    {'name': 'Mood Booster', 'url': 'https://open.spotify.com/playlist/37i9dQZF1DX3rxVfibe1L0',
                     'platform': 'Spotify', 'description': 'Uplifting songs to elevate your mood'},
                    {'name': 'Cheer Up',
                     'url': 'https://music.youtube.com/playlist?list=PLMC9KNkIncKvYin_USF1qoJQnIyMAfRxl',
                     'platform': 'YouTube Music', 'description': 'Positive vibes to help you feel better'},
                    {'name': 'Rainy Day Jazz', 'url': 'https://open.spotify.com/playlist/37i9dQZF1DWVqfgj8NZEp1',
                     'platform': 'Spotify', 'description': 'Smooth jazz for a contemplative mood'}
                ],
                'books': [
                    {'title': 'The Upward Spiral', 'author': 'Alex Korb',
                     'description': 'Using neuroscience to reverse the course of depression, one small change at a time'},
                    {'title': 'Feeling Good', 'author': 'David D. Burns',
                     'description': 'The new mood therapy to conquer depression'},
                    {'title': 'The Mindful Way Through Depression',
                     'author': 'Mark Williams, John Teasdale, Zindel Segal, and Jon Kabat-Zinn',
                     'description': 'Freeing Yourself from Chronic Unhappiness'}
                ],
                'activities': [
                    'Take a nature walk',
                    'Practice mindfulness meditation',
                    'Reach out to a friend or family member',
                    'Create art or write in a journal'
                ]
            },

            'neutral': {
                'websites': [
                    {'name': '7 Cups', 'url': 'https://www.7cups.com/',
                     'description': 'Free emotional support and counseling'},
                    {'name': 'Calm', 'url': 'https://www.calm.com/', 'description': 'Meditation and relaxation app'},
                    {'name': 'Moodfit', 'url': 'https://www.getmoodfit.com/',
                     'description': 'Tools and insights to shape up your mood'}
                ],
                'playlists': [
                    {'name': 'Mood Booster', 'url': 'https://open.spotify.com/playlist/37i9dQZF1DX3rxVfibe1L0',
                     'platform': 'Spotify', 'description': 'Uplifting songs to elevate your mood'},
                    {'name': 'Cheer Up',
                     'url': 'https://music.youtube.com/playlist?list=PLMC9KNkIncKvYin_USF1qoJQnIyMAfRxl',
                     'platform': 'YouTube Music', 'description': 'Positive vibes to help you feel better'},
                    {'name': 'Rainy Day Jazz', 'url': 'https://open.spotify.com/playlist/37i9dQZF1DWVqfgj8NZEp1',
                     'platform': 'Spotify', 'description': 'Smooth jazz for a contemplative mood'}
                ],
                'books': [
                    {'title': 'The Upward Spiral', 'author': 'Alex Korb',
                     'description': 'Using neuroscience to reverse the course of depression, one small change at a time'},
                    {'title': 'Feeling Good', 'author': 'David D. Burns',
                     'description': 'The new mood therapy to conquer depression'},
                    {'title': 'The Mindful Way Through Depression',
                     'author': 'Mark Williams, John Teasdale, Zindel Segal, and Jon Kabat-Zinn',
                     'description': 'Freeing Yourself from Chronic Unhappiness'}
                ],
                'activities': [
                    'Take a nature walk',
                    'Practice mindfulness meditation',
                    'Reach out to a friend or family member',
                    'Create art or write in a journal'
                ]
            },

            'angry': {
                'websites': [
                    {'name': '7 Cups', 'url': 'https://www.7cups.com/',
                     'description': 'Free emotional support and counseling'},
                    {'name': 'Calm', 'url': 'https://www.calm.com/', 'description': 'Meditation and relaxation app'},
                    {'name': 'Moodfit', 'url': 'https://www.getmoodfit.com/',
                     'description': 'Tools and insights to shape up your mood'}
                ],
                'playlists': [
                    {'name': 'Mood Booster', 'url': 'https://open.spotify.com/playlist/37i9dQZF1DX3rxVfibe1L0',
                     'platform': 'Spotify', 'description': 'Uplifting songs to elevate your mood'},
                    {'name': 'Cheer Up',
                     'url': 'https://music.youtube.com/playlist?list=PLMC9KNkIncKvYin_USF1qoJQnIyMAfRxl',
                     'platform': 'YouTube Music', 'description': 'Positive vibes to help you feel better'},
                    {'name': 'Rainy Day Jazz', 'url': 'https://open.spotify.com/playlist/37i9dQZF1DWVqfgj8NZEp1',
                     'platform': 'Spotify', 'description': 'Smooth jazz for a contemplative mood'}
                ],
                'books': [
                    {'title': 'The Upward Spiral', 'author': 'Alex Korb',
                     'description': 'Using neuroscience to reverse the course of depression, one small change at a time'},
                    {'title': 'Feeling Good', 'author': 'David D. Burns',
                     'description': 'The new mood therapy to conquer depression'},
                    {'title': 'The Mindful Way Through Depression',
                     'author': 'Mark Williams, John Teasdale, Zindel Segal, and Jon Kabat-Zinn',
                     'description': 'Freeing Yourself from Chronic Unhappiness'}
                ],
                'activities': [
                    'Take a nature walk',
                    'Practice mindfulness meditation',
                    'Reach out to a friend or family member',
                    'Create art or write in a journal'
                ]
            },

            'surprise': {
                'websites': [
                    {'name': '7 Cups', 'url': 'https://www.7cups.com/',
                     'description': 'Free emotional support and counseling'},
                    {'name': 'Calm', 'url': 'https://www.calm.com/', 'description': 'Meditation and relaxation app'},
                    {'name': 'Moodfit', 'url': 'https://www.getmoodfit.com/',
                     'description': 'Tools and insights to shape up your mood'}
                ],
                'playlists': [
                    {'name': 'Mood Booster', 'url': 'https://open.spotify.com/playlist/37i9dQZF1DX3rxVfibe1L0',
                     'platform': 'Spotify', 'description': 'Uplifting songs to elevate your mood'},
                    {'name': 'Cheer Up',
                     'url': 'https://music.youtube.com/playlist?list=PLMC9KNkIncKvYin_USF1qoJQnIyMAfRxl',
                     'platform': 'YouTube Music', 'description': 'Positive vibes to help you feel better'},
                    {'name': 'Rainy Day Jazz', 'url': 'https://open.spotify.com/playlist/37i9dQZF1DWVqfgj8NZEp1',
                     'platform': 'Spotify', 'description': 'Smooth jazz for a contemplative mood'}
                ],
                'books': [
                    {'title': 'The Upward Spiral', 'author': 'Alex Korb',
                     'description': 'Using neuroscience to reverse the course of depression, one small change at a time'},
                    {'title': 'Feeling Good', 'author': 'David D. Burns',
                     'description': 'The new mood therapy to conquer depression'},
                    {'title': 'The Mindful Way Through Depression',
                     'author': 'Mark Williams, John Teasdale, Zindel Segal, and Jon Kabat-Zinn',
                     'description': 'Freeing Yourself from Chronic Unhappiness'}
                ],
                'activities': [
                    'Take a nature walk',
                    'Practice mindfulness meditation',
                    'Reach out to a friend or family member',
                    'Create art or write in a journal'
                ]
            },

            'fear': {
                'websites': [
                    {'name': '7 Cups', 'url': 'https://www.7cups.com/',
                     'description': 'Free emotional support and counseling'},
                    {'name': 'Calm', 'url': 'https://www.calm.com/', 'description': 'Meditation and relaxation app'},
                    {'name': 'Moodfit', 'url': 'https://www.getmoodfit.com/',
                     'description': 'Tools and insights to shape up your mood'}
                ],
                'playlists': [
                    {'name': 'Mood Booster', 'url': 'https://open.spotify.com/playlist/37i9dQZF1DX3rxVfibe1L0',
                     'platform': 'Spotify', 'description': 'Uplifting songs to elevate your mood'},
                    {'name': 'Cheer Up',
                     'url': 'https://music.youtube.com/playlist?list=PLMC9KNkIncKvYin_USF1qoJQnIyMAfRxl',
                     'platform': 'YouTube Music', 'description': 'Positive vibes to help you feel better'},
                    {'name': 'Rainy Day Jazz', 'url': 'https://open.spotify.com/playlist/37i9dQZF1DWVqfgj8NZEp1',
                     'platform': 'Spotify', 'description': 'Smooth jazz for a contemplative mood'}
                ],
                'books': [
                    {'title': 'The Upward Spiral', 'author': 'Alex Korb',
                     'description': 'Using neuroscience to reverse the course of depression, one small change at a time'},
                    {'title': 'Feeling Good', 'author': 'David D. Burns',
                     'description': 'The new mood therapy to conquer depression'},
                    {'title': 'The Mindful Way Through Depression',
                     'author': 'Mark Williams, John Teasdale, Zindel Segal, and Jon Kabat-Zinn',
                     'description': 'Freeing Yourself from Chronic Unhappiness'}
                ],
                'activities': [
                    'Take a nature walk',
                    'Practice mindfulness meditation',
                    'Reach out to a friend or family member',
                    'Create art or write in a journal'
                ]
            },

            'disgust': {
                'websites': [
                    {'name': '7 Cups', 'url': 'https://www.7cups.com/',
                     'description': 'Free emotional support and counseling'},
                    {'name': 'Calm', 'url': 'https://www.calm.com/', 'description': 'Meditation and relaxation app'},
                    {'name': 'Moodfit', 'url': 'https://www.getmoodfit.com/',
                     'description': 'Tools and insights to shape up your mood'}
                ],
                'playlists': [
                    {'name': 'Mood Booster', 'url': 'https://open.spotify.com/playlist/37i9dQZF1DX3rxVfibe1L0',
                     'platform': 'Spotify', 'description': 'Uplifting songs to elevate your mood'},
                    {'name': 'Cheer Up',
                     'url': 'https://music.youtube.com/playlist?list=PLMC9KNkIncKvYin_USF1qoJQnIyMAfRxl',
                     'platform': 'YouTube Music', 'description': 'Positive vibes to help you feel better'},
                    {'name': 'Rainy Day Jazz', 'url': 'https://open.spotify.com/playlist/37i9dQZF1DWVqfgj8NZEp1',
                     'platform': 'Spotify', 'description': 'Smooth jazz for a contemplative mood'}
                ],
                'books': [
                    {'title': 'The Upward Spiral', 'author': 'Alex Korb',
                     'description': 'Using neuroscience to reverse the course of depression, one small change at a time'},
                    {'title': 'Feeling Good', 'author': 'David D. Burns',
                     'description': 'The new mood therapy to conquer depression'},
                    {'title': 'The Mindful Way Through Depression',
                     'author': 'Mark Williams, John Teasdale, Zindel Segal, and Jon Kabat-Zinn',
                     'description': 'Freeing Yourself from Chronic Unhappiness'}
                ],
                'activities': [
                    'Take a nature walk',
                    'Practice mindfulness meditation',
                    'Reach out to a friend or family member',
                    'Create art or write in a journal'
                ]
            },
            # ... (add more emotions and their corresponding recommendations) ...
        }

    def get_suggestions(self, emotion):
        if emotion.lower() in self.content_map:
            return self.content_map[emotion.lower()]
        else:
            return None

# Usage example
suggestion_system = ContentSuggestionSystem()

def get_content_suggestions(emotion):
    suggestions = suggestion_system.get_suggestions(emotion)
    if suggestions:
        return suggestions
    else:
        return "No specific suggestions for this emotion."


# Integration with emotion detection
def process_detected_emotion(emotion):
    suggestions = get_content_suggestions(emotion)
    return suggestions

@app.route('/')
def home():
    if 'user_id' in session:
        return render_template('home.html', logged_in=True)
    return render_template('home.html', logged_in=False)


@app.route('/register', methods=['GET', 'POST'])
def register():
    if request.method == 'POST':
        name = request.form['name']
        email = request.form['email']
        password = request.form['password']
        hashed_password = generate_password_hash(password)

        cur = mysql.connection.cursor()
        try:
            cur.execute("INSERT INTO users (name, email, password) VALUES (%s, %s, %s)", (name, email, hashed_password))
            mysql.connection.commit()
            flash('Registration successful. Please log in.', 'success')
            return redirect(url_for('login'))
        except:
            flash('Email already exists. Please use a different email.', 'danger')
        finally:
            cur.close()

    return render_template('register.html')


@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        email = request.form['email']
        password = request.form['password']

        cur = mysql.connection.cursor()
        cur.execute("SELECT * FROM users WHERE email = %s", (email,))
        user = cur.fetchone()
        cur.close()

        if user and check_password_hash(user[3], password):
            session['user_id'] = user[0]
            flash('Logged in successfully.', 'success')
            return redirect(url_for('home'))
        else:
            flash('Invalid email or password.', 'danger')

    return render_template('login.html')


@app.route('/logout')
def logout():
    session.pop('user_id', None)
    flash('Logged out successfully.', 'success')
    return redirect(url_for('home'))


@app.route('/AboutUs')
def AboutUs():
    if 'user_id' in session:
        return render_template('AboutUs.html', logged_in=True)
    return render_template('AboutUs.html', logged_in=False)

@app.route('/Blog')
def Blog():
    if 'user_id' in session:
        return render_template('blog.html', logged_in=True)
    return render_template('blog.html', logged_in=False)


@app.route('/Contact', methods=['GET', 'POST'])
def Contact():
    success_message = None
    if request.method == 'POST':
        email = request.form['email']
        message = request.form['message']

        cur = mysql.connection.cursor()
        try:
            cur.execute("INSERT INTO contact (email, message) VALUES (%s, %s)", (email, message))
            mysql.connection.commit()
            success_message = 'Your message has been sent successfully.'
        except Exception as e:
            flash('An error occurred while sending your message. Please try again.', 'danger')
            print(e)  # For debugging purposes
        finally:
            cur.close()

    return render_template('contact.html', success_message=success_message)


@app.route('/profile')
def profile():
    if 'user_id' not in session:
        flash('Please log in to view your profile.', 'warning')
        return redirect(url_for('login'), logged_in=False)

    cur = mysql.connection.cursor()
    cur.execute("SELECT name, email, age, gender, occupation, phone, address, mental_health_history, profile_picture FROM users WHERE id = %s", (session['user_id'],))
    user_data = cur.fetchone()
    cur.close()

    return render_template('profile.html', user=user_data, logged_in=True)


@app.route('/update_profile', methods=['GET', 'POST'])
def update_profile():
    if 'user_id' not in session:
        flash('Please log in to update your profile.', 'warning')
        return redirect(url_for('login'))

    cur = mysql.connection.cursor()

    if request.method == 'POST':
        # Get form data
        name = request.form.get('name')
        email = request.form.get('email')
        age = request.form.get('age')
        gender = request.form.get('gender')
        occupation = request.form.get('occupation')
        phone = request.form.get('phone')
        address = request.form.get('address')
        mental_health_history = request.form.get('mental_health_history')
        current_password = request.form.get('current_password')
        new_password = request.form.get('new_password')

        # Verify current password if user is changing it
        if current_password and new_password:
            cur.execute("SELECT password FROM users WHERE id = %s", (session['user_id'],))
            stored_password = cur.fetchone()[0]

            if not check_password_hash(stored_password, current_password):
                flash('Current password is incorrect.', 'danger')
                cur.close()
                return redirect(url_for('update_profile'))

            # Update password
            hashed_new_password = generate_password_hash(new_password)
            cur.execute("""
                UPDATE users 
                SET password = %s
                WHERE id = %s
            """, (hashed_new_password, session['user_id']))

        # Update user information
        try:
            cur.execute("""
                UPDATE users 
                SET name = %s, 
                    email = %s, 
                    age = %s,
                    gender = %s,
                    occupation = %s,
                    phone = %s,
                    address = %s,
                    mental_health_history = %s
                WHERE id = %s
            """, (name, email, age, gender, occupation, phone, address, mental_health_history, session['user_id']))

            mysql.connection.commit()
            flash('Profile updated successfully!', 'success')

        except Exception as e:
            print(f"Error updating profile: {e}")
            mysql.connection.rollback()
            flash('Error updating profile. Please try again.', 'danger')

        finally:
            cur.close()
            return redirect(url_for('profile'))

    # GET request - show current profile data
    cur.execute("""
        SELECT name, email, age, gender, occupation, phone, address, mental_health_history, profile_picture 
        FROM users 
        WHERE id = %s
    """, (session['user_id'],))

    user_data = cur.fetchone()
    cur.close()

    # Ensure that user data is not None
    if not user_data:
        flash('User not found.', 'danger')
        return redirect(url_for('profile'))

    # Pass the user data as a dictionary for easier access in the template
    user = {
        'name': user_data[0],
        'email': user_data[1],
        'age': user_data[2],
        'gender': user_data[3],
        'occupation': user_data[4],
        'phone': user_data[5],
        'address': user_data[6],
        'mental_health_history': user_data[7],
        'profile_picture': user_data[8]
    }

    return render_template('update_profile.html', user=user,logged_in=True)




@app.route('/upload_profile_picture', methods=['POST'])
def upload_profile_picture():
    if 'user_id' not in session:
        flash('Please log in to upload a profile picture.', 'warning')
        return redirect(url_for('login'))

    if 'profile_picture' not in request.files:
        flash('No file selected', 'danger')
        return redirect(url_for('update_profile'))

    file = request.files['profile_picture']

    if file.filename == '':
        flash('No file selected', 'danger')
        return redirect(url_for('update_profile'))

    if file and allowed_profile_picture(file.filename):
        try:
            # Generate secure filename
            filename = secure_filename(f"profile_pic_{session['user_id']}{os.path.splitext(file.filename)[1]}")

            # Save file
            file_path = os.path.join(app.config['UPLOAD_FOLDER'], 'profile_pictures', filename)
            os.makedirs(os.path.dirname(file_path), exist_ok=True)
            file.save(file_path)

            # Update database with new profile picture path
            cur = mysql.connection.cursor()
            cur.execute("UPDATE users SET profile_picture = %s WHERE id = %s",
                        (filename, session['user_id']))
            mysql.connection.commit()
            cur.close()

            flash('Profile picture updated successfully!', 'success')

        except Exception as e:
            print(f"Error uploading profile picture: {e}")
            flash('Error uploading profile picture. Please try again.', 'danger')

    else:
        flash('Invalid file type. Please upload a valid image file.', 'danger')

    return redirect(url_for('profile'))  # Corrected to match your actual route


def allowed_profile_picture(filename):
    return '.' in filename and \
        filename.rsplit('.', 1)[1].lower() in {'png', 'jpg', 'jpeg', 'gif'}


from flask import render_template, request
import random


# Add these functions to your app.py

def get_emotion_based_activities(emotion):
    emotion_activities = {
        'happy': [
    {
        'title': 'Joy Journaling',
        'description': 'Document and amplify your positive emotions',
        'icon': 'fas fa-journal-whills',
        'category': 'Mindfulness',
        'mood_boost': '💖',
        'url': '/joy-journaling'
    },
    {
        'title': 'Group Dance Session',
        'description': 'Share your joy through movement',
        'icon': 'fas fa-dancing',
        'category': 'Physical',
        'mood_boost': '💃',
        'url': '/activities/group-dance-session'
    }
],

        'sad': [
            {
                'title': 'Comfort Art Session',
                'description': 'Express emotions through gentle creativity',
                'icon': 'fas fa-paint-brush',
                'category': 'Creative',
                'mood_boost': '🎨'
            },
            {
                'title': 'Support Circle',
                'description': 'Connect with others in a caring environment',
                'icon': 'fas fa-hands-helping',
                'category': 'Social',
                'mood_boost': '🤝'
            }
        ],
        'angry': [
            {
                'title': 'Stress Relief Exercise',
                'description': 'Channel energy into positive movement',
                'icon': 'fas fa-dumbbell',
                'category': 'Physical',
                'mood_boost': '💪'
            },
            {
                'title': 'Calm Corner',
                'description': 'Guided breathing and relaxation',
                'icon': 'fas fa-wind',
                'category': 'Relaxation',
                'mood_boost': '🧘'
            }
        ],
        'anxious': [
            {
                'title': 'Grounding Techniques',
                'description': 'Practice present-moment awareness',
                'icon': 'fas fa-tree',
                'category': 'Mindfulness',
                'mood_boost': '🌳'
            },
            {
                'title': 'Worry Release Workshop',
                'description': 'Learn practical anxiety management tools',
                'icon': 'fas fa-cloud-sun',
                'category': 'Educational',
                'mood_boost': '☀️'
            }
        ],
        'neutral': [
            {
                'title': 'Joy Journaling',
                'description': 'Document and amplify your positive emotions',
                'icon': 'fas fa-journal-whills',
                'category': 'Mindfulness',
                'mood_boost': '💖',
                'url': '/joy-journaling'
            },
            {
                'title': 'Group Dance Session',
                'description': 'Share your joy through movement',
                'icon': 'fas fa-dancing',
                'category': 'Physical',
                'mood_boost': '💃',
                'url': '/activities/group-dance-session'
            }
        ]
    }
    return emotion_activities.get(emotion, emotion_activities['happy'])


def get_activities_by_age(age):
    activities = {
        'teen': [
            {
                'title': 'Group Art Therapy',
                'description': 'Express emotions through creative art sessions',
                'icon': 'fas fa-paint-brush',
                'category': 'Creative',
                'intensity': 'Low',
                'duration': '1 hour'
            },
            {
                'title': 'Teen Music Workshop',
                'description': 'Create and share music with peers',
                'icon': 'fas fa-music',
                'category': 'Creative',
                'intensity': 'Medium',
                'duration': '2 hours'
            },
            {
                'title': 'Digital Story Creation',
                'description': 'Express emotions through digital media',
                'icon': 'fas fa-video',
                'category': 'Technology',
                'intensity': 'Medium',
                'duration': '1.5 hours'
            }
        ],
        'young_adult': [
            {
                'title': 'Mindfulness Meditation',
                'description': 'Guided meditation sessions',
                'icon': 'fas fa-brain',
                'category': 'Wellness',
                'intensity': 'Low',
                'duration': '30 mins'
            },
            {
                'title': 'Career Balance Workshop',
                'description': 'Managing work-life emotional balance',
                'icon': 'fas fa-briefcase',
                'category': 'Professional',
                'intensity': 'Medium',
                'duration': '2 hours'
            },
            {
                'title': 'Adventure Sports',
                'description': 'Channel emotions through exciting activities',
                'icon': 'fas fa-hiking',
                'category': 'Adventure',
                'intensity': 'High',
                'duration': '3 hours'
            }
        ],
        'adult': [
            {
                'title': 'Mindfulness Meditation',
                'description': 'Guided meditation sessions',
                'icon': 'fas fa-brain',
                'category': 'Wellness',
                'intensity': 'Low',
                'duration': '30 mins'
            },
            {
                'title': 'Career Balance Workshop',
                'description': 'Managing work-life emotional balance',
                'icon': 'fas fa-briefcase',
                'category': 'Professional',
                'intensity': 'Medium',
                'duration': '2 hours'
            },
            {
                'title': 'Adventure Sports',
                'description': 'Channel emotions through exciting activities',
                'icon': 'fas fa-hiking',
                'category': 'Adventure',
                'intensity': 'High',
                'duration': '3 hours'
            }
        ],
        'senior': [
            {
                'title': 'Mindfulness Meditation',
                'description': 'Guided meditation sessions',
                'icon': 'fas fa-brain',
                'category': 'Wellness',
                'intensity': 'Low',
                'duration': '30 mins'
            },
            {
                'title': 'Career Balance Workshop',
                'description': 'Managing work-life emotional balance',
                'icon': 'fas fa-briefcase',
                'category': 'Professional',
                'intensity': 'Medium',
                'duration': '2 hours'
            },
            {
                'title': 'Adventure Sports',
                'description': 'Channel emotions through exciting activities',
                'icon': 'fas fa-hiking',
                'category': 'Adventure',
                'intensity': 'High',
                'duration': '3 hours'
            }
        ],
        # Add more age groups as needed
    }


    def get_age_group(age):
        if age < 18:
            return 'teen'
        elif age < 30:
            return 'young_adult'
        elif age < 50:
            return 'adult'
        else:
            return 'senior'

    return activities.get(get_age_group(age), activities['adult'])


@app.route('/joy-journaling')
def joy_journaling():
    # Activity details for Joy Journaling
    activity = {
        'title': 'Joy Journaling',
        'description': 'Document and amplify your positive emotions by journaling.',
        'icon': 'fas fa-journal-whills',
        'category': 'Mindfulness',
        'mood_boost': '💖',
        'benefits': [
            'Reflect on positive experiences',
            'Increase gratitude and self-awareness',
            'Enhance emotional resilience'
        ],
        'tips': [
            'Write in the morning to set a positive tone for the day.',
            'List three things you’re grateful for each day.',
            'Reflect on a joyful memory and describe it in detail.'
        ],
        'suggested_prompts': [
            'What made you smile recently?',
            'Describe a person who brings you joy.',
            'What are three things you’re grateful for today?'
        ]
    }

    return render_template('joy_journaling.html', activity=activity)


@app.route('/explore_activities')
def explore_activities():
    if 'user_id' not in session:
        flash('Please log in to view activities.', 'warning')
        return redirect(url_for('login'))

    # Retrieve user's age and other information from the database
    cur = mysql.connection.cursor()
    cur.execute("SELECT name, age, profile_picture FROM users WHERE id = %s", (session['user_id'],))
    user_data = cur.fetchone()
    cur.close()

    if not user_data:
        flash('User information could not be retrieved.', 'danger')
        return redirect(url_for('profile'))

    # Unpack user data
    user = {
        'name': user_data[0],
        'age': user_data[1] if user_data[1] else 30,
        'profile_picture': user_data[2]
    }

    detected_emotion = request.args.get('emotion', 'neutral')

    # Fetch age-based and emotion-based activities
    age_activities = get_activities_by_age(user['age'])
    emotion_activities = get_emotion_based_activities(detected_emotion)

    return render_template('activities.html',
                           age_activities=age_activities,
                           emotion_activities=emotion_activities,
                           user=user,
                           current_emotion=detected_emotion)

if __name__ == '__main__':
    app.run(debug=True)