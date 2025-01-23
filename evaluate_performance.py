import os
import cv2
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from PIL import Image
import torch
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from tqdm import tqdm
import time
import json
from datetime import datetime

# Import your models and configurations from app.py
from app import (
    mtcnn,
    extractor,
    model,
    face_classifier,
    classifier,
    emotion_labels,
    detect_emotions,
    process_video
)


class EmotionRecognitionEvaluator:
    def __init__(self, test_data_path='test_data'):
        self.test_data_path = test_data_path
        self.results_path = 'evaluation_results'
        os.makedirs(self.results_path, exist_ok=True)

        # Initialize models from app.py
        self.mtcnn = mtcnn
        self.extractor = extractor
        self.model = model
        self.face_classifier = face_classifier
        self.classifier = classifier
        self.emotion_labels = emotion_labels

    def evaluate_image_recognition(self):
        """Evaluate performance on test images"""
        image_path = os.path.join(self.test_data_path, 'images')
        if not os.path.exists(image_path):
            print(f"Test images directory not found: {image_path}")
            return None

        results = {
            'predictions': [],
            'processing_times': [],
            'confidence_scores': []
        }

        # Load ground truth labels if available
        labels_file = os.path.join(image_path, 'labels.txt')
        true_labels = {}
        if os.path.exists(labels_file):
            with open(labels_file, 'r') as f:
                for line in f:
                    img_name, label = line.strip().split(',')
                    true_labels[img_name] = label

        print("Evaluating Image Recognition...")
        for img_file in tqdm(os.listdir(image_path)):
            if img_file.lower().endswith(('.png', '.jpg', '.jpeg')):
                img_path = os.path.join(image_path, img_file)

                # Process image
                start_time = time.time()
                image = Image.open(img_path)
                face, class_probabilities = detect_emotions(image)

                processing_time = time.time() - start_time

                if face and class_probabilities:
                    dominant_emotion = max(class_probabilities.items(), key=lambda x: x[1])[0]
                    confidence = max(class_probabilities.values())
                else:
                    dominant_emotion = 'no_face'
                    confidence = 0.0

                results['predictions'].append({
                    'image': img_file,
                    'predicted_emotion': dominant_emotion,
                    'true_emotion': true_labels.get(img_file, 'unknown'),
                    'confidence': confidence
                })
                results['processing_times'].append(processing_time)
                results['confidence_scores'].append(confidence)

        # Calculate metrics
        metrics = {
            'avg_processing_time': np.mean(results['processing_times']),
            'std_processing_time': np.std(results['processing_times']),
            'avg_confidence': np.mean(results['confidence_scores']),
            'face_detection_rate': len(
                [p for p in results['predictions'] if p['predicted_emotion'] != 'no_face']) / len(
                results['predictions'])
        }

        # Calculate accuracy if ground truth is available
        if true_labels:
            true = [p['true_emotion'] for p in results['predictions'] if p['true_emotion'] != 'unknown']
            pred = [p['predicted_emotion'] for p in results['predictions'] if p['true_emotion'] != 'unknown']
            metrics['accuracy'] = accuracy_score(true, pred)
            metrics['confusion_matrix'] = confusion_matrix(true, pred, labels=emotion_labels).tolist()
            metrics['classification_report'] = classification_report(true, pred, labels=emotion_labels)

        return results, metrics

    def evaluate_video_recognition(self):
        """Evaluate performance on test videos"""
        video_path = os.path.join(self.test_data_path, 'videos')
        if not os.path.exists(video_path):
            print(f"Test videos directory not found: {video_path}")
            return None

        results = {
            'video_analysis': [],
            'processing_times': [],
            'frame_rates': []
        }

        print("Evaluating Video Recognition...")
        for video_file in tqdm(os.listdir(video_path)):
            if video_file.lower().endswith(('.mp4', '.avi', '.mov')):
                video_file_path = os.path.join(video_path, video_file)

                # Process video
                start_time = time.time()
                all_class_probabilities, overall_percentages, dominant_emotion = process_video(video_file_path)
                processing_time = time.time() - start_time

                # Calculate frame rate
                cap = cv2.VideoCapture(video_file_path)
                frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
                fps = frame_count / processing_time
                cap.release()

                results['video_analysis'].append({
                    'video': video_file,
                    'dominant_emotion': dominant_emotion,
                    'emotion_percentages': overall_percentages,
                    'processing_time': processing_time,
                    'frame_count': frame_count,
                    'fps': fps
                })
                results['processing_times'].append(processing_time)
                results['frame_rates'].append(fps)

        # Calculate metrics
        metrics = {
            'avg_processing_time': np.mean(results['processing_times']),
            'std_processing_time': np.std(results['processing_times']),
            'avg_fps': np.mean(results['frame_rates']),
            'min_fps': min(results['frame_rates']),
            'max_fps': max(results['frame_rates'])
        }

        return results, metrics

    def evaluate_live_recognition(self, duration=30):
        """Evaluate performance of live recognition"""
        print(f"Evaluating Live Recognition for {duration} seconds...")

        results = {
            'frames': [],
            'processing_times': [],
            'detected_emotions': []
        }

        cap = cv2.VideoCapture(0)
        start_time = time.time()
        frame_count = 0

        while (time.time() - start_time) < duration:
            frame_start = time.time()
            ret, frame = cap.read()
            if not ret:
                break

            frame_count += 1
            # Process frame
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            faces = face_classifier.detectMultiScale(gray)

            frame_emotions = []
            for (x, y, w, h) in faces:
                roi_gray = gray[y:y + h, x:x + w]
                roi_gray = cv2.resize(roi_gray, (48, 48))
                roi = roi_gray.astype('float') / 255.0
                roi = np.expand_dims(roi, axis=0)

                prediction = classifier.predict(roi)[0]
                emotion = emotion_labels[prediction.argmax()]
                frame_emotions.append(emotion)

            frame_time = time.time() - frame_start
            results['processing_times'].append(frame_time)
            results['detected_emotions'].extend(frame_emotions)

            # Display FPS
            cv2.putText(frame, f'FPS: {1 / frame_time:.2f}', (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            cv2.imshow('Live Evaluation', frame)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        cap.release()
        cv2.destroyAllWindows()

        total_time = time.time() - start_time

        # Calculate metrics
        metrics = {
            'avg_frame_time': np.mean(results['processing_times']),
            'std_frame_time': np.std(results['processing_times']),
            'avg_fps': frame_count / total_time,
            'face_detection_rate': len(results['detected_emotions']) / frame_count,
            'emotion_distribution': pd.Series(results['detected_emotions']).value_counts().to_dict()
        }

        return results, metrics

    def save_results(self, image_results, video_results, live_results):
        """Save evaluation results and generate plots"""
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        results_dir = os.path.join(self.results_path, timestamp)
        os.makedirs(results_dir, exist_ok=True)

        # Save metrics to JSON
        all_metrics = {
            'image_recognition': image_results[1],
            'video_recognition': video_results[1],
            'live_recognition': live_results[1]
        }

        with open(os.path.join(results_dir, 'metrics.json'), 'w') as f:
            json.dump(all_metrics, f, indent=4)

        # Generate and save plots
        self._plot_processing_times(image_results[0]['processing_times'],
                                    'Image Processing Times',
                                    os.path.join(results_dir, 'image_processing_times.png'))

        self._plot_processing_times(video_results[0]['processing_times'],
                                    'Video Processing Times',
                                    os.path.join(results_dir, 'video_processing_times.png'))

        self._plot_emotion_distribution(live_results[1]['emotion_distribution'],
                                        os.path.join(results_dir, 'live_emotion_distribution.png'))

        print(f"Results saved to: {results_dir}")

    def _plot_processing_times(self, times, title, save_path):
        plt.figure(figsize=(10, 6))
        plt.hist(times, bins=30)
        plt.title(title)
        plt.xlabel('Processing Time (seconds)')
        plt.ylabel('Frequency')
        plt.savefig(save_path)
        plt.close()

    def _plot_emotion_distribution(self, distribution, save_path):
        plt.figure(figsize=(10, 6))
        emotions = list(distribution.keys())
        counts = list(distribution.values())
        plt.bar(emotions, counts)
        plt.title('Emotion Distribution in Live Recognition')
        plt.xlabel('Emotions')
        plt.ylabel('Count')
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.savefig(save_path)
        plt.close()


def main():
    # Create evaluator instance
    evaluator = EmotionRecognitionEvaluator()

    # Run evaluations
    print("\nStarting Performance Evaluation...")

    print("\n1. Evaluating Image Recognition...")
    image_results = evaluator.evaluate_image_recognition()

    print("\n2. Evaluating Video Recognition...")
    video_results = evaluator.evaluate_video_recognition()

    print("\n3. Evaluating Live Recognition...")
    live_results = evaluator.evaluate_live_recognition(duration=30)

    # Save results
    print("\nSaving results and generating plots...")
    evaluator.save_results(image_results, video_results, live_results)

    print("\nEvaluation complete!")


if __name__ == "__main__":
    main()