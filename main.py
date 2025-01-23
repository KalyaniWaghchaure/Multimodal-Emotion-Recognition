def calculate_accuracy_metrics(results):
    """
    Calculate comprehensive accuracy metrics for emotion recognition system
    """
    metrics = {
        "Image Recognition": {
            "Face Detection Accuracy": results["image_recognition"]["face_detection_rate"] * 100,
            "Confidence Score": results["image_recognition"]["avg_confidence"] * 100,
            "Processing Speed": f"{1/results['image_recognition']['avg_processing_time']:.2f} FPS"
        },
        "Video Recognition": {
            "Processing Performance": {
                "Frame Rate": f"{results['video_recognition']['avg_fps']:.2f} FPS",
                "Processing Time": f"{results['video_recognition']['avg_processing_time']:.2f} seconds"
            }
        },
        "Live Recognition": {
            "Real-time Performance": {
                "Face Detection Rate": f"{results['live_recognition']['face_detection_rate'] * 100:.2f}%",
                "Frame Rate": f"{results['live_recognition']['avg_fps']:.2f} FPS",
                "Frame Processing Time": f"{results['live_recognition']['avg_frame_time'] * 1000:.2f} ms"
            },
            "Emotion Distribution": {
                "Total Detections": sum(results["live_recognition"]["emotion_distribution"].values()),
                "Emotion Breakdown": {
                    emotion: f"{count/sum(results['live_recognition']['emotion_distribution'].values())*100:.2f}%"
                    for emotion, count in results["live_recognition"]["emotion_distribution"].items()
                }
            }
        }
    }
    return metrics

# Example usage with your results
results = {
    "image_recognition": {
        "avg_processing_time": 5.953442811965942,
        "std_processing_time": 0.0,
        "avg_confidence": 0.9877966046333313,
        "face_detection_rate": 1.0
    },
    "video_recognition": {
        "avg_processing_time": 66.17844605445862,
        "std_processing_time": 0.0,
        "avg_fps": 2.1003817449206363,
        "min_fps": 2.1003817449206363,
        "max_fps": 2.1003817449206363
    },
    "live_recognition": {
        "avg_frame_time": 0.24449560642242432,
        "std_frame_time": 0.22807584848388143,
        "avg_fps": 3.9242435437879197,
        "face_detection_rate": 0.9916666666666667,
        "emotion_distribution": {
            "Neutral": 107,
            "Happy": 12
        }
    }
}

detailed_metrics = calculate_accuracy_metrics(results)