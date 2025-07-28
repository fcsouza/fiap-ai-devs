import json
import os
from datetime import datetime
from typing import Any, Dict, List, Tuple

import cv2
import numpy as np

CONFIG = {
    "face_detection_confidence": 0.5,
    "emotion_confidence_threshold": 0.6,
    "activity_confidence_threshold": 0.7,
    "anomaly_threshold": 0.8,
    "max_faces": 10,
    "frame_skip": 3,
    "output_dir": "output",
    "temp_dir": "temp",
    "max_frame_width": 640,
    "emotion_cache_size": 100,
    "enable_emotion_caching": True,
    "parallel_processing": False,
    "skip_emotion_analysis": False,
    "skip_anomaly_detection": False,
}

EMOTIONS = ["angry", "disgust", "fear", "happy", "sad", "surprise", "neutral"]

ACTIVITIES = [
    "walking",
    "sitting",
    "standing",
    "gesturing",
    "talking",
    "writing",
    "typing",
]


def create_directories():
    directories = [CONFIG["output_dir"], CONFIG["temp_dir"]]
    for directory in directories:
        os.makedirs(directory, exist_ok=True)


def get_video_info(video_path: str) -> Dict[str, Any]:
    cap = cv2.VideoCapture(video_path)

    if not cap.isOpened():
        raise ValueError(f"Could not open video file: {video_path}")

    info = {
        "fps": cap.get(cv2.CAP_PROP_FPS),
        "frame_count": int(cap.get(cv2.CAP_PROP_FRAME_COUNT)),
        "width": int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
        "height": int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
        "duration": int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) / cap.get(cv2.CAP_PROP_FPS),
    }

    cap.release()
    return info


def preprocess_frame(frame: np.ndarray) -> np.ndarray:
    height, width = frame.shape[:2]
    max_width = CONFIG.get("max_frame_width", 1280)

    if width > max_width:
        scale = max_width / width
        new_width = int(width * scale)
        new_height = int(height * scale)
        frame = cv2.resize(frame, (new_width, new_height))

    return frame


def draw_face_boxes(
    frame: np.ndarray, faces: List[Dict], emotions: List[str] = None
) -> np.ndarray:
    frame_copy = frame.copy()

    for i, face in enumerate(faces):
        x, y, w, h = face["bbox"]
        confidence = face.get("confidence", 0)

        color = (
            (0, 255, 0)
            if confidence > CONFIG["face_detection_confidence"]
            else (0, 165, 255)
        )

        cv2.rectangle(frame_copy, (x, y), (x + w, y + h), color, 2)

        label = f"Face {i+1}: {confidence:.2f}"
        if emotions and i < len(emotions):
            if isinstance(emotions[i], dict):
                emotion = emotions[i].get("emotion", "unknown")
                confidence_emotion = emotions[i].get("confidence", 0)
                label += f" | {emotion}: {confidence_emotion:.2f}"
            else:
                label += f" | {emotions[i]}"

        cv2.putText(
            frame_copy,
            label,
            (x, y - 10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            color,
            1,
        )

    return frame_copy


def save_results(results: Dict[str, Any], filename: str = None):
    if filename is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"analysis_results_{timestamp}.json"

    filepath = os.path.join(CONFIG["output_dir"], filename)

    with open(filepath, "w") as f:
        json.dump(results, f, indent=2, default=str)

    return filepath


def calculate_statistics(data: List[Any]) -> Dict[str, Any]:
    if not data:
        return {"count": 0, "mean": 0, "std": 0, "min": 0, "max": 0}

    numeric_data = [float(x) for x in data if isinstance(x, (int, float))]

    if not numeric_data:
        return {"count": len(data), "mean": 0, "std": 0, "min": 0, "max": 0}

    mean = sum(numeric_data) / len(numeric_data)
    variance = sum((x - mean) ** 2 for x in numeric_data) / len(numeric_data)
    std = variance**0.5

    return {
        "count": len(numeric_data),
        "mean": mean,
        "std": std,
        "min": min(numeric_data),
        "max": max(numeric_data),
    }


def format_duration(seconds: float) -> str:
    if seconds < 60:
        return f"{seconds:.1f}s"
    elif seconds < 3600:
        minutes = seconds / 60
        return f"{minutes:.1f}m"
    else:
        hours = seconds / 3600
        return f"{hours:.1f}h"
