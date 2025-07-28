from typing import Dict, List, Optional

import cv2
import numpy as np
from deepface import DeepFace

from .utils import CONFIG, EMOTIONS


class EmotionAnalyzer:
    def __init__(self):
        self.emotion_models = ["fer2013"]
        self.emotion_cache = {}
        self.cache_size = CONFIG.get("emotion_cache_size", 100)
        self.enable_caching = CONFIG.get("enable_emotion_caching", True)

    def analyze_emotion(self, face_region: np.ndarray) -> Optional[Dict]:
        if self.enable_caching:
            face_hash = hash(face_region.tobytes())
            if face_hash in self.emotion_cache:
                return self.emotion_cache[face_hash]

        try:
            face_rgb = cv2.cvtColor(face_region, cv2.COLOR_BGR2RGB)

            try:
                result = DeepFace.analyze(
                    face_rgb,
                    actions=["emotion"],
                    model=self.emotion_models[0],
                    enforce_detection=False,
                )
            except TypeError:
                result = DeepFace.analyze(
                    face_rgb,
                    actions=["emotion"],
                    enforce_detection=False,
                )

            if isinstance(result, list):
                result = result[0]

            emotions = result.get("emotion", {})
            dominant_emotion = max(emotions.items(), key=lambda x: x[1])[0]
            confidence = emotions.get(dominant_emotion, 0)

            if confidence >= CONFIG["emotion_confidence_threshold"]:
                emotion_result = {
                    "emotion": dominant_emotion,
                    "confidence": confidence,
                    "all_emotions": emotions,
                }

                if self.enable_caching:
                    face_hash = hash(face_region.tobytes())
                    if len(self.emotion_cache) >= self.cache_size:
                        oldest_key = next(iter(self.emotion_cache))
                        del self.emotion_cache[oldest_key]
                    self.emotion_cache[face_hash] = emotion_result

                return emotion_result

        except Exception as e:
            print(f"Error analyzing emotion: {e}")
            return None

    def analyze_faces_emotions(
        self, frame: np.ndarray, faces: List[Dict]
    ) -> List[Dict]:
        emotions = []

        for face in faces:
            try:
                face_region = self._extract_face_region(frame, face)
                if face_region is not None and face_region.size > 0:
                    emotion_result = self.analyze_emotion(face_region)
                    if emotion_result:
                        emotion_result["face_id"] = face.get("track_id", "unknown")
                        emotions.append(emotion_result)
            except Exception as e:
                print(f"Error processing face for emotion: {e}")
                continue

        return emotions

    def _extract_face_region(
        self, frame: np.ndarray, face: Dict
    ) -> Optional[np.ndarray]:
        try:
            x, y, w, h = face["bbox"]

            x = max(0, x)
            y = max(0, y)
            w = min(w, frame.shape[1] - x)
            h = min(h, frame.shape[0] - y)

            if w <= 0 or h <= 0:
                return None

            face_region = frame[y : y + h, x : x + w]

            if face_region.size == 0:
                return None

            min_size = 48
            if face_region.shape[0] < min_size or face_region.shape[1] < min_size:
                face_region = cv2.resize(face_region, (min_size, min_size))

            return face_region

        except Exception as e:
            print(f"Error extracting face region: {e}")
            return None

    def _update_emotion_track(
        self, emotion_tracks: Dict, emotion: str, confidence: float
    ):
        if emotion not in emotion_tracks:
            emotion_tracks[emotion] = {
                "count": 0,
                "total_confidence": 0,
                "max_confidence": 0,
            }

        emotion_tracks[emotion]["count"] += 1
        emotion_tracks[emotion]["total_confidence"] += confidence
        emotion_tracks[emotion]["max_confidence"] = max(
            emotion_tracks[emotion]["max_confidence"], confidence
        )

    def get_dominant_emotion(self, emotions: List[Dict]) -> Optional[str]:
        if not emotions:
            return None

        emotion_counts = {}
        for emotion_data in emotions:
            emotion = emotion_data.get("emotion", "unknown")
            emotion_counts[emotion] = emotion_counts.get(emotion, 0) + 1

        if emotion_counts:
            return max(emotion_counts.items(), key=lambda x: x[1])[0]
        return None

    def get_emotion_statistics(self) -> Dict:
        emotion_tracks = {}
        total_emotions = 0
        total_confidence = 0

        for emotion_data in self.emotion_cache.values():
            emotion = emotion_data.get("emotion", "unknown")
            confidence = emotion_data.get("confidence", 0)

            self._update_emotion_track(emotion_tracks, emotion, confidence)
            total_emotions += 1
            total_confidence += confidence

        dominant_emotion = None
        max_count = 0
        for emotion, stats in emotion_tracks.items():
            if stats["count"] > max_count:
                max_count = stats["count"]
                dominant_emotion = emotion

        avg_confidence = total_confidence / total_emotions if total_emotions > 0 else 0

        return {
            "total_emotions": total_emotions,
            "dominant_emotion": dominant_emotion,
            "average_confidence": avg_confidence,
            "emotion_distribution": emotion_tracks,
        }

    def detect_emotion_changes(
        self, emotions: List[Dict], threshold: float = 0.3
    ) -> List[Dict]:
        changes = []

        if len(emotions) < 2:
            return changes

        for i in range(1, len(emotions)):
            prev_emotion = emotions[i - 1].get("emotion", "unknown")
            curr_emotion = emotions[i].get("emotion", "unknown")

            if prev_emotion != curr_emotion:
                prev_confidence = emotions[i - 1].get("confidence", 0)
                curr_confidence = emotions[i].get("confidence", 0)

                confidence_change = abs(curr_confidence - prev_confidence)

                if confidence_change > threshold:
                    changes.append(
                        {
                            "from_emotion": prev_emotion,
                            "to_emotion": curr_emotion,
                            "confidence_change": confidence_change,
                            "frame_index": i,
                        }
                    )

        return changes

    def draw_emotions(self, frame: np.ndarray, emotions: List[Dict]) -> np.ndarray:
        frame_copy = frame.copy()

        for emotion_data in emotions:
            emotion = emotion_data.get("emotion", "unknown")
            confidence = emotion_data.get("confidence", 0)
            face_id = emotion_data.get("face_id", "unknown")

            color = self._get_emotion_color(emotion)

            label = f"{emotion}: {confidence:.2f}"

            cv2.putText(
                frame_copy,
                label,
                (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                color,
                2,
            )

        return frame_copy

    def _get_emotion_color(self, emotion: str) -> tuple:
        emotion_colors = {
            "happy": (0, 255, 0),
            "sad": (255, 0, 0),
            "angry": (0, 0, 255),
            "fear": (255, 255, 0),
            "surprise": (255, 0, 255),
            "disgust": (0, 255, 255),
            "neutral": (128, 128, 128),
        }
        return emotion_colors.get(emotion, (255, 255, 255))
