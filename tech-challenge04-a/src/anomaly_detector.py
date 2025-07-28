from typing import Dict, List, Optional

import cv2
import numpy as np
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler

from .utils import CONFIG


class AnomalyDetector:
    def __init__(self):
        self.isolation_forest = IsolationForest(
            contamination=0.1,
            random_state=42,
            n_estimators=100,
        )
        self.scaler = StandardScaler()

        self.movement_history = []
        self.pose_history = []
        self.emotion_history = []
        self.activity_history = []

        self.anomaly_threshold = CONFIG["anomaly_threshold"]
        self.history_size = 50

    def detect_anomalies(
        self,
        frame: np.ndarray,
        faces: List[Dict],
        emotions: List[Dict],
        activities: List[Dict],
    ) -> List[Dict]:
        anomalies = []

        movement_anomalies = self._detect_movement_anomalies(faces)
        anomalies.extend(movement_anomalies)

        pose_anomalies = self._detect_pose_anomalies(activities)
        anomalies.extend(pose_anomalies)

        emotion_anomalies = self._detect_emotion_anomalies(emotions)
        anomalies.extend(emotion_anomalies)

        activity_anomalies = self._detect_activity_anomalies(activities)
        anomalies.extend(activity_anomalies)

        return anomalies

    def _update_movement_tracking(self, faces: List[Dict]):
        if not faces:
            return

        current_movements = []
        for face in faces:
            center = face.get("center")
            if center is None:
                continue

            current_movements.append(center)

        if current_movements:
            self.movement_history.append(current_movements)

            if len(self.movement_history) > self.history_size:
                self.movement_history.pop(0)

    def _detect_movement_anomalies(self, faces: List[Dict]) -> List[Dict]:
        self._update_movement_tracking(faces)
        anomalies = []

        if len(self.movement_history) < 5:
            return anomalies

        for i, face in enumerate(faces):
            center = face.get("center")
            if center is None:
                continue

            movement_pattern = self._calculate_movement_pattern(center, i)

            if self._is_unusual_movement(movement_pattern):
                anomalies.append(
                    {
                        "type": "unusual_movement",
                        "confidence": 0.8,
                        "face_id": face.get("track_id", "unknown"),
                        "description": "Detected unusual movement pattern",
                    }
                )

        return anomalies

    def _detect_pose_anomalies(self, activities: List[Dict]) -> List[Dict]:
        anomalies = []

        for activity in activities:
            pose_confidence = activity.get("pose_confidence", 0)

            if pose_confidence < 0.3:
                anomalies.append(
                    {
                        "type": "low_pose_confidence",
                        "confidence": 0.7,
                        "person_id": activity.get("person_id", "unknown"),
                        "description": "Low pose detection confidence",
                    }
                )

        return anomalies

    def _detect_emotion_anomalies(self, emotions: List[Dict]) -> List[Dict]:
        anomalies = []

        for emotion in emotions:
            emotion_type = emotion.get("emotion", "unknown")
            confidence = emotion.get("confidence", 0)

            if emotion_type in ["fear", "angry", "disgust"] and confidence > 0.7:
                anomalies.append(
                    {
                        "type": "negative_emotion",
                        "confidence": confidence,
                        "face_id": emotion.get("face_id", "unknown"),
                        "description": f"Detected {emotion_type} emotion",
                    }
                )

        return anomalies

    def _detect_activity_anomalies(self, activities: List[Dict]) -> List[Dict]:
        anomalies = []

        for activity in activities:
            activity_type = activity.get("activity", "unknown")
            velocity = activity.get("velocity", 0)

            if velocity > 0.3:
                anomalies.append(
                    {
                        "type": "high_movement",
                        "confidence": min(velocity, 1.0),
                        "person_id": activity.get("person_id", "unknown"),
                        "description": "Detected high movement activity",
                    }
                )

        return anomalies

    def _calculate_movement_pattern(
        self, current_center: tuple, face_index: int
    ) -> np.ndarray:
        if len(self.movement_history) < 3:
            return np.array([0, 0])

        recent_movements = []
        for movements in self.movement_history[-3:]:
            if face_index < len(movements):
                recent_movements.append(movements[face_index])

        if len(recent_movements) < 2:
            return np.array([0, 0])

        velocities = []
        for i in range(1, len(recent_movements)):
            prev = recent_movements[i - 1]
            curr = recent_movements[i]

            velocity = np.sqrt((curr[0] - prev[0]) ** 2 + (curr[1] - prev[1]) ** 2)
            velocities.append(velocity)

        return np.array(velocities)

    def _is_unusual_movement(self, movement_pattern: np.ndarray) -> bool:
        if len(movement_pattern) == 0:
            return False

        if len(self.movement_history) < 10:
            return False

        try:
            if not hasattr(self, "_movement_model_fitted"):
                self._fit_movement_model()

            features = movement_pattern.reshape(1, -1)
            features_scaled = self.scaler.transform(features)
            prediction = self.isolation_forest.predict(features_scaled)

            return prediction[0] == -1
        except Exception:
            return False

    def _fit_movement_model(self):
        if len(self.movement_history) < 10:
            return

        all_movements = []
        for movements in self.movement_history:
            for movement in movements:
                if len(movement) >= 2:
                    all_movements.append([movement[0], movement[1]])

        if len(all_movements) < 5:
            return

        try:
            movements_array = np.array(all_movements)
            movements_scaled = self.scaler.fit_transform(movements_array)
            self.isolation_forest.fit(movements_scaled)
            self._movement_model_fitted = True
        except Exception:
            pass

    def get_anomaly_statistics(self) -> Dict:
        total_anomalies = 0
        anomaly_types = {}
        total_confidence = 0

        for anomaly in self._get_all_anomalies():
            anomaly_type = anomaly.get("type", "unknown")
            confidence = anomaly.get("confidence", 0)

            if anomaly_type not in anomaly_types:
                anomaly_types[anomaly_type] = {
                    "count": 0,
                    "total_confidence": 0,
                    "max_confidence": 0,
                }

            anomaly_types[anomaly_type]["count"] += 1
            anomaly_types[anomaly_type]["total_confidence"] += confidence
            anomaly_types[anomaly_type]["max_confidence"] = max(
                anomaly_types[anomaly_type]["max_confidence"], confidence
            )

            total_anomalies += 1
            total_confidence += confidence

        avg_confidence = (
            total_confidence / total_anomalies if total_anomalies > 0 else 0
        )

        return {
            "total_anomalies": total_anomalies,
            "average_confidence": avg_confidence,
            "anomaly_types": anomaly_types,
        }

    def _get_all_anomalies(self) -> List[Dict]:
        return []

    def draw_anomalies(self, frame: np.ndarray, anomalies: List[Dict]) -> np.ndarray:
        frame_copy = frame.copy()

        for anomaly in anomalies:
            anomaly_type = anomaly.get("type", "unknown")
            confidence = anomaly.get("confidence", 0)

            color = self._get_anomaly_color(anomaly_type)

            label = f"Anomaly: {anomaly_type} ({confidence:.2f})"

            cv2.putText(
                frame_copy,
                label,
                (10, 90),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                color,
                2,
            )

        return frame_copy

    def _get_anomaly_color(self, anomaly_type: str) -> tuple:
        anomaly_colors = {
            "unusual_movement": (0, 0, 255),
            "low_pose_confidence": (255, 0, 0),
            "negative_emotion": (0, 255, 255),
            "high_movement": (255, 255, 0),
        }
        return anomaly_colors.get(anomaly_type, (255, 255, 255))
