from typing import Dict, List, Tuple

import cv2
import mediapipe as mp
import numpy as np
from ultralytics import YOLO

from .utils import ACTIVITIES, CONFIG


class ActivityDetector:
    def __init__(self):
        self.mp_pose = mp.solutions.pose
        self.pose = self.mp_pose.Pose(
            static_image_mode=False,
            model_complexity=1,
            smooth_landmarks=True,
            enable_segmentation=False,
            smooth_segmentation=True,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5,
        )

        self.yolo_model = YOLO("yolov8n.pt")

        self.activity_tracks = {}
        self.next_activity_id = 0

    def detect_activities(self, frame: np.ndarray) -> List[Dict]:
        activities = []

        people = self._detect_people(frame)

        for person in people:
            person_region = self._extract_person_region(frame, person)
            if person_region is not None:
                pose_analysis = self._analyze_pose(person_region)
                activity = self._classify_activity(pose_analysis, person)

                if activity:
                    activity["person_id"] = person.get("track_id", "unknown")
                    activities.append(activity)

        return activities

    def _detect_people(self, frame: np.ndarray) -> List[Dict]:
        results = self.yolo_model(frame, verbose=False)
        people = []

        for result in results:
            boxes = result.boxes
            if boxes is not None:
                for box in boxes:
                    if box.cls == 0:  # person class
                        x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                        confidence = box.conf[0].cpu().numpy()

                        if confidence > CONFIG["activity_confidence_threshold"]:
                            people.append(
                                {
                                    "bbox": [
                                        int(x1),
                                        int(y1),
                                        int(x2 - x1),
                                        int(y2 - y1),
                                    ],
                                    "confidence": float(confidence),
                                    "center": (int((x1 + x2) / 2), int((y1 + y2) / 2)),
                                }
                            )

        return people

    def _extract_person_region(self, frame: np.ndarray, person: Dict) -> np.ndarray:
        x, y, w, h = person["bbox"]

        x = max(0, x)
        y = max(0, y)
        w = min(w, frame.shape[1] - x)
        h = min(h, frame.shape[0] - y)

        if w <= 0 or h <= 0:
            return None

        person_region = frame[y : y + h, x : x + w]
        return person_region

    def _analyze_pose(self, person_region: np.ndarray) -> Dict:
        rgb_region = cv2.cvtColor(person_region, cv2.COLOR_BGR2RGB)
        results = self.pose.process(rgb_region)

        pose_data = {
            "landmarks": [],
            "movement_velocity": 0,
            "pose_confidence": 0,
        }

        if results.pose_landmarks:
            landmarks = results.pose_landmarks.landmark
            pose_data["landmarks"] = landmarks
            pose_data["pose_confidence"] = results.pose_landmarks.landmark[0].visibility

            if len(landmarks) > 0:
                velocity = self._calculate_movement_velocity(landmarks)
                pose_data["movement_velocity"] = velocity

        return pose_data

    def _classify_activity(self, pose_analysis: Dict, person: Dict) -> Dict:
        if not pose_analysis["landmarks"]:
            return None

        landmarks = pose_analysis["landmarks"]
        velocity = pose_analysis["movement_velocity"]
        confidence = pose_analysis["pose_confidence"]

        if confidence < 0.5:
            return None

        activity = "unknown"
        activity_confidence = 0.0

        if velocity > 0.1:
            activity = "walking"
            activity_confidence = min(velocity * 2, 1.0)
        elif velocity > 0.05:
            activity = "gesturing"
            activity_confidence = min(velocity * 3, 1.0)
        else:
            activity = "sitting"
            activity_confidence = 0.8

        if activity_confidence >= CONFIG["activity_confidence_threshold"]:
            return {
                "activity": activity,
                "confidence": activity_confidence,
                "velocity": velocity,
                "pose_confidence": confidence,
            }

        return None

    def _calculate_movement_velocity(self, landmarks: List) -> float:
        if len(landmarks) < 17:
            return 0.0

        key_points = [0, 11, 12, 13, 14, 15, 16]
        velocities = []

        for i in range(len(key_points) - 1):
            point1 = landmarks[key_points[i]]
            point2 = landmarks[key_points[i + 1]]

            if point1.visibility > 0.5 and point2.visibility > 0.5:
                distance = np.sqrt(
                    (point1.x - point2.x) ** 2 + (point1.y - point2.y) ** 2
                )
                velocities.append(distance)

        return np.mean(velocities) if velocities else 0.0

    def _update_activity_track(
        self, activity_tracks: Dict, activity: str, confidence: float
    ):
        if activity not in activity_tracks:
            activity_tracks[activity] = {
                "count": 0,
                "total_confidence": 0,
                "max_confidence": 0,
            }

        activity_tracks[activity]["count"] += 1
        activity_tracks[activity]["total_confidence"] += confidence
        activity_tracks[activity]["max_confidence"] = max(
            activity_tracks[activity]["max_confidence"], confidence
        )

    def get_activity_statistics(self) -> Dict:
        activity_tracks = {}
        total_activities = 0
        total_confidence = 0

        for activity_data in self.activity_tracks.values():
            activity = activity_data.get("activity", "unknown")
            confidence = activity_data.get("confidence", 0)

            self._update_activity_track(activity_tracks, activity, confidence)
            total_activities += 1
            total_confidence += confidence

        dominant_activity = None
        max_count = 0
        for activity, stats in activity_tracks.items():
            if stats["count"] > max_count:
                max_count = stats["count"]
                dominant_activity = activity

        avg_confidence = (
            total_confidence / total_activities if total_activities > 0 else 0
        )

        return {
            "total_activities": total_activities,
            "dominant_activity": dominant_activity,
            "average_confidence": avg_confidence,
            "activity_distribution": activity_tracks,
        }

    def draw_activities(self, frame: np.ndarray, activities: List[Dict]) -> np.ndarray:
        frame_copy = frame.copy()

        for activity_data in activities:
            activity = activity_data.get("activity", "unknown")
            confidence = activity_data.get("confidence", 0)
            person_id = activity_data.get("person_id", "unknown")

            color = self._get_activity_color(activity)

            label = f"{activity}: {confidence:.2f}"

            cv2.putText(
                frame_copy,
                label,
                (10, 60),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                color,
                2,
            )

        return frame_copy

    def _get_activity_color(self, activity: str) -> tuple:
        activity_colors = {
            "walking": (0, 255, 0),
            "sitting": (255, 0, 0),
            "standing": (0, 0, 255),
            "gesturing": (255, 255, 0),
            "talking": (255, 0, 255),
            "writing": (0, 255, 255),
            "typing": (128, 128, 128),
        }
        return activity_colors.get(activity, (255, 255, 255))
