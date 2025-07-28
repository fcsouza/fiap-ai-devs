from typing import Dict, List, Tuple

import cv2
import mediapipe as mp
import numpy as np

from .utils import CONFIG


class FaceDetector:
    def __init__(self):
        self.mp_face_detection = mp.solutions.face_detection
        self.mp_drawing = mp.solutions.drawing_utils

        self.face_detection = self.mp_face_detection.FaceDetection(
            model_selection=1,
            min_detection_confidence=CONFIG["face_detection_confidence"],
        )

        self.face_tracks = {}
        self.next_track_id = 0
        self.max_track_age = 30
        self.min_track_hits = 3
        self.max_distance = 150
        self.face_embeddings = {}

        self.face_clusters = {}
        self.cluster_embeddings = {}
        self.cluster_threshold = 0.5
        self.next_cluster_id = 0

    def detect_faces(self, frame: np.ndarray) -> List[Dict]:
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        results = self.face_detection.process(rgb_frame)

        faces = []
        if results.detections:
            height, width = frame.shape[:2]

            for detection in results.detections:
                bbox = detection.location_data.relative_bounding_box
                x = int(bbox.xmin * width)
                y = int(bbox.ymin * height)
                w = int(bbox.width * width)
                h = int(bbox.height * height)

                x = max(0, x)
                y = max(0, y)
                w = min(w, width - x)
                h = min(h, height - y)

                face_info = {
                    "bbox": [x, y, w, h],
                    "confidence": detection.score[0],
                    "center": (x + w // 2, y + h // 2),
                    "area": w * h,
                    "frame_width": width,
                    "frame_height": height,
                }

                if detection.location_data.relative_keypoints:
                    keypoints = []
                    for kp in detection.location_data.relative_keypoints:
                        kp_x = int(kp.x * width)
                        kp_y = int(kp.y * height)
                        keypoints.append((kp_x, kp_y))
                    face_info["keypoints"] = keypoints

                faces.append(face_info)

        return faces

    def track_faces(self, faces: List[Dict], frame_number: int) -> List[Dict]:
        if not faces:
            for track_id in list(self.face_tracks.keys()):
                self.face_tracks[track_id]["lost_frames"] = (
                    self.face_tracks[track_id].get("lost_frames", 0) + 1
                )
                if self.face_tracks[track_id]["lost_frames"] > self.max_track_age:
                    del self.face_tracks[track_id]
            return []

        face_embeddings = []
        for face in faces:
            embedding = self._calculate_face_embedding(face)
            face_embeddings.append(embedding)

        for track_id in list(self.face_tracks.keys()):
            track = self.face_tracks[track_id]
            track["lost_frames"] = track.get("lost_frames", 0) + 1

            if track["lost_frames"] > self.max_track_age:
                del self.face_tracks[track_id]
                continue

            best_match_idx = self._find_best_match(track, faces, face_embeddings)
            if best_match_idx >= 0:
                matched_face = faces[best_match_idx]
                matched_embedding = face_embeddings[best_match_idx]

                track["bbox"] = matched_face["bbox"]
                track["confidence"] = matched_face["confidence"]
                track["center"] = matched_face["center"]
                track["last_center"] = matched_face["center"]
                track["last_embedding"] = matched_embedding
                track["lost_frames"] = 0
                track["hits"] = track.get("hits", 0) + 1

                faces.pop(best_match_idx)
                face_embeddings.pop(best_match_idx)

        for face, embedding in zip(faces, face_embeddings):
            track_id = self.next_track_id
            self.next_track_id += 1

            self.face_tracks[track_id] = {
                "track_id": track_id,
                "bbox": face["bbox"],
                "confidence": face["confidence"],
                "center": face["center"],
                "last_center": face["center"],
                "last_embedding": embedding,
                "hits": 1,
                "lost_frames": 0,
            }

        tracked_faces = []
        for track_id, track in self.face_tracks.items():
            if track.get("hits", 0) >= self.min_track_hits:
                cluster_id = self._assign_cluster_id(track)
                tracked_faces.append(
                    {
                        "bbox": track["bbox"],
                        "confidence": track["confidence"],
                        "track_id": track["track_id"],
                        "center": track["last_center"],
                        "cluster_id": cluster_id,
                    }
                )

        return tracked_faces

    def _calculate_face_embedding(self, face: Dict) -> np.ndarray:
        x, y, w, h = face["bbox"]
        center_x, center_y = face["center"]
        confidence = face["confidence"]

        aspect_ratio = w / h if h > 0 else 1.0
        area = w * h
        area_ratio = area / (
            face.get("frame_width", 640) * face.get("frame_height", 480)
        )

        embedding = np.array(
            [
                center_x / 1000.0,
                center_y / 1000.0,
                w / 1000.0,
                h / 1000.0,
                aspect_ratio,
                area_ratio,
                confidence,
            ]
        )
        return embedding

    def _find_best_match(
        self, track: Dict, faces: List[Dict], face_embeddings: List[np.ndarray]
    ) -> int:
        if "last_embedding" not in track:
            return -1

        track_embedding = track["last_embedding"]
        track_center = track.get("last_center", (0, 0))

        best_match_idx = -1
        min_distance = float("inf")

        for i, (face, embedding) in enumerate(zip(faces, face_embeddings)):
            face_center = face["center"]

            embedding_distance = np.linalg.norm(track_embedding - embedding)
            spatial_distance = np.linalg.norm(
                np.array(track_center) - np.array(face_center)
            )

            if track.get("hits", 0) >= 3:
                distance = embedding_distance * 0.7 + spatial_distance * 0.3
            else:
                distance = embedding_distance * 0.3 + spatial_distance * 0.7

            if distance < min_distance and distance < self.max_distance:
                min_distance = distance
                best_match_idx = i

        return best_match_idx

    def _assign_cluster_id(self, track: Dict) -> int:
        if "last_embedding" not in track:
            return -1

        track_embedding = track["last_embedding"]
        track_id = track["track_id"]

        for cluster_id, track_ids in self.face_clusters.items():
            if track_id in track_ids:
                self._update_cluster_embedding(cluster_id, track_embedding)
                return cluster_id

        best_cluster_id = -1
        min_distance = float("inf")

        for cluster_id, cluster_embedding in self.cluster_embeddings.items():
            distance = np.linalg.norm(track_embedding - cluster_embedding)
            if distance < min_distance and distance < self.cluster_threshold:
                min_distance = distance
                best_cluster_id = cluster_id

        if best_cluster_id >= 0:
            self.face_clusters[best_cluster_id].append(track_id)
            self._update_cluster_embedding(best_cluster_id, track_embedding)
            return best_cluster_id
        else:
            new_cluster_id = self.next_cluster_id
            self.next_cluster_id += 1
            self.cluster_embeddings[new_cluster_id] = track_embedding
            self.face_clusters[new_cluster_id] = [track_id]
            return new_cluster_id

    def _update_cluster_embedding(self, cluster_id: int, new_embedding: np.ndarray):
        if cluster_id in self.cluster_embeddings:
            current_embedding = self.cluster_embeddings[cluster_id]
            cluster_size = len(self.face_clusters.get(cluster_id, []))
            alpha = 1.0 / (cluster_size + 1)
            updated_embedding = (1 - alpha) * current_embedding + alpha * new_embedding
            self.cluster_embeddings[cluster_id] = updated_embedding
        else:
            self.cluster_embeddings[cluster_id] = new_embedding

    def get_cluster_statistics(self) -> Dict:
        self._merge_similar_clusters()

        stats = {
            "total_clusters": len(self.face_clusters),
            "total_tracks": len(self.face_tracks),
            "clusters": {},
        }

        for cluster_id, track_ids in self.face_clusters.items():
            stats["clusters"][cluster_id] = {
                "track_count": len(track_ids),
                "track_ids": track_ids,
            }
        return stats

    def _merge_similar_clusters(self):
        clusters_to_merge = []

        for cluster_id1 in list(self.cluster_embeddings.keys()):
            for cluster_id2 in list(self.cluster_embeddings.keys()):
                if cluster_id1 >= cluster_id2:
                    continue

                embedding1 = self.cluster_embeddings[cluster_id1]
                embedding2 = self.cluster_embeddings[cluster_id2]
                distance = np.linalg.norm(embedding1 - embedding2)

                if distance < self.cluster_threshold * 0.7:
                    clusters_to_merge.append((cluster_id1, cluster_id2))

        for cluster_id1, cluster_id2 in clusters_to_merge:
            if cluster_id1 in self.face_clusters and cluster_id2 in self.face_clusters:
                self.face_clusters[cluster_id1].extend(self.face_clusters[cluster_id2])

                embedding1 = self.cluster_embeddings[cluster_id1]
                embedding2 = self.cluster_embeddings[cluster_id2]
                merged_embedding = (embedding1 + embedding2) / 2
                self.cluster_embeddings[cluster_id1] = merged_embedding

                del self.face_clusters[cluster_id2]
                del self.cluster_embeddings[cluster_id2]

    def extract_face_region(self, frame: np.ndarray, face: Dict) -> np.ndarray:
        x, y, w, h = face["bbox"]
        face_region = frame[y : y + h, x : x + w]
        return face_region

    def draw_faces(self, frame: np.ndarray, faces: List[Dict]) -> np.ndarray:
        frame_copy = frame.copy()

        for i, face in enumerate(faces):
            x, y, w, h = face["bbox"]
            confidence = face.get("confidence", 0)
            track_id = face.get("track_id", i)
            cluster_id = face.get("cluster_id", -1)

            color = (0, 255, 0) if confidence > 0.7 else (0, 165, 255)

            cv2.rectangle(frame_copy, (x, y), (x + w, y + h), color, 2)

            label = f"Face {track_id}"
            if cluster_id >= 0:
                label += f" (C{cluster_id})"
            label += f": {confidence:.2f}"

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

    def get_face_statistics(self, faces: List[Dict]) -> Dict:
        if not faces:
            return {
                "total_faces": 0,
                "average_confidence": 0.0,
                "face_sizes": [],
                "dominant_face_size": "unknown",
            }

        confidences = [face.get("confidence", 0) for face in faces]
        areas = [face.get("area", 0) for face in faces]

        avg_confidence = sum(confidences) / len(confidences) if confidences else 0

        size_categories = []
        for area in areas:
            if area < 1000:
                size_categories.append("small")
            elif area < 5000:
                size_categories.append("medium")
            else:
                size_categories.append("large")

        dominant_size = (
            max(set(size_categories), key=size_categories.count)
            if size_categories
            else "unknown"
        )

        return {
            "total_faces": len(faces),
            "average_confidence": avg_confidence,
            "face_sizes": size_categories,
            "dominant_face_size": dominant_size,
        }
