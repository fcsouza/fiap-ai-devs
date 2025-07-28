from typing import Dict, Generator, List, Tuple

import cv2
import numpy as np

from .utils import CONFIG, format_duration, get_video_info


class VideoProcessor:
    def __init__(self, video_path: str):
        self.video_path = video_path
        self.cap = None
        self.frame_skip = CONFIG.get("frame_skip", 1)

    def __enter__(self):
        self.cap = cv2.VideoCapture(self.video_path)
        if not self.cap.isOpened():
            raise ValueError(f"Could not open video file: {self.video_path}")
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        if self.cap:
            self.cap.release()

    def get_frame_generator(self) -> Generator[Tuple[int, np.ndarray], None, None]:
        frame_number = 0
        while True:
            ret, frame = self.cap.read()
            if not ret:
                break

            if frame_number % self.frame_skip == 0:
                frame = self._preprocess_frame(frame)
                yield frame_number, frame

            frame_number += 1

    def extract_frames(
        self, start_time: float = 0, end_time: float = None
    ) -> List[np.ndarray]:
        frames = []

        fps = self.cap.get(cv2.CAP_PROP_FPS)
        start_frame = int(start_time * fps)

        if end_time:
            end_frame = int(end_time * fps)
        else:
            end_frame = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))

        self.cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)

        frame_number = start_frame
        while frame_number < end_frame:
            ret, frame = self.cap.read()
            if not ret:
                break

            if frame_number % self.frame_skip == 0:
                frame = self._preprocess_frame(frame)
                frames.append(frame)

            frame_number += 1

        return frames

    def get_frame_at_time(self, time_seconds: float) -> np.ndarray:
        fps = self.cap.get(cv2.CAP_PROP_FPS)
        frame_number = int(time_seconds * fps)

        self.cap.set(cv2.CAP_PROP_POS_FRAMES, frame_number)
        ret, frame = self.cap.read()

        if ret:
            return self._preprocess_frame(frame)
        return None

    def save_frame(self, frame: np.ndarray, output_path: str) -> bool:
        try:
            cv2.imwrite(output_path, frame)
            return True
        except Exception as e:
            print(f"Error saving frame: {e}")
            return False

    def get_video_summary(self) -> Dict:
        video_info = get_video_info(self.video_path)

        return {
            "file_path": self.video_path,
            "duration": video_info["duration"],
            "duration_formatted": format_duration(video_info["duration"]),
            "total_frames": video_info["frame_count"],
            "fps": video_info["fps"],
            "resolution": f"{video_info['width']}x{video_info['height']}",
            "width": video_info["width"],
            "height": video_info["height"],
        }

    def _preprocess_frame(self, frame: np.ndarray) -> np.ndarray:
        height, width = frame.shape[:2]
        max_width = CONFIG.get("max_frame_width", 1280)

        if width > max_width:
            scale = max_width / width
            new_width = int(width * scale)
            new_height = int(height * scale)
            frame = cv2.resize(frame, (new_width, new_height))

        return frame
