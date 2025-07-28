#!/usr/bin/env python3

import argparse
import os
import sys
import time
from pathlib import Path

sys.path.append(os.path.join(os.path.dirname(__file__), "src"))

from src.activity_detector import ActivityDetector
from src.anomaly_detector import AnomalyDetector
from src.emotion_analyzer import EmotionAnalyzer
from src.face_detector import FaceDetector
from src.report_generator import ReportGenerator
from src.utils import CONFIG, create_directories
from src.video_processor import VideoProcessor


class VideoAnalyzer:
    def __init__(
        self,
        video_path: str,
        output_dir: str = None,
        show_preview: bool = False,
        fast_mode: bool = False,
    ):
        self.video_path = video_path
        self.output_dir = output_dir or CONFIG["output_dir"]
        self.show_preview = show_preview
        self.fast_mode = fast_mode

        if fast_mode:
            CONFIG["frame_skip"] = 5
            CONFIG["max_frame_width"] = 480
            CONFIG["skip_emotion_analysis"] = True
            CONFIG["skip_anomaly_detection"] = True
            CONFIG["enable_emotion_caching"] = False

        create_directories()

        self.face_detector = FaceDetector()
        self.emotion_analyzer = EmotionAnalyzer()
        self.activity_detector = ActivityDetector()
        self.anomaly_detector = AnomalyDetector()
        self.report_generator = ReportGenerator(self.output_dir)

        self.analysis_results = {
            "frames_processed": 0,
            "faces_detected": 0,
            "emotions_analyzed": 0,
            "activities_detected": 0,
            "anomalies_detected": 0,
            "processing_time": 0,
            "unique_faces": set(),
            "unique_emotions": set(),
            "unique_activities": set(),
            "unique_anomalies": set(),
            "frames_with_faces": 0,
            "frames_with_emotions": 0,
            "frames_with_activities": 0,
            "frames_with_anomalies": 0,
        }

    def analyze_video(self) -> dict:
        print("Starting video analysis...")
        start_time = time.time()

        try:
            with VideoProcessor(self.video_path) as video_processor:
                video_info = video_processor.get_video_summary()
                print(
                    f"Video: {video_info['duration_formatted']} ({video_info['total_frames']} frames)"
                )

                frame_count = 0
                for frame_number, frame in video_processor.get_frame_generator():
                    frame_count += 1

                    if frame_count % 30 == 0:
                        elapsed_time = time.time() - start_time
                        fps = frame_count / elapsed_time if elapsed_time > 0 else 0
                        eta = (
                            (video_info["total_frames"] - frame_count) / fps
                            if fps > 0
                            else 0
                        )
                        print(
                            f"Processing frame {frame_number}/{video_info['total_frames']} "
                            f"({frame_count/video_info['total_frames']*100:.1f}%) "
                            f"- Speed: {fps:.1f} fps - ETA: {eta/60:.1f} min"
                        )

                    try:
                        faces = self.face_detector.detect_faces(frame)
                        tracked_faces = self.face_detector.track_faces(
                            faces, frame_number
                        )
                    except Exception as e:
                        print(f"Error in face detection: {e}")
                        faces = []
                        tracked_faces = []

                    emotions = []
                    if tracked_faces and not CONFIG.get("skip_emotion_analysis", False):
                        try:
                            emotions = self.emotion_analyzer.analyze_faces_emotions(
                                frame, tracked_faces
                            )
                        except Exception as e:
                            print(f"Error in emotion analysis: {e}")
                            emotions = []

                    activities = []
                    try:
                        activities = self.activity_detector.detect_activities(frame)
                    except Exception as e:
                        print(f"Error in activity detection: {e}")
                        activities = []

                    anomalies = []
                    if not CONFIG.get("skip_anomaly_detection", False):
                        try:
                            anomalies = self.anomaly_detector.detect_anomalies(
                                frame, tracked_faces, emotions, activities
                            )
                        except Exception as e:
                            print(f"Error in anomaly detection: {e}")
                            anomalies = []

                    self.analysis_results["frames_processed"] += 1

                    if tracked_faces:
                        self.analysis_results["frames_with_faces"] += 1
                    if emotions:
                        self.analysis_results["frames_with_emotions"] += 1
                    if activities:
                        self.analysis_results["frames_with_activities"] += 1
                    if anomalies:
                        self.analysis_results["frames_with_anomalies"] += 1

                    if tracked_faces:
                        for face in tracked_faces:
                            if "cluster_id" in face and face["cluster_id"] >= 0:
                                face_id = f"cluster_{face['cluster_id']}"
                                self.analysis_results["unique_faces"].add(face_id)
                            elif "track_id" in face:
                                face_id = f"track_{face['track_id']}"
                                self.analysis_results["unique_faces"].add(face_id)

                    for emotion in emotions:
                        if isinstance(emotion, dict) and "emotion" in emotion:
                            self.analysis_results["unique_emotions"].add(
                                emotion["emotion"]
                            )
                        elif isinstance(emotion, str):
                            self.analysis_results["unique_emotions"].add(emotion)

                    for activity in activities:
                        if isinstance(activity, dict) and "activity" in activity:
                            self.analysis_results["unique_activities"].add(
                                activity["activity"]
                            )
                        elif isinstance(activity, str):
                            self.analysis_results["unique_activities"].add(activity)

                    for anomaly in anomalies:
                        if isinstance(anomaly, dict) and "type" in anomaly:
                            self.analysis_results["unique_anomalies"].add(
                                anomaly["type"]
                            )
                        elif isinstance(anomaly, str):
                            self.analysis_results["unique_anomalies"].add(anomaly)

                    self.analysis_results["faces_detected"] = len(
                        self.analysis_results["unique_faces"]
                    )
                    self.analysis_results["emotions_analyzed"] = len(
                        self.analysis_results["unique_emotions"]
                    )
                    self.analysis_results["activities_detected"] = len(
                        self.analysis_results["unique_activities"]
                    )
                    self.analysis_results["anomalies_detected"] = len(
                        self.analysis_results["unique_anomalies"]
                    )

                    if frame_count % 30 == 0:
                        cluster_ids = [
                            face.get("cluster_id", face.get("track_id", "N/A"))
                            for face in tracked_faces
                        ]
                        print(
                            f"Frame {frame_number}: {len(tracked_faces)} faces (clusters: {cluster_ids}), {len(emotions)} emotions, {len(activities)} activities, {len(anomalies)} anomalies"
                        )

                    if self.show_preview:
                        self._show_preview(
                            frame, tracked_faces, emotions, activities, anomalies
                        )

                processing_time = time.time() - start_time
                self.analysis_results["processing_time"] = processing_time

                print(f"\nAnalysis completed in {processing_time:.2f} seconds")
                print(f"Processed {self.analysis_results['frames_processed']} frames")

                if hasattr(self.face_detector, "face_tracks"):
                    active_tracks = len(
                        [
                            t
                            for t in self.face_detector.face_tracks.values()
                            if t.get("hits", 0) >= 3
                        ]
                    )
                    total_tracks = len(self.face_detector.face_tracks)
                    print(
                        f"Face tracking: {active_tracks} active tracks, {total_tracks} total tracks created"
                    )

                    if hasattr(self.face_detector, "get_cluster_statistics"):
                        cluster_stats = self.face_detector.get_cluster_statistics()
                        print(
                            f"Face re-identification: {cluster_stats['total_clusters']} unique people identified"
                        )

                        for cluster_id, cluster_info in cluster_stats[
                            "clusters"
                        ].items():
                            print(
                                f"  Cluster {cluster_id}: {cluster_info['track_count']} tracks"
                            )

                print("Generating report...")
                report = self.report_generator.generate_report(
                    video_info,
                    self.face_detector,
                    self.emotion_analyzer,
                    self.activity_detector,
                    self.anomaly_detector,
                    self.analysis_results,
                )

                self._print_summary(report)
                return report

        except Exception as e:
            print(f"Error during analysis: {e}")
            return {}

    def _show_preview(self, frame, faces, emotions, activities, anomalies):
        import cv2

        display_frame = frame.copy()

        if faces:
            display_frame = self.face_detector.draw_faces(display_frame, faces)

        if emotions:
            display_frame = self.emotion_analyzer.draw_emotions(display_frame, emotions)

        if activities:
            display_frame = self.activity_detector.draw_activities(
                display_frame, activities
            )

        if anomalies:
            display_frame = self.anomaly_detector.draw_anomalies(
                display_frame, anomalies
            )

        cv2.imshow("Video Analysis Preview", display_frame)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            cv2.destroyAllWindows()
            sys.exit()

    def _print_summary(self, report: dict):
        print("\n" + "=" * 60)
        print("VIDEO ANALYSIS SUMMARY")
        print("=" * 60)
        print(
            f"Video Duration: {report['statistics']['video_info']['duration_formatted']}"
        )
        print(
            f"Frames Analyzed: {report['statistics']['video_info']['total_frames']:,}"
        )
        print(f"Unique Faces Detected: {self.analysis_results['faces_detected']}")
        print(f"Unique Emotions Found: {self.analysis_results['emotions_analyzed']}")
        print(
            f"Unique Activities Detected: {self.analysis_results['activities_detected']}"
        )
        print(f"Unique Anomalies Found: {self.analysis_results['anomalies_detected']}")
        print(f"Frames with Faces: {self.analysis_results['frames_with_faces']}")
        print(f"Frames with Emotions: {self.analysis_results['frames_with_emotions']}")
        print(
            f"Frames with Activities: {self.analysis_results['frames_with_activities']}"
        )
        print(
            f"Frames with Anomalies: {self.analysis_results['frames_with_anomalies']}"
        )
        print(
            f"Processing Time: {self.analysis_results['processing_time']:.2f} seconds"
        )

        try:
            emotion_stats = report["statistics"].get("emotion_statistics", {})
            activity_stats = report["statistics"].get("activity_statistics", {})

            if emotion_stats.get("dominant_emotion"):
                print(f"Dominant Emotion: {emotion_stats['dominant_emotion']}")

            if activity_stats.get("dominant_activity"):
                print(f"Dominant Activity: {activity_stats['dominant_activity']}")
        except Exception as e:
            print(f"Error displaying statistics: {e}")

        print("\nReports saved to:")
        for report_type, filepath in report.get("files", {}).items():
            print(f"  - {report_type}: {filepath}")
        print("=" * 60)
        print("\nAnalysis completed successfully!")


def main():
    parser = argparse.ArgumentParser(
        description="Video Analysis with Facial Recognition, Emotion Analysis, Activity Detection, and Anomaly Detection"
    )
    parser.add_argument("video_path", help="Path to the video file")
    parser.add_argument(
        "--output-dir", "-o", help="Output directory for reports", default="output"
    )
    parser.add_argument(
        "--show-preview", "-p", action="store_true", help="Show real-time preview"
    )
    parser.add_argument(
        "--fast-mode",
        "-f",
        action="store_true",
        help="Enable fast mode for quicker analysis",
    )

    args = parser.parse_args()

    if not os.path.exists(args.video_path):
        print(f"Error: Video file not found: {args.video_path}")
        sys.exit(1)

    try:
        analyzer = VideoAnalyzer(
            video_path=args.video_path,
            output_dir=args.output_dir,
            show_preview=args.show_preview,
            fast_mode=args.fast_mode,
        )
        analyzer.analyze_video()
    except KeyboardInterrupt:
        print("\nAnalysis interrupted by user")
    except Exception as e:
        print(f"Error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
