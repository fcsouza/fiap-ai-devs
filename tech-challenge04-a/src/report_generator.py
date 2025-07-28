import json
import os
from datetime import datetime
from typing import Dict, List

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from .utils import ACTIVITIES, CONFIG, EMOTIONS


class ReportGenerator:
    def __init__(self, output_dir: str):
        self.output_dir = output_dir
        self.timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    def generate_report(
        self,
        video_info: Dict,
        face_detector,
        emotion_analyzer,
        activity_detector,
        anomaly_detector,
        analysis_results: Dict,
    ) -> Dict:
        self.video_info = video_info
        self.analysis_results = analysis_results

        face_stats = face_detector.get_face_statistics([])
        emotion_stats = emotion_analyzer.get_emotion_statistics()
        activity_stats = activity_detector.get_activity_statistics()
        anomaly_stats = anomaly_detector.get_anomaly_statistics()

        self.statistics = {
            "video_info": video_info,
            "face_statistics": face_stats,
            "emotion_statistics": emotion_stats,
            "activity_statistics": activity_stats,
            "anomaly_statistics": anomaly_stats,
        }

        self.statistics["analysis_summary"] = self._generate_analysis_summary()

        text_report = self._generate_text_report()
        json_report = self._generate_json_report()
        visualizations = self._generate_visualizations()

        files = {
            "text_report": text_report,
            "json_report": json_report,
        }
        files.update(visualizations)

        return {
            "statistics": self.statistics,
            "files": files,
        }

    def _generate_analysis_summary(self) -> Dict:
        summary = self.statistics["video_info"].copy()
        summary.update(
            {
                "total_frames_analyzed": self.analysis_results["frames_processed"],
                "total_faces_detected": self.analysis_results["faces_detected"],
                "total_emotions_analyzed": self.analysis_results["emotions_analyzed"],
                "total_activities_detected": self.analysis_results[
                    "activities_detected"
                ],
                "total_anomalies_detected": self.analysis_results["anomalies_detected"],
                "processing_time": self.analysis_results["processing_time"],
            }
        )

        return summary

    def _generate_text_report(self) -> str:
        filename = f"video_analysis_report_{self.timestamp}.txt"
        filepath = os.path.join(self.output_dir, filename)

        with open(filepath, "w", encoding="utf-8") as f:
            f.write("VIDEO ANALYSIS REPORT\n")
            f.write("=" * 50 + "\n\n")

            f.write("VIDEO INFORMATION:\n")
            f.write(f"- File Path: {self.video_info['file_path']}\n")
            f.write(f"- Duration: {self.video_info['duration_formatted']}\n")
            f.write(f"- Total Frames: {self.video_info['total_frames']:,}\n")
            f.write(f"- Resolution: {self.video_info['resolution']}\n")
            f.write(f"- FPS: {self.video_info['fps']:.2f}\n\n")

            f.write("ANALYSIS RESULTS:\n")
            f.write("=" * 50 + "\n\n")

            face_stats = self.statistics["face_statistics"]
            f.write("FACIAL RECOGNITION:\n")
            f.write(f"- Total Faces Detected: {face_stats.get('total_faces', 0)}\n")
            f.write(
                f"- Average Detection Confidence: {face_stats.get('average_confidence', 0):.2f}\n\n"
            )

            emotion_stats = self.statistics["emotion_statistics"]
            f.write("EMOTION ANALYSIS:\n")
            f.write(
                f"- Total Emotions Analyzed: {emotion_stats.get('total_emotions', 0)}\n"
            )
            f.write(
                f"- Dominant Emotion: {emotion_stats.get('dominant_emotion', 'Unknown')}\n"
            )
            f.write(
                f"- Average Emotion Confidence: {emotion_stats.get('average_confidence', 0):.2f}\n\n"
            )

            f.write("Emotion Distribution:\n")
            emotion_dist = emotion_stats.get("emotion_distribution", {})
            for emotion, stats in emotion_dist.items():
                f.write(
                    f"  - {emotion}: {stats.get('count', 0)} ({stats.get('max_confidence', 0):.2f})\n"
                )
            f.write("\n")

            activity_stats = self.statistics["activity_statistics"]
            f.write("ACTIVITY DETECTION:\n")
            f.write(
                f"- Total Activities Detected: {activity_stats.get('total_activities', 0)}\n"
            )
            f.write(
                f"- Dominant Activity: {activity_stats.get('dominant_activity', 'Unknown')}\n"
            )
            f.write(
                f"- Average Activity Confidence: {activity_stats.get('average_confidence', 0):.2f}\n\n"
            )

            f.write("Activity Distribution:\n")
            activity_dist = activity_stats.get("activity_distribution", {})
            for activity, stats in activity_dist.items():
                f.write(
                    f"  - {activity}: {stats.get('count', 0)} ({stats.get('max_confidence', 0):.2f})\n"
                )
            f.write("\n")

            anomaly_stats = self.statistics["anomaly_statistics"]
            f.write("ANOMALY DETECTION:\n")
            f.write(
                f"- Total Anomalies Detected: {anomaly_stats.get('total_anomalies', 0)}\n"
            )
            f.write(
                f"- Average Anomaly Confidence: {anomaly_stats.get('average_confidence', 0):.2f}\n\n"
            )

            f.write("Anomaly Types:\n")
            anomaly_types = anomaly_stats.get("anomaly_types", {})
            for anomaly_type, stats in anomaly_types.items():
                f.write(
                    f"  - {anomaly_type}: {stats.get('count', 0)} ({stats.get('max_confidence', 0):.2f})\n"
                )
            f.write("\n")

            insights = self._generate_insights()
            f.write("KEY INSIGHTS:\n")
            f.write("=" * 50 + "\n")
            for i, insight in enumerate(insights, 1):
                f.write(f"{i}. {insight}\n")

            f.write(f"\nANALYSIS COMPLETED: {datetime.now().isoformat()}\n")

        return filepath

    def _generate_json_report(self) -> str:
        filename = f"video_analysis_report_{self.timestamp}.json"
        filepath = os.path.join(self.output_dir, filename)

        with open(filepath, "w", encoding="utf-8") as f:
            json.dump(self.statistics, f, indent=2, default=str)

        return filepath

    def _generate_insights(self) -> List[str]:
        insights = []

        emotion_stats = self.statistics["emotion_statistics"]
        activity_stats = self.statistics["activity_statistics"]

        dominant_emotion = emotion_stats.get("dominant_emotion")
        dominant_activity = activity_stats.get("dominant_activity")

        if dominant_emotion:
            insights.append(f"The most frequent emotion was '{dominant_emotion}'")

        if dominant_activity:
            insights.append(f"The most frequent activity was '{dominant_activity}'")

        total_frames = self.analysis_results["frames_processed"]
        if total_frames > 0:
            frames_per_second = total_frames / self.video_info["duration"]
            insights.append(
                f"Analysis processed {frames_per_second:.1f} frames per second on average"
            )

        return insights

    def _generate_visualizations(self) -> Dict[str, str]:
        visualizations = {}

        try:
            emotion_chart = self._create_emotion_chart()
            if emotion_chart:
                visualizations["emotion_chart"] = emotion_chart

            activity_chart = self._create_activity_chart()
            if activity_chart:
                visualizations["activity_chart"] = activity_chart

            anomaly_chart = self._create_anomaly_chart()
            if anomaly_chart:
                visualizations["anomaly_chart"] = anomaly_chart

            summary_table = self._create_summary_table()
            if summary_table:
                visualizations["summary_table"] = summary_table

        except Exception as e:
            print(f"Error generating visualizations: {e}")

        return visualizations

    def _create_emotion_chart(self) -> str:
        emotion_stats = self.statistics["emotion_statistics"]
        emotion_dist = emotion_stats.get("emotion_distribution", {})

        if not emotion_dist:
            return None

        emotions = list(emotion_dist.keys())
        counts = [stats.get("count", 0) for stats in emotion_dist.values()]

        plt.figure(figsize=(10, 6))
        plt.bar(emotions, counts, color="skyblue")
        plt.title("Emotion Distribution", fontsize=16)
        plt.xlabel("Emotions", fontsize=12)
        plt.ylabel("Count", fontsize=12)
        plt.xticks(rotation=45)
        plt.tight_layout()

        filename = f"emotion_distribution_{self.timestamp}.png"
        filepath = os.path.join(self.output_dir, filename)
        plt.savefig(filepath, dpi=300, bbox_inches="tight")
        plt.close()

        return filepath

    def _create_activity_chart(self) -> str:
        activity_stats = self.statistics["activity_statistics"]
        activity_dist = activity_stats.get("activity_distribution", {})

        if not activity_dist:
            return None

        activities = list(activity_dist.keys())
        counts = [stats.get("count", 0) for stats in activity_dist.values()]

        plt.figure(figsize=(10, 6))
        plt.bar(activities, counts, color="lightgreen")
        plt.title("Activity Distribution", fontsize=16)
        plt.xlabel("Activities", fontsize=12)
        plt.ylabel("Count", fontsize=12)
        plt.xticks(rotation=45)
        plt.tight_layout()

        filename = f"activity_distribution_{self.timestamp}.png"
        filepath = os.path.join(self.output_dir, filename)
        plt.savefig(filepath, dpi=300, bbox_inches="tight")
        plt.close()

        return filepath

    def _create_anomaly_chart(self) -> str:
        anomaly_stats = self.statistics["anomaly_statistics"]
        anomaly_types = anomaly_stats.get("anomaly_types", {})

        if not anomaly_types:
            return None

        types = list(anomaly_types.keys())
        counts = [stats.get("count", 0) for stats in anomaly_types.values()]

        plt.figure(figsize=(10, 6))
        plt.bar(types, counts, color="lightcoral")
        plt.title("Anomaly Types", fontsize=16)
        plt.xlabel("Anomaly Types", fontsize=12)
        plt.ylabel("Count", fontsize=12)
        plt.xticks(rotation=45)
        plt.tight_layout()

        filename = f"anomaly_types_{self.timestamp}.png"
        filepath = os.path.join(self.output_dir, filename)
        plt.savefig(filepath, dpi=300, bbox_inches="tight")
        plt.close()

        return filepath

    def _create_summary_table(self) -> str:
        fig, ax = plt.subplots(figsize=(12, 8))
        ax.axis("tight")
        ax.axis("off")

        data = [
            ["Video Duration", self.video_info["duration_formatted"]],
            ["Total Frames", f"{self.video_info['total_frames']:,}"],
            ["Frames Analyzed", f"{self.analysis_results['frames_processed']:,}"],
            ["Faces Detected", self.analysis_results["faces_detected"]],
            ["Emotions Analyzed", self.analysis_results["emotions_analyzed"]],
            ["Activities Detected", self.analysis_results["activities_detected"]],
            ["Anomalies Found", self.analysis_results["anomalies_detected"]],
            ["Processing Time", f"{self.analysis_results['processing_time']:.2f}s"],
        ]

        table = ax.table(
            cellText=data, colLabels=["Metric", "Value"], cellLoc="left", loc="center"
        )
        table.auto_set_font_size(False)
        table.set_fontsize(12)
        table.scale(1.2, 1.5)

        plt.title("Video Analysis Summary", fontsize=16, pad=20)
        plt.tight_layout()

        filename = f"analysis_summary_{self.timestamp}.png"
        filepath = os.path.join(self.output_dir, filename)
        plt.savefig(filepath, dpi=300, bbox_inches="tight")
        plt.close()

        return filepath
