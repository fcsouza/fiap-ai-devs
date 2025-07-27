import os
from video_processing import process_video
from transcription_analysis import process_transcription
from summarizer import save_summary
from utils.transcribe_audio import extract_audio_from_video, transcribe_audio_to_text

if __name__ == "__main__":
    video_path = "../data/video.mp4"
    audio_path = "../data/audio.wav"
    transcription_path = "../data/transcription.txt"

    if not os.path.exists(audio_path):
        extract_audio_from_video(video_path, audio_path)
    
    if not os.path.exists(transcription_path):
        transcribe_audio_to_text(audio_path, transcription_path)

    total_frames, activities_count, emotion_count, anomaly_count = process_video(video_path, transcription_path)
    save_summary(total_frames, activities_count, emotion_count, anomaly_count)