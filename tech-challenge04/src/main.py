import os
from video_processing import process_video
from summarizer import save_summary
from utils.transcribe_audio import extract_audio_from_video, transcribe_audio_to_text, transcribe_with_timestamps, save_transcription_json

if __name__ == "__main__":
    video_path = "input/video.mp4"
    audio_path = "output/audio.wav"
    transcription_path = "output/transcription.json"
    transcription_text = "output/transcription.txt"

    if not os.path.exists(audio_path):
        extract_audio_from_video(video_path, audio_path)
    
    if not os.path.exists(transcription_path):
        transcribe_audio_to_text(audio_path, transcription_text)
        result = transcribe_with_timestamps(audio_path)
        save_transcription_json(result, transcription_path)

    total_frames, activities_count, emotion_count, anomaly_count, transcription = process_video(video_path, transcription_path)
    save_summary(total_frames, activities_count, emotion_count, anomaly_count, transcription)