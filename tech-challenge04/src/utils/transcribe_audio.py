from moviepy.video.io.VideoFileClip import VideoFileClip
import speech_recognition as sr
import os

def extract_audio_from_video(video_path, audio_path):
    video = VideoFileClip(video_path)
    video.audio.write_audiofile(audio_path)

def transcribe_audio_to_text(audio_path, text_output_path):
    recognizer = sr.Recognizer()
    with sr.AudioFile(audio_path) as source:
        audio = recognizer.record(source)  # lê todo o áudio do arquivo
        try:
            # Usa o serviço de reconhecimento de fala do Google com configuração para português do Brasil
            text = recognizer.recognize_google(audio, language="en-US")
            print("Transcrição: " + text)
            # Salva a transcrição em um arquivo de texto
            with open(text_output_path, 'w', encoding='utf-8') as file:
                file.write(text)
        except sr.UnknownValueError:
            print("Google Speech Recognition não conseguiu entender o áudio")
        except sr.RequestError as e:
            print(f"Erro ao solicitar resultados do serviço de reconhecimento de fala do Google; {e}")