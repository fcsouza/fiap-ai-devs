import json

def process_transcription(transcription_path):
    transcription_data = {}
    with open(transcription_path, 'r', encoding='utf-8') as file:
        try:
            data = json.load(file)  # tenta carregar como JSON
            for segment in data:
                timestamp = float(segment["start"])
                text = segment["text"]
                transcription_data[timestamp] = text
        except json.JSONDecodeError:
            # Se não for JSON, tenta como txt (fallback antigo)
            file.seek(0)
            for line in file:
                parts = line.strip().split(': ', 1)
                if len(parts) == 2:
                    timestamp_str, text = parts
                    try:
                        timestamp = float(timestamp_str)
                        transcription_data[timestamp] = text
                    except ValueError:
                        print(f"Erro ao converter timestamp: {timestamp_str}")
    return transcription_data


def analyze_transcription(transcribed_text, activities_count, emotion_count):
    EMOTION_KEYWORDS = {
        "happiness": "Happiness",
        "betrayal": "Sadness",
        "smiles": "Happiness",
        "genuine": "Happiness",
        "fear": "Fear",
        "angry": "Anger"
    }
    ACTIVITY_KEYWORDS = {
        "reading": "Reading",
        "conversation": "Conversation",
        "talking": "Conversation",
        "walking": "Walking",
        "gestures": "Gesturing",
        "animated": "Gesturing"
    }

    text = transcribed_text.lower()

    for keyword, label in EMOTION_KEYWORDS.items():
        if keyword in text:
            emotion_count[label] = emotion_count.get(label, 0) + 1

    for keyword, label in ACTIVITY_KEYWORDS.items():
        if keyword in text:
            activities_count[label] = activities_count.get(label, 0) + 1
