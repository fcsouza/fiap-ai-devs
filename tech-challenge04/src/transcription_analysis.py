def process_transcription(transcription_path):
    transcription_data = {}
    with open(transcription_path, 'r', encoding='utf-8') as file:
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
        "betrayal": "Sadness"
    }
    ACTIVITY_KEYWORDS = {
        "reading": "Reading",
        "conversation": "Conversation",
        "walking": "Walking"
    }

    for word in transcribed_text.lower().split():
        if word in EMOTION_KEYWORDS:
            emotion = EMOTION_KEYWORDS[word]
            emotion_count[emotion] = emotion_count.get(emotion, 0) + 1

        if word in ACTIVITY_KEYWORDS:
            activity = ACTIVITY_KEYWORDS[word]
            activities_count[activity] = activities_count.get(activity, 0) + 1