from transformers import pipeline

summarizer = pipeline("summarization", model="facebook/bart-large-cnn")

def generate_text_summary(transcription, activities_count, emotion_count, anomaly_count, total_frames):

    if isinstance(transcription, dict):
        transcription_text = ' '.join(transcription.values())
    else:
        transcription_text = transcription

    context = (
        f"{transcription_text}\n\n"
        f"Durante o vídeo, foram analisados {total_frames} frames.\n"
        f"Foram detectadas as seguintes atividades: {activities_count}.\n"
        f"As emoções observadas foram: {emotion_count}.\n"
        f"O número de anomalias detectadas foi: {anomaly_count}.\n"
    )

    # Resumo com modelo da Hugging Face
    summary = summarizer(context, max_length=150, min_length=60, do_sample=False)[0]['summary_text']
    return summary


def save_summary(total_frames, activities_count, emotion_count, anomaly_count, transcription):

    summary = generate_text_summary(
        transcription,
        activities_count,
        emotion_count,
        anomaly_count,
        total_frames
    )

    with open("./reports/relatorio_final.txt", "w", encoding="utf-8") as f:
        f.write("==== Relatório Técnico ====\n")
        f.write(f"Total de frames analisados: {total_frames}\n")
        f.write(f"Atividades detectadas: {activities_count}\n")
        f.write(f"Emoções detectadas: {emotion_count}\n")
        f.write(f"Número de anomalias detectadas: {anomaly_count}\n\n")
        f.write("==== Resumo Natural ====\n")
        f.write(summary)
