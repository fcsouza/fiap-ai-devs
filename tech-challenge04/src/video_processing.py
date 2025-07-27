import cv2
from tqdm import tqdm
from transcription_analysis import process_transcription, analyze_transcription
from utils.anomaly_detection import detect_anomaly
from utils.face_recognition import detect_faces
from utils.emotion_analysis import analyze_emotions
from utils.activity_detection import detect_activity_per_person

def draw_emotions(frame, faces, names, emotions_list, attributes_list):
    for face, name, emotions, attributes in zip(faces, names, emotions_list, attributes_list):
        y1, x2, y2, x1 = face
        cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 0, 0), 2)
        # Use full names for emotion text
        emotion_text = emotions  # Ensure this is full names
        name_text = name if name else "Unknown"
        
        dominant_emotion = attributes.get('dominant_emotion', 'Unknown')
        age = attributes.get('age', 'N/A')
        gender = attributes.get('dominant_gender', 'N/A')

        display_text = f'{name_text}: {dominant_emotion} ({emotion_text}), Age: {age}, Gender: {gender}'


        cv2.putText(frame, display_text, (x1, y1 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, (36, 255, 12), 2)

def draw_activities(frame, landmarks_list):
    # Draw landmarks for each person
    if landmarks_list:
        for res in landmarks_list:
            if res and res.pose_landmarks:
                for pt in res.pose_landmarks.landmark:
                    cx, cy = int(pt.x * frame.shape[1]), int(pt.y * frame.shape[0])
                    cv2.circle(frame, (cx, cy), 3, (0, 255, 0), -1)

def process_emotions(video_path):
    cap = cv2.VideoCapture(video_path)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    emotion_count = {}

    for _ in tqdm(range(total_frames), desc="Processando emoções"):
        ret, frame = cap.read()
        if not ret:
            break

        face_locations, face_names, face_attributes = detect_faces(frame)
        emotions = analyze_emotions(frame, face_locations)

        for i, emotion in enumerate(emotions):
            for emo in emotion:
                emotion_count[emo] = emotion_count.get(emo, 0) + 1

        draw_emotions(frame, face_locations, face_names, emotions, face_attributes)

        # Display the processed frame
        cv2.imshow('Emotions Video', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
    return emotion_count

def process_activities(video_path):
    cap = cv2.VideoCapture(video_path)

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    activities_count = {}
    anomaly_count = 0

    prev_landmarks_list = None

    for _ in tqdm(range(total_frames), desc="Processando atividades"):
        ret, frame = cap.read()
        if not ret:
            break

        face_locations, _, _ = detect_faces(frame)
        activities, curr_landmarks_list = detect_activity_per_person(
            frame, prev_landmarks_list, face_locations, []
        )
        prev_landmarks_list = curr_landmarks_list

        for activity in activities:
            activities_count[activity] = activities_count.get(activity, 0) + 1

            if detect_anomaly(activity):
                anomaly_count += 1

        draw_activities(frame, curr_landmarks_list)

        # Display the processed frame
        cv2.imshow('Activities Video', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
    return activities_count, anomaly_count

def process_transcription_segments(transcription_path):
    segments = []
    with open(transcription_path, 'r', encoding='utf-8') as f:
        for line in f:
            # Exemplo de linha: "start_time,end_time,text"
            parts = line.strip().split(',', 2)
            if len(parts) == 3:
                try:
                    start = float(parts[0])
                    end = float(parts[1])
                    text = parts[2]
                    segments.append({'start': start, 'end': end, 'text': text})
                except ValueError:
                    pass
    return segments

def get_text_for_timestamp(segments, timestamp):
    """
    Retorna o texto correspondente ao intervalo de tempo do frame atual
    """
    for segment in segments:
        if segment['start'] <= timestamp <= segment['end']:
            return segment['text']
    return ""

def get_closest_transcription_text(transcription_data, timestamp, max_diff=1.0):
    """
    Busca no dict transcription_data o texto com timestamp mais próximo do passado,
    considerando uma tolerância max_diff (em segundos).
    Retorna o texto se achar dentro do max_diff, ou string vazia.
    """
    candidates = [(abs(ts - timestamp), ts) for ts in transcription_data.keys() if ts <= timestamp]
    if not candidates:
        return ""
    # pegar o timestamp mais próximo, mas menor ou igual ao timestamp atual
    _, best_ts = min(candidates)
    if abs(best_ts - timestamp) <= max_diff:
        return transcription_data[best_ts]
    return ""


def process_video(video_path, transcription_path):
    cap = cv2.VideoCapture(video_path)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    activities_count = {}
    emotion_count = {}
    anomaly_count = 0

    transcription_data = process_transcription(transcription_path)
    last_text = ""
    prev_landmarks_list = None

    for _ in tqdm(range(total_frames), desc="Processando vídeo"):
        ret, frame = cap.read()
        if not ret:
            break

        current_time = cap.get(cv2.CAP_PROP_POS_MSEC) / 1000.0  # tempo em segundos

        # Detect faces, emotions, and activities
        face_locations, face_names, face_attributes = detect_faces(frame)
        emotions = analyze_emotions(frame, face_locations)
        activities, curr_landmarks_list = detect_activity_per_person(
            frame, prev_landmarks_list, face_locations, emotions
        )
        prev_landmarks_list = curr_landmarks_list

        # Update counts for emotions
        for emotion in enumerate(emotions):
            emotion_count[emotion] = emotion_count.get(emotion, 0) + 1

        # Update counts for activities
        for activity in activities:
            activities_count[activity] = activities_count.get(activity, 0) + 1
            if detect_anomaly(activity):
                anomaly_count += 1

        # Draw both emotions and activities annotations
        draw_emotions(frame, face_locations, face_names, emotions, face_attributes)
        draw_activities(frame, curr_landmarks_list)

        # Buscar texto da transcrição no momento atual
        transcribed_text = get_closest_transcription_text(transcription_data, current_time)
        if transcribed_text and transcribed_text != last_text:
            analyze_transcription(transcribed_text, activities_count, emotion_count)
            last_text = transcribed_text  # Atualiza o último texto analisado
       
        cv2.putText(frame, f"{last_text}", (10, 3cv2.putText(frame, f"{last_text}", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 1)0),
            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 1)

        # Display the processed frame
        cv2.imshow('Video Preview', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

    return total_frames, activities_count, emotion_count, anomaly_count, transcription_data
    