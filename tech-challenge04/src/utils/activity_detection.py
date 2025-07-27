import cv2
import mediapipe as mp
import logging
from ultralytics import YOLO

# Inicialização de logs
logging.basicConfig(level=logging.INFO)

# Modelos
mp_holistic = mp.solutions.holistic
holistic = mp_holistic.Holistic(static_image_mode=False)
yolo = YOLO("yolov8n.pt")  # Use 'yolov8s.pt' para mais precisão

# Função para detectar pessoas com YOLO
def detect_persons_with_yolo(frame):
    results = yolo(frame)[0]
    boxes = []
    for r in results.boxes.data.tolist():
        x1, y1, x2, y2, conf, cls = r
        if int(cls) == 0 and conf > 0.5:
            boxes.append((int(x1), int(y1), int(x2), int(y2)))
    return boxes

# Função para extrair landmarks da imagem recortada
def get_landmarks_from_crop(crop):
    rgb = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
    return holistic.process(rgb)

# Funções de atividade

def is_reading(landmarks):
    if not landmarks or not landmarks.pose_landmarks:
        return False
    try:
        nose = landmarks.pose_landmarks.landmark[mp_holistic.PoseLandmark.NOSE.value]
        left_eye = landmarks.pose_landmarks.landmark[mp_holistic.PoseLandmark.LEFT_EYE.value]
        left_wrist = landmarks.pose_landmarks.landmark[mp_holistic.PoseLandmark.LEFT_WRIST.value]
        right_wrist = landmarks.pose_landmarks.landmark[mp_holistic.PoseLandmark.RIGHT_WRIST.value]
        head_down = nose.y > left_eye.y
        hand_near_face = (abs(left_wrist.y - nose.y) < 0.15 or abs(right_wrist.y - nose.y) < 0.15)
        return head_down and hand_near_face
    except IndexError:
        return False

def is_gesturing(landmarks):
    if not landmarks or not landmarks.pose_landmarks:
        return False
    try:
        left_wrist = landmarks.pose_landmarks.landmark[mp_holistic.PoseLandmark.LEFT_WRIST.value]
        right_wrist = landmarks.pose_landmarks.landmark[mp_holistic.PoseLandmark.RIGHT_WRIST.value]
        left_shoulder = landmarks.pose_landmarks.landmark[mp_holistic.PoseLandmark.LEFT_SHOULDER.value]
        right_shoulder = landmarks.pose_landmarks.landmark[mp_holistic.PoseLandmark.RIGHT_SHOULDER.value]
        return (left_wrist.y < left_shoulder.y or right_wrist.y < right_shoulder.y)
    except (AttributeError, IndexError):
        return False

def is_walking(landmarks, prev_landmarks):
    if not landmarks or not landmarks.pose_landmarks or not prev_landmarks or not prev_landmarks.pose_landmarks:
        return False
    try:
        left_hip = landmarks.pose_landmarks.landmark[mp_holistic.PoseLandmark.LEFT_HIP.value]
        prev_left_hip = prev_landmarks.pose_landmarks.landmark[mp_holistic.PoseLandmark.LEFT_HIP.value]
        return abs(left_hip.x - prev_left_hip.x) > 0.02
    except IndexError:
        return False

def is_conversing(faces, emotions):
    return len(faces) > 1 and any(e in ['happy', 'surprised'] for e in emotions)

# Função principal para detecção de atividade por pessoa
def detect_activity_per_person(frame, prev_landmarks_list=None, faces=None, emotions=None):
    activities = []
    landmarks_list = []

    person_boxes = detect_persons_with_yolo(frame)
    if not person_boxes:
        return ["Unknown"], []

    for i, (x1, y1, x2, y2) in enumerate(person_boxes):
        crop = frame[y1:y2, x1:x2]
        results = get_landmarks_from_crop(crop)

        if not results or not results.pose_landmarks:
            activities.append("Unknown")
            landmarks_list.append(None)
            continue

        lm = results

        if is_reading(lm):
            activities.append("Reading")
        elif is_gesturing(lm):
            activities.append("Gesturing")
        elif is_walking(lm, prev_landmarks_list[i] if prev_landmarks_list and i < len(prev_landmarks_list) else None):
            activities.append("Walking")
        elif faces and emotions and i < len(emotions) and is_conversing(faces, emotions[i]):
            activities.append("Conversing")
        else:
            activities.append("Other")

        landmarks_list.append(lm)

    return activities, landmarks_list
