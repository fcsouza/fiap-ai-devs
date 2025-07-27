from deepface import DeepFace

def analyze_emotions(frame, faces):
    emotions = []
    for (y, x2, y2, x1) in faces:  # Assuming `faces` contains tuples of (top, right, bottom, left)
        face_img = frame[y:y2, x1:x2]
        result = DeepFace.analyze(face_img, actions=['emotion'], enforce_detection=False)
        if isinstance(result, list):
            result = result[0]
        emotions.append(result['dominant_emotion'])
    return emotions