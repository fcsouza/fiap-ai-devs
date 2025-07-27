from deepface import DeepFace

def detect_faces(
        frame,
        actions=['emotion', 'age', 'gender'],
        enforce_detection=True,
        detector_backend='retinaface'
    ):
        """
        Detect faces and analyze attributes in a frame.

        Parameters:
            frame (np.ndarray): The input image/frame.
            actions (list): List of DeepFace actions to analyze.
            enforce_detection (bool): Whether to enforce face detection.
            detector_backend (str): DeepFace detector backend.

        Returns:
            locations (list): List of face bounding boxes.
            names (list): List of dominant gender labels.
            attributes (list): List of analysis results.
        """
        try:
            analysis = DeepFace.analyze(
                frame,
                actions=actions,
                enforce_detection=enforce_detection,
                detector_backend=detector_backend
            )
            if not isinstance(analysis, list):
                analysis = [analysis]
            locations = [(
                a['region']['y'],
                a['region']['x'] + a['region']['w'],
                a['region']['y'] + a['region']['h'],
                a['region']['x']
            ) for a in analysis]
            # Use gender if available, else fallback to 'Person'
            names = [
                a.get('dominant_gender', 'Person').capitalize()
                for a in analysis
            ]
            attributes = analysis
        except Exception:
            locations, names, attributes = [], [], []
        return locations, names, attributes