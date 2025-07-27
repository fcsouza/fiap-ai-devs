def detect_anomaly(activity):
    expected = ["Reading", "Gesturing", "Walking", "Conversing", "Other"]
    return activity not in expected