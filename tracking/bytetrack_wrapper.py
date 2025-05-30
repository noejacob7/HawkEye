import numpy as np
from norfair import Detection, Tracker

class ByteTrackWrapper:
    def __init__(self, distance_function=None, distance_threshold=30):
        self.tracker = Tracker(
            distance_function=distance_function or "euclidean",
            distance_threshold=distance_threshold
        )
        self.latest_boxes = []

    def update(self, boxes):
        self.latest_boxes = boxes
        detections = [
            Detection(
                points=np.array([[x + w / 2, y + h / 2]]),
                scores=np.array([1.0])
            )
            for x, y, w, h in boxes
        ]
        return self.tracker.update(detections)
