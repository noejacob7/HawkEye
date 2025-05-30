# tracking/nanotrack_wrapper.py
from nanotrack import NanoTracker

class NanoTrackWrapper:
    def __init__(self):
        self.tracker = NanoTracker()

    def update(self, boxes, scores=None, classes=None):
        """
        boxes: list of [x, y, w, h]
        scores: list of confidence scores (optional)
        classes: list of class ids (optional)
        """
        # NanoTracker expects boxes as [x1, y1, x2, y2]
        xyxy_boxes = []
        for box in boxes:
            x, y, w, h = box
            x1, y1 = x, y
            x2, y2 = x + w, y + h
            xyxy_boxes.append([x1, y1, x2, y2])

        # Run tracking
        results = self.tracker.update(xyxy_boxes)

        # Each result is a dictionary: {'id': ID, 'box': [x1, y1, x2, y2]}
        return results
