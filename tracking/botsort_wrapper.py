# tracking/botsort_wrapper.py
from BoT_SORT.tracker.bot_sort import BoTSORT

class BoTSORTWrapper:
    def __init__(self, args):
        self.tracker = BoTSORT(args)

    def update(self, detections, img_info, img_size):
        """
        detections: numpy array of shape [N, 6] with columns [x1, y1, x2, y2, score, class]
        img_info: tuple (height, width)
        img_size: tuple (height, width)
        """
        online_targets = self.tracker.update(detections, img_info, img_size)
        results = []
        for t in online_targets:
            tlwh = t.tlwh
            tid = t.track_id
            results.append((tlwh, tid))
        return results
