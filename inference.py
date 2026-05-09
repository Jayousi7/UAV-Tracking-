import sys
import os
sys.path.append(os.path.join(os.getcwd(), 'trackers'))

_cudnn_bin = os.path.join(sys.prefix, 'Lib', 'site-packages', 'nvidia', 'cudnn', 'bin')
_cublas_bin = os.path.join(sys.prefix, 'Lib', 'site-packages', 'nvidia', 'cublas', 'bin')
for _p in (_cudnn_bin, _cublas_bin):
    if os.path.isdir(_p) and _p not in os.environ.get('PATH', ''):
        os.environ['PATH'] = _p + os.pathsep + os.environ.get('PATH', '')

import numpy as np
import cv2 as cv
import onnxruntime as ort

from trackers.ocsort.ocsort import OCSort
from trackers.botsort.botsort import BoTSORT


class InferenceEngine:
    TRACKER_TYPES = {'ocsort': 'OC-SORT', 'botsort': 'BoT-SORT'}

    def __init__(self, model_path, tracker_type='ocsort', det_thresh=0.3, max_age=50, min_hits=1):
        providers = ['CUDAExecutionProvider', 'CPUExecutionProvider']
        self.session = ort.InferenceSession(model_path, providers=providers)

        self.tracker_type = tracker_type
        self._det_thresh = det_thresh
        self._max_age = max_age
        self._min_hits = min_hits
        self.tracker = self._create_tracker(tracker_type)

        self.img_size = 640
        self.input_name = self.session.get_inputs()[0].name
        self.output_names = [o.name for o in self.session.get_outputs()]
        self.model_input_type = self.session.get_inputs()[0].type
        print(f"Model initialized. Precision: {self.model_input_type} | Tracker: {self.TRACKER_TYPES[tracker_type]}")

        self.class_memory = {}
        self.classes = {0: 'Person', 1: 'Car'}

        self._lb_scale = 1.0
        self._lb_pad_x = 0
        self._lb_pad_y = 0

    def _create_tracker(self, tracker_type):
        if tracker_type == 'ocsort':
            return OCSort(det_thresh=self._det_thresh, max_age=self._max_age, min_hits=self._min_hits)
        elif tracker_type == 'botsort':
            return BoTSORT(
                track_high_thresh=self._det_thresh,
                track_low_thresh=0.1,
                new_track_thresh=self._det_thresh + 0.1,
                track_buffer=self._max_age,
                with_reid=True,
                cmc_method='sparseOptFlow',
            )
        else:
            raise ValueError(f"Unknown tracker type: {tracker_type}")

    def switch_tracker(self, tracker_type):
        if tracker_type not in self.TRACKER_TYPES:
            raise ValueError(f"Unknown tracker: {tracker_type}")
        self.tracker_type = tracker_type
        self.tracker = self._create_tracker(tracker_type)

    def switch_model(self, model_path):
        providers = ['CUDAExecutionProvider', 'CPUExecutionProvider']
        self.session = ort.InferenceSession(model_path, providers=providers)
        self.input_name = self.session.get_inputs()[0].name
        self.output_names = [o.name for o in self.session.get_outputs()]
        self.model_input_type = self.session.get_inputs()[0].type
        self.tracker = self._create_tracker(self.tracker_type)

    def preprocess(self, img: np.ndarray) -> np.ndarray:
        h0, w0 = img.shape[:2]
        scale = min(self.img_size / h0, self.img_size / w0)
        new_w, new_h = int(w0 * scale), int(h0 * scale)

        resized = cv.resize(img, (new_w, new_h))
        canvas = np.full((self.img_size, self.img_size, 3), 114, dtype=np.uint8)
        pad_top = (self.img_size - new_h) // 2
        pad_left = (self.img_size - new_w) // 2
        canvas[pad_top:pad_top + new_h, pad_left:pad_left + new_w] = resized

        self._lb_scale = scale
        self._lb_pad_x = pad_left
        self._lb_pad_y = pad_top

        canvas = cv.cvtColor(canvas, cv.COLOR_BGR2RGB)
        canvas = canvas.transpose((2, 0, 1))

        if self.model_input_type == 'tensor(float16)':
            canvas = canvas.astype(np.float16) / 255.0
        else:
            canvas = canvas.astype(np.float32) / 255.0

        return np.expand_dims(canvas, axis=0)

    def detect(self, img: np.ndarray):
        return self.session.run(self.output_names, {self.input_name: img})

    def postprocess(self, outputs, orig_shape):
        predictions = np.atleast_2d(np.squeeze(outputs[0]))
        tracker_input = []
        class_labels = []

        if len(predictions) == 0:
            return np.empty((0, 5)), []

        # Check if output is normalized cx, cy, w, h (values <= 1.1)
        # DT models output [cx, cy, w, h] normalized 0-1. YOLO outputs [x1, y1, x2, y2] unnormalized.
        is_normalized = np.max(predictions[:, :4]) <= 1.1

        for det in predictions:
            x1, y1, x2, y2, score, label = det
            if score > self._det_thresh:
                if is_normalized:
                    # Convert cx, cy, w, h to x1, y1, x2, y2
                    cx, cy, w, h = x1, y1, x2, y2
                    x1 = (cx - w / 2) * self.img_size
                    y1 = (cy - h / 2) * self.img_size
                    x2 = (cx + w / 2) * self.img_size
                    y2 = (cy + h / 2) * self.img_size

                x1 = (x1 - self._lb_pad_x) / self._lb_scale
                x2 = (x2 - self._lb_pad_x) / self._lb_scale
                y1 = (y1 - self._lb_pad_y) / self._lb_scale
                y2 = (y2 - self._lb_pad_y) / self._lb_scale
                tracker_input.append([x1, y1, x2, y2, score])
                class_labels.append(int(label))

        dets = np.array(tracker_input) if tracker_input else np.empty((0, 5))
        return dets, class_labels

    def update_tracker(self, detections, orig_shape, frame=None):
        if self.tracker_type == 'botsort':
            stracks = self.tracker.update(detections, frame)
            if len(stracks) > 0:
                tracks = [[t.tlbr[0], t.tlbr[1], t.tlbr[2], t.tlbr[3], t.track_id]
                           for t in stracks if t.is_activated]
                return np.array(tracks) if tracks else np.empty((0, 5))
            return np.empty((0, 5))
        else:
            return self.tracker.update(
                output_results=detections, img_info=orig_shape, img_size=orig_shape
            )

    def set_threshold(self, thresh):
        """Dynamically update the detection confidence threshold."""
        self._det_thresh = max(0.05, min(0.95, thresh))

    def _assign_classes(self, tracks, detections, class_labels):
        """Assign class labels to tracks by matching with detections via IoU."""
        if len(tracks) == 0 or len(detections) == 0:
            return
        for track in tracks:
            tid = int(track[4])
            tx1, ty1, tx2, ty2 = track[:4]
            best_iou, best_cls = 0, -1
            for i, det in enumerate(detections):
                dx1, dy1, dx2, dy2 = det[:4]
                ix1, iy1 = max(tx1, dx1), max(ty1, dy1)
                ix2, iy2 = min(tx2, dx2), min(ty2, dy2)
                inter = max(0, ix2 - ix1) * max(0, iy2 - iy1)
                union = (tx2 - tx1) * (ty2 - ty1) + (dx2 - dx1) * (dy2 - dy1) - inter
                iou = inter / max(union, 1e-6)
                if iou > best_iou:
                    best_iou = iou
                    best_cls = class_labels[i]
            if best_cls >= 0:
                self.class_memory[tid] = best_cls

    def process_frame(self, frame):
        orig_shape = (frame.shape[0], frame.shape[1])
        input_tensor = self.preprocess(frame)
        raw_output = self.detect(input_tensor)
        detections, class_labels = self.postprocess(raw_output, orig_shape)
        tracks = self.update_tracker(detections, orig_shape, frame=frame)
        self._assign_classes(tracks, detections, class_labels)
        return tracks
