"""YOLO leaf detection through ONNX Runtime, with no PyTorch dependency.

The Raspberry Pi runs this module. It needs onnxruntime, numpy and OpenCV only,
which together are tens of megabytes rather than the multiple gigabytes that
ultralytics and PyTorch occupy.

Weights are converted to ONNX on a desktop machine with export_for_pi.py. The
export produces raw YOLO output, so letterbox preprocessing, confidence
thresholding and non-maximum suppression all happen here.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np


DEFAULT_CLASSES = ("Leaf", "Pile of leaves")


@dataclass(frozen=True)
class Detection:
    """One detected leaf, in pixel coordinates of the original frame."""

    x1: float
    y1: float
    x2: float
    y2: float
    confidence: float
    label: str

    @property
    def width(self) -> float:
        return max(0.0, self.x2 - self.x1)

    @property
    def height(self) -> float:
        return max(0.0, self.y2 - self.y1)

    @property
    def area(self) -> float:
        return self.width * self.height

    @property
    def center_x(self) -> float:
        return (self.x1 + self.x2) / 2.0

    @property
    def center_y(self) -> float:
        return (self.y1 + self.y2) / 2.0


def letterbox(
    image: np.ndarray,
    new_shape: tuple[int, int] = (320, 320),
    colour: tuple[int, int, int] = (114, 114, 114),
) -> tuple[np.ndarray, float, tuple[float, float]]:
    """Resize keeping aspect ratio and pad to new_shape.

    Returns the padded image, the scale applied, and the (x, y) padding so the
    caller can map boxes back to the original frame.
    """
    height, width = image.shape[:2]
    ratio = min(new_shape[0] / height, new_shape[1] / width)

    unpadded = (int(round(width * ratio)), int(round(height * ratio)))
    pad_x = (new_shape[1] - unpadded[0]) / 2.0
    pad_y = (new_shape[0] - unpadded[1]) / 2.0

    if (width, height) != unpadded:
        image = cv2.resize(image, unpadded, interpolation=cv2.INTER_LINEAR)

    top = int(round(pad_y - 0.1))
    bottom = int(round(pad_y + 0.1))
    left = int(round(pad_x - 0.1))
    right = int(round(pad_x + 0.1))

    padded = cv2.copyMakeBorder(
        image, top, bottom, left, right, cv2.BORDER_CONSTANT, value=colour
    )
    return padded, ratio, (left, top)


def nms(
    boxes: np.ndarray, scores: np.ndarray, iou_threshold: float
) -> np.ndarray:
    """Greedy non-maximum suppression over xyxy boxes; returns kept indices."""
    if len(boxes) == 0:
        return np.empty(0, dtype=int)

    x1, y1, x2, y2 = boxes[:, 0], boxes[:, 1], boxes[:, 2], boxes[:, 3]
    areas = np.maximum(0.0, x2 - x1) * np.maximum(0.0, y2 - y1)

    order = scores.argsort()[::-1]
    keep: list[int] = []

    while order.size > 0:
        current = order[0]
        keep.append(int(current))

        xx1 = np.maximum(x1[current], x1[order[1:]])
        yy1 = np.maximum(y1[current], y1[order[1:]])
        xx2 = np.minimum(x2[current], x2[order[1:]])
        yy2 = np.minimum(y2[current], y2[order[1:]])

        inter = np.maximum(0.0, xx2 - xx1) * np.maximum(0.0, yy2 - yy1)
        union = areas[current] + areas[order[1:]] - inter
        iou = np.where(union > 0.0, inter / union, 0.0)

        order = order[np.where(iou <= iou_threshold)[0] + 1]

    return np.asarray(keep, dtype=int)


class OnnxLeafDetector:
    """Runs a YOLO ONNX model exported at a fixed square input size."""

    def __init__(
        self,
        model_path: str | Path,
        class_names: tuple[str, ...] = DEFAULT_CLASSES,
        confidence: float = 0.35,
        iou: float = 0.45,
        imgsz: int = 320,
    ) -> None:
        import onnxruntime as ort

        self.model_path = Path(model_path)
        if not self.model_path.is_file():
            raise FileNotFoundError(f"ONNX model not found: {self.model_path}")

        options = ort.SessionOptions()
        options.intra_op_num_threads = 0
        self.session = ort.InferenceSession(
            str(self.model_path),
            sess_options=options,
            providers=["CPUExecutionProvider"],
        )
        self.input_name = self.session.get_inputs()[0].name

        self.class_names = tuple(class_names)
        self.confidence = confidence
        self.iou = iou
        self.imgsz = imgsz

    def preprocess(
        self, frame: np.ndarray
    ) -> tuple[np.ndarray, float, float, float]:
        """Letterbox and normalise a BGR frame into an NCHW float32 tensor."""
        padded, ratio, (pad_x, pad_y) = letterbox(
            frame, (self.imgsz, self.imgsz)
        )
        rgb = cv2.cvtColor(padded, cv2.COLOR_BGR2RGB)
        tensor = rgb.astype(np.float32) / 255.0
        tensor = np.transpose(tensor, (2, 0, 1))[None]
        return np.ascontiguousarray(tensor), ratio, pad_x, pad_y

    def postprocess(
        self,
        raw: object,
        ratio: float,
        pad_x: float,
        pad_y: float,
        frame_shape: tuple[int, ...],
    ) -> list[Detection]:
        """Turn raw YOLO output into detections in original-frame pixels."""
        predictions = np.asarray(raw)
        predictions = np.squeeze(predictions, axis=0) if predictions.ndim == 3 else predictions

        # YOLOv8/v11 export as (4 + classes, anchors); normalise to (N, 4 + classes).
        if predictions.shape[0] < predictions.shape[1]:
            predictions = predictions.transpose(1, 0)

        boxes_xywh = predictions[:, :4]
        class_scores = predictions[:, 4:]
        class_ids = np.argmax(class_scores, axis=1)
        confidences = class_scores[np.arange(class_scores.shape[0]), class_ids]

        keep = confidences >= self.confidence
        boxes_xywh, class_ids, confidences = (
            boxes_xywh[keep],
            class_ids[keep],
            confidences[keep],
        )
        if boxes_xywh.shape[0] == 0:
            return []

        centre_x, centre_y, width, height = (
            boxes_xywh[:, 0],
            boxes_xywh[:, 1],
            boxes_xywh[:, 2],
            boxes_xywh[:, 3],
        )
        xyxy = np.stack(
            [
                centre_x - width / 2.0,
                centre_y - height / 2.0,
                centre_x + width / 2.0,
                centre_y + height / 2.0,
            ],
            axis=1,
        )

        kept = nms(xyxy, confidences, self.iou)
        xyxy, class_ids, confidences = xyxy[kept], class_ids[kept], confidences[kept]

        # Undo the letterbox: remove padding, then divide by the resize scale.
        xyxy[:, [0, 2]] = (xyxy[:, [0, 2]] - pad_x) / ratio
        xyxy[:, [1, 3]] = (xyxy[:, [1, 3]] - pad_y) / ratio

        frame_height, frame_width = frame_shape[:2]
        xyxy[:, [0, 2]] = np.clip(xyxy[:, [0, 2]], 0, frame_width - 1)
        xyxy[:, [1, 3]] = np.clip(xyxy[:, [1, 3]], 0, frame_height - 1)

        detections: list[Detection] = []
        for box, class_id, score in zip(xyxy, class_ids, confidences):
            index = int(class_id)
            label = (
                self.class_names[index]
                if 0 <= index < len(self.class_names)
                else f"class{index}"
            )
            detections.append(
                Detection(
                    x1=float(box[0]),
                    y1=float(box[1]),
                    x2=float(box[2]),
                    y2=float(box[3]),
                    confidence=float(score),
                    label=label,
                )
            )
        return detections

    def detect(self, frame: np.ndarray) -> list[Detection]:
        """Run inference on one BGR frame and return its detections."""
        tensor, ratio, pad_x, pad_y = self.preprocess(frame)
        raw = self.session.run(None, {self.input_name: tensor})[0]
        return self.postprocess(raw, ratio, pad_x, pad_y, frame.shape)