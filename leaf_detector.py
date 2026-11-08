"""YOLO leaf detection through ONNX Runtime, with no PyTorch dependency.

The Raspberry Pi runs this module. It needs onnxruntime, numpy and OpenCV only,
which together are tens of megabytes rather than the multiple gigabytes that
ultralytics and PyTorch occupy.

Weights are converted to ONNX on a desktop machine with export_for_pi.py. The
export produces raw YOLO output, so letterbox preprocessing, confidence
thresholding and non-maximum suppression all happen here.
"""

from __future__ import annotations

import base64
import http.client
import json
import os
import time
import urllib.parse
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np


DEFAULT_CLASSES = ("Leaf", "Pile of leaves")

# Hosted workflow that wraps this project's trained leaf detector.
DEFAULT_API_URL = "https://serverless.roboflow.com"
DEFAULT_WORKSPACE = "object-detection-bdal6"
DEFAULT_WORKFLOW_ID = (
    "leave-detection-jaybk-mkz2c-mzw7j-1-rfdetr-small-t1-"
    "vleave-detection-jaybk-mkz2c-mzw7j-1-rfdetr-small-t1-logic"
)


class RoboflowError(RuntimeError):
    """Base class for failures when calling a hosted Roboflow workflow."""


class RoboflowAuthError(RoboflowError):
    """The API key was missing, rejected, or lacks access to the workflow."""


class RoboflowTimeoutError(RoboflowError):
    """The workflow did not answer in time. Safe to retry."""


class RoboflowResponseError(RoboflowError):
    """The workflow replied with something that could not be interpreted."""


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

    def close(self) -> None:
        """Release the inference session. Shared interface with the hosted detector."""
        self.session = None

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

        expected = 4 + len(self.class_names)
        if class_scores.shape[1] != len(self.class_names):
            raise ValueError(
                f"model output has {class_scores.shape[1]} class channels but "
                f"{len(self.class_names)} class names were supplied (expected "
                f"{expected} values per anchor). Pass --classes matching the "
                "model, or re-export it with the right class count."
            )
        
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

        # Suppress overlaps within each class only. Boxes of different classes may
        # legitimately overlap, so a class-agnostic pass would drop valid ones.
        kept_indices: list[int] = []
        for class_id in np.unique(class_ids):
            same_class = np.where(class_ids == class_id)[0]
            picked = nms(xyxy[same_class], confidences[same_class], self.iou)
            kept_indices.extend(same_class[picked].tolist())

        if not kept_indices:
            return []
        kept_indices.sort()
        xyxy = xyxy[kept_indices]
        class_ids = class_ids[kept_indices]
        confidences = confidences[kept_indices]

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
        if self.session is None:
            raise RuntimeError("this detector has already been closed")
        tensor, ratio, pad_x, pad_y = self.preprocess(frame)
        raw = self.session.run(None, {self.input_name: tensor})[0]
        return self.postprocess(raw, ratio, pad_x, pad_y, frame.shape)


_BOX_FIELDS = frozenset({"x", "y", "width", "height", "confidence"})


def _is_box_list(value: object) -> bool:
    """True when value is a non-empty list of detection dicts."""
    return (
        isinstance(value, list)
        and bool(value)
        and isinstance(value[0], dict)
        and _BOX_FIELDS <= set(value[0])
    )


def _find_box_list(value: object) -> list[dict] | None:
    """Search value, then one level inside it, for a non-empty detection list."""
    if _is_box_list(value):
        return value
    if isinstance(value, dict):
        for nested in value.values():
            if _is_box_list(nested):
                return nested
    return None


def extract_predictions(payload: object) -> list[dict]:
    """Find the detection list inside a workflow response.

    The output's name and the wrappers around it are read from the payload
    rather than assumed, so a renamed output or an extra layer does not break
    parsing.

    An empty scene is a normal result, not an error: the workflow reports it as
    an empty list sitting beside the image metadata, which this returns as [].
    """
    outputs = payload.get("outputs") if isinstance(payload, dict) else None
    if not isinstance(outputs, list):
        raise RoboflowResponseError("workflow response has no 'outputs' list")

    empty_result: list[dict] | None = None
    for entry in outputs:
        if not isinstance(entry, dict):
            continue
        for value in entry.values():
            found = _find_box_list(value)
            if found is not None:
                return found
            if isinstance(value, dict) and "image" in value:
                # The detection list is always paired with the image it refers
                # to, even when that list is empty.
                for nested in value.values():
                    if isinstance(nested, list) and not nested:
                        empty_result = nested

    if empty_result is not None:
        return empty_result
    raise RoboflowResponseError("workflow response contained no detection list")


class RoboflowWorkflowDetector:
    """Detects leaves by calling this project's hosted Roboflow workflow.

    Inference happens on Roboflow's servers, so the device needs neither local
    weights nor PyTorch. Each call uploads one JPEG frame and waits for boxes.

    The workflow declares a single input, `image`, and no parameters, so the
    confidence cut is applied here rather than being sent to the server.
    """

    def __init__(
        self,
        api_key: str,
        workspace: str = DEFAULT_WORKSPACE,
        workflow_id: str = DEFAULT_WORKFLOW_ID,
        api_url: str = DEFAULT_API_URL,
        class_names: tuple[str, ...] = DEFAULT_CLASSES,
        confidence: float = 0.35,
        timeout: float = 15.0,
        retries: int = 2,
        backoff: float = 1.5,
        jpeg_quality: int = 85,
    ) -> None:
        if not api_key:
            raise RoboflowAuthError(
                "no API key: set ROBOFLOW_API_KEY or pass one explicitly"
            )

        self.api_key = api_key
        self.workspace = workspace
        self.workflow_id = workflow_id
        self.api_url = api_url.rstrip("/")
        self.endpoint = f"{self.api_url}/{workspace}/workflows/{workflow_id}"

        # Keep one connection alive across calls. Rebuilding it per frame costs a
        # full TCP and TLS handshake, which on a Raspberry Pi over WiFi is a
        # large fraction of the time the control loop is blocked.
        split = urllib.parse.urlsplit(self.api_url)
        self._host = split.netloc
        self._secure = split.scheme != "http"
        self._path = f"{split.path.rstrip('/')}/{workspace}/workflows/{workflow_id}"
        self._connection: http.client.HTTPConnection | None = None

        self.class_names = tuple(class_names)
        self.confidence = confidence
        self.timeout = timeout
        self.retries = max(0, retries)
        self.backoff = backoff
        self.jpeg_quality = jpeg_quality

    def close(self) -> None:
        """Drop the pooled connection, if any."""
        connection, self._connection = self._connection, None
        if connection is not None:
            connection.close()

    def _connect(self) -> http.client.HTTPConnection:
        if self._secure:
            return http.client.HTTPSConnection(self._host, timeout=self.timeout)
        return http.client.HTTPConnection(self._host, timeout=self.timeout)

    def _post(self, payload: dict) -> dict:
        """One request, mapped to a typed error. The key never leaves the header."""
        body = json.dumps(payload).encode("utf-8")
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
            "Accept": "application/json",
        }

        try:
            connection = self._connection or self._connect()
            self._connection = connection
            connection.request("POST", self._path, body=body, headers=headers)
            response = connection.getresponse()
            status = response.status
            raw = response.read().decode("utf-8", "replace")
        except (http.client.HTTPException, OSError) as exc:
            # A pooled connection can be closed by the server between calls.
            # Drop it so the retry dials a fresh one instead of reusing a corpse.
            self.close()
            raise RoboflowTimeoutError(f"could not reach Roboflow: {exc}") from exc

        if status in (401, 403):
            raise RoboflowAuthError(f"Roboflow rejected the API key (HTTP {status})")
        if status == 429 or status >= 500:
            raise RoboflowTimeoutError(
                f"Roboflow is unavailable (HTTP {status}); safe to retry"
            )
        if status >= 400:
            raise RoboflowResponseError(f"Roboflow rejected the request (HTTP {status})")

        try:
            return json.loads(raw)
        except json.JSONDecodeError as exc:
            raise RoboflowResponseError("Roboflow returned a non-JSON body") from exc

    def _post_with_retries(self, payload: dict) -> dict:
        """Retry transient failures with exponential backoff; never retry auth."""
        last_error: RoboflowError | None = None
        for attempt in range(self.retries + 1):
            try:
                return self._post(payload)
            except RoboflowTimeoutError as exc:
                last_error = exc
                if attempt < self.retries:
                    time.sleep(self.backoff * (2**attempt))
        assert last_error is not None
        raise last_error

    def build_payload(self, frame: np.ndarray) -> dict:
        """Wrap one BGR frame as a base64 image input for the workflow."""
        encoded, buffer = cv2.imencode(
            ".jpg", frame, [int(cv2.IMWRITE_JPEG_QUALITY), self.jpeg_quality]
        )
        if not encoded:
            raise RoboflowResponseError("could not encode the frame as JPEG")
        return {
            "inputs": {
                "image": {
                    "type": "base64",
                    "value": base64.b64encode(buffer.tobytes()).decode("ascii"),
                }
            }
        }

    def to_detections(
        self, payload: dict, frame_shape: tuple[int, ...]
    ) -> list[Detection]:
        """Convert a workflow response into detections in frame pixels."""
        frame_height, frame_width = frame_shape[:2]
        scale_x = scale_y = 1.0

        # The workflow reports boxes in its own image space. Scale only if that
        # space differs from the frame we sent.
        outputs = payload.get("outputs")
        if isinstance(outputs, list) and outputs and isinstance(outputs[0], dict):
            for value in outputs[0].values():
                if isinstance(value, dict) and isinstance(value.get("image"), dict):
                    reported = value["image"]
                    width = reported.get("width")
                    height = reported.get("height")
                    if width and height:
                        scale_x = frame_width / float(width)
                        scale_y = frame_height / float(height)
                    break

        detections: list[Detection] = []
        for prediction in extract_predictions(payload):
            score = float(prediction.get("confidence", 0.0))
            if score < self.confidence:
                continue

            centre_x = float(prediction["x"]) * scale_x
            centre_y = float(prediction["y"]) * scale_y
            width = float(prediction["width"]) * scale_x
            height = float(prediction["height"]) * scale_y

            label = prediction.get("class")
            if not label:
                index = int(prediction.get("class_id", 0))
                label = (
                    self.class_names[index]
                    if 0 <= index < len(self.class_names)
                    else f"class{index}"
                )

            detections.append(
                Detection(
                    x1=max(0.0, centre_x - width / 2.0),
                    y1=max(0.0, centre_y - height / 2.0),
                    x2=min(float(frame_width - 1), centre_x + width / 2.0),
                    y2=min(float(frame_height - 1), centre_y + height / 2.0),
                    confidence=score,
                    label=str(label),
                )
            )
        return detections

    def detect(self, frame: np.ndarray) -> list[Detection]:
        """Run the hosted workflow on one BGR frame and return its detections."""
        payload = self._post_with_retries(self.build_payload(frame))
        return self.to_detections(payload, frame.shape)


def build_detector(
    backend: str,
    *,
    model_path: str | Path | None = None,
    api_key: str | None = None,
    workspace: str = DEFAULT_WORKSPACE,
    workflow_id: str = DEFAULT_WORKFLOW_ID,
    api_url: str = DEFAULT_API_URL,
    class_names: tuple[str, ...] = DEFAULT_CLASSES,
    confidence: float = 0.35,
    iou: float = 0.45,
    imgsz: int = 320,
    timeout: float = 15.0,
):
    """Create the detector named by backend.

    'roboflow' calls the hosted workflow and needs no local weights.
    'onnx' runs a local export and needs no network.
    """
    if backend == "roboflow":
        return RoboflowWorkflowDetector(
            api_key=api_key or os.environ.get("ROBOFLOW_API_KEY", ""),
            workspace=workspace,
            workflow_id=workflow_id,
            api_url=api_url,
            class_names=class_names,
            confidence=confidence,
            timeout=timeout,
        )
    if backend == "onnx":
        if model_path is None:
            raise RoboflowError("the onnx backend needs a model path")
        return OnnxLeafDetector(
            model_path,
            class_names=class_names,
            confidence=confidence,
            iou=iou,
            imgsz=imgsz,
        )
    raise ValueError(f"unknown backend: {backend}")