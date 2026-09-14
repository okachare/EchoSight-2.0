"""Image preprocessing, inference execution, and normalized results."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from time import perf_counter
from typing import Any

import numpy as np
from PIL import Image

from .model_loader import ModelInfo, ModelLoader, TaskType


@dataclass(frozen=True)
class Detection:
    label_id: int
    label: str
    confidence: float
    box: tuple[float, float, float, float]
    mask: np.ndarray | None = None


@dataclass(frozen=True)
class Classification:
    label_id: int
    label: str
    confidence: float


@dataclass(frozen=True)
class InferenceResult:
    source: Path | None
    task_type: TaskType
    image_size: tuple[int, int]
    input_size: tuple[int, int]
    duration_ms: float
    detections: tuple[Detection, ...] = ()
    classifications: tuple[Classification, ...] = ()
    anomaly_score: float | None = None
    anomaly_map: np.ndarray | None = None
    output_shapes: tuple[tuple[str, tuple[int, ...]], ...] = ()

    @property
    def summary(self) -> str:
        if self.task_type in {TaskType.DETECTION, TaskType.INSTANCE_SEGMENTATION}:
            return f"{len(self.detections)} annotation(s)"
        if self.task_type is TaskType.ANOMALY and self.anomaly_score is not None:
            return f"Anomaly score {self.anomaly_score:.1%}"
        if self.classifications:
            best = self.classifications[0]
            return f"{best.label} {best.confidence:.1%}"
        return "Raw outputs captured"


@dataclass(frozen=True)
class PreparedImage:
    tensor: np.ndarray
    original_size: tuple[int, int]
    input_size: tuple[int, int]
    resized_size: tuple[int, int] | None = None
    padding: tuple[int, int] = (0, 0)


class InferenceEngine:
    def __init__(
        self,
        model_info: ModelInfo,
        compiled_model: Any,
        confidence_threshold: float | None = None,
    ) -> None:
        self.model_info = model_info
        self.compiled_model = compiled_model
        self.confidence_threshold = model_info.confidence_threshold if confidence_threshold is None else confidence_threshold

    @classmethod
    def load(
        cls,
        model_path: Path,
        device: str = "CPU",
        confidence_threshold: float | None = None,
    ) -> InferenceEngine:
        info, compiled_model = ModelLoader(device=device).compile(model_path)
        return cls(info, compiled_model, confidence_threshold)

    def infer(self, image: Image.Image, source: Path | None = None) -> InferenceResult:
        prepared = self.prepare_image(image)
        started = perf_counter()
        raw_result = self.compiled_model([prepared.tensor])
        duration_ms = (perf_counter() - started) * 1000.0
        outputs = {port.any_name: np.asarray(value) for port, value in raw_result.items()}
        return self.normalize(outputs, prepared, duration_ms, source)

    def prepare_image(self, image: Image.Image) -> PreparedImage:
        input_size = self.model_info.input_size
        if input_size is None:
            raise ValueError("Dynamic model input dimensions are not supported yet.")

        input_height, input_width = input_size
        source = image.convert("RGB")
        metadata = dict(self.model_info.metadata)
        resized, resized_size, padding = self._resize_image(source, input_size, metadata)
        array = np.asarray(resized)
        input_shape = self.model_info.inputs[0].shape
        channels_first = len(input_shape) == 4 and input_shape[1] in {1, 3, 4}

        if channels_first and input_shape[1] == 1:
            array = np.asarray(resized.convert("L"))[..., None]
        element_type = self.model_info.inputs[0].element_type.lower()
        if "u8" not in element_type:
            array = self._apply_metadata_preprocessing(array, metadata)
        if channels_first:
            array = np.transpose(array, (2, 0, 1))
        array = np.expand_dims(array, axis=0)

        dtype = np.uint8 if "u8" in element_type else np.float32
        tensor = np.ascontiguousarray(array, dtype=dtype)
        return PreparedImage(
            tensor=tensor,
            original_size=source.size,
            input_size=(input_width, input_height),
            resized_size=resized_size,
            padding=padding,
        )

    @staticmethod
    def _resize_image(
        source: Image.Image,
        input_size: tuple[int, int],
        metadata: Mapping[str, str],
    ) -> tuple[Image.Image, tuple[int, int], tuple[int, int]]:
        input_height, input_width = input_size
        resize_type = metadata.get("resize_type", "standard").lower()
        if resize_type not in {"fit_to_window", "fit_to_window_letterbox"}:
            return source.resize((input_width, input_height), Image.Resampling.BILINEAR), (input_width, input_height), (0, 0)

        ratio = min(input_width / source.width, input_height / source.height)
        resized_size = (max(1, round(source.width * ratio)), max(1, round(source.height * ratio)))
        content = source.resize(resized_size, Image.Resampling.BILINEAR)
        pad_value = round(float(metadata.get("pad_value", "0")))
        canvas = Image.new("RGB", (input_width, input_height), (pad_value, pad_value, pad_value))
        padding = ((input_width - resized_size[0]) // 2, (input_height - resized_size[1]) // 2)
        canvas.paste(content, padding)
        return canvas, resized_size, padding

    @classmethod
    def _apply_metadata_preprocessing(cls, array: np.ndarray, metadata: Mapping[str, str]) -> np.ndarray:
        values = array.astype(np.float32, copy=False)
        if metadata.get("intensity_mode", "").lower() == "scale_to_unit":
            minimum = float(metadata.get("intensity_min_value", "0"))
            maximum = float(metadata.get("intensity_max_value", "255"))
            scale_factor = float(metadata.get("intensity_scale_factor", "1"))
            if maximum > minimum:
                values = np.clip(values, minimum, maximum)
                values = (values - minimum) / (maximum - minimum) * scale_factor
        if metadata.get("reverse_input_channels", "false").lower() == "true" and values.shape[-1] > 1:
            values = values[..., ::-1]

        mean = cls._metadata_vector(metadata.get("mean_values"), values.shape[-1], 0.0)
        scale = cls._metadata_vector(metadata.get("scale_values"), values.shape[-1], 1.0)
        scale = np.where(scale == 0, 1.0, scale)
        return (values - mean) / scale

    @staticmethod
    def _metadata_vector(value: str | None, channels: int, default: float) -> np.ndarray:
        numbers = np.fromstring(value or "", sep=" ", dtype=np.float32)
        if numbers.size == 1:
            numbers = np.repeat(numbers, channels)
        if numbers.size != channels:
            numbers = np.full(channels, default, dtype=np.float32)
        return numbers.reshape(1, 1, channels)

    def normalize(
        self,
        outputs: Mapping[str, np.ndarray],
        prepared: PreparedImage,
        duration_ms: float,
        source: Path | None = None,
    ) -> InferenceResult:
        output_shapes = tuple((name, tuple(value.shape)) for name, value in outputs.items())
        common = {
            "source": source,
            "task_type": self.model_info.task_type,
            "image_size": prepared.original_size,
            "input_size": prepared.input_size,
            "duration_ms": duration_ms,
            "output_shapes": output_shapes,
        }

        if self.model_info.task_type in {TaskType.DETECTION, TaskType.INSTANCE_SEGMENTATION}:
            return InferenceResult(detections=self._normalize_detections(outputs, prepared), **common)
        if self.model_info.task_type is TaskType.ANOMALY:
            anomaly_map = self._first_output(outputs)
            anomaly_map = np.squeeze(anomaly_map).astype(np.float32, copy=False)
            anomaly_score = float(np.max(anomaly_map))
            metadata = dict(self.model_info.metadata)
            anomaly_map = self._normalize_anomaly_values(anomaly_map, metadata.get("pixel_threshold"), metadata.get("normalization_scale"))
            anomaly_score = float(self._normalize_anomaly_values(anomaly_score, metadata.get("image_threshold"), metadata.get("normalization_scale")))
            return InferenceResult(
                anomaly_score=anomaly_score,
                anomaly_map=anomaly_map,
                **common,
            )
        if self.model_info.task_type is TaskType.CLASSIFICATION:
            return InferenceResult(classifications=self._normalize_classification(outputs), **common)
        return InferenceResult(**common)

    def _normalize_detections(
        self,
        outputs: Mapping[str, np.ndarray],
        prepared: PreparedImage,
    ) -> tuple[Detection, ...]:
        boxes = self._find_output(outputs, "boxes")
        labels = self._find_output(outputs, "labels")
        masks = self._optional_output(outputs, "masks")
        boxes = np.asarray(boxes).reshape(-1, boxes.shape[-1])
        labels = np.asarray(labels).reshape(-1)
        if masks is not None and masks.ndim >= 4:
            masks = masks[0]

        source_width, source_height = prepared.original_size
        resized_width, resized_height = prepared.resized_size or prepared.input_size
        padding_x, padding_y = prepared.padding
        scale_x = source_width / resized_width
        scale_y = source_height / resized_height
        detections: list[Detection] = []

        for index, box in enumerate(boxes):
            if box.size < 5:
                continue
            confidence = float(box[4])
            if not np.isfinite(confidence) or confidence < self.confidence_threshold:
                continue
            label_id = int(labels[index]) if index < labels.size else 0
            label = self.model_info.labels[label_id] if 0 <= label_id < len(self.model_info.labels) else f"Class {label_id}"
            coordinates = (
                (float(box[0]) - padding_x) * scale_x,
                (float(box[1]) - padding_y) * scale_y,
                (float(box[2]) - padding_x) * scale_x,
                (float(box[3]) - padding_y) * scale_y,
            )
            mask = np.asarray(masks[index]) if masks is not None and index < len(masks) else None
            detections.append(Detection(label_id, label, confidence, coordinates, mask))
        return tuple(detections)

    def _normalize_classification(self, outputs: Mapping[str, np.ndarray]) -> tuple[Classification, ...]:
        values = self._first_output(outputs).astype(np.float64, copy=False).reshape(-1)
        if values.size == 0:
            return ()
        if np.any(values < 0) or np.any(values > 1) or not np.isclose(values.sum(), 1.0, atol=0.01):
            shifted = values - np.max(values)
            values = np.exp(shifted) / np.exp(shifted).sum()
        order = np.argsort(values)[::-1]
        results = []
        for label_id in order:
            label = self.model_info.labels[label_id] if label_id < len(self.model_info.labels) else f"Class {label_id}"
            results.append(Classification(int(label_id), label, float(values[label_id])))
        return tuple(results)

    @staticmethod
    def _find_output(outputs: Mapping[str, np.ndarray], name: str) -> np.ndarray:
        for output_name, value in outputs.items():
            if output_name.lower() == name:
                return value
        raise ValueError(f"Required model output is missing: {name}")

    @staticmethod
    def _optional_output(outputs: Mapping[str, np.ndarray], name: str) -> np.ndarray | None:
        for output_name, value in outputs.items():
            if output_name.lower() == name:
                return value
        return None

    @staticmethod
    def _first_output(outputs: Mapping[str, np.ndarray]) -> np.ndarray:
        try:
            return next(iter(outputs.values()))
        except StopIteration as error:
            raise ValueError("The model returned no outputs.") from error

    @staticmethod
    def _normalize_anomaly_values(values: Any, threshold: str | None, scale: str | None) -> Any:
        try:
            threshold_value = float(threshold) if threshold is not None else None
            scale_value = float(scale) if scale is not None else None
        except ValueError:
            return values
        if threshold_value is None or scale_value is None or scale_value <= 0:
            return values
        return np.clip((values - threshold_value) / scale_value + 0.5, 0.0, 1.0)
