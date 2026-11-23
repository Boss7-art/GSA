from __future__ import annotations

from dataclasses import dataclass, field
import importlib
import importlib.metadata
from pathlib import Path
from typing import Any, Mapping, Protocol
import warnings

import numpy as np


class GeometryBackendError(RuntimeError):
    pass


@dataclass
class GeometryResult:
    points: np.ndarray
    depth: np.ndarray
    normals: np.ndarray | None
    valid: np.ndarray
    intrinsics: np.ndarray
    confidence: np.ndarray | None = None
    metadata: dict[str, Any] = field(default_factory=dict)
    extras: dict[str, np.ndarray] = field(default_factory=dict)


class GeometryBackend(Protocol):
    def infer(self, rgb: np.ndarray) -> GeometryResult: ...


def _torch():
    try:
        import torch
    except (ImportError, OSError) as exc:
        raise GeometryBackendError(
            "PyTorch is unavailable. Install the project requirements into its "
            "Python environment; no geometry prediction was produced."
        ) from exc
    return torch


def select_device(device: str = "auto") -> str:
    torch = _torch()
    device = str(device or "auto").lower()
    mps = getattr(torch.backends, "mps", None)
    if device == "auto":
        return "cuda" if torch.cuda.is_available() else (
            "mps" if mps is not None and mps.is_available() else "cpu"
        )
    try:
        requested = torch.device(device)
    except (TypeError, RuntimeError) as exc:
        raise ValueError(f"Invalid geometry device: {device!r}") from exc
    if requested.type == "cuda":
        if not torch.cuda.is_available():
            raise GeometryBackendError("CUDA was requested, but PyTorch has no available CUDA device.")
        if requested.index is not None and requested.index >= torch.cuda.device_count():
            raise GeometryBackendError(f"CUDA device index {requested.index} is unavailable.")
    elif requested.type == "mps":
        if mps is None or not mps.is_available():
            raise GeometryBackendError("MPS was requested, but this PyTorch runtime cannot use MPS.")
    elif requested.type != "cpu":
        raise ValueError("Supported geometry devices are auto, cpu, cuda[:index], and mps.")
    return str(requested)


get_device = select_device


def _validate_rgb(rgb: np.ndarray) -> np.ndarray:
    rgb = np.asarray(rgb)
    if rgb.ndim != 3 or rgb.shape[-1] != 3 or rgb.dtype != np.uint8:
        raise ValueError("Geometry inference expects an H x W x 3 RGB uint8 array.")
    if min(rgb.shape[:2]) < 3:
        raise ValueError("Geometry inference requires an image of at least 3 x 3 pixels.")
    rgb = np.ascontiguousarray(rgb)
    return rgb if rgb.flags.writeable else rgb.copy()


def _numpy(value: Any) -> np.ndarray:
    if hasattr(value, "detach"):
        value = value.detach().float().cpu().numpy()
    return np.asarray(value)


def _version(package: str) -> str | None:
    try:
        return importlib.metadata.version(package)
    except importlib.metadata.PackageNotFoundError:
        return None


def normals_from_points(
    points: np.ndarray, valid: np.ndarray, max_relative_depth_jump: float = 0.15
) -> np.ndarray:
    points = np.asarray(points, dtype=np.float32)
    valid = np.asarray(valid, dtype=bool)
    if points.shape != valid.shape + (3,):
        raise ValueError("Point and validity map shapes disagree.")
    if not np.isfinite(max_relative_depth_jump) or max_relative_depth_jump <= 0:
        raise ValueError("max_relative_depth_jump must be finite and positive.")
    normals = np.full(points.shape, np.nan, dtype=np.float32)
    if min(valid.shape) < 3:
        return normals
    finite = valid & np.all(np.isfinite(points), axis=-1) & (points[..., 2] > 0)
    stencil = (finite[1:-1, 1:-1] & finite[1:-1, :-2] & finite[1:-1, 2:]
               & finite[:-2, 1:-1] & finite[2:, 1:-1])
    safe = np.where(finite[..., None], points, 0.0)
    center = safe[1:-1, 1:-1]
    z = np.maximum(center[..., 2], np.finfo(np.float32).tiny)
    for neighbor in (safe[1:-1, :-2], safe[1:-1, 2:], safe[:-2, 1:-1], safe[2:, 1:-1]):
        stencil &= np.abs(neighbor[..., 2] - center[..., 2]) <= max_relative_depth_jump * z
    dx = safe[1:-1, 2:] - safe[1:-1, :-2]
    dy = safe[2:, 1:-1] - safe[:-2, 1:-1]
    n = np.cross(dx, dy)
    length = np.linalg.norm(n, axis=-1)
    stencil &= np.isfinite(length) & (length > 1e-12)
    n = n / np.maximum(length[..., None], 1e-12)
    n[np.sum(n * center, axis=-1) > 0] *= -1
    normals[1:-1, 1:-1] = np.where(stencil[..., None], n, np.nan)
    return normals


def estimated_intrinsics(height: int, width: int, fov_degrees: float) -> np.ndarray:
    if not np.isfinite(fov_degrees) or not 1 < fov_degrees < 179:
        raise ValueError("estimated_fov_degrees must be between 1 and 179 degrees.")
    fx = 0.5 / np.tan(np.deg2rad(fov_degrees) / 2.0)
    fy = fx * width / height
    return np.array([[fx, 0, 0.5], [0, fy, 0.5], [0, 0, 1]], dtype=np.float32)


def depth_to_points(depth: np.ndarray, intrinsics: np.ndarray) -> np.ndarray:
    depth = np.asarray(depth, dtype=np.float32)
    intrinsics = np.asarray(intrinsics, dtype=np.float32)
    if depth.ndim != 2 or intrinsics.shape != (3, 3):
        raise ValueError("Expected H x W depth and 3 x 3 normalized intrinsics.")
    h, w = depth.shape
    y, x = np.indices((h, w), dtype=np.float32)
    uv = np.stack([(x + 0.5) / w, (y + 0.5) / h, np.ones_like(x)], axis=-1)
    ray = uv @ np.linalg.inv(intrinsics).T
    ray = ray / ray[..., 2:3]
    return (ray * depth[..., None]).astype(np.float32)


class MoGeBackend:

    DEFAULT_MODELS = {
        "v1": "Ruicheng/moge-vitl",
        "v2": "Ruicheng/moge-2-vits-normal",
        "v3": "Ruicheng/moge-3-vitl",
    }

    def __init__(self, config: Mapping[str, Any], device: str, offline: bool = False):
        self.config = dict(config)
        self.device_request = device
        self.offline = offline
        self.version = str(config.get("moge_version", "v2")).lower()
        if self.version in {"1", "2", "3"}:
            self.version = "v" + self.version
        if self.version not in self.DEFAULT_MODELS:
            raise ValueError("moge_version must be v1, v2, or v3.")
        self.checkpoint = str(config.get("moge_model") or self.DEFAULT_MODELS[self.version])
        self.resolution = int(config.get("moge_resolution_level", 3))
        self.refine_steps = int(config.get("moge_refine_steps", 3))
        if not 0 <= self.resolution <= 9 or self.refine_steps < 0:
            raise ValueError("MoGe resolution must be 0..9 and refine steps nonnegative.")
        self.model = None
        self.device = None

    def _load(self):
        if self.model is not None:
            return
        self.device = select_device(self.device_request)
        try:
            model_class = importlib.import_module(f"moge.model.{self.version}").MoGeModel
            kwargs = {"local_files_only": self.offline}
            if self.config.get("moge_revision"):
                kwargs["revision"] = self.config["moge_revision"]
            with warnings.catch_warnings(record=True) as captured:
                warnings.simplefilter("always")
                model = model_class.from_pretrained(self.checkpoint, **kwargs)
            for warning in captured:
                message = str(warning.message)
                if "absent from the checkpoint" in message or "have no counterpart" in message:
                    raise GeometryBackendError(
                        "MoGe checkpoint and model version disagree: " + message
                    )
            if self.version == "v3" and self.refine_steps > 0 and not hasattr(model, "refiner"):
                raise GeometryBackendError(
                    "The MoGe-3 checkpoint has no trained refiner; set moge_refine_steps to 0."
                )
            self.model = model.to(self.device).eval()
        except GeometryBackendError:
            raise
        except Exception as exc:
            mode = "cached/local weights only" if self.offline else "checkpoint download enabled"
            raise GeometryBackendError(
                f"Cannot load MoGe {self.version} checkpoint {self.checkpoint!r} ({mode}). "
                "Install the optional official MoGe package, supply a matching local .pt "
                f"checkpoint, or select depth-anything. Original error: {exc}"
            ) from exc

    def infer(self, rgb: np.ndarray) -> GeometryResult:
        rgb = _validate_rgb(rgb)
        self._load()
        torch = _torch()
        image = torch.from_numpy(rgb).permute(2, 0, 1).to(self.device, dtype=torch.float32) / 255.0
        infer_kwargs = {
            "resolution_level": self.resolution,
            "apply_mask": True,
            "force_projection": True,
            "use_fp16": bool(self.config.get("moge_use_fp16", self.version != "v3"))
                          and self.device.startswith("cuda"),
        }
        if self.version == "v3":
            infer_kwargs["refine_steps"] = self.refine_steps
        try:
            with torch.inference_mode():
                output = self.model.infer(image, **infer_kwargs)
            points = _numpy(output["points"]).astype(np.float32)
            depth = _numpy(output["depth"]).astype(np.float32)
            intrinsics = _numpy(output["intrinsics"]).astype(np.float32)
            if points.shape != rgb.shape[:2] + (3,) or depth.shape != rgb.shape[:2]:
                raise ValueError("The model output dimensions disagree with the input image.")
            if intrinsics.shape != (3, 3) or not np.all(np.isfinite(intrinsics)):
                raise ValueError("The model did not return finite 3 x 3 intrinsics.")
            valid = np.isfinite(depth) & (depth > 0) & np.all(np.isfinite(points), axis=-1)
            if output.get("mask") is not None:
                model_mask = _numpy(output["mask"]).astype(bool)
                if model_mask.shape != valid.shape:
                    raise ValueError("The model validity mask has an unexpected shape.")
                valid &= model_mask
            if not np.any(valid):
                raise ValueError("MoGe returned no finite, positive valid camera points.")
            points[~valid] = np.nan
            depth[~valid] = np.nan
            if output.get("normal") is not None:
                normals = _numpy(output["normal"]).astype(np.float32)
                if normals.shape != points.shape:
                    raise ValueError("The model normal map has an unexpected shape.")
                length = np.linalg.norm(normals, axis=-1)
                normal_valid = valid & np.isfinite(length) & (length > 1e-8)
                normals /= np.maximum(length[..., None], 1e-8)
                facing_away = np.sum(normals * np.where(valid[..., None], points, 0), axis=-1) > 0
                normals[facing_away] *= -1
                normals[~normal_valid] = np.nan
                normal_source = "pretrained_model"
            else:
                normals = normals_from_points(points, valid)
                normal_source = "central_difference_of_point_map"
            return GeometryResult(
                points=points, depth=depth, normals=normals, valid=valid, intrinsics=intrinsics,
                metadata={
                    "backend": "moge", "model": self.checkpoint, "model_version": self.version,
                    "device": self.device, "resolution_level": self.resolution,
                    "refine_steps": self.refine_steps if self.version == "v3" else 0,
                    "depth_kind": "scale_invariant" if self.version == "v1" else "metric_prediction",
                    "depth_units": "arbitrary" if self.version == "v1" else "meters_predicted",
                    "intrinsics_source": "model_estimated", "intrinsics_normalized": True,
                    "coordinate_system": "opencv_x_right_y_down_z_forward",
                    "normal_source": normal_source, "normal_orientation": "faces_camera",
                    "confidence_available": False, "package_version": _version("moge"),
                    "caution": "Monocular predicted scale and camera intrinsics require validation; "
                               "they are not surveyed measurements.",
                },
            )
        except Exception as exc:
            raise GeometryBackendError(f"MoGe inference failed: {exc}") from exc


class DepthAnythingV2Backend:

    def __init__(self, config: Mapping[str, Any], device: str, offline: bool = False):
        self.config = dict(config)
        self.device_request = device
        self.offline = offline
        self.checkpoint = str(config.get("depth_anything_model") or "depth-anything/Depth-Anything-V2-Small-hf")
        self.fov = float(config.get("estimated_fov_degrees", 70.0))
        self.inverse_scale = float(config.get("inverse_depth_scale", 1.0))
        self.inverse_shift = float(config.get("inverse_depth_shift", 0.1))
        if not np.isfinite(self.inverse_scale) or self.inverse_scale <= 0:
            raise ValueError("inverse_depth_scale must be finite and strictly positive.")
        if not np.isfinite(self.inverse_shift):
            raise ValueError("inverse_depth_shift must be finite.")
        estimated_intrinsics(3, 3, self.fov)
        self.model = None
        self.processor = None
        self.device = None

    def _load(self):
        if self.model is not None:
            return
        self.device = select_device(self.device_request)
        try:
            from transformers import AutoImageProcessor, AutoModelForDepthEstimation
            kwargs = {"local_files_only": self.offline}
            processor = AutoImageProcessor.from_pretrained(self.checkpoint, **kwargs)
            model = AutoModelForDepthEstimation.from_pretrained(self.checkpoint, **kwargs)
            kind = getattr(model.config, "depth_estimation_type", "relative")
            if kind != "relative":
                raise GeometryBackendError(
                    f"Expected a relative Depth Anything V2 checkpoint, got {kind!r}. "
                    "Use depth-anything/Depth-Anything-V2-Small-hf or a local copy."
                )
            self.processor = processor
            self.model = model.to(self.device).eval()
        except GeometryBackendError:
            raise
        except Exception as exc:
            mode = "cached/local weights only" if self.offline else "checkpoint download enabled"
            raise GeometryBackendError(
                f"Cannot load Depth Anything V2 {self.checkpoint!r} ({mode}). Install "
                "PyTorch and Transformers, or provide a cached/local Hugging Face model "
                f"directory. Original error: {exc}"
            ) from exc

    def infer(self, rgb: np.ndarray) -> GeometryResult:
        rgb = _validate_rgb(rgb)
        self._load()
        torch = _torch()
        try:
            from PIL import Image
            inputs = self.processor(images=Image.fromarray(rgb), return_tensors="pt").to(self.device)
            with torch.inference_mode():
                outputs = self.model(**inputs)
            raw = torch.nn.functional.interpolate(
                outputs.predicted_depth.unsqueeze(1), size=rgb.shape[:2],
                mode="bicubic", align_corners=False,
            )[0, 0]
            raw = _numpy(raw).astype(np.float32)
            depth, valid, conversion = inverse_prediction_to_pseudo_depth(
                raw, self.inverse_scale, self.inverse_shift
            )
            intrinsics = estimated_intrinsics(*rgb.shape[:2], self.fov)
            points = depth_to_points(depth, intrinsics)
            normals = normals_from_points(points, valid)
            return GeometryResult(
                points=points, depth=depth, normals=normals, valid=valid, intrinsics=intrinsics,
                metadata={
                    "backend": "depth-anything", "model": self.checkpoint,
                    "model_version": "v2", "device": self.device,
                    "depth_kind": "pseudo_depth_from_relative_inverse_depth",
                    "depth_units": "arbitrary_median_one", "metric_depth": False,
                    "raw_prediction_kind": "relative_inverse_depth",
                    "intrinsics_source": "assumed_horizontal_fov", "intrinsics_normalized": True,
                    "estimated_fov_degrees": self.fov,
                    "coordinate_system": "opencv_x_right_y_down_z_forward",
                    "normal_source": "central_difference_of_pseudo_point_map",
                    "normal_orientation": "faces_camera", "confidence_available": False,
                    "package_version": _version("transformers"), **conversion,
                    "caution": "Relative inverse depth has an unknown affine scale/shift. "
                               "The chosen inverse-depth shift and assumed FOV change plane "
                               "geometry. These points/normals are exploratory, not metric "
                               "reconstruction or evidence of an opening by themselves.",
                },
                extras={"raw_prediction": raw, "raw_inverse_depth": raw},
            )
        except GeometryBackendError:
            raise
        except Exception as exc:
            raise GeometryBackendError(f"Depth Anything V2 inference failed: {exc}") from exc


def inverse_prediction_to_pseudo_depth(
    raw_prediction: np.ndarray, inverse_scale: float = 1.0, inverse_shift: float = 0.1
) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    raw = np.asarray(raw_prediction, dtype=np.float32)
    if raw.ndim != 2:
        raise ValueError("Expected a two-dimensional raw inverse-depth prediction.")
    if not np.isfinite(inverse_scale) or inverse_scale <= 0 or not np.isfinite(inverse_shift):
        raise ValueError("Inverse-depth scale must be positive and both parameters finite.")
    denominator = inverse_scale * raw + inverse_shift
    valid = np.isfinite(raw) & np.isfinite(denominator) & (denominator > 1e-8)
    if not np.any(valid):
        raise GeometryBackendError(
            "The configured inverse-depth scale/shift produces no positive denominator."
        )
    depth = np.full(raw.shape, np.nan, dtype=np.float32)
    depth[valid] = 1.0 / denominator[valid]
    median_before_scaling = float(np.median(depth[valid]))
    if not np.isfinite(median_before_scaling) or median_before_scaling <= 0:
        raise GeometryBackendError("Cannot scale the fallback pseudo-depth by a positive median.")
    depth[valid] /= median_before_scaling
    return depth, valid, {
        "inverse_depth_scale": float(inverse_scale), "inverse_depth_shift": float(inverse_shift),
        "pseudo_depth_formula": "z = 1/(inverse_depth_scale*raw_prediction + inverse_depth_shift); "
                                "z = z/median(z[valid])",
        "pseudo_depth_median_before_scaling": median_before_scaling,
    }


class AutomaticGeometryBackend:

    def __init__(self, config: Mapping[str, Any], device: str, offline: bool = False):
        self.primary = MoGeBackend(config, device, offline)
        self.fallback = DepthAnythingV2Backend(config, device, offline)
        self.active: GeometryBackend = self.primary
        self.fallback_reason: str | None = None

    def infer(self, rgb: np.ndarray) -> GeometryResult:
        rgb = _validate_rgb(rgb)
        if self.fallback_reason is None:
            try:
                return self.primary.infer(rgb)
            except GeometryBackendError as exc:
                self.fallback_reason = str(exc)
                self.active = self.fallback
                warnings.warn("MoGe unavailable; using relative Depth Anything V2 pseudo-geometry. "
                              + self.fallback_reason, RuntimeWarning, stacklevel=2)
        try:
            result = self.active.infer(rgb)
        except GeometryBackendError as exc:
            raise GeometryBackendError(
                f"Both geometry backends failed. MoGe: {self.fallback_reason}; "
                f"Depth Anything V2: {exc}"
            ) from exc
        result.metadata["requested_backend"] = "auto"
        result.metadata["fallback_reason"] = self.fallback_reason
        return result


def create_geometry_backend(
    name: str, config: Mapping[str, Any] | None = None, device: str = "auto", offline: bool = False
) -> GeometryBackend:
    config = dict(config or {})
    config = dict(config.get("geometry", config))
    name = str(name).lower().replace("_", "-")
    if name == "moge":
        return MoGeBackend(config, device, offline)
    if name in {"depth-anything", "depth-anything-v2", "da2", "dav2"}:
        return DepthAnythingV2Backend(config, device, offline)
    if name == "auto":
        return AutomaticGeometryBackend(config, device, offline)
    raise ValueError("Unknown geometry backend; choose moge, depth-anything, or auto.")
