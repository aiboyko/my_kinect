"""Cross-platform utilities for capturing Kinect v2 (Kinect One) frames."""

from __future__ import annotations

import ctypes
import platform
import sys
import time
from contextlib import AbstractContextManager
from pathlib import Path
from typing import Callable, List, Optional, Tuple

import numpy as np

__all__ = [
    "capture_kinect_snapshot",
    "normalize_depth",
    "KinectRuntimeError",
]

DEFAULT_WINDOWS_SDK_PATH = Path(
    r"C:\Program Files\Microsoft SDKs\Kinect\v2.0_1409\Assemblies"
)


class KinectRuntimeError(RuntimeError):
    """Raised when Kinect frames cannot be captured."""


class KinectBackend(AbstractContextManager):
    """Base class for Kinect backends."""

    def start(self) -> None:
        raise NotImplementedError

    def stop(self) -> None:
        raise NotImplementedError

    def capture(self, initial_wait: float, timeout: float) -> Tuple[np.ndarray, np.ndarray]:
        raise NotImplementedError

    def __enter__(self) -> "KinectBackend":
        self.start()
        return self

    def __exit__(self, exc_type, exc, tb) -> None:  # type: ignore[override]
        self.stop()


def capture_kinect_snapshot(
    *,
    initial_wait: float = 1.5,
    timeout: float = 12.0,
    sdk_path: Optional[Path] = None,
    prefer_backend: Optional[str] = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """Capture a single color and depth frame from a Kinect v2 sensor.

    Returns a tuple ``(color_rgb, depth_mm)``, where ``color_rgb`` is an RGB
    uint8 array of shape ``(H, W, 3)`` and ``depth_mm`` is a float32 array of
    depth values in millimetres with invalid pixels set to ``np.nan``.
    """

    backend = _get_backend(prefer_backend=prefer_backend, sdk_path=sdk_path)
    with backend as device:
        return device.capture(initial_wait=initial_wait, timeout=timeout)


def normalize_depth(
    depth_mm: np.ndarray,
    *,
    lower_percentile: float = 2.0,
    upper_percentile: float = 98.0,
) -> np.ndarray:
    """Normalize a depth map (in millimetres) for visualisation."""

    depth = np.asarray(depth_mm, dtype=np.float32)
    with np.errstate(invalid="ignore"):
        depth[depth <= 0] = np.nan

    valid = depth[np.isfinite(depth)]
    if valid.size == 0:
        raise KinectRuntimeError("Depth frame contains no valid pixels to normalise.")

    depth_min, depth_max = np.nanpercentile(valid, [lower_percentile, upper_percentile])
    if not np.isfinite(depth_min) or not np.isfinite(depth_max) or depth_max <= depth_min:
        raise KinectRuntimeError("Depth frame dynamic range is insufficient for normalisation.")

    normalized = (depth - depth_min) / (depth_max - depth_min)
    return np.clip(normalized, 0.0, 1.0)


def _get_backend(*, prefer_backend: Optional[str], sdk_path: Optional[Path]) -> KinectBackend:
    system = platform.system().lower()
    candidates: List[Callable[[], KinectBackend]] = []

    if prefer_backend:
        key = prefer_backend.strip().lower()
        if key in {"win", "windows", "microsoft"}:
            candidates.append(lambda: WindowsKinectBackend(sdk_path=sdk_path))
        elif key in {"linux", "ubuntu", "freenect2"}:
            candidates.append(UbuntuFreenect2Backend)
        else:
            raise KinectRuntimeError(f"Unknown backend preference '{prefer_backend}'.")
    else:
        if system == "windows":
            candidates.append(lambda: WindowsKinectBackend(sdk_path=sdk_path))
        if system == "linux":
            candidates.append(UbuntuFreenect2Backend)

    if not candidates:
        raise KinectRuntimeError(f"No Kinect backend available for platform '{system}'.")

    errors = []
    for factory in candidates:
        try:
            return factory()
        except (ImportError, KinectRuntimeError) as exc:
            errors.append(str(exc))
            continue

    raise KinectRuntimeError("Failed to initialise any Kinect backend: " + "; ".join(errors))


def _ensure_path_in_sys_path(path: Path) -> None:
    path_str = str(path)
    if path_str not in sys.path:
        sys.path.append(path_str)


class WindowsKinectBackend(KinectBackend):
    """Backend that uses the official Kinect SDK via pythonnet on Windows."""

    def __init__(self, *, sdk_path: Optional[Path]):
        self.sdk_path = sdk_path
        self.sensor = None
        self.color_reader = None
        self.depth_reader = None
        self.System = None
        self.ColorImageFormat = None

    def start(self) -> None:
        if platform.system().lower() != "windows":
            raise KinectRuntimeError("The Windows Kinect backend can only be used on Windows.")

        try:
            import clr  # type: ignore
        except ImportError as exc:
            raise KinectRuntimeError(
                "pythonnet is required for the Windows Kinect backend. Install it with 'pip install pythonnet'."
            ) from exc

        candidate_paths = []
        for path in (self.sdk_path, DEFAULT_WINDOWS_SDK_PATH):
            if path and path.exists():
                candidate_paths.append(path)
        for path in candidate_paths:
            _ensure_path_in_sys_path(path)

        try:
            clr.AddReference("System")
            clr.AddReference("Microsoft.Kinect")
        except Exception as exc:
            raise KinectRuntimeError(
                "Unable to load Microsoft.Kinect assembly. Ensure the Kinect SDK is installed."
            ) from exc

        import System  # type: ignore
        from Microsoft.Kinect import KinectSensor, ColorImageFormat  # type: ignore

        sensor = KinectSensor.GetDefault()
        if sensor is None:
            raise KinectRuntimeError("No Kinect sensor detected. Check USB/power connections.")

        sensor.Open()
        start_time = time.time()
        while not sensor.IsAvailable:
            if time.time() - start_time > 5.0:
                sensor.Close()
                raise KinectRuntimeError("Kinect sensor failed to become available within 5 seconds.")
            time.sleep(0.1)

        color_reader = sensor.ColorFrameSource.OpenReader()
        depth_reader = sensor.DepthFrameSource.OpenReader()
        if color_reader is None or depth_reader is None:
            self._safe_dispose(color_reader)
            self._safe_dispose(depth_reader)
            sensor.Close()
            raise KinectRuntimeError("Failed to open Kinect color/depth readers.")

        self.sensor = sensor
        self.color_reader = color_reader
        self.depth_reader = depth_reader
        self.System = System
        self.ColorImageFormat = ColorImageFormat

    def stop(self) -> None:
        self._safe_dispose(self.color_reader)
        self._safe_dispose(self.depth_reader)
        if self.sensor is not None:
            try:
                self.sensor.Close()
            except Exception:
                pass
        self.sensor = None
        self.color_reader = None
        self.depth_reader = None
        self.System = None
        self.ColorImageFormat = None

    def capture(self, initial_wait: float, timeout: float) -> Tuple[np.ndarray, np.ndarray]:
        if self.sensor is None or self.color_reader is None or self.depth_reader is None:
            raise KinectRuntimeError("Windows backend has not been started.")

        if initial_wait > 0:
            time.sleep(initial_wait)

        deadline = time.time() + timeout
        color_frame = None
        depth_frame = None

        try:
            while time.time() < deadline:
                if color_frame is None:
                    latest_color = self.color_reader.AcquireLatestFrame()
                    if latest_color is not None:
                        color_frame = latest_color

                if depth_frame is None:
                    latest_depth = self.depth_reader.AcquireLatestFrame()
                    if latest_depth is not None:
                        depth_frame = latest_depth

                if color_frame is not None and depth_frame is not None:
                    break

                time.sleep(0.05)

            if color_frame is None or depth_frame is None:
                raise KinectRuntimeError("Timed out waiting for Kinect frames on Windows backend.")

            color_desc = color_frame.FrameDescription
            system = self.System
            color_buffer = system.Array.CreateInstance(
                system.Byte, color_desc.Width * color_desc.Height * 4
            )
            color_frame.CopyConvertedFrameDataToArray(color_buffer, self.ColorImageFormat.Bgra)
            color_np = np.frombuffer(bytearray(color_buffer), dtype=np.uint8).reshape(
                (color_desc.Height, color_desc.Width, 4)
            )
            color_rgb = color_np[:, :, [2, 1, 0]].copy()

            depth_desc = depth_frame.FrameDescription
            locked_depth = depth_frame.LockImageBuffer()
            try:
                depth_ptr = locked_depth.UnderlyingBuffer.ToInt64()
                if depth_ptr == 0:
                    raise KinectRuntimeError("Depth buffer pointer is null.")
                depth_size = locked_depth.Size
                depth_pointer = ctypes.cast(
                    ctypes.c_void_p(depth_ptr), ctypes.POINTER(ctypes.c_uint16)
                )
                depth_np = np.ctypeslib.as_array(
                    depth_pointer, shape=(depth_size // 2,)
                ).reshape((depth_desc.Height, depth_desc.Width)).copy()
            finally:
                locked_depth.Dispose()

            depth_mm = depth_np.astype(np.float32)
            depth_mm[depth_mm <= 0] = np.nan

            return color_rgb, depth_mm
        finally:
            self._safe_dispose(color_frame)
            self._safe_dispose(depth_frame)

    @staticmethod
    def _safe_dispose(obj) -> None:
        if obj is None:
            return
        for method_name in ("Dispose", "Close"):
            method = getattr(obj, method_name, None)
            if callable(method):
                try:
                    method()
                except Exception:
                    pass
                break


class UbuntuFreenect2Backend(KinectBackend):
    """Backend that uses libfreenect2 via pylibfreenect2 on Linux."""

    def __init__(self) -> None:
        self.fn = None
        self.device = None
        self.listener = None
        self.pipeline = None
        self.FrameType = None

    def start(self) -> None:
        if platform.system().lower() != "linux":
            raise KinectRuntimeError("The libfreenect2 backend currently supports Linux only.")

        try:
            from pylibfreenect2 import (  # type: ignore
                Freenect2,
                SyncMultiFrameListener,
                FrameType,
                OpenCLPacketPipeline,
                OpenGLPacketPipeline,
                CpuPacketPipeline,
            )
        except ImportError as exc:
            raise KinectRuntimeError(
                "pylibfreenect2 is required on Linux. Install libfreenect2 and 'pip install pylibfreenect2'."
            ) from exc

        self.Freenect2 = Freenect2
        self.SyncMultiFrameListener = SyncMultiFrameListener
        self.FrameType = FrameType
        self._pipeline_classes = [
            OpenCLPacketPipeline,
            OpenGLPacketPipeline,
            CpuPacketPipeline,
        ]

        fn = Freenect2()
        device_count = fn.enumerateDevices()
        if device_count == 0:
            raise KinectRuntimeError("No Kinect v2 devices detected by libfreenect2.")

        serial = fn.getDeviceSerialNumber(0)
        pipeline = self._create_pipeline()
        device = fn.openDevice(serial, pipeline=pipeline)
        if device is None:
            raise KinectRuntimeError(f"Unable to open Kinect device with serial '{serial}'.")

        listener = SyncMultiFrameListener(FrameType.Color | FrameType.Depth)
        device.setColorFrameListener(listener)
        device.setIrAndDepthFrameListener(listener)
        device.start()

        self.fn = fn
        self.device = device
        self.listener = listener
        self.pipeline = pipeline

    def stop(self) -> None:
        if self.device is not None:
            try:
                self.device.stop()
            except Exception:
                pass
            try:
                self.device.close()
            except Exception:
                pass
        self.device = None
        self.listener = None
        self.fn = None
        self.pipeline = None
        self.FrameType = None

    def capture(self, initial_wait: float, timeout: float) -> Tuple[np.ndarray, np.ndarray]:
        if self.listener is None or self.FrameType is None:
            raise KinectRuntimeError("Linux backend has not been started.")

        if initial_wait > 0:
            time.sleep(initial_wait)

        frames = None
        deadline = time.time() + timeout
        while time.time() < deadline:
            remaining_ms = max(1, int((deadline - time.time()) * 1000))
            try:
                frames = self.listener.waitForNewFrame(remaining_ms)
            except RuntimeError:
                frames = None
            if frames:
                break

        if not frames:
            raise KinectRuntimeError("Timed out waiting for Kinect frames via libfreenect2.")

        try:
            color_frame = frames[self.FrameType.Color]
            depth_frame = frames[self.FrameType.Depth]

            color_np = color_frame.asarray(np.uint8)
            if color_np.ndim != 3 or color_np.shape[-1] not in (3, 4):
                raise KinectRuntimeError("Unexpected color frame format from libfreenect2.")

            if color_np.shape[-1] == 4:
                color_np = color_np[:, :, :3]
            color_rgb = color_np[:, :, ::-1].copy()  # BGR -> RGB

            depth_mm = depth_frame.asarray(np.float32).copy()
            depth_mm[depth_mm <= 0] = np.nan

            return color_rgb, depth_mm
        finally:
            if frames is not None:
                self.listener.release(frames)

    def _create_pipeline(self):
        for pipeline_cls in self._pipeline_classes:
            if pipeline_cls is None:
                continue
            try:
                return pipeline_cls()
            except Exception:
                continue
        raise KinectRuntimeError("Failed to initialise any libfreenect2 packet pipeline.")

