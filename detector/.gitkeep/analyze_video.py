"""Estimate a remote pulse signal from face-region color changes in a video."""

from __future__ import annotations

import json
import math
import sys
from pathlib import Path
from typing import Any

import cv2
import numpy as np
from scipy.signal import butter, detrend, sosfiltfilt

MIN_DURATION_SECONDS = 10.0
MAX_DURATION_SECONDS = 60.0
MAX_SAMPLE_RATE_HZ = 12.0
LOW_FREQUENCY_HZ = 0.7
HIGH_FREQUENCY_HZ = 3.3
MAX_GRAPH_POINTS = 160


def _result(
    *,
    status: str,
    verdict: str,
    message: str,
    duration_seconds: float,
    face_detection_rate: float,
    bpm: float | None = None,
    signal_quality: float = 0.0,
    pulse_strength: float | None = None,
    pulse_signal: list[float] | None = None,
) -> dict[str, Any]:
    return {
        "status": status,
        "verdict": verdict,
        "message": message,
        "bpm": bpm,
        "signalQuality": round(signal_quality, 1),
        "pulseStrength": pulse_strength,
        "durationSeconds": round(duration_seconds, 2),
        "faceDetectionRate": round(face_detection_rate, 3),
        "pulseSignal": pulse_signal or [],
    }


def _face_regions(frame: np.ndarray, face: tuple[int, int, int, int]) -> list[np.ndarray]:
    """Return forehead and cheek crops relative to the detected face rectangle."""
    x, y, width, height = face
    bounds = (
        (0.25, 0.18, 0.75, 0.36),
        (0.18, 0.48, 0.42, 0.70),
        (0.58, 0.48, 0.82, 0.70),
    )
    regions = []
    for left, top, right, bottom in bounds:
        x1 = max(0, x + int(width * left))
        y1 = max(0, y + int(height * top))
        x2 = min(frame.shape[1], x + int(width * right))
        y2 = min(frame.shape[0], y + int(height * bottom))
        regions.append(frame[y1:y2, x1:x2])
    return regions


def _analyze_signals(
    signals: np.ndarray,
    sample_rate: float,
    duration_seconds: float,
    face_detection_rate: float,
) -> dict[str, Any]:
    """Filter three green-channel traces and estimate a shared pulse frequency."""
    empty = _result(
        status="insufficient_signal",
        verdict="INSUFFICIENT_SIGNAL",
        message="A stable face signal was not available. Try a steady, well-lit video facing the camera.",
        duration_seconds=duration_seconds,
        face_detection_rate=face_detection_rate,
    )
    if duration_seconds < MIN_DURATION_SECONDS:
        empty.update(
            status="video_too_short",
            verdict="VIDEO_TOO_SHORT",
            message="Use a face video that is at least 10 seconds long.",
        )
        return empty
    if sample_rate / 2 <= HIGH_FREQUENCY_HZ or signals.shape[0] < int(sample_rate * 8):
        return empty

    filter_sos = butter(
        3,
        [LOW_FREQUENCY_HZ, HIGH_FREQUENCY_HZ],
        btype="bandpass",
        fs=sample_rate,
        output="sos",
    )
    filtered_regions = []
    for region_signal in signals.T:
        clean = detrend(region_signal.astype(np.float64), type="linear")
        filtered_regions.append(sosfiltfilt(filter_sos, clean))
    filtered = np.stack(filtered_regions, axis=1)

    region_spectra = np.abs(np.fft.rfft(filtered, axis=0)) ** 2
    frequencies = np.fft.rfftfreq(filtered.shape[0], d=1 / sample_rate)
    band = (frequencies >= LOW_FREQUENCY_HZ) & (frequencies <= HIGH_FREQUENCY_HZ)
    band_frequencies = frequencies[band]
    band_spectra = region_spectra[band, :]
    if band_spectra.size == 0 or np.max(band_spectra) <= 0:
        return empty

    region_peak_indices = np.argmax(band_spectra, axis=0)
    region_peak_frequencies = band_frequencies[region_peak_indices]
    shared_frequency = float(np.median(region_peak_frequencies))
    agreeing_regions = int(np.count_nonzero(np.abs(region_peak_frequencies - shared_frequency) <= 0.25))
    spectrum = np.mean(band_spectra, axis=1)
    peak_index = int(np.argmin(np.abs(band_frequencies - shared_frequency)))
    peak_power = float(spectrum[peak_index])
    mean_power = float(np.mean(spectrum))
    pulse_strength = peak_power / mean_power if mean_power > 0 else 0.0
    band_power = float(np.sum(spectrum))
    spectral_concentration = peak_power / band_power if band_power > 0 else 0.0

    region_scales = np.std(filtered, axis=0)
    usable_regions = region_scales > 1e-8
    normalized_regions = filtered[:, usable_regions] / region_scales[usable_regions]
    combined = np.mean(normalized_regions, axis=1) if normalized_regions.size else np.mean(filtered, axis=1)
    graph_stride = max(1, math.ceil(len(combined) / MAX_GRAPH_POINTS))
    graph_values = combined[::graph_stride]
    graph_scale = float(np.max(np.abs(graph_values))) if graph_values.size else 0.0
    graph = (
        [round(float(value / graph_scale), 4) for value in graph_values]
        if graph_scale > 0
        else []
    )

    agreement = agreeing_regions / signals.shape[1]
    quality = min(
        100.0,
        max(0.0, 100.0 * agreement * min(1.0, face_detection_rate) * min(1.0, spectral_concentration * 5)),
    )
    has_consensus = agreeing_regions >= 2 and face_detection_rate >= 0.6
    if has_consensus:
        return _result(
            status="pulse_estimated",
            verdict="PULSE_ESTIMATED",
            message="A pulse-like color variation was detected across multiple face regions. This is not a real/fake determination.",
            duration_seconds=duration_seconds,
            face_detection_rate=face_detection_rate,
            bpm=round(shared_frequency * 60, 1),
            signal_quality=quality,
            pulse_strength=round(pulse_strength, 2),
            pulse_signal=graph,
        )

    return _result(
        status="low_signal_quality",
        verdict="LOW_SIGNAL_QUALITY",
        message="The video did not produce a pulse estimate consistent across enough face regions.",
        duration_seconds=duration_seconds,
        face_detection_rate=face_detection_rate,
        bpm=round(shared_frequency * 60, 1),
        signal_quality=quality,
        pulse_strength=round(pulse_strength, 2),
        pulse_signal=graph,
    )


def analyze_video(video_path: str | Path) -> dict[str, Any]:
    """Read a video, track its largest face, and estimate pulse from green values."""
    capture = cv2.VideoCapture(str(video_path))
    if not capture.isOpened():
        return _result(
            status="video_unreadable",
            verdict="VIDEO_UNREADABLE",
            message="OpenCV could not read this video. Try a standard MP4 or WebM file.",
            duration_seconds=0,
            face_detection_rate=0,
        )

    fps = float(capture.get(cv2.CAP_PROP_FPS))
    frame_count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
    if not math.isfinite(fps) or fps <= 0 or frame_count <= 0:
        capture.release()
        return _result(
            status="video_metadata_invalid",
            verdict="VIDEO_METADATA_INVALID",
            message="The video is missing usable frame-rate or duration metadata.",
            duration_seconds=0,
            face_detection_rate=0,
        )

    duration_seconds = frame_count / fps
    if duration_seconds < MIN_DURATION_SECONDS:
        capture.release()
        return _analyze_signals(
            np.empty((0, 3)),
            0,
            duration_seconds,
            0,
        )
    if duration_seconds > MAX_DURATION_SECONDS:
        capture.release()
        return _result(
            status="video_too_long",
            verdict="VIDEO_TOO_LONG",
            message="For this demo, use a video no longer than 60 seconds.",
            duration_seconds=duration_seconds,
            face_detection_rate=0,
        )

    sample_stride = max(1, math.ceil(fps / MAX_SAMPLE_RATE_HZ))
    sample_rate = fps / sample_stride
    cascade_path = Path(cv2.data.haarcascades) / "haarcascade_frontalface_default.xml"
    try:
        face_detector = cv2.CascadeClassifier(str(cascade_path))
    except Exception:
        capture.release()
        raise
    if face_detector.empty():
        capture.release()
        raise RuntimeError("OpenCV's frontal-face cascade could not be loaded.")

    green_samples: list[list[float] | None] = []
    sampled_frames = 0
    frame_index = 0
    while True:
        success, frame = capture.read()
        if not success:
            break
        if frame_index % sample_stride == 0:
            sampled_frames += 1
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            faces = face_detector.detectMultiScale(
                gray,
                scaleFactor=1.1,
                minNeighbors=5,
                minSize=(80, 80),
            )
            if len(faces):
                largest_face = max(faces, key=lambda face: face[2] * face[3])
                region_values = []
                for region in _face_regions(frame, largest_face):
                    if region.size == 0:
                        region_values = []
                        break
                    region_values.append(float(np.mean(region[:, :, 1])))
                green_samples.append(region_values if len(region_values) == 3 else None)
            else:
                green_samples.append(None)
        frame_index += 1
    capture.release()

    face_detection_rate = (
        sum(sample is not None for sample in green_samples) / sampled_frames
        if sampled_frames
        else 0.0
    )
    if face_detection_rate < 0.6:
        return _result(
            status="face_not_tracked",
            verdict="FACE_NOT_TRACKED",
            message="A face was not tracked reliably. Use a single, visible face, facing the camera in steady light.",
            duration_seconds=duration_seconds,
            face_detection_rate=face_detection_rate,
        )

    valid_samples = np.array([sample for sample in green_samples if sample is not None])
    sample_positions = np.flatnonzero([sample is not None for sample in green_samples])
    if valid_samples.shape[0] < 2:
        return _result(
            status="insufficient_signal",
            verdict="INSUFFICIENT_SIGNAL",
            message="Not enough face-region samples were collected to estimate a pulse.",
            duration_seconds=duration_seconds,
            face_detection_rate=face_detection_rate,
        )

    all_positions = np.arange(sampled_frames)
    interpolated = np.column_stack(
        [
            np.interp(all_positions, sample_positions, valid_samples[:, region_index])
            for region_index in range(3)
        ]
    )
    return _analyze_signals(
        interpolated,
        sample_rate,
        duration_seconds,
        face_detection_rate,
    )


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print(json.dumps({"error": "Usage: analyze_video.py VIDEO_PATH"}))
        sys.exit(2)
    try:
        print(json.dumps(analyze_video(sys.argv[1]), allow_nan=False))
    except Exception as error:
        print(json.dumps({"error": f"Video analysis failed: {error}"}))
        sys.exit(1)
