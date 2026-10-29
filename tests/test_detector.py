import unittest
import tempfile
from pathlib import Path

import cv2
import numpy as np

from detector.analyze_video import _analyze_signals, analyze_video


class PulseSignalTests(unittest.TestCase):
    def test_estimates_shared_synthetic_pulse_frequency(self):
        sample_rate = 12.0
        duration = 20.0
        timestamps = np.arange(int(sample_rate * duration)) / sample_rate
        pulse_hz = 1.2
        signals = np.column_stack(
            [
                100 + 2.0 * np.sin(2 * np.pi * pulse_hz * timestamps),
                120 + 1.5 * np.sin(2 * np.pi * pulse_hz * timestamps + 0.2),
                90 + 2.5 * np.sin(2 * np.pi * pulse_hz * timestamps - 0.1),
            ]
        )

        result = _analyze_signals(signals, sample_rate, duration, 1.0)

        self.assertEqual(result["status"], "pulse_estimated")
        self.assertEqual(result["verdict"], "PULSE_ESTIMATED")
        self.assertAlmostEqual(result["bpm"], 72.0, delta=3.0)
        self.assertGreater(result["signalQuality"], 0)
        self.assertLessEqual(result["signalQuality"], 100)
        self.assertTrue(result["pulseSignal"])

    def test_reports_low_quality_for_disagreeing_regions(self):
        sample_rate = 12.0
        duration = 20.0
        timestamps = np.arange(int(sample_rate * duration)) / sample_rate
        signals = np.column_stack(
            [
                np.sin(2 * np.pi * 0.8 * timestamps),
                np.sin(2 * np.pi * 1.5 * timestamps),
                np.sin(2 * np.pi * 2.8 * timestamps),
            ]
        )

        result = _analyze_signals(signals, sample_rate, duration, 1.0)

        self.assertEqual(result["status"], "low_signal_quality")
        self.assertEqual(result["verdict"], "LOW_SIGNAL_QUALITY")

    def test_rejects_video_shorter_than_ten_seconds(self):
        result = _analyze_signals(np.empty((0, 3)), 0, 5.0, 0)

        self.assertEqual(result["status"], "video_too_short")
        self.assertIsNone(result["bpm"])

    def test_reads_decodable_video_without_a_face(self):
        with tempfile.TemporaryDirectory() as directory:
            video_path = Path(directory) / "blank.mp4"
            writer = cv2.VideoWriter(
                str(video_path),
                cv2.VideoWriter_fourcc(*"mp4v"),
                12,
                (160, 120),
            )
            if not writer.isOpened():
                self.skipTest("This OpenCV build has no MP4 video encoder.")
            for _ in range(120):
                writer.write(np.zeros((120, 160, 3), dtype=np.uint8))
            writer.release()

            result = analyze_video(video_path)

        self.assertEqual(result["status"], "face_not_tracked")
        self.assertIsNone(result["bpm"])


if __name__ == "__main__":
    unittest.main()
