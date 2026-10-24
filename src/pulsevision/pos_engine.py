from dataclasses import dataclass

import numpy as np
from scipy.signal import butter, detrend, find_peaks, sosfiltfilt


@dataclass
class PulseResult:
    bpm: float | None
    signal_quality: float
    status: str
    waveform: np.ndarray


class POSEngine:
    """Experimental POS-based remote pulse estimator."""

    def __init__(
        self,
        min_bpm=45,
        max_bpm=180,
        min_seconds=20,
        max_seconds=30,
    ):
        self.min_bpm = float(min_bpm)
        self.max_bpm = float(max_bpm)
        self.min_seconds = float(min_seconds)
        self.max_seconds = float(max_seconds)

    @staticmethod
    def _fail(status, waveform=None, quality=0.0):
        return PulseResult(
            None,
            float(quality),
            status,
            np.asarray(
                waveform if waveform is not None else [],
                dtype=float,
            ),
        )

    @staticmethod
    def _clean_samples(timestamps, rgb):
        timestamps = np.asarray(timestamps, dtype=float).reshape(-1)
        rgb = np.asarray(rgb, dtype=float)

        if rgb.ndim != 2 or rgb.shape[1] != 3:
            return None, None

        if len(timestamps) != len(rgb):
            return None, None

        valid = (
            np.isfinite(timestamps)
            & np.all(np.isfinite(rgb), axis=1)
        )

        timestamps = timestamps[valid]
        rgb = rgb[valid]

        if len(timestamps) < 3:
            return None, None

        order = np.argsort(timestamps)
        timestamps = timestamps[order]
        rgb = rgb[order]

        unique_times, inverse = np.unique(
            timestamps,
            return_inverse=True,
        )

        if len(unique_times) != len(timestamps):
            sums = np.zeros((len(unique_times), 3))
            counts = np.zeros(len(unique_times))
            np.add.at(sums, inverse, rgb)
            np.add.at(counts, inverse, 1)
            rgb = sums / counts[:, None]
            timestamps = unique_times

        if len(timestamps) < 3 or timestamps[-1] <= timestamps[0]:
            return None, None

        return timestamps, rgb

    def _resample(self, timestamps, rgb):
        timestamps, rgb = self._clean_samples(timestamps, rgb)

        if timestamps is None:
            return None, None, None, "Insufficient valid samples"

        duration = timestamps[-1] - timestamps[0]

        if duration < self.min_seconds:
            return None, None, None, "Collecting signal"

        intervals = np.diff(timestamps)
        intervals = intervals[
            np.isfinite(intervals) & (intervals > 0)
        ]

        if len(intervals) < 2:
            return None, None, None, "Invalid sample timestamps"

        median_interval = float(np.median(intervals))
        measured_fps = 1.0 / median_interval

        if not np.isfinite(measured_fps) or measured_fps < 5:
            return None, None, measured_fps, "Sampling rate too low"

        gaps = np.diff(timestamps)
        gap_fraction = float(
            np.mean(gaps > max(0.5, 3.0 * median_interval))
        )

        if gap_fraction > 0.10:
            return None, None, measured_fps, "Too many dropped frames"

        fps = min(measured_fps, 15.0)

        uniform_times = np.arange(
            timestamps[0],
            timestamps[-1],
            1.0 / fps,
        )

        if len(uniform_times) < int(self.min_seconds * fps):
            return None, None, fps, "Collecting signal"

        uniform_rgb = np.column_stack([
            np.interp(
                uniform_times,
                timestamps,
                rgb[:, channel],
            )
            for channel in range(3)
        ])

        return uniform_times, uniform_rgb, fps, None

    @staticmethod
    def _pos(rgb):
        means = np.mean(rgb, axis=0)

        if not np.all(np.isfinite(means)) or np.any(means <= 1e-8):
            return None

        normalized = rgb / means - 1.0
        red, green, blue = normalized.T

        x = green - blue
        y = green + blue - 2.0 * red

        std_x = float(np.std(x))
        std_y = float(np.std(y))

        if std_x < 1e-8 or std_y < 1e-8:
            return None

        return detrend(
            x + (std_x / std_y) * y,
            type="linear",
        )

    @staticmethod
    def _interpolated_peak(frequencies, power, index):
        if index <= 0 or index >= len(power) - 1:
            return float(frequencies[index])

        values = np.log(
            np.maximum(power[index - 1:index + 2], 1e-12)
        )

        y0, y1, y2 = values
        denominator = y0 - 2.0 * y1 + y2

        if abs(denominator) < 1e-12:
            return float(frequencies[index])

        offset = 0.5 * (y0 - y2) / denominator
        offset = float(np.clip(offset, -1.0, 1.0))

        spacing = frequencies[1] - frequencies[0]

        return float(
            frequencies[index] + offset * spacing
        )

    def estimate(self, timestamps, rgb, fs=None):
        times, samples, fps, error = self._resample(
            timestamps,
            rgb,
        )

        if error:
            return self._fail(error)

        if times is None or samples is None or fps is None:
            return self._fail("Insufficient valid samples")

        keep = times >= times[-1] - self.max_seconds
        times = times[keep]
        samples = samples[keep]

        duration = float(times[-1] - times[0])

        if duration < self.min_seconds:
            return self._fail("Collecting signal")

        pulse = self._pos(samples)

        if pulse is None:
            return self._fail("Poor RGB signal")

        low_hz = self.min_bpm / 60.0
        high_hz = min(self.max_bpm / 60.0, fps * 0.45)

        if high_hz <= low_hz:
            return self._fail("Sampling rate insufficient")

        try:
            sos = butter(
                3,
                [low_hz, high_hz],
                btype="bandpass",
                fs=fps,
                output="sos",
            )

            filtered = sosfiltfilt(sos, pulse)

        except (ValueError, FloatingPointError):
            return self._fail("Signal window too short", pulse)

        filtered = detrend(filtered, type="linear")
        amplitude = float(np.std(filtered))

        if not np.isfinite(amplitude) or amplitude < 1e-7:
            return self._fail("Weak pulse signal", filtered)

        sample_count = len(filtered)
        segment_len = min(
            sample_count,
            max(int(10 * fps), int(0.6 * sample_count)),
        )

        if segment_len < int(self.min_seconds * fps * 0.75):
            segment_len = sample_count

        step = max(1, segment_len // 2)
        power_sum = None
        segment_count = 0

        for start in range(
            0,
            sample_count - segment_len + 1,
            step,
        ):
            segment = filtered[start:start + segment_len]
            segment = (segment - np.mean(segment)) * np.hanning(
                len(segment)
            )

            power = np.abs(np.fft.rfft(segment)) ** 2

            power_sum = (
                power
                if power_sum is None
                else power_sum + power
            )

            segment_count += 1

        if power_sum is None:
            segment = (
                filtered - np.mean(filtered)
            ) * np.hanning(sample_count)

            power_sum = np.abs(np.fft.rfft(segment)) ** 2
        else:
            power_sum /= max(1, segment_count)

        frequencies = np.fft.rfftfreq(
            segment_len,
            d=1.0 / fps,
        )

        band = (
            (frequencies >= low_hz)
            & (frequencies <= high_hz)
        )

        indices = np.flatnonzero(band)

        if len(indices) < 3:
            return self._fail(
                "Insufficient frequency resolution",
                filtered,
            )

        band_power = power_sum[band]
        peak_index = int(indices[int(np.argmax(band_power))])

        peak_hz = self._interpolated_peak(
            frequencies,
            power_sum,
            peak_index,
        )

        if not np.isfinite(peak_hz) or peak_hz <= 0:
            return self._fail("No credible pulse peak", filtered)

        total_power = float(np.sum(band_power))
        peak_power = float(power_sum[peak_index])

        quality = (
            peak_power / total_power
            if total_power > 0
            else 0.0
        )

        quality = float(np.clip(quality, 0.0, 1.0))
        spectral_bpm = peak_hz * 60.0

        peaks, _ = find_peaks(
            filtered,
            distance=max(1, int(0.33 * fps)),
        )

        peak_bpm = None

        if len(peaks) >= 4:
            intervals = np.diff(peaks) / fps

            intervals = intervals[
                (intervals >= 60.0 / self.max_bpm)
                & (intervals <= 60.0 / self.min_bpm)
            ]

            if len(intervals) >= 3:
                peak_bpm = 60.0 / float(np.median(intervals))

        # Keep a provisional reading visible but flag disagreement.
        if quality < 0.03:
            return self._fail(
                "Very low signal quality; hold still",
                filtered,
                quality,
            )

        if (
            peak_bpm is not None
            and abs(spectral_bpm - peak_bpm) > 12.0
        ):
            return PulseResult(
                round(float(spectral_bpm), 1),
                round(quality, 3),
                "Unstable provisional estimate; hold still",
                filtered,
            )

        if quality < 0.08:
            return PulseResult(
                round(float(spectral_bpm), 1),
                round(quality, 3),
                "Low signal quality; provisional estimate",
                filtered,
            )

        bpm = (
            spectral_bpm
            if peak_bpm is None
            else 0.7 * spectral_bpm + 0.3 * peak_bpm
        )

        status = (
            "Experimental estimate"
            if quality >= 0.15
            else "Low-quality experimental estimate"
        )

        return PulseResult(
            round(float(bpm), 1),
            round(quality, 3),
            status,
            filtered,
        )
