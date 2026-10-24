
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
    """
    Experimental POS-based remote photoplethysmography (rPPG).

    Input: RGB samples shaped (N, 3), ordered as red, green, blue.
    Output: estimated BPM, a rough signal-quality score, and waveform.

    This is a prototype, not a medical measurement device.
    """

    def __init__(
        self,
        min_bpm=45,
        max_bpm=180,
        min_seconds=20,
        max_seconds=30,
    ):
        self.min_bpm = min_bpm
        self.max_bpm = max_bpm
        self.min_seconds = min_seconds
        self.max_seconds = max_seconds

    @staticmethod
    def _resample(timestamps, rgb, fs):
        timestamps = np.asarray(timestamps, dtype=float)
        rgb = np.asarray(rgb, dtype=float)

        valid = (
            np.isfinite(timestamps)
            & np.all(np.isfinite(rgb), axis=1)
        )
        timestamps = timestamps[valid]
        rgb = rgb[valid]

        if len(timestamps) < 2:
            return None, None

        order = np.argsort(timestamps)
        timestamps = timestamps[order]
        rgb = rgb[order]

        # Remove duplicate timestamps.
        timestamps, unique_indices = np.unique(
            timestamps, return_index=True
        )
        rgb = rgb[unique_indices]

        if len(timestamps) < 2:
            return None, None

        duration = timestamps[-1] - timestamps[0]
        if duration <= 0:
            return None, None

        uniform_times = np.arange(
            timestamps[0],
            timestamps[-1],
            1.0 / fs,
        )

        if len(uniform_times) < 2:
            return None, None

        uniform_rgb = np.column_stack([
            np.interp(uniform_times, timestamps, rgb[:, channel])
            for channel in range(3)
        ])

        return uniform_times, uniform_rgb

    def _pos(self, rgb):
        # Normalize each colour channel by its temporal mean.
        means = np.mean(rgb, axis=0)
        if np.any(means <= 1e-8):
            return None

        normalized = rgb / means - 1.0
        red, green, blue = normalized.T

        # POS projection.
        x = green - blue
        y = green + blue - 2.0 * red

        std_x = np.std(x)
        std_y = np.std(y)

        if std_x < 1e-8 or std_y < 1e-8:
            return None

        alpha = std_x / std_y
        pulse = x + alpha * y

        return detrend(pulse, type="linear")

    def estimate(self, timestamps, rgb, fs=30.0):
        """
        timestamps: seconds from a consistent clock, shape (N,)
        rgb: per-frame RGB means, shape (N, 3)
        fs: target resampling frequency in Hz
        """
        times, samples = self._resample(timestamps, rgb, fs)

        if times is None:
            return PulseResult(
                None, 0.0, "Insufficient data", np.array([])
            )

        duration = times[-1] - times[0]

        if duration < self.min_seconds:
            return PulseResult(
                None, 0.0, "Collecting signal", np.array([])
            )

        # Use the latest portion of the recording.
        keep = times >= times[-1] - self.max_seconds
        times = times[keep]
        samples = samples[keep]

        pulse = self._pos(samples)

        if pulse is None:
            return PulseResult(
                None, 0.0, "Poor RGB signal", np.array([])
            )

        low_hz = self.min_bpm / 60.0
        high_hz = self.max_bpm / 60.0

        try:
            sos = butter(
                3,
                [low_hz, high_hz],
                btype="bandpass",
                fs=fs,
                output="sos",
            )
            filtered = sosfiltfilt(sos, pulse)
        except ValueError:
            return PulseResult(
                None, 0.0, "Signal window too short", pulse
            )

        filtered = detrend(filtered, type="linear")
        amplitude = np.std(filtered)

        if not np.isfinite(amplitude) or amplitude < 1e-6:
            return PulseResult(
                None, 0.0, "Weak pulse signal", filtered
            )

        # Window the waveform before calculating its spectrum.
        windowed = filtered * np.hanning(len(filtered))
        spectrum = np.abs(np.fft.rfft(windowed)) ** 2
        frequencies = np.fft.rfftfreq(len(windowed), d=1.0 / fs)

        band = (frequencies >= low_hz) & (frequencies <= high_hz)
        band_indices = np.flatnonzero(band)

        if len(band_indices) < 2:
            return PulseResult(
                None, 0.0, "Insufficient frequency resolution", filtered
            )

        band_power = spectrum[band]
        peak_index = int(np.argmax(band_power))
        peak_bin = band_indices[peak_index]
        peak_hz = frequencies[peak_bin]

        if peak_hz <= 0:
            return PulseResult(
                None, 0.0, "No credible pulse peak", filtered
            )

        total_power = float(np.sum(band_power))
        peak_power = float(spectrum[peak_bin])
        quality = (
            peak_power / total_power if total_power > 0 else 0.0
        )
        quality = float(np.clip(quality, 0.0, 1.0))

        # This is only a rough spectral concentration score,
        # not a calibrated probability of correctness.
        bpm = float(peak_hz * 60.0)

        return PulseResult(
            round(bpm, 1),
            round(quality, 3),
            "Experimental estimate",
            filtered,
        )
