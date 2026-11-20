"""Real-time spectrum and waterfall visualizer for PyFSK-Modem."""

from __future__ import annotations

import logging
import os
import queue
import sys
from typing import Optional
import numpy as np

logger = logging.getLogger(__name__)


class SpectrumWaterfallVisualizer:
    """Real-time audio spectrum and rolling waterfall display using Matplotlib."""

    def __init__(
        self,
        sample_rate: int = 44100,
        mark_freq: float = 1800.0,
        space_freq: float = 2200.0,
        fft_size: int = 2048,
        history_frames: int = 80,
    ) -> None:
        """Initialize visualizer parameters.

        Args:
            sample_rate: Audio sampling frequency in Hz.
            mark_freq: Frequency for binary 1 (Hz).
            space_freq: Frequency for binary 0 (Hz).
            fft_size: Number of samples per FFT window.
            history_frames: Number of rolling time steps in the waterfall.
        """
        self.sample_rate = sample_rate
        self.mark_freq = mark_freq
        self.space_freq = space_freq
        self.fft_size = fft_size
        self.history_frames = history_frames

        self.freqs = np.fft.rfftfreq(fft_size, 1.0 / sample_rate)
        self.window = np.hanning(fft_size)

        # Buffer for incoming audio
        self.audio_buf = np.zeros(fft_size, dtype=np.float32)
        # Waterfall 2D array: (history_frames, rfft_bins)
        self.waterfall_data = np.full((history_frames, len(self.freqs)), -100.0, dtype=np.float32)

    def is_display_available(self) -> bool:
        """Check if a graphical display environment is present."""
        if sys.platform.startswith("linux") and not os.environ.get("DISPLAY") and not os.environ.get("WAYLAND_DISPLAY"):
            return False
        return True

    def run(self, audio_queue: queue.Queue[np.ndarray], stop_event: Optional[Any] = None) -> None:
        """Launch the live Matplotlib GUI window and consume audio frames from audio_queue.

        Args:
            audio_queue: Queue delivering incoming 1D float32 audio chunks.
            stop_event: Optional threading.Event to signal shutdown.
        """
        if not self.is_display_available():
            logger.warning("No GUI display detected ($DISPLAY not set). Running without graphical visualizer.")
            return

        os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")
        try:
            import matplotlib.pyplot as plt
            from matplotlib.animation import FuncAnimation
        except Exception as exc:
            logger.warning(f"Could not initialize Matplotlib visualizer: {exc}")
            return

        fig, (ax_spec, ax_wf) = plt.subplots(2, 1, figsize=(10, 7), gridspec_kw={"height_ratios": [1, 1.5]})
        fig.canvas.manager.set_window_title("PyFSK-Modem: Live Audio Spectrum & Waterfall")

        # Focus frequency range around mark and space (+/- 3 kHz margin)
        min_f = max(0.0, min(self.mark_freq, self.space_freq) - 2500.0)
        max_f = min(self.sample_rate / 2.0, max(self.mark_freq, self.space_freq) + 2500.0)

        # Top plot: Spectrum
        (line_spec,) = ax_spec.plot(self.freqs, np.zeros_like(self.freqs), color="cyan", linewidth=1.2)
        ax_spec.axvline(self.mark_freq, color="green", linestyle="--", alpha=0.8, label=f"Mark ({self.mark_freq:.0f} Hz)")
        ax_spec.axvline(self.space_freq, color="magenta", linestyle="--", alpha=0.8, label=f"Space ({self.space_freq:.0f} Hz)")
        ax_spec.set_xlim(min_f, max_f)
        ax_spec.set_ylim(-90, 10)
        ax_spec.set_ylabel("Power (dBFS)")
        ax_spec.set_title("Live FFT Spectrum")
        ax_spec.grid(True, linestyle=":", alpha=0.6)
        ax_spec.legend(loc="upper right")
        ax_spec.set_facecolor("#111118")

        # Bottom plot: Waterfall
        im_wf = ax_wf.imshow(
            self.waterfall_data,
            aspect="auto",
            origin="lower",
            extent=[self.freqs[0], self.freqs[-1], 0, self.history_frames],
            cmap="inferno",
            vmin=-90,
            vmax=0,
        )
        ax_wf.set_xlim(min_f, max_f)
        ax_wf.set_xlabel("Frequency (Hz)")
        ax_wf.set_ylabel("Time History (Frames)")
        ax_wf.set_title("Spectrogram Waterfall")
        fig.tight_layout()

        def update_frame(_):
            # Drain queue and append to audio buffer
            has_new = False
            while True:
                try:
                    chunk = audio_queue.get_nowait()
                    n = len(chunk)
                    if n >= self.fft_size:
                        self.audio_buf = chunk[-self.fft_size:].copy()
                    else:
                        self.audio_buf = np.roll(self.audio_buf, -n)
                        self.audio_buf[-n:] = chunk
                    has_new = True
                except queue.Empty:
                    break

            if has_new:
                # Compute windowed FFT
                windowed = self.audio_buf * self.window
                fft_complex = np.fft.rfft(windowed)
                power = np.abs(fft_complex) ** 2 + 1e-12
                power_db = 10.0 * np.log10(power / (self.fft_size ** 2))

                line_spec.set_ydata(power_db)

                # Roll waterfall
                self.waterfall_data = np.roll(self.waterfall_data, -1, axis=0)
                self.waterfall_data[-1, :] = power_db
                im_wf.set_data(self.waterfall_data)

            return line_spec, im_wf

        anim = FuncAnimation(fig, update_frame, interval=40, blit=True, cache_frame_data=False)
        plt.show()
