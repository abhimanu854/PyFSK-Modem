"""BFSK Modulator module for PyFSK-Modem."""

from __future__ import annotations

from typing import List, Sequence
import numpy as np
from pyfsk_modem.config import ModemConfig
from pyfsk_modem.framing import build_packet


class Modulator:
    """Continuous-phase BFSK acoustic modulator.

    Converts binary bit streams into phase-continuous acoustic waveforms with
    raised-cosine windowed edges and silence guards.
    """

    def __init__(self, config: ModemConfig | None = None) -> None:
        """Initialize modulator with configuration."""
        self.config = config or ModemConfig.create()

    def modulate_bits(self, bits: Sequence[int]) -> np.ndarray:
        """Synthesize a continuous-phase BFSK audio waveform from a bit sequence.

        Args:
            bits: Sequence of 0s and 1s.

        Returns:
            np.ndarray of float32 audio samples normalized to [-amplitude, +amplitude].
        """
        if not bits:
            return np.zeros(self.config.guard_samples * 2, dtype=np.float32)

        bits_arr = np.asarray(bits, dtype=np.int32)
        samples_per_symbol = self.config.samples_per_symbol
        sample_rate = self.config.sample_rate

        # Map bits: 1 -> mark_freq, 0 -> space_freq
        symbol_freqs = np.where(bits_arr == 1, self.config.mark_freq, self.config.space_freq)

        # Expand per-symbol frequencies across samples
        freq_series = np.repeat(symbol_freqs, samples_per_symbol)

        # Instantaneous phase calculation via cumulative integration (CPFSK)
        phase_increments = 2.0 * np.pi * freq_series / sample_rate
        phase = np.cumsum(phase_increments) % (2.0 * np.pi)

        waveform = (self.config.amplitude * np.sin(phase)).astype(np.float32)

        # Apply raised-cosine ramps at start and end of transmission
        ramp_len = min(self.config.ramp_samples, len(waveform) // 2)
        if ramp_len > 1:
            ramp_up = 0.5 * (1.0 - np.cos(np.pi * np.arange(ramp_len) / ramp_len))
            ramp_down = 0.5 * (1.0 + np.cos(np.pi * np.arange(ramp_len) / ramp_len))
            waveform[:ramp_len] *= ramp_up.astype(np.float32)
            waveform[-ramp_len:] *= ramp_down.astype(np.float32)

        # Prepend and append silence guard intervals
        guard_samples = self.config.guard_samples
        if guard_samples > 0:
            guard = np.zeros(guard_samples, dtype=np.float32)
            waveform = np.concatenate([guard, waveform, guard])

        return waveform

    def modulate(self, text_or_bytes: str | bytes) -> np.ndarray:
        """Convenience method to frame text/bytes into a packet and modulate.

        Args:
            text_or_bytes: ASCII text message or raw bytes.

        Returns:
            np.ndarray of audio samples ready for transmission.
        """
        bits = build_packet(text_or_bytes, self.config)
        return self.modulate_bits(bits)
