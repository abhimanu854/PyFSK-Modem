"""Configuration module for PyFSK-Modem."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import List


class ModemMode(str, Enum):
    """Acoustic frequency band modes."""
    AUDIBLE = "audible"
    ULTRASONIC = "ultrasonic"


@dataclass
class ModemConfig:
    """Tunable parameters for BFSK acoustic transmission and reception.

    Attributes:
        mode: Operating mode ('audible' or 'ultrasonic').
        sample_rate: Audio sampling frequency in Hz (default: 44100).
        baud_rate: Transmission speed in symbols per second (default: 100, up to 300).
        mark_freq: Frequency representing binary '1' in Hz.
        space_freq: Frequency representing binary '0' in Hz.
        amplitude: Peak output amplitude (0.0 to 1.0, default: 0.8 to avoid clipping).
        preamble_bits: Number of alternating preamble bits (1010...; default: 16).
        sync_word: 16-bit synchronization word (default: 0x2DD4).
        ramp_duration_s: Duration of raised-cosine fade-in and fade-out in seconds (default: 0.01).
        guard_duration_s: Duration of leading/trailing silence guard in seconds (default: 0.05).
        squelch_threshold: Minimum signal energy ratio relative to noise floor to trigger detection.
        max_payload_bytes: Maximum payload bytes per packet (default: 255).
    """

    mode: ModemMode = ModemMode.AUDIBLE
    sample_rate: int = 44100
    baud_rate: int = 100
    mark_freq: float = 1800.0
    space_freq: float = 2200.0
    amplitude: float = 0.8
    preamble_bits: int = 16
    sync_word: int = 0x2DD4
    ramp_duration_s: float = 0.01
    guard_duration_s: float = 0.05
    squelch_threshold: float = 2.0
    max_payload_bytes: int = 255

    # Derived attributes computed post-init
    samples_per_symbol: int = field(init=False)
    ramp_samples: int = field(init=False)
    guard_samples: int = field(init=False)

    def __post_init__(self) -> None:
        """Calculate derived sample counts and validate parameters."""
        if isinstance(self.mode, str):
            self.mode = ModemMode(self.mode.lower())

        if self.baud_rate <= 0:
            raise ValueError(f"Baud rate must be positive, got {self.baud_rate}")
        if self.sample_rate <= 0:
            raise ValueError(f"Sample rate must be positive, got {self.sample_rate}")
        if not (0.0 < self.amplitude <= 1.0):
            raise ValueError(f"Amplitude must be between 0.0 and 1.0, got {self.amplitude}")
        if self.max_payload_bytes <= 0 or self.max_payload_bytes > 255:
            raise ValueError(f"Max payload bytes must be between 1 and 255, got {self.max_payload_bytes}")

        self.samples_per_symbol = int(round(self.sample_rate / self.baud_rate))
        self.ramp_samples = max(1, int(round(self.ramp_duration_s * self.sample_rate)))
        self.guard_samples = int(round(self.guard_duration_s * self.sample_rate))

    @classmethod
    def create(
        cls,
        mode: str | ModemMode = ModemMode.AUDIBLE,
        baud_rate: int = 100,
        sample_rate: int = 44100,
        amplitude: float = 0.8,
        **kwargs,
    ) -> ModemConfig:
        """Factory method to instantiate configuration for a given mode."""
        mode_enum = ModemMode(mode.lower()) if isinstance(mode, str) else mode
        if mode_enum == ModemMode.AUDIBLE:
            mark_freq = 1800.0
            space_freq = 2200.0
        elif mode_enum == ModemMode.ULTRASONIC:
            # Ultrasonic acoustic band below 44.1 kHz Nyquist (22.05 kHz)
            mark_freq = 18500.0
            space_freq = 19500.0
        else:
            raise ValueError(f"Unknown modem mode: {mode}")

        config_params = {
            "mode": mode_enum,
            "sample_rate": sample_rate,
            "baud_rate": baud_rate,
            "mark_freq": mark_freq,
            "space_freq": space_freq,
            "amplitude": amplitude,
        }
        config_params.update(kwargs)
        return cls(**config_params)

    @property
    def preamble_bit_pattern(self) -> List[int]:
        """Generate the alternating 1010... preamble bit sequence."""
        return [1 if i % 2 == 0 else 0 for i in range(self.preamble_bits)]

    @property
    def sync_word_bits(self) -> List[int]:
        """Generate 16-bit sync word as a list of integers (MSB first)."""
        return [(self.sync_word >> (15 - i)) & 1 for i in range(16)]
