"""Unit tests for BFSK Modulator: frequency accuracy, phase continuity, and envelope shaping."""

import numpy as np
import pytest
from pyfsk_modem.config import ModemConfig, ModemMode
from pyfsk_modem.modulator import Modulator


@pytest.mark.parametrize("mode", [ModemMode.AUDIBLE, ModemMode.ULTRASONIC])
def test_modulator_frequency_peaks(mode: ModemMode):
    """Verify that modulated mark and space symbols exhibit FFT peaks at expected frequencies."""
    baud = 50  # Lower baud -> longer symbol -> higher FFT bin resolution
    cfg = ModemConfig.create(mode=mode, baud_rate=baud, guard_duration_s=0.0, ramp_duration_s=0.0)
    mod = Modulator(cfg)

    # Modulate a long series of 1s (mark)
    mark_bits = [1] * 20
    mark_wave = mod.modulate_bits(mark_bits)

    # Compute FFT
    n = len(mark_wave)
    fft_vals = np.abs(np.fft.rfft(mark_wave))
    fft_freqs = np.fft.rfftfreq(n, 1.0 / cfg.sample_rate)

    peak_freq_mark = fft_freqs[np.argmax(fft_vals)]
    assert abs(peak_freq_mark - cfg.mark_freq) < (cfg.sample_rate / n) * 2, (
        f"Mark peak frequency mismatch: expected {cfg.mark_freq}, got {peak_freq_mark}"
    )

    # Modulate a long series of 0s (space)
    space_bits = [0] * 20
    space_wave = mod.modulate_bits(space_bits)
    fft_vals_s = np.abs(np.fft.rfft(space_wave))
    peak_freq_space = fft_freqs[np.argmax(fft_vals_s)]
    assert abs(peak_freq_space - cfg.space_freq) < (cfg.sample_rate / n) * 2, (
        f"Space peak frequency mismatch: expected {cfg.space_freq}, got {peak_freq_space}"
    )


def test_modulator_phase_continuity():
    """Verify phase continuity across symbol transitions (no sudden discontinuities)."""
    cfg = ModemConfig.create(baud_rate=100, guard_duration_s=0.0, ramp_duration_s=0.0)
    mod = Modulator(cfg)

    # Alternating bits to force frequent symbol boundaries
    bits = [1, 0, 1, 0, 1, 1, 0, 0]
    wave = mod.modulate_bits(bits)

    # Numerical derivative of waveform: max slope bounded by 2 * pi * f_max * amplitude
    f_max = max(cfg.mark_freq, cfg.space_freq)
    diff = np.diff(wave)
    max_expected_diff = (2.0 * np.pi * f_max / cfg.sample_rate) * cfg.amplitude * 1.25

    assert np.max(np.abs(diff)) <= max_expected_diff, (
        f"Detected possible phase discontinuity: max diff {np.max(np.abs(diff))} > {max_expected_diff}"
    )


def test_modulator_silence_guards():
    """Verify silence guards are prepended and appended."""
    guard_s = 0.05
    cfg = ModemConfig.create(guard_duration_s=guard_s)
    mod = Modulator(cfg)

    wave = mod.modulate("Hello")
    guard_samples = cfg.guard_samples

    # Check that first guard_samples and last guard_samples are pure zeros
    assert np.all(wave[:guard_samples] == 0.0)
    assert np.all(wave[-guard_samples:] == 0.0)


def test_modulator_envelope_ramp():
    """Verify raised-cosine smoothing ramps at waveform boundaries."""
    cfg = ModemConfig.create(ramp_duration_s=0.02, guard_duration_s=0.0)
    mod = Modulator(cfg)

    wave = mod.modulate("A")
    # Amplitude at sample 0 should be 0.0 due to raised-cosine ramp
    assert abs(wave[0]) < 1e-4
    assert abs(wave[-1]) < 1e-4
