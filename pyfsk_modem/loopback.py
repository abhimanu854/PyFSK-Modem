"""In-memory loopback simulation and Bit Error Rate (BER) testing for PyFSK-Modem."""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple
import numpy as np

from pyfsk_modem.config import ModemConfig
from pyfsk_modem.demodulator import Demodulator, PacketResult
from pyfsk_modem.framing import build_packet, bytes_to_bits
from pyfsk_modem.modulator import Modulator


@dataclass
class LoopbackResult:
    """Outcome of a single loopback transmission test."""
    target_snr_db: float
    measured_snr_db: float
    transmitted_text: str
    received_text: str
    crc_ok: bool
    packet_detected: bool
    bit_errors: int
    total_bits: int
    ber: float
    status: str

    def format_summary(self) -> str:
        """Format loopback result for display."""
        crc_str = "PASS" if self.crc_ok else "FAIL"
        det_str = "YES" if self.packet_detected else "NO"
        return (
            f"Target SNR: {self.target_snr_db:4.1f} dB | Measured SNR: {self.measured_snr_db:4.1f} dB | "
            f"Detected: {det_str} | CRC: {crc_str} | BER: {self.ber * 100:5.2f}% "
            f"({self.bit_errors}/{self.total_bits} bits) | RX: '{self.received_text}'"
        )


def add_awgn_noise(signal_array: np.ndarray, target_snr_db: float) -> np.ndarray:
    """Add Additive White Gaussian Noise (AWGN) to achieve a specified SNR.

    Args:
        signal_array: Clean audio sample array.
        target_snr_db: Target Signal-to-Noise Ratio in decibels.

    Returns:
        Noisy audio sample array.
    """
    sig_power = float(np.mean(signal_array ** 2))
    if sig_power <= 1e-12:
        return signal_array.copy()

    snr_linear = 10.0 ** (target_snr_db / 10.0)
    noise_power = sig_power / snr_linear
    noise_std = np.sqrt(noise_power)

    noise = np.random.normal(0.0, noise_std, len(signal_array)).astype(np.float32)
    return (signal_array + noise).astype(np.float32)


def run_loopback(
    payload: str = "PyFSK Air-Gap Modem",
    target_snr_db: float = 15.0,
    config: Optional[ModemConfig] = None,
    delay_samples: int = 1500,
) -> LoopbackResult:
    """Run an end-to-end loopback simulation through a software AWGN channel.

    Args:
        payload: ASCII text to transmit.
        target_snr_db: Desired channel SNR in dB.
        config: ModemConfig instance (defaults to audible mode).
        delay_samples: Lead-in noise delay samples to test timing recovery.

    Returns:
        LoopbackResult detailing reception fidelity, BER, and CRC validity.
    """
    cfg = config or ModemConfig.create()
    mod = Modulator(cfg)
    demod = Demodulator(cfg)

    tx_bits = build_packet(payload, cfg)
    clean_audio = mod.modulate_bits(tx_bits)

    # Add AWGN channel degradation
    noisy_audio = add_awgn_noise(clean_audio, target_snr_db)

    # Add leading and trailing arbitrary noise to test timing recovery
    noise_power = float(np.mean(noisy_audio ** 2)) / (10.0 ** (target_snr_db / 10.0))
    noise_std = np.sqrt(max(1e-8, noise_power))
    prefix_noise = np.random.normal(0, noise_std, delay_samples).astype(np.float32)
    suffix_noise = np.random.normal(0, noise_std, delay_samples).astype(np.float32)
    channel_audio = np.concatenate([prefix_noise, noisy_audio, suffix_noise])

    # Demodulate
    packets = demod.process_buffer(channel_audio)

    # Calculate payload bit errors
    tx_payload_bits = bytes_to_bits(payload.encode("ascii", errors="replace"))
    total_bits = len(tx_payload_bits)

    if packets:
        best_pkt = packets[0]
        rx_payload_bits = bytes_to_bits(best_pkt.payload)

        # Pad or truncate to compute bit errors
        min_len = min(len(tx_payload_bits), len(rx_payload_bits))
        bit_errors = sum(tx_payload_bits[i] != rx_payload_bits[i] for i in range(min_len))
        bit_errors += abs(len(tx_payload_bits) - len(rx_payload_bits)) * 8

        ber = bit_errors / total_bits if total_bits > 0 else 0.0

        return LoopbackResult(
            target_snr_db=target_snr_db,
            measured_snr_db=best_pkt.snr_db,
            transmitted_text=payload,
            received_text=best_pkt.text,
            crc_ok=best_pkt.crc_ok,
            packet_detected=True,
            bit_errors=bit_errors,
            total_bits=total_bits,
            ber=ber,
            status=best_pkt.status,
        )
    else:
        # Packet undetected due to heavy noise
        return LoopbackResult(
            target_snr_db=target_snr_db,
            measured_snr_db=0.0,
            transmitted_text=payload,
            received_text="",
            crc_ok=False,
            packet_detected=False,
            bit_errors=total_bits,
            total_bits=total_bits,
            ber=1.0,
            status="Undetected / Squelched",
        )


def sweep_snr_ber(
    snr_range: List[float],
    payload: str = "PyFSK Transmission Test 2026",
    trials_per_snr: int = 5,
    save_plot_path: Optional[str] = None,
    config: Optional[ModemConfig] = None,
) -> Dict[float, Dict[str, float]]:
    """Sweep a list of SNR levels, compute average BER and PER, and save plot.

    Args:
        snr_range: List of SNR values in dB (e.g. [20, 15, 10, 5, 0]).
        payload: Test text payload.
        trials_per_snr: Repetitions per SNR point for statistical averaging.
        save_plot_path: Optional file path to save Matplotlib PNG plot.
        config: ModemConfig instance.

    Returns:
        Dictionary mapping SNR (dB) to performance metrics (mean_ber, per, success_rate).
    """
    results: Dict[float, Dict[str, float]] = {}

    for snr in snr_range:
        bers: List[float] = []
        crc_oks = 0
        detected = 0

        for trial in range(trials_per_snr):
            # Vary random noise seed across trials
            res = run_loopback(payload, snr, config, delay_samples=1000 + trial * 250)
            bers.append(res.ber)
            if res.crc_ok:
                crc_oks += 1
            if res.packet_detected:
                detected += 1

        results[snr] = {
            "mean_ber": float(np.mean(bers)),
            "packet_success_rate": crc_oks / trials_per_snr,
            "detection_rate": detected / trials_per_snr,
        }

    if save_plot_path:
        os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")
        import matplotlib.pyplot as plt

        snrs = sorted(results.keys())
        mean_bers = [results[s]["mean_ber"] for s in snrs]
        pkt_rates = [results[s]["packet_success_rate"] * 100.0 for s in snrs]

        fig, ax1 = plt.subplots(figsize=(8, 5))

        color = "tab:blue"
        ax1.set_xlabel("Channel SNR (dB)", fontsize=12)
        ax1.set_ylabel("Bit Error Rate (BER)", color=color, fontsize=12)
        
        # Clamp zero values for log scale display
        clamped_bers = [max(b, 1e-4) for b in mean_bers]
        ax1.plot(snrs, clamped_bers, marker="o", color=color, linewidth=2, label="BER")
        ax1.set_yscale("log")
        ax1.set_ylim(bottom=5e-5, top=1.2)
        ax1.tick_params(axis="y", labelcolor=color)
        ax1.grid(True, which="both", linestyle="--", alpha=0.5)

        ax2 = ax1.twinx()
        color = "tab:green"
        ax2.set_ylabel("Packet Success Rate (%)", color=color, fontsize=12)
        ax2.plot(snrs, pkt_rates, marker="s", color=color, linestyle="--", linewidth=2, label="Packet Success")
        ax2.set_ylim(-5, 105)
        ax2.tick_params(axis="y", labelcolor=color)

        plt.title(f"PyFSK-Modem: BER vs SNR Performance ({config.mode.value.upper() if config else 'AUDIBLE'})")
        fig.tight_layout()
        plt.savefig(save_plot_path, dpi=150)
        plt.close(fig)

    return results
