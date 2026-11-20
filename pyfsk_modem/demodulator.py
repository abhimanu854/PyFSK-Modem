"""BFSK Demodulator and frame decoder state machine for PyFSK-Modem."""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import List, Optional, Tuple
import numpy as np

from pyfsk_modem.config import ModemConfig
from pyfsk_modem.framing import (
    bits_to_bytes,
    compute_crc16_ccitt,
    parse_packet_bytes,
)

logger = logging.getLogger(__name__)


class RxState(Enum):
    """Receiver state machine states."""
    IDLE = auto()
    PREAMBLE_DETECT = auto()
    SYNC = auto()
    LENGTH = auto()
    PAYLOAD = auto()
    CRC_CHECK = auto()


@dataclass
class PacketResult:
    """Result of a decoded packet."""
    payload: bytes
    text: str
    crc_ok: bool
    expected_crc: int
    received_crc: int
    snr_db: float
    timestamp: float = field(default_factory=time.time)
    status: str = ""
    bit_count: int = 0

    def format_log(self) -> str:
        """Format packet result for human-readable console logging."""
        t_str = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(self.timestamp))
        status_tag = "[CRC OK]" if self.crc_ok else "[CRC FAIL]"
        return (
            f"[{t_str}] {status_tag} SNR: {self.snr_db:5.1f} dB | "
            f"Len: {len(self.payload)} B | Msg: '{self.text}' ({self.status})"
        )


def goertzel_power(
    samples: np.ndarray,
    target_freq: float,
    sample_rate: int,
    window: Optional[np.ndarray] = None,
) -> float:
    """Compute the signal power at a target frequency using the Goertzel algorithm."""
    n = len(samples)
    if n == 0:
        return 0.0

    if window is not None:
        samples = samples * window

    k = int(round(n * target_freq / sample_rate))
    omega = (2.0 * np.pi * k) / n
    coeff = 2.0 * np.cos(omega)

    s_prev = 0.0
    s_prev2 = 0.0

    for sample in samples:
        s = float(sample) + coeff * s_prev - s_prev2
        s_prev2 = s_prev
        s_prev = s

    power = s_prev * s_prev + s_prev2 * s_prev2 - coeff * s_prev * s_prev2
    return max(0.0, float(power))


def dft_power_pair(
    samples: np.ndarray,
    freq_mark: float,
    freq_space: float,
    sample_rate: int,
    window: Optional[np.ndarray] = None,
) -> Tuple[float, float]:
    """Compute power at mark and space frequencies using windowed DFT correlators."""
    n = len(samples)
    if n == 0:
        return 0.0, 0.0

    t = np.arange(n)
    w_samples = samples * window if window is not None else samples

    basis_m = np.exp(-2j * np.pi * freq_mark * t / sample_rate)
    basis_s = np.exp(-2j * np.pi * freq_space * t / sample_rate)

    dot_m = np.dot(w_samples, basis_m)
    dot_s = np.dot(w_samples, basis_s)

    return float(np.abs(dot_m) ** 2), float(np.abs(dot_s) ** 2)


class Demodulator:
    """BFSK acoustic demodulator and state machine decoder.

    Detects acoustic packet preambles, tracks symbol timing, acquires frame
    synchronization, unpacks length and payload bytes, and validates CRC-16 checksums.
    """

    def __init__(self, config: ModemConfig | None = None) -> None:
        """Initialize demodulator with modem configuration."""
        self.config = config or ModemConfig.create()
        self.state = RxState.IDLE

        self.samples_per_symbol = self.config.samples_per_symbol
        self.sample_rate = self.config.sample_rate
        self.mark_freq = self.config.mark_freq
        self.space_freq = self.config.space_freq

        # Full-symbol window
        self.window = np.hanning(self.samples_per_symbol).astype(np.float32)
        n = np.arange(self.samples_per_symbol)
        self._basis_m = (self.window * np.exp(-2j * np.pi * self.mark_freq * n / self.sample_rate)).astype(np.complex64)
        self._basis_s = (self.window * np.exp(-2j * np.pi * self.space_freq * n / self.sample_rate)).astype(np.complex64)

        # Centered sub-symbol window (75% of symbol length = integer cycles of carrier delta)
        self.window_len = max(8, int(round(self.samples_per_symbol * 0.75)))
        self._sub_window = np.hanning(self.window_len).astype(np.float32)
        nw = np.arange(self.window_len)
        self._sub_basis_m = (self._sub_window * np.exp(-2j * np.pi * self.mark_freq * nw / self.sample_rate)).astype(np.complex64)
        self._sub_basis_s = (self._sub_window * np.exp(-2j * np.pi * self.space_freq * nw / self.sample_rate)).astype(np.complex64)
        self._center_offset = max(0, (self.samples_per_symbol - self.window_len) // 2)

        # Internal audio stream buffer
        self._buffer: np.ndarray = np.array([], dtype=np.float32)

        # Noise floor tracking
        self._noise_floor = 1e-4
        self._last_carrier_log = 0.0

    def reset(self) -> None:
        """Reset state machine to IDLE state."""
        self.state = RxState.IDLE
        self._buffer = np.array([], dtype=np.float32)
        self._noise_floor = 1e-4

    def evaluate_symbol_power(self, chunk: np.ndarray) -> Tuple[float, float]:
        """Compute mark and space power for an audio chunk of symbol length."""
        if len(chunk) < self.samples_per_symbol:
            return 0.0, 0.0
        c = chunk[:self.samples_per_symbol]
        dot_m = np.dot(c, self._basis_m)
        dot_s = np.dot(c, self._basis_s)
        return float(np.abs(dot_m) ** 2), float(np.abs(dot_s) ** 2)

    def evaluate_bit_centered(
        self,
        audio: np.ndarray,
        sym_start_idx: int,
        calib_m: float = 1.0,
        calib_s: float = 1.0,
    ) -> Tuple[int, float]:
        """Evaluate a symbol using the centered window with channel calibration.

        Returns:
            Tuple of (bit_val, snr_db).
        """
        idx = sym_start_idx + self._center_offset
        if idx + self.window_len > len(audio) or idx < 0:
            return 0, 0.0

        c = audio[idx : idx + self.window_len]
        dot_m = np.dot(c, self._sub_basis_m)
        dot_s = np.dot(c, self._sub_basis_s)

        em = float(np.abs(dot_m) ** 2) / max(1e-9, calib_m)
        es = float(np.abs(dot_s) ** 2) / max(1e-9, calib_s)

        bit = 1 if em > es else 0
        sig_p = max(em, es)
        noise_p = min(em, es) + 1e-9
        snr_db = 10.0 * np.log10(sig_p / noise_p)
        return bit, snr_db

    def process_buffer(self, audio: np.ndarray) -> List[PacketResult]:
        """Demodulate a batch audio buffer and return all decoded packets."""
        packets: List[PacketResult] = []
        total_len = len(audio)
        Ns = self.samples_per_symbol
        preamble_len = self.config.preamble_bits
        preamble_pat = np.array([1 if i % 2 == 0 else -1 for i in range(preamble_len)])

        # Minimum required samples for preamble + sync (16) + length (8) + crc (16)
        min_packet_symbols = preamble_len + 16 + 8 + 16
        min_packet_samples = min_packet_symbols * Ns

        if total_len < min_packet_samples:
            return packets

        ptr = 0
        step = max(1, Ns // 4)
        sync_target_bits = self.config.sync_word_bits

        while ptr <= total_len - min_packet_samples:
            # Measure in-band energy
            em, es = self.evaluate_symbol_power(audio[ptr : ptr + Ns])
            in_band_energy = em + es

            # Squelch check
            if in_band_energy < self.config.squelch_threshold * self._noise_floor:
                self._noise_floor = 0.99 * self._noise_floor + 0.01 * max(1e-6, in_band_energy)
                ptr += step
                continue

            # Update noise floor slowly
            self._noise_floor = 0.995 * self._noise_floor + 0.005 * max(1e-6, in_band_energy)

            # Check preamble alternating pattern
            preamble_score = 0.0
            for k in range(preamble_len):
                idx = ptr + k * Ns
                if idx + Ns > total_len:
                    break
                p_em, p_es = self.evaluate_symbol_power(audio[idx : idx + Ns])
                preamble_score += (p_em - p_es) * preamble_pat[k]

            # Trigger condition: contrast score is positive and significant
            if preamble_score > 0.15 * preamble_len * in_band_energy:
                now = time.time()
                if now - self._last_carrier_log > 1.5:
                    print("  [CARRIER] Detected acoustic signal burst, searching for sync...", flush=True)
                    self._last_carrier_log = now

                # Search for the 16-bit sync word (0x2DD4) across sub-symbol offsets
                # Sync word starts immediately after the 16 preamble bits
                sync_nominal = ptr + preamble_len * Ns
                search_window_start = max(0, sync_nominal - Ns)
                search_window_end = min(total_len - 32 * Ns, sync_nominal + 2 * Ns)
                fine_step = max(1, Ns // 16)

                best_sync_score = -1
                best_sync_pos = None

                for cand in range(search_window_start, search_window_end, fine_step):
                    matches = 0
                    for k in range(16):
                        bit, _ = self.evaluate_bit_centered(audio, cand + k * Ns)
                        if bit == sync_target_bits[k]:
                            matches += 1
                    if matches > best_sync_score:
                        best_sync_score = matches
                        best_sync_pos = cand

                # If sync score matches with at most 2 bit errors (out of 16):
                if best_sync_score >= 14 and best_sync_pos is not None:
                    # Perform Channel Equalization across the sync word
                    m_powers = []
                    s_powers = []
                    for k in range(16):
                        idx = best_sync_pos + k * Ns + self._center_offset
                        if idx + self.window_len <= total_len:
                            c = audio[idx : idx + self.window_len]
                            dot_m = np.dot(c, self._sub_basis_m)
                            dot_s = np.dot(c, self._sub_basis_s)
                            if sync_target_bits[k] == 1:
                                m_powers.append(float(np.abs(dot_m) ** 2))
                            else:
                                s_powers.append(float(np.abs(dot_s) ** 2))

                    calib_m = max(1e-6, float(np.mean(m_powers))) if m_powers else 1.0
                    calib_s = max(1e-6, float(np.mean(s_powers))) if s_powers else 1.0

                    # Re-align sync offset using calibrated channel gains
                    fine_sub_step = max(1, Ns // 32)
                    refined_pos = best_sync_pos
                    refined_score = -1
                    for cand in range(max(0, best_sync_pos - Ns // 4), min(total_len - 32 * Ns, best_sync_pos + Ns // 4), fine_sub_step):
                        matches = 0
                        for k in range(16):
                            b, _ = self.evaluate_bit_centered(audio, cand + k * Ns, calib_m, calib_s)
                            if b == sync_target_bits[k]:
                                matches += 1
                        if matches > refined_score:
                            refined_score = matches
                            refined_pos = cand

                    # SYNC acquired! Decode LENGTH (8 bits)
                    data_start = refined_pos + 16 * Ns
                    length_bits = []
                    snrs = []
                    for k in range(8):
                        bit, snr = self.evaluate_bit_centered(audio, data_start + k * Ns, calib_m, calib_s)
                        length_bits.append(bit)
                        snrs.append(snr)

                    length = int(bits_to_bytes(length_bits)[0])
                    if length == 0 or length > self.config.max_payload_bytes:
                        # Invalid length byte, advance pointer past this burst
                        ptr = data_start + 8 * Ns
                        continue

                    # Ensure the entire payload and CRC have arrived in buffer
                    expected_end_sample = data_start + (8 + length * 8 + 16) * Ns
                    if expected_end_sample > total_len:
                        # Full packet has not arrived yet; wait for more stream chunks
                        break

                    # Read PAYLOAD (length * 8 bits)
                    payload_start = data_start + 8 * Ns
                    payload_bits = []
                    for k in range(length * 8):
                        bit, snr = self.evaluate_bit_centered(audio, payload_start + k * Ns, calib_m, calib_s)
                        payload_bits.append(bit)
                        snrs.append(snr)

                    payload_bytes = bits_to_bytes(payload_bits)

                    # Read CRC-16 (16 bits)
                    crc_start = payload_start + (length * 8) * Ns
                    crc_bits = []
                    for k in range(16):
                        bit, snr = self.evaluate_bit_centered(audio, crc_start + k * Ns, calib_m, calib_s)
                        crc_bits.append(bit)
                        snrs.append(snr)

                    received_crc = int.from_bytes(bits_to_bytes(crc_bits), byteorder="big")
                    expected_crc = compute_crc16_ccitt(bytes([length]) + payload_bytes)
                    crc_ok = (received_crc == expected_crc)
                    avg_snr = float(np.mean(snrs)) if snrs else 0.0

                    text = payload_bytes.decode("ascii", errors="replace")
                    status = "CRC-16 OK" if crc_ok else f"CRC mismatch: exp 0x{expected_crc:04X}, rx 0x{received_crc:04X}"

                    packets.append(
                        PacketResult(
                            payload=payload_bytes,
                            text=text,
                            crc_ok=crc_ok,
                            expected_crc=expected_crc,
                            received_crc=received_crc,
                            snr_db=avg_snr,
                            status=status,
                            bit_count=len(length_bits) + len(payload_bits) + len(crc_bits),
                        )
                    )

                    # Advance pointer past this packet
                    ptr = crc_start + 16 * Ns
                else:
                    ptr += step
            else:
                ptr += step

        return packets

    def feed_stream(self, chunk: np.ndarray) -> List[PacketResult]:
        """Feed real-time audio chunk from microphone stream and return any decoded packets."""
        if len(self._buffer) == 0:
            self._buffer = chunk.astype(np.float32)
        else:
            self._buffer = np.concatenate([self._buffer, chunk.astype(np.float32)])

        max_buf = self.sample_rate * 8
        if len(self._buffer) > max_buf:
            self._buffer = self._buffer[-max_buf:]

        min_samples = (self.config.preamble_bits + 16 + 8 + 16) * self.samples_per_symbol
        if len(self._buffer) < min_samples:
            return []

        packets = self.process_buffer(self._buffer)
        if packets:
            keep_len = min(len(self._buffer), self.sample_rate // 2)
            self._buffer = self._buffer[-keep_len:]

        return packets
