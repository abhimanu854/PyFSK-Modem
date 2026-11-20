"""BFSK Demodulator and frame decoder state machine for PyFSK-Modem."""

from __future__ import annotations

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
    """Compute the signal power at a target frequency using the Goertzel algorithm.

    Args:
        samples: 1D array of audio samples.
        target_freq: Target frequency in Hz.
        sample_rate: Audio sampling frequency in Hz.
        window: Optional window array (e.g. Hann) of same length as samples.

    Returns:
        Power (squared magnitude) at target frequency.
    """
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
    """Compute power at mark and space frequencies using windowed DFT correlators.

    Args:
        samples: 1D audio sample array of length N.
        freq_mark: Frequency for binary 1 (Hz).
        freq_space: Frequency for binary 0 (Hz).
        sample_rate: Sampling frequency (Hz).
        window: Optional window array (e.g. Hann).

    Returns:
        Tuple of (power_mark, power_space).
    """
    n = len(samples)
    if n == 0:
        return 0.0, 0.0

    t = np.arange(n)
    if window is not None:
        w_samples = samples * window
    else:
        w_samples = samples

    basis_m = np.exp(-2j * np.pi * freq_mark * t / sample_rate)
    basis_s = np.exp(-2j * np.pi * freq_space * t / sample_rate)

    dot_m = np.dot(w_samples, basis_m)
    dot_s = np.dot(w_samples, basis_s)

    power_m = float(np.abs(dot_m) ** 2)
    power_s = float(np.abs(dot_s) ** 2)

    return power_m, power_s


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

        # Precompute Hann window and bases
        self.window = np.hanning(self.samples_per_symbol).astype(np.float32)
        n = np.arange(self.samples_per_symbol)
        self._basis_m = (self.window * np.exp(-2j * np.pi * self.mark_freq * n / self.sample_rate)).astype(np.complex64)
        self._basis_s = (self.window * np.exp(-2j * np.pi * self.space_freq * n / self.sample_rate)).astype(np.complex64)

        # Internal audio stream buffer
        self._buffer: np.ndarray = np.array([], dtype=np.float32)

        # Noise floor tracking
        self._noise_floor = 1e-4

        # Frame assembly variables
        self._sync_register = 0
        self._sync_bits_collected = 0
        self._expected_length = 0
        self._payload_bits: List[int] = []
        self._length_bits: List[int] = []
        self._crc_bits: List[int] = []
        self._snr_measurements: List[float] = []

        # Symbol timing index
        self._symbol_sample_index = 0

    def reset(self) -> None:
        """Reset state machine to IDLE state."""
        self.state = RxState.IDLE
        self._sync_register = 0
        self._sync_bits_collected = 0
        self._expected_length = 0
        self._payload_bits.clear()
        self._length_bits.clear()
        self._crc_bits.clear()
        self._snr_measurements.clear()
        self._symbol_sample_index = 0

    def evaluate_symbol_power(self, chunk: np.ndarray) -> Tuple[float, float]:
        """Compute mark and space power for an audio chunk of symbol length."""
        if len(chunk) < self.samples_per_symbol:
            return 0.0, 0.0
        c = chunk[:self.samples_per_symbol]
        dot_m = np.dot(c, self._basis_m)
        dot_s = np.dot(c, self._basis_s)
        return float(np.abs(dot_m) ** 2), float(np.abs(dot_s) ** 2)

    def evaluate_bit(self, chunk: np.ndarray) -> Tuple[int, float]:
        """Determine bit value (0 or 1) and instantaneous SNR in dB for a symbol chunk."""
        em, es = self.evaluate_symbol_power(chunk)
        bit = 1 if em > es else 0
        sig_p = max(em, es)
        noise_p = min(em, es) + 1e-9
        snr_db = 10.0 * np.log10(sig_p / noise_p)
        return bit, snr_db

    def process_buffer(self, audio: np.ndarray) -> List[PacketResult]:
        """Demodulate a batch audio buffer and return all decoded packets.

        Args:
            audio: 1D array of float32 audio samples.

        Returns:
            List of PacketResult instances decoded from the audio.
        """
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

        while ptr <= total_len - min_packet_samples:
            # Squelch & preamble correlation check
            # Coarse search for preamble start
            em, es = self.evaluate_symbol_power(audio[ptr : ptr + Ns])
            in_band_energy = em + es

            # Check if energy is above squelch threshold
            if in_band_energy < self.config.squelch_threshold * self._noise_floor:
                # Update noise floor estimate when idle
                self._noise_floor = 0.99 * self._noise_floor + 0.01 * max(1e-6, in_band_energy)
                ptr += step
                continue

            # Check preamble pattern over preamble_len symbols
            preamble_score = 0.0
            for k in range(preamble_len):
                idx = ptr + k * Ns
                if idx + Ns > total_len:
                    break
                p_em, p_es = self.evaluate_symbol_power(audio[idx : idx + Ns])
                preamble_score += (p_em - p_es) * preamble_pat[k]

            # Threshold for preamble detection confidence
            if preamble_score > 0.3 * preamble_len * in_band_energy:
                # Preamble detected! Perform fine timing recovery in neighborhood
                fine_step = max(1, Ns // 16)
                fine_range = range(
                    max(0, ptr - Ns // 2),
                    min(total_len - min_packet_samples, ptr + Ns // 2 + 1),
                    fine_step,
                )
                best_offset = ptr
                best_score = preamble_score
                for cand in fine_range:
                    score = 0.0
                    for k in range(preamble_len):
                        p_em, p_es = self.evaluate_symbol_power(audio[cand + k * Ns : cand + (k + 1) * Ns])
                        score += (p_em - p_es) * preamble_pat[k]
                    if score > best_score:
                        best_score = score
                        best_offset = cand

                # Transition to SYNC: scan for sync word starting after preamble
                sync_found = False
                sync_sym_offset = best_offset + preamble_len * Ns
                sync_reg = 0

                # Search within a window of symbols for sync word
                search_limit = sync_sym_offset + 32 * Ns
                sym_ptr = sync_sym_offset

                while sym_ptr + Ns <= min(total_len, search_limit):
                    bit, _ = self.evaluate_bit(audio[sym_ptr : sym_ptr + Ns])
                    sync_reg = ((sync_reg << 1) & 0xFFFF) | bit
                    sym_ptr += Ns

                    # Check match with sync_word (0x2DD4)
                    diff = bin(sync_reg ^ self.config.sync_word).count("1")
                    if diff <= 1:  # Allow 1-bit tolerance for acoustic channel noise
                        sync_found = True
                        break

                if not sync_found:
                    # False alarm, advance pointer past this region
                    ptr += preamble_len * Ns
                    continue

                # SYNC acquired! Read LENGTH (8 bits)
                length_bits: List[int] = []
                snrs: List[float] = []
                for _ in range(8):
                    if sym_ptr + Ns > total_len:
                        break
                    bit, snr = self.evaluate_bit(audio[sym_ptr : sym_ptr + Ns])
                    length_bits.append(bit)
                    snrs.append(snr)
                    sym_ptr += Ns

                if len(length_bits) < 8:
                    ptr = sym_ptr
                    continue

                length = int(bits_to_bytes(length_bits)[0])
                if length > self.config.max_payload_bytes:
                    # Invalid length, discard
                    ptr = sym_ptr
                    continue

                # Read PAYLOAD (length * 8 bits)
                payload_bits: List[int] = []
                for _ in range(length * 8):
                    if sym_ptr + Ns > total_len:
                        break
                    bit, snr = self.evaluate_bit(audio[sym_ptr : sym_ptr + Ns])
                    payload_bits.append(bit)
                    snrs.append(snr)
                    sym_ptr += Ns

                if len(payload_bits) < length * 8:
                    ptr = sym_ptr
                    continue

                payload_bytes = bits_to_bytes(payload_bits)

                # Read CRC-16 (16 bits)
                crc_bits: List[int] = []
                for _ in range(16):
                    if sym_ptr + Ns > total_len:
                        break
                    bit, snr = self.evaluate_bit(audio[sym_ptr : sym_ptr + Ns])
                    crc_bits.append(bit)
                    snrs.append(snr)
                    sym_ptr += Ns

                if len(crc_bits) < 16:
                    ptr = sym_ptr
                    continue

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

                # Advance pointer to continue scanning for subsequent packets
                ptr = sym_ptr
            else:
                ptr += step

        return packets

    def feed_stream(self, chunk: np.ndarray) -> List[PacketResult]:
        """Feed real-time audio chunk from microphone stream and return any decoded packets.

        Args:
            chunk: 1D array of newly captured audio samples.

        Returns:
            List of newly decoded PacketResults (if any).
        """
        # Append incoming samples to stream buffer
        if len(self._buffer) == 0:
            self._buffer = chunk.astype(np.float32)
        else:
            self._buffer = np.concatenate([self._buffer, chunk.astype(np.float32)])

        # Keep buffer manageable (e.g. up to ~10 seconds of audio)
        max_buf = self.sample_rate * 10
        if len(self._buffer) > max_buf:
            self._buffer = self._buffer[-max_buf:]

        # Minimum required to decode a packet
        min_samples = (self.config.preamble_bits + 16 + 8 + 16) * self.samples_per_symbol
        if len(self._buffer) < min_samples:
            return []

        packets = self.process_buffer(self._buffer)
        if packets:
            # Discard processed samples to prevent re-decoding
            # Keep the trailing 1 second of audio in case next preamble started
            keep_len = min(len(self._buffer), self.sample_rate)
            self._buffer = self._buffer[-keep_len:]

        return packets
