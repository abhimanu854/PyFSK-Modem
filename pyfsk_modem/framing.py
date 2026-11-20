"""Framing, packet serialization/deserialization, and CRC-16-CCITT for PyFSK-Modem."""

from __future__ import annotations

from typing import List, Tuple
from pyfsk_modem.config import ModemConfig


class FramingError(Exception):
    """Base exception for framing errors."""
    pass


class InvalidPacketLengthError(FramingError):
    """Raised when the packet length field is invalid or does not match payload."""
    pass


class CRCMismatchError(FramingError):
    """Raised when the received CRC does not match the computed CRC."""
    pass


def compute_crc16_ccitt(data: bytes, poly: int = 0x1021, init: int = 0xFFFF) -> int:
    """Compute 16-bit CRC-CCITT (poly 0x1021, initial 0xFFFF) over data bytes.

    Args:
        data: Input bytes.
        poly: CRC polynomial (default: 0x1021).
        init: Initial shift register value (default: 0xFFFF).

    Returns:
        16-bit integer CRC checksum.
    """
    crc = init
    for byte in data:
        crc ^= (byte << 8)
        for _ in range(8):
            if crc & 0x8000:
                crc = ((crc << 1) ^ poly) & 0xFFFF
            else:
                crc = (crc << 1) & 0xFFFF
    return crc


def bytes_to_bits(data: bytes) -> List[int]:
    """Convert bytes to a list of bits (MSB first per byte)."""
    bits: List[int] = []
    for b in data:
        for i in range(7, -1, -1):
            bits.append((b >> i) & 1)
    return bits


def bits_to_bytes(bits: List[int]) -> bytes:
    """Convert a list of bits (multiple of 8, MSB first) to bytes.

    If the bit count is not a multiple of 8, trailing bits are discarded.
    """
    byte_count = len(bits) // 8
    byte_array = bytearray(byte_count)
    for i in range(byte_count):
        val = 0
        for bit_idx in range(8):
            val = (val << 1) | (bits[i * 8 + bit_idx] & 1)
        byte_array[i] = val
    return bytes(byte_array)


def build_packet(payload: bytes | str, config: ModemConfig | None = None) -> List[int]:
    """Build a complete framed binary bit sequence ready for modulation.

    Packet layout:
    [Preamble: 16 bits 1010...] [Sync Word: 16 bits 0x2DD4]
    [Length: 1 byte] [Payload: N bytes] [CRC-16-CCITT: 2 bytes]

    Args:
        payload: ASCII string or raw bytes (up to 255 bytes).
        config: ModemConfig instance (defaults to default config).

    Returns:
        List of integer bits (0 and 1).
    """
    if config is None:
        config = ModemConfig.create()

    if isinstance(payload, str):
        payload_bytes = payload.encode("ascii", errors="replace")
    else:
        payload_bytes = bytes(payload)

    if len(payload_bytes) > config.max_payload_bytes:
        raise InvalidPacketLengthError(
            f"Payload length ({len(payload_bytes)}) exceeds maximum ({config.max_payload_bytes})"
        )

    length_byte = bytes([len(payload_bytes)])
    data_for_crc = length_byte + payload_bytes
    crc = compute_crc16_ccitt(data_for_crc)
    crc_bytes = bytes([(crc >> 8) & 0xFF, crc & 0xFF])

    # Convert components to bit streams
    preamble_bits = config.preamble_bit_pattern
    sync_bits = config.sync_word_bits
    data_bits = bytes_to_bits(data_for_crc + crc_bytes)

    return preamble_bits + sync_bits + data_bits


def parse_packet_bytes(packet_body: bytes) -> Tuple[bytes, bool, str]:
    """Parse and validate packet body: [Length: 1B] [Payload: NB] [CRC-16: 2B].

    Args:
        packet_body: Bytes starting immediately after the sync word.

    Returns:
        Tuple of (payload_bytes, crc_valid, status_message).
    """
    if len(packet_body) < 3:
        return b"", False, f"Packet too short ({len(packet_body)} bytes, minimum 3)"

    length = packet_body[0]
    expected_total_len = 1 + length + 2
    if len(packet_body) < expected_total_len:
        return (
            b"",
            False,
            f"Incomplete packet: expected {expected_total_len} bytes, got {len(packet_body)}",
        )

    payload = packet_body[1 : 1 + length]
    received_crc = (packet_body[1 + length] << 8) | packet_body[2 + length]

    data_for_crc = bytes([length]) + payload
    calculated_crc = compute_crc16_ccitt(data_for_crc)

    if received_crc == calculated_crc:
        return payload, True, "CRC-16 OK"
    else:
        return (
            payload,
            False,
            f"CRC mismatch: expected 0x{calculated_crc:04X}, received 0x{received_crc:04X}",
        )
