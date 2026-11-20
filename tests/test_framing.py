"""Unit tests for packet framing and CRC-16-CCITT."""

import pytest
from pyfsk_modem.config import ModemConfig
from pyfsk_modem.framing import (
    CRCMismatchError,
    InvalidPacketLengthError,
    bits_to_bytes,
    build_packet,
    bytes_to_bits,
    compute_crc16_ccitt,
    parse_packet_bytes,
)


def test_crc16_ccitt_standard():
    """Verify CRC-16-CCITT computation against known test vectors."""
    # Standard CCITT with init=0xFFFF:
    # "123456789" -> 0x29B1
    test_data = b"123456789"
    crc = compute_crc16_ccitt(test_data)
    assert crc == 0x29B1, f"Expected 0x29B1, got 0x{crc:04X}"


def test_crc16_bit_flip_detection():
    """Verify that any single bit flip alters the CRC checksum."""
    data = b"Hello, PyFSK!"
    original_crc = compute_crc16_ccitt(data)

    for byte_idx in range(len(data)):
        for bit_idx in range(8):
            corrupted = bytearray(data)
            corrupted[byte_idx] ^= (1 << bit_idx)
            corrupted_crc = compute_crc16_ccitt(bytes(corrupted))
            assert corrupted_crc != original_crc, f"CRC collision on byte {byte_idx} bit {bit_idx}"


def test_bits_to_bytes_roundtrip():
    """Verify bit serialization and deserialization roundtrip."""
    original = b"Test binary conversion \x00\xFF\x55\xAA"
    bits = bytes_to_bits(original)
    recovered = bits_to_bytes(bits)
    assert recovered == original


def test_packet_structure():
    """Verify packet bit layout: preamble (16), sync (16), length (8), payload (8*N), crc (16)."""
    cfg = ModemConfig.create()
    payload = "AirGap"
    bits = build_packet(payload, cfg)

    expected_len_bits = 16 + 16 + 8 + (len(payload) * 8) + 16
    assert len(bits) == expected_len_bits

    # Preamble: alternating 1010101010101010
    preamble = bits[:16]
    assert preamble == [1, 0] * 8

    # Sync word: 0x2DD4 = 0010 1101 1101 0100
    sync_bits = bits[16:32]
    expected_sync = [(0x2DD4 >> (15 - i)) & 1 for i in range(16)]
    assert sync_bits == expected_sync

    # Length byte: len("AirGap") = 6
    length_byte = bits_to_bytes(bits[32:40])[0]
    assert length_byte == 6

    # Payload
    payload_recovered = bits_to_bytes(bits[40 : 40 + 6 * 8])
    assert payload_recovered == b"AirGap"


def test_parse_valid_packet():
    """Verify parse_packet_bytes accepts a valid length+payload+CRC sequence."""
    payload = b"Acoustic Modem 2026"
    length_byte = bytes([len(payload)])
    crc = compute_crc16_ccitt(length_byte + payload)
    crc_bytes = bytes([(crc >> 8) & 0xFF, crc & 0xFF])

    packet_body = length_byte + payload + crc_bytes
    parsed_payload, crc_ok, status = parse_packet_bytes(packet_body)

    assert crc_ok is True
    assert parsed_payload == payload
    assert "OK" in status


def test_parse_corrupted_crc_rejected():
    """Verify that corrupting any bit in the payload or CRC causes rejection."""
    payload = b"Super Secure Password"
    length_byte = bytes([len(payload)])
    crc = compute_crc16_ccitt(length_byte + payload)
    crc_bytes = bytes([(crc >> 8) & 0xFF, crc & 0xFF])

    packet_body = bytearray(length_byte + payload + crc_bytes)
    # Corrupt one byte in payload
    packet_body[5] ^= 0x01

    parsed_payload, crc_ok, status = parse_packet_bytes(bytes(packet_body))
    assert crc_ok is False
    assert "mismatch" in status.lower()


def test_max_payload_exceeded():
    """Verify error raised when payload exceeds max_payload_bytes."""
    cfg = ModemConfig.create(max_payload_bytes=255)
    too_long = "X" * 256
    with pytest.raises(InvalidPacketLengthError):
        build_packet(too_long, cfg)
