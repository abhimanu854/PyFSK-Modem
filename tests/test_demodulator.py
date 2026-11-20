"""Unit tests for BFSK Demodulator and state machine."""

import numpy as np
import pytest
from pyfsk_modem.config import ModemConfig, ModemMode
from pyfsk_modem.demodulator import Demodulator
from pyfsk_modem.modulator import Modulator


@pytest.mark.parametrize("mode", [ModemMode.AUDIBLE, ModemMode.ULTRASONIC])
def test_clean_roundtrip(mode: ModemMode):
    """Verify clean signal demodulation roundtrip in both audible and ultrasonic modes."""
    cfg = ModemConfig.create(mode=mode, baud_rate=100)
    mod = Modulator(cfg)
    demod = Demodulator(cfg)

    message = f"Hello PyFSK in {mode.value} mode!"
    audio = mod.modulate(message)

    packets = demod.process_buffer(audio)
    assert len(packets) == 1
    pkt = packets[0]
    assert pkt.crc_ok is True
    assert pkt.text == message
    assert pkt.snr_db > 20.0


def test_roundtrip_with_offset_and_noise():
    """Verify demodulator recovers symbol timing with arbitrary delay and background noise."""
    cfg = ModemConfig.create(mode=ModemMode.AUDIBLE, baud_rate=100)
    mod = Modulator(cfg)
    demod = Demodulator(cfg)

    message = "Acoustic Air-Gap 42"
    audio = mod.modulate(message)

    np.random.seed(12345)
    noise_in = np.random.normal(0, 0.03, 3147).astype(np.float32)
    noise_out = np.random.normal(0, 0.03, 2000).astype(np.float32)
    rx_audio = np.concatenate([noise_in, audio, noise_out])

    packets = demod.process_buffer(rx_audio)
    assert len(packets) == 1
    pkt = packets[0]
    assert pkt.crc_ok is True
    assert pkt.text == message


def test_squelch_ambient_noise_rejection():
    """Verify pure ambient noise does not trigger false packet detections."""
    cfg = ModemConfig.create(mode=ModemMode.AUDIBLE, baud_rate=100)
    demod = Demodulator(cfg)

    np.random.seed(42)
    # 2 seconds of pure Gaussian white noise
    noise = np.random.normal(0, 0.05, cfg.sample_rate * 2).astype(np.float32)

    packets = demod.process_buffer(noise)
    assert len(packets) == 0, f"Expected 0 packets from pure noise, got {len(packets)}"


def test_corrupted_packet_rejected():
    """Verify that if payload bits are damaged, the demodulator reports CRC failure."""
    cfg = ModemConfig.create(mode=ModemMode.AUDIBLE, baud_rate=100)
    mod = Modulator(cfg)
    demod = Demodulator(cfg)

    message = "Original Secret"
    audio = mod.modulate(message)

    # Overwrite samples in the payload region with noise to corrupt bits
    # Preamble (16) + sync (16) + length (8) = 40 symbols
    # Payload starts around symbol 40.
    Ns = cfg.samples_per_symbol
    corrupt_start = cfg.guard_samples + 42 * Ns
    corrupt_end = corrupt_start + 4 * Ns
    audio[corrupt_start:corrupt_end] = np.random.normal(0, 1.5, corrupt_end - corrupt_start).astype(np.float32)

    packets = demod.process_buffer(audio)
    assert len(packets) == 1
    pkt = packets[0]
    assert pkt.crc_ok is False
    assert "mismatch" in pkt.status.lower()


def test_streaming_chunk_feed():
    """Verify real-time chunk streaming with feed_stream."""
    cfg = ModemConfig.create(mode=ModemMode.AUDIBLE, baud_rate=100)
    mod = Modulator(cfg)
    demod = Demodulator(cfg)

    message = "Stream Chunk 123"
    audio = mod.modulate(message)

    # Add random leading delay
    stream = np.concatenate([np.zeros(2000, dtype=np.float32), audio, np.zeros(2000, dtype=np.float32)])

    chunk_size = 512
    decoded_packets = []
    for i in range(0, len(stream), chunk_size):
        chunk = stream[i : i + chunk_size]
        new_pkts = demod.feed_stream(chunk)
        if new_pkts:
            decoded_packets.extend(new_pkts)

    assert len(decoded_packets) == 1
    assert decoded_packets[0].crc_ok is True
    assert decoded_packets[0].text == message
