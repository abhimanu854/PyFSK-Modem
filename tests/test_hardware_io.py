"""Tests for audio hardware abstraction, device enumeration, and error handling."""

from unittest.mock import MagicMock, patch
import numpy as np
import pytest
from pyfsk_modem.config import ModemConfig
from pyfsk_modem.receiver import Receiver, ReceiverError
from pyfsk_modem.transmitter import AudioDeviceError, Transmitter


def test_device_enumeration():
    """Verify input and output device queries return structured lists."""
    inputs = Receiver.get_devices()
    outputs = Transmitter.get_devices()

    assert isinstance(inputs, list)
    assert isinstance(outputs, list)

    for dev in inputs:
        assert "index" in dev
        assert "name" in dev
        assert dev["channels"] > 0

    for dev in outputs:
        assert "index" in dev
        assert "name" in dev
        assert dev["channels"] > 0


def test_transmitter_error_handling_when_sd_missing():
    """Verify Transmitter raises AudioDeviceError when sounddevice is not available."""
    with patch("pyfsk_modem.transmitter.sd", None):
        tx = Transmitter()
        with pytest.raises(AudioDeviceError, match="sounddevice"):
            tx.play(np.zeros(100, dtype=np.float32))


def test_receiver_error_handling_when_sd_missing():
    """Verify Receiver raises ReceiverError when sounddevice is not available."""
    with patch("pyfsk_modem.receiver.sd", None):
        rx = Receiver()
        with pytest.raises(ReceiverError, match="sounddevice"):
            rx.listen(duration_s=0.1)


def test_audio_callback_queue_enqueue():
    """Verify Receiver audio callback enqueues mono samples without blocking."""
    rx = Receiver()
    # 2 channels stereo mock chunk
    dummy_input = np.ones((512, 2), dtype=np.float32) * 0.42
    status_mock = MagicMock()
    status_mock.input_overflow = False

    rx._audio_callback(dummy_input, 512, None, status_mock)

    assert not rx.audio_queue.empty()
    queued = rx.audio_queue.get_nowait()
    assert len(queued) == 512
    assert np.allclose(queued, 0.42)
