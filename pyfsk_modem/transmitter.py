"""Acoustic BFSK audio transmitter using sounddevice."""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional
import numpy as np

try:
    import sounddevice as sd
except ImportError:
    sd = None  # Handled gracefully if sounddevice is unavailable

from pyfsk_modem.config import ModemConfig
from pyfsk_modem.modulator import Modulator

logger = logging.getLogger(__name__)


class AudioDeviceError(Exception):
    """Raised when an audio hardware or stream error occurs."""
    pass


class Transmitter:
    """Plays modulated acoustic BFSK audio packets through speakers."""

    def __init__(self, config: Optional[ModemConfig] = None) -> None:
        """Initialize transmitter with modem configuration."""
        self.config = config or ModemConfig.create()
        self.modulator = Modulator(self.config)

    @staticmethod
    def get_devices() -> List[Dict[str, Any]]:
        """List all available audio output devices."""
        if sd is None:
            raise AudioDeviceError("sounddevice library is not available.")
        try:
            device_list = sd.query_devices()
            outputs = []
            for idx, dev in enumerate(device_list):
                if dev.get("max_output_channels", 0) > 0:
                    outputs.append({
                        "index": idx,
                        "name": dev["name"],
                        "channels": dev["max_output_channels"],
                        "default_samplerate": dev.get("default_samplerate", 44100),
                        "hostapi": dev.get("hostapi", 0),
                    })
            return outputs
        except Exception as exc:
            raise AudioDeviceError(f"Failed to query output devices: {exc}") from exc

    def play(
        self,
        audio: np.ndarray,
        device: Optional[int | str] = None,
        blocking: bool = True,
        sample_rate: Optional[int] = None,
    ) -> None:
        """Play audio waveform through the specified or default output device.

        Args:
            audio: 1D array of float32 samples.
            device: Output device index or name substring (None for default).
            blocking: Whether to wait until playback finishes before returning.
            sample_rate: Audio sampling frequency in Hz (defaults to config.sample_rate).
        """
        if sd is None:
            raise AudioDeviceError("sounddevice library is not available.")

        sr = sample_rate or self.config.sample_rate
        try:
            sd.play(audio, samplerate=sr, device=device, blocking=blocking)
        except sd.PortAudioError as exc:
            raise AudioDeviceError(f"PortAudio playback error: {exc}") from exc
        except Exception as exc:
            raise AudioDeviceError(f"Audio playback error: {exc}") from exc

    def transmit(
        self,
        payload: str | bytes,
        device: Optional[int | str] = None,
        blocking: bool = True,
    ) -> np.ndarray:
        """Modulate text/bytes and transmit via speaker.

        Args:
            payload: ASCII text string or raw bytes to transmit.
            device: Output audio device.
            blocking: Block until audio finishes playing.

        Returns:
            The synthesized float32 audio waveform.
        """
        # Validate or adapt sample rate
        sample_rate = self.config.sample_rate
        try:
            sd.check_output_settings(device=device, samplerate=sample_rate)
        except Exception:
            dev_info = sd.query_devices(device=device, kind="output")
            hw_sr = int(dev_info.get("default_samplerate", 44100))
            if hw_sr != sample_rate:
                logger.warning(
                    f"Sample rate {sample_rate} Hz unsupported by output hardware; "
                    f"auto-adapting to hardware default {hw_sr} Hz."
                )
                self.config = ModemConfig.create(
                    mode=self.config.mode,
                    baud_rate=self.config.baud_rate,
                    sample_rate=hw_sr,
                    amplitude=self.config.amplitude,
                )
                self.modulator = Modulator(self.config)

        audio = self.modulator.modulate(payload)
        self.play(audio, device=device, blocking=blocking, sample_rate=self.config.sample_rate)
        return audio

    def save_wav(self, payload: str | bytes, wav_path: str) -> np.ndarray:
        """Modulate text/bytes and export to a standard WAV audio file.

        Args:
            payload: ASCII text string or raw bytes to transmit.
            wav_path: Target path for .wav file.

        Returns:
            The synthesized float32 audio waveform.
        """
        import wave

        audio = self.modulator.modulate(payload)
        # Convert float32 [-1.0, 1.0] to 16-bit PCM
        pcm16 = (np.clip(audio, -1.0, 1.0) * 32767).astype(np.int16)

        with wave.open(wav_path, "wb") as wf:
            wf.setnchannels(1)  # Mono
            wf.setsampwidth(2)  # 16-bit
            wf.setframerate(self.config.sample_rate)
            wf.writeframes(pcm16.tobytes())

        return audio
