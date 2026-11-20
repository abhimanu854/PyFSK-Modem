"""Acoustic BFSK audio receiver and real-time capture stream."""

from __future__ import annotations

import logging
import queue
import threading
import time
from typing import Any, Callable, Dict, List, Optional
import numpy as np

try:
    import sounddevice as sd
except ImportError:
    sd = None

from pyfsk_modem.config import ModemConfig
from pyfsk_modem.demodulator import Demodulator, PacketResult

logger = logging.getLogger(__name__)


class ReceiverError(Exception):
    """Raised when an audio capture or hardware error occurs."""
    pass


class Receiver:
    """Captures microphone audio stream and demodulates BFSK acoustic packets in real time."""

    def __init__(self, config: Optional[ModemConfig] = None) -> None:
        """Initialize receiver with modem configuration."""
        self.config = config or ModemConfig.create()
        self.demodulator = Demodulator(self.config)
        self.audio_queue: queue.Queue[np.ndarray] = queue.Queue(maxsize=100)
        self.visualizer_queue: Optional[queue.Queue[np.ndarray]] = None
        self._is_running = False

    @staticmethod
    def get_devices() -> List[Dict[str, Any]]:
        """List all available audio input devices (microphones)."""
        if sd is None:
            raise ReceiverError("sounddevice library is not available.")
        try:
            device_list = sd.query_devices()
            inputs = []
            for idx, dev in enumerate(device_list):
                if dev.get("max_input_channels", 0) > 0:
                    inputs.append({
                        "index": idx,
                        "name": dev["name"],
                        "channels": dev["max_input_channels"],
                        "default_samplerate": dev.get("default_samplerate", 44100),
                        "hostapi": dev.get("hostapi", 0),
                    })
            return inputs
        except Exception as exc:
            raise ReceiverError(f"Failed to query input devices: {exc}") from exc

    def _audio_callback(
        self,
        indata: np.ndarray,
        frames: int,
        time_info: Any,
        status: sd.CallbackFlags,
    ) -> None:
        """PortAudio capture callback running in audio thread."""
        if status:
            if status.input_overflow:
                logger.warning("Audio input buffer overflow detected.")

        # Extract mono channel
        chunk = indata[:, 0].copy()

        # Place in processing queue without blocking
        try:
            self.audio_queue.put_nowait(chunk)
        except queue.Full:
            try:
                self.audio_queue.get_nowait()  # Drop oldest frame to avoid lag
                self.audio_queue.put_nowait(chunk)
            except (queue.Empty, queue.Full):
                pass

        # Optional visualizer tap
        if self.visualizer_queue is not None:
            try:
                self.visualizer_queue.put_nowait(chunk)
            except queue.Full:
                pass

    def listen(
        self,
        device: Optional[int | str] = None,
        duration_s: Optional[float] = None,
        on_packet: Optional[Callable[[PacketResult], None]] = None,
        block_size: int = 1024,
    ) -> None:
        """Start microphone listening loop and decode packets in real time.

        Args:
            device: Audio input device index or substring.
            duration_s: Seconds to listen (None for indefinite until KeyboardInterrupt).
            on_packet: Optional callback invoked whenever a packet is decoded.
            block_size: Audio buffer chunk size in samples.
        """
        if sd is None:
            raise ReceiverError("sounddevice library is not available.")

        self._is_running = True
        start_time = time.time()

        # Validate or negotiate hardware sample rate
        sample_rate = self.config.sample_rate
        try:
            sd.check_input_settings(device=device, samplerate=sample_rate)
        except Exception as check_exc:
            dev_info = sd.query_devices(device=device, kind="input")
            hw_sr = int(dev_info.get("default_samplerate", 44100))
            if hw_sr != sample_rate:
                logger.warning(
                    f"Sample rate {sample_rate} Hz unsupported by audio hardware; "
                    f"auto-adapting to hardware default {hw_sr} Hz."
                )
                self.config = ModemConfig.create(
                    mode=self.config.mode,
                    baud_rate=self.config.baud_rate,
                    sample_rate=hw_sr,
                    squelch_threshold=self.config.squelch_threshold,
                )
                self.demodulator = Demodulator(self.config)
                sample_rate = hw_sr

        self.demodulator.reset()

        try:
            with sd.InputStream(
                samplerate=self.config.sample_rate,
                channels=1,
                dtype="float32",
                blocksize=block_size,
                device=device,
                callback=self._audio_callback,
            ):
                logger.info(
                    f"Listening for BFSK packets [{self.config.mode.value.upper()}] "
                    f"@ {self.config.baud_rate} baud (sample rate {self.config.sample_rate} Hz)..."
                )

                while self._is_running:
                    if duration_s is not None and (time.time() - start_time) >= duration_s:
                        break

                    try:
                        chunk = self.audio_queue.get(timeout=0.1)
                    except queue.Empty:
                        continue

                    # Feed audio chunk to state machine
                    packets = self.demodulator.feed_stream(chunk)
                    for pkt in packets:
                        print(pkt.format_log(), flush=True)
                        if on_packet:
                            on_packet(pkt)

        except KeyboardInterrupt:
            logger.info("Listening stopped by user.")
        except sd.PortAudioError as exc:
            raise ReceiverError(f"PortAudio microphone error: {exc}") from exc
        except Exception as exc:
            raise ReceiverError(f"Microphone listening error: {exc}") from exc
        finally:
            self._is_running = False

    def stop(self) -> None:
        """Stop listening loop."""
        self._is_running = False
