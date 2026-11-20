"""Command Line Interface (CLI) for PyFSK-Modem."""

from __future__ import annotations

import argparse
import logging
import queue
import sys
import threading
from typing import List, Optional

from pyfsk_modem.config import ModemConfig, ModemMode
from pyfsk_modem.loopback import run_loopback, sweep_snr_ber
from pyfsk_modem.receiver import Receiver, ReceiverError
from pyfsk_modem.transmitter import AudioDeviceError, Transmitter
from pyfsk_modem.visualizer import SpectrumWaterfallVisualizer
from pyfsk_modem.web_app import start_web_transmitter


def setup_logging(verbose: bool = False) -> None:
    """Configure console logging level and format."""
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format="%(asctime)s [%(levelname)s] %(message)s",
        datefmt="%H:%M:%S",
    )


def cmd_devices(args: argparse.Namespace) -> int:
    """List audio input and output devices."""
    print("=" * 65)
    print("PyFSK-Modem: Available Audio Hardware Devices")
    print("=" * 65)

    try:
        inputs = Receiver.get_devices()
        print("\n--- Audio Input Devices (Microphones) ---")
        if not inputs:
            print("  (No microphone devices detected)")
        for dev in inputs:
            print(f"  [{dev['index']:2d}] {dev['name']} (Channels: {dev['channels']}, Default SR: {dev['default_samplerate']} Hz)")
    except ReceiverError as exc:
        print(f"Error querying input devices: {exc}")

    try:
        outputs = Transmitter.get_devices()
        print("\n--- Audio Output Devices (Speakers) ---")
        if not outputs:
            print("  (No speaker devices detected)")
        for dev in outputs:
            print(f"  [{dev['index']:2d}] {dev['name']} (Channels: {dev['channels']}, Default SR: {dev['default_samplerate']} Hz)")
    except AudioDeviceError as exc:
        print(f"Error querying output devices: {exc}")

    print("=" * 65)
    return 0


def cmd_send(args: argparse.Namespace) -> int:
    """Transmit text message as BFSK audio packet over speaker."""
    setup_logging(args.verbose)
    mode = ModemMode(args.mode.lower())

    cfg = ModemConfig.create(
        mode=mode,
        baud_rate=args.baud,
        sample_rate=args.sample_rate,
        amplitude=args.amplitude,
    )

    print(f"\n[TX] Preparing acoustic transmission:")
    print(f"     Payload:    '{args.message}' ({len(args.message)} bytes)")
    print(f"     Mode:       {cfg.mode.value.upper()} (Mark: {cfg.mark_freq} Hz, Space: {cfg.space_freq} Hz)")
    print(f"     Baud Rate:  {cfg.baud_rate} symbols/s ({cfg.samples_per_symbol} samples/symbol)")
    print(f"     Sample Rate:{cfg.sample_rate} Hz")

    tx = Transmitter(cfg)
    try:
        if args.wav:
            print(f"[TX] Exporting modulated waveform to WAV file: '{args.wav}'...")
            tx.save_wav(args.message, args.wav)
            print(f"[TX] WAV file saved successfully: '{args.wav}'! You can play this audio on your phone.\n")
            return 0

        print("[TX] Transmitting audio over air gap...")
        tx.transmit(args.message, device=args.device, blocking=True)
        print("[TX] Transmission complete!\n")
        return 0
    except AudioDeviceError as exc:
        print(f"[ERROR] Audio playback failed: {exc}", file=sys.stderr)
        return 1
    except Exception as exc:
        print(f"[ERROR] Failed to save WAV file: {exc}", file=sys.stderr)
        return 1


def cmd_web(args: argparse.Namespace) -> int:
    """Launch local web server for mobile phone acoustic transmission."""
    start_web_transmitter(port=args.port)
    return 0


def cmd_listen(args: argparse.Namespace) -> int:
    """Listen on microphone and demodulate incoming BFSK packets in real time."""
    setup_logging(args.verbose)
    mode = ModemMode(args.mode.lower())

    cfg = ModemConfig.create(
        mode=mode,
        baud_rate=args.baud,
        sample_rate=args.sample_rate,
        squelch_threshold=args.squelch,
    )

    print(f"\n[RX] PyFSK Acoustic Receiver Active")
    print(f"     Mode:       {cfg.mode.value.upper()} (Mark: {cfg.mark_freq} Hz, Space: {cfg.space_freq} Hz)")
    print(f"     Baud Rate:  {cfg.baud_rate} symbols/s ({cfg.samples_per_symbol} samples/symbol)")
    print(f"     Squelch:    {cfg.squelch_threshold:.1f}x noise floor")
    print("     Press Ctrl+C to stop listening.\n")

    rx = Receiver(cfg)

    # Handle optional live visualizer
    if args.visualize:
        vis = SpectrumWaterfallVisualizer(
            sample_rate=cfg.sample_rate,
            mark_freq=cfg.mark_freq,
            space_freq=cfg.space_freq,
        )
        if vis.is_display_available():
            vis_queue: queue.Queue = queue.Queue(maxsize=50)
            rx.visualizer_queue = vis_queue
            # Run receiver in background thread while GUI runs in main thread
            rx_thread = threading.Thread(
                target=rx.listen,
                kwargs={"device": args.device, "duration_s": args.duration},
                daemon=True,
            )
            rx_thread.start()
            try:
                vis.run(vis_queue)
            except KeyboardInterrupt:
                pass
            finally:
                rx.stop()
            return 0
        else:
            print("[WARN] Graphical display unavailable ($DISPLAY not set); continuing in CLI mode.")

    try:
        rx.listen(device=args.device, duration_s=args.duration)
        return 0
    except ReceiverError as exc:
        print(f"[ERROR] Microphone capture failed: {exc}", file=sys.stderr)
        return 1


def cmd_loopback(args: argparse.Namespace) -> int:
    """Run in-memory AWGN channel simulation without audio hardware."""
    setup_logging(args.verbose)
    mode = ModemMode(args.mode.lower())

    cfg = ModemConfig.create(
        mode=mode,
        baud_rate=args.baud,
        sample_rate=args.sample_rate,
        amplitude=args.amplitude,
    )

    if args.sweep:
        print(f"\n[LOOPBACK] Running Multi-SNR Performance Sweep ({cfg.mode.value.upper()} @ {cfg.baud_rate} baud)")
        snr_points = [25.0, 20.0, 15.0, 10.0, 5.0, 0.0]
        results = sweep_snr_ber(
            snr_range=snr_points,
            payload=args.message,
            trials_per_snr=args.trials,
            save_plot_path=args.plot_file,
            config=cfg,
        )
        print("-" * 65)
        print(f"{'Target SNR':>12} | {'Mean BER':>10} | {'Packet Success':>14} | {'Detection':>10}")
        print("-" * 65)
        for s in snr_points:
            res = results[s]
            print(
                f"{s:10.1f} dB | {res['mean_ber'] * 100:9.2f}% | "
                f"{res['packet_success_rate'] * 100:13.1f}% | {res['detection_rate'] * 100:9.1f}%"
            )
        print("-" * 65)
        if args.plot_file:
            print(f"[LOOPBACK] BER vs SNR plot saved to: {args.plot_file}\n")
        return 0

    print(f"\n[LOOPBACK] In-Memory Simulation @ Target SNR: {args.snr:.1f} dB")
    print(f"           Mode: {cfg.mode.value.upper()} | Baud: {cfg.baud_rate}")
    print(f"           TX Payload: '{args.message}'\n")

    res = run_loopback(payload=args.message, target_snr_db=args.snr, config=cfg)
    print(f"[LOOPBACK RESULT]")
    print(f"  {res.format_summary()}")

    if res.crc_ok:
        print("  Status: SUCCESS (CRC-16 Verified, payload decoded perfectly)\n")
        return 0
    else:
        print(f"  Status: CRC ERROR or Undetected ({res.status})\n")
        return 1 if res.packet_detected else 2


def build_parser() -> argparse.ArgumentParser:
    """Build command line argument parser."""
    parser = argparse.ArgumentParser(
        prog="pyfsk_modem",
        description="PyFSK-Modem: Software-Defined Acoustic Air-Gap Modem",
    )
    subparsers = parser.add_subparsers(dest="command", help="Available subcommands")

    # devices
    subparsers.add_parser("devices", help="List available audio input and output devices")

    # send
    p_send = subparsers.add_parser("send", help="Transmit text message over speaker")
    p_send.add_argument("message", help="ASCII text message to transmit")
    p_send.add_argument("--mode", choices=["audible", "ultrasonic"], default="audible", help="Acoustic band mode")
    p_send.add_argument("--baud", type=int, default=100, help="Transmission baud rate (default: 100)")
    p_send.add_argument("--device", help="Output audio device index or substring")
    p_send.add_argument("--amplitude", type=float, default=0.8, help="Output peak amplitude 0.0-1.0")
    p_send.add_argument("--sample-rate", type=int, default=44100, help="Sample rate in Hz (default: 44100)")
    p_send.add_argument("--wav", help="Export to a .wav audio file instead of playing immediately (play on phone)")
    p_send.add_argument("-v", "--verbose", action="store_true", help="Verbose debug logging")

    # listen
    p_listen = subparsers.add_parser("listen", help="Listen for acoustic packets on microphone")
    p_listen.add_argument("--mode", choices=["audible", "ultrasonic"], default="audible", help="Acoustic band mode")
    p_listen.add_argument("--baud", type=int, default=100, help="Transmission baud rate (default: 100)")
    p_listen.add_argument("--device", help="Input audio device index or substring")
    p_listen.add_argument("--sample-rate", type=int, default=44100, help="Sample rate in Hz (default: 44100)")
    p_listen.add_argument("--squelch", type=float, default=2.0, help="Squelch energy multiplier (default: 2.0)")
    p_listen.add_argument("--duration", type=float, default=None, help="Listening duration in seconds")
    p_listen.add_argument("--visualize", action="store_true", help="Show live FFT spectrum and waterfall display")
    p_listen.add_argument("-v", "--verbose", action="store_true", help="Verbose debug logging")

    # loopback
    p_loop = subparsers.add_parser("loopback", help="Run in-memory simulation with AWGN noise")
    p_loop.add_argument("message", nargs="?", default="Hello, PyFSK Air-Gap Modem!", help="Text message to transmit")
    p_loop.add_argument("--snr", type=float, default=15.0, help="Target channel SNR in dB (default: 15.0)")
    p_loop.add_argument("--mode", choices=["audible", "ultrasonic"], default="audible", help="Acoustic band mode")
    p_loop.add_argument("--baud", type=int, default=100, help="Transmission baud rate (default: 100)")
    p_loop.add_argument("--sample-rate", type=int, default=44100, help="Sample rate in Hz")
    p_loop.add_argument("--amplitude", type=float, default=0.8, help="Peak signal amplitude")
    p_loop.add_argument("--sweep", action="store_true", help="Run multi-SNR sweep and print BER table")
    p_loop.add_argument("--trials", type=int, default=5, help="Number of trials per SNR in sweep")
    p_loop.add_argument("--plot-file", default="ber_vs_snr.png", help="Path to save BER plot (default: ber_vs_snr.png)")
    p_loop.add_argument("-v", "--verbose", action="store_true", help="Verbose debug logging")

    # web (mobile phone transmitter)
    p_web = subparsers.add_parser("web", help="Launch mobile phone Web Audio transmitter interface")
    p_web.add_argument("--port", type=int, default=8080, help="Port to host web transmitter on (default: 8080)")

    return parser


def main(args: Optional[List[str]] = None) -> int:
    """Main CLI entry point."""
    parser = build_parser()
    parsed_args = parser.parse_args(args)

    if parsed_args.command == "devices":
        return cmd_devices(parsed_args)
    elif parsed_args.command == "send":
        return cmd_send(parsed_args)
    elif parsed_args.command == "listen":
        return cmd_listen(parsed_args)
    elif parsed_args.command == "loopback":
        return cmd_loopback(parsed_args)
    elif parsed_args.command == "web":
        return cmd_web(parsed_args)
    else:
        parser.print_help()
        return 0


if __name__ == "__main__":
    sys.exit(main())
