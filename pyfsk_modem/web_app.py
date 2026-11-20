"""Mobile Web Audio Transmitter for PyFSK-Modem.

Serves an HTML5 Web Audio transmitter interface over the local network so any
smartphone (iPhone / Android) can transmit acoustic BFSK packets over its speaker.
"""

from __future__ import annotations

import http.server
import logging
import socket
import sys
from typing import Optional

logger = logging.getLogger(__name__)

HTML_PAGE = """<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
  <title>PyFSK Mobile Transmitter</title>
  <style>
    * { box-sizing: border-box; margin: 0; padding: 0; font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif; }
    body { background: #0f111a; color: #e6edf3; display: flex; flex-direction: column; align-items: center; justify-content: center; min-height: 100vh; padding: 20px; }
    .card { background: #161b22; border: 1px solid #30363d; border-radius: 16px; padding: 28px; width: 100%; max-width: 440px; box-shadow: 0 12px 32px rgba(0,0,0,0.4); }
    h1 { font-size: 1.5rem; text-align: center; color: #58a6ff; margin-bottom: 6px; }
    p.sub { font-size: 0.85rem; text-align: center; color: #8b949e; margin-bottom: 24px; }
    label { display: block; font-size: 0.82rem; font-weight: 600; text-transform: uppercase; letter-spacing: 0.5px; color: #8b949e; margin-bottom: 6px; }
    input[type="text"], select { width: 100%; padding: 12px 14px; background: #0d1117; border: 1px solid #30363d; border-radius: 8px; color: #f0f6fc; font-size: 1rem; margin-bottom: 18px; outline: none; }
    input[type="text"]:focus, select:focus { border-color: #58a6ff; }
    .grid { display: grid; grid-template-columns: 1fr 1fr; gap: 12px; }
    button { width: 100%; padding: 16px; background: #238636; border: none; border-radius: 12px; color: #fff; font-size: 1.1rem; font-weight: 700; cursor: pointer; transition: all 0.2s ease; margin-top: 8px; }
    button:hover { background: #2ea043; transform: translateY(-1px); }
    button:active { transform: translateY(1px); }
    button:disabled { background: #21262d; color: #8b949e; cursor: not-allowed; }
    .status { margin-top: 20px; padding: 12px; background: #0d1117; border-radius: 8px; font-size: 0.85rem; border-left: 4px solid #58a6ff; min-height: 48px; display: flex; align-items: center; }
    .badge { display: inline-block; padding: 3px 8px; border-radius: 6px; font-size: 0.72rem; font-weight: bold; background: #1f6feb; color: white; margin-left: 6px; }
  </style>
</head>
<body>
  <div class="card">
    <h1>📡 PyFSK Phone TX</h1>
    <p class="sub">Acoustic Air-Gap Transmitter</p>

    <label for="msg">Message to Transmit</label>
    <input type="text" id="msg" value="Hello from Phone!" placeholder="Type ASCII text...">

    <div class="grid">
      <div>
        <label for="mode">Acoustic Mode</label>
        <select id="mode">
          <option value="audible" selected>Audible (1.8/2.2 kHz)</option>
          <option value="ultrasonic">Ultrasonic (18.5/19.5 kHz)</option>
        </select>
      </div>
      <div>
        <label for="baud">Baud Rate</label>
        <select id="baud">
          <option value="100" selected>100 baud</option>
          <option value="50">50 baud</option>
          <option value="150">150 baud</option>
        </select>
      </div>
    </div>

    <button id="txBtn" onclick="transmitAudio()">🔊 TRANSMIT OVER SPEAKER</button>

    <div class="status" id="statusBox">Ready to transmit acoustic packet.</div>
  </div>

  <script>
    function computeCRC16(bytes) {
      let crc = 0xFFFF;
      for (let i = 0; i < bytes.length; i++) {
        crc ^= (bytes[i] << 8);
        for (let j = 0; j < 8; j++) {
          if (crc & 0x8000) {
            crc = ((crc << 1) ^ 0x1021) & 0xFFFF;
          } else {
            crc = (crc << 1) & 0xFFFF;
          }
        }
      }
      return crc;
    }

    function buildPacketBits(text) {
      const encoder = new TextEncoder();
      const payloadBytes = encoder.encode(text).slice(0, 255);
      const len = payloadBytes.length;

      const dataForCrc = new Uint8Array(1 + len);
      dataForCrc[0] = len;
      dataForCrc.set(payloadBytes, 1);

      const crc = computeCRC16(dataForCrc);
      const crcBytes = [(crc >> 8) & 0xFF, crc & 0xFF];

      // Preamble: 16 alternating bits (1, 0, 1, 0...)
      const bits = [];
      for (let i = 0; i < 16; i++) {
        bits.push(i % 2 === 0 ? 1 : 0);
      }

      // Sync word: 0x2DD4 (16 bits)
      for (let i = 15; i >= 0; i--) {
        bits.push((0x2DD4 >> i) & 1);
      }

      // Length + Payload + CRC bits
      const fullBytes = [...dataForCrc, ...crcBytes];
      for (let b of fullBytes) {
        for (let i = 7; i >= 0; i--) {
          bits.push((b >> i) & 1);
        }
      }

      return bits;
    }

    let audioCtx = null;

    async function transmitAudio() {
      const btn = document.getElementById("txBtn");
      const statusBox = document.getElementById("statusBox");
      const text = document.getElementById("msg").value || "Hello";
      const mode = document.getElementById("mode").value;
      const baud = parseInt(document.getElementById("baud").value, 10);

      btn.disabled = true;
      statusBox.innerText = "Synthesizing BFSK audio waveform...";

      if (!audioCtx) {
        audioCtx = new (window.AudioContext || window.webkitAudioContext)();
      }
      if (audioCtx.state === "suspended") {
        await audioCtx.resume();
      }

      const sampleRate = audioCtx.sampleRate;
      const markFreq = mode === "ultrasonic" ? 18500.0 : 1800.0;
      const spaceFreq = mode === "ultrasonic" ? 19500.0 : 2200.0;
      const samplesPerSymbol = Math.round(sampleRate / baud);

      const bits = buildPacketBits(text);
      const numSymbols = bits.length;
      const toneSamples = numSymbols * samplesPerSymbol;

      const guardSamples = Math.round(0.05 * sampleRate);
      const rampSamples = Math.round(0.01 * sampleRate);
      const totalSamples = guardSamples + toneSamples + guardSamples;

      const buffer = audioCtx.createBuffer(1, totalSamples, sampleRate);
      const channelData = buffer.getChannelData(0);

      const amp = 0.85;
      let phase = 0.0;

      // Synthesize Continuous-Phase BFSK
      for (let s = 0; s < numSymbols; s++) {
        const bit = bits[s];
        const freq = (bit === 1) ? markFreq : spaceFreq;
        const phaseInc = 2.0 * Math.PI * freq / sampleRate;
        const startIdx = guardSamples + s * samplesPerSymbol;

        for (let i = 0; i < samplesPerSymbol; i++) {
          phase += phaseInc;
          channelData[startIdx + i] = amp * Math.sin(phase);
        }
      }

      // Raised-cosine window smoothing
      const rampLen = Math.min(rampSamples, Math.floor(toneSamples / 2));
      for (let i = 0; i < rampLen; i++) {
        const factorUp = 0.5 * (1.0 - Math.cos(Math.PI * i / rampLen));
        const factorDown = 0.5 * (1.0 + Math.cos(Math.PI * i / rampLen));
        channelData[guardSamples + i] *= factorUp;
        channelData[guardSamples + toneSamples - rampLen + i] *= factorDown;
      }

      statusBox.innerHTML = `Transmitting ${bits.length} bits [${mode.toUpperCase()}] via speaker...`;

      const source = audioCtx.createBufferSource();
      source.buffer = buffer;
      source.connect(audioCtx.destination);
      source.onended = () => {
        btn.disabled = false;
        statusBox.innerHTML = `Transmission finished! Sent <b>'${text}'</b> (${bits.length} bits).`;
      };
      source.start();
    }
  </script>
</body>
</html>
"""


def get_local_ip() -> str:
    """Detect the local LAN IP address of this machine."""
    try:
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        s.connect(("8.8.8.8", 80))
        ip = s.getsockname()[0]
        s.close()
        return ip
    except Exception:
        return "127.0.0.1"


class FSKRequestHandler(http.server.BaseHTTPRequestHandler):
    """Serves the single-page mobile transmitter HTML application."""

    def do_GET(self) -> None:
        """Handle HTTP GET request."""
        self.send_response(200)
        self.send_header("Content-Type", "text/html; charset=utf-8")
        self.send_header("Content-Length", str(len(HTML_PAGE.encode("utf-8"))))
        self.end_headers()
        self.wfile.write(HTML_PAGE.encode("utf-8"))

    def log_message(self, format: str, *args: object) -> None:
        """Quiet HTTP server logging."""
        pass


def start_web_transmitter(port: int = 8080) -> None:
    """Start local web server hosting the phone transmitter interface."""
    local_ip = get_local_ip()
    url = f"http://{local_ip}:{port}"

    print("=" * 65)
    print("PyFSK-Modem: Mobile Phone Transmitter Web Interface")
    print("=" * 65)
    print(f"\n  1. Connect your phone to the same Wi-Fi / LAN.")
    print(f"  2. Open this URL in your phone's browser (Safari/Chrome):")
    print(f"\n     👉  \033[1;32m{url}\033[0m  👈\n")
    print(f"  3. Type any message on your phone and tap TRANSMIT!")
    print(f"  4. Your laptop's `pyfsk_modem listen` will pick it up!")
    print(f"\n(Press Ctrl+C to stop the web server)\n")
    print("=" * 65)

    server = http.server.ThreadingHTTPServer(("0.0.0.0", port), FSKRequestHandler)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nWeb transmitter stopped.")
    finally:
        server.server_close()
