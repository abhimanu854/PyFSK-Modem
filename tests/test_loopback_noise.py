"""Loopback SNR sweep and noise tolerance unit tests."""

import os
import pytest
from pyfsk_modem.config import ModemConfig, ModemMode
from pyfsk_modem.loopback import run_loopback, sweep_snr_ber


def test_loopback_high_snr_zero_ber():
    """Verify that at 20 dB SNR, transmission achieves 0% BER and valid CRC."""
    cfg = ModemConfig.create(mode=ModemMode.AUDIBLE, baud_rate=100)
    result = run_loopback("AirGap 20dB Test", target_snr_db=20.0, config=cfg)

    assert result.packet_detected is True
    assert result.crc_ok is True
    assert result.bit_errors == 0
    assert result.ber == 0.0
    assert result.received_text == "AirGap 20dB Test"


def test_loopback_medium_snr_10db():
    """Verify that at 10 dB SNR, packet is detected and decoded reliably."""
    cfg = ModemConfig.create(mode=ModemMode.AUDIBLE, baud_rate=100)
    result = run_loopback("AirGap 10dB Test", target_snr_db=10.0, config=cfg)

    assert result.packet_detected is True
    assert result.ber < 0.05  # Less than 5% BER in 10 dB AWGN
    if result.crc_ok:
        assert result.received_text == "AirGap 10dB Test"


def test_loopback_snr_sweep_and_plot(tmp_path):
    """Run loopback sweep across 20, 10, 5 dB and save ber_vs_snr.png."""
    cfg = ModemConfig.create(mode=ModemMode.AUDIBLE, baud_rate=100)
    plot_file = str(tmp_path / "ber_vs_snr.png")

    results = sweep_snr_ber(
        snr_range=[20.0, 10.0, 5.0],
        payload="PyFSK AWGN Test",
        trials_per_snr=3,
        save_plot_path=plot_file,
        config=cfg,
    )

    assert 20.0 in results
    assert 10.0 in results
    assert 5.0 in results

    # 20 dB should have 0 bit error rate
    assert results[20.0]["mean_ber"] == 0.0
    assert results[20.0]["packet_success_rate"] == 1.0

    # Ensure plot was generated and has non-zero size
    assert os.path.exists(plot_file)
    assert os.path.getsize(plot_file) > 1000

    # Also save a copy in the workspace root for the user/documentation
    workspace_plot = "ber_vs_snr.png"
    sweep_snr_ber(
        snr_range=[20.0, 15.0, 10.0, 5.0],
        payload="PyFSK Air-Gap Modem",
        trials_per_snr=3,
        save_plot_path=workspace_plot,
        config=cfg,
    )
    assert os.path.exists(workspace_plot)
