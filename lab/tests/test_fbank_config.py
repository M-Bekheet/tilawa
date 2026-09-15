"""Lock LHOTSE_FBANK_CONFIG keys/values (no node, no torchaudio required)."""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from shared.fbank import LHOTSE_FBANK_CONFIG

_EXPECTED = {
    "sampling_rate": 16000,
    "num_filters": 80,
    "dither": 0.0,
    "snip_edges": False,
    "preemph_coeff": 0.97,
    "window_type": "povey",
    "remove_dc_offset": True,
    "low_freq": 20,
    "high_freq": -400,
    "use_energy": False,
    "torchaudio_compatible_mel_scale": True,
}


def test_lhotse_fbank_config_frozen_keys_and_values():
    assert set(LHOTSE_FBANK_CONFIG) == set(_EXPECTED)
    assert LHOTSE_FBANK_CONFIG == _EXPECTED
