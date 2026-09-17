"""Kaldi / lhotse-equivalent log-mel fbank (training frontend).

`compute_fbank` is a thin wrapper around `torchaudio.compliance.kaldi.fbank`
with the same geometry as the shipped TS inference frontend
(`packages/core/src/recitation/fbank.ts`): 16 kHz,
25 ms / 10 ms, 80 mel bins, povey window, 512-FFT, snip_edges=False,
pre-emphasis 0.97, remove_dc_offset, dither 0, low 20 Hz, high −400
(= 7600 Hz), log mel energies, no CMVN.

Waveform is float32 in [-1, 1]. Neither torchaudio nor kaldi-native-fbank
(the lhotse default extractor) scales to int16; the JS port also takes
[-1, 1]. Log floor is torch.finfo(float32).eps ≈ 1.1921e-07, matching
JS `Math.log(Math.max(e, 1.1920929e-7))`.
"""

from __future__ import annotations

import numpy as np
import torch
import torchaudio

# Exact kwargs for torchaudio.compliance.kaldi.fbank. sample_frequency is
# passed from compute_fbank(sr=...) so a caller can override the rate.
KALDI_FBANK_KWARGS = dict(
    num_mel_bins=80,
    frame_length=25.0,
    frame_shift=10.0,
    dither=0.0,
    energy_floor=0.0,
    snip_edges=False,
    preemphasis_coefficient=0.97,
    remove_dc_offset=True,
    window_type="povey",
    low_freq=20.0,
    high_freq=-400.0,
    use_energy=False,
    sample_frequency=16000,
    round_to_power_of_two=True,
)

# lhotse FbankConfig kwargs that produce identical features. lhotse's default
# extractor is kaldi-native-fbank (knf); torchaudio_compatible_mel_scale=True
# makes knf use the same HTK-style mel banks as torchaudio.compliance.kaldi.
# Not used locally (lhotse is not installed); exported for the training recipe.
LHOTSE_FBANK_CONFIG = dict(
    sampling_rate=16000,
    num_filters=80,
    dither=0.0,
    snip_edges=False,
    preemph_coeff=0.97,
    window_type="povey",
    remove_dc_offset=True,
    low_freq=20,
    high_freq=-400,
    use_energy=False,
    torchaudio_compatible_mel_scale=True,
)


def compute_fbank(wave: np.ndarray, sr: int = 16000) -> np.ndarray:
    """Log-mel fbank matching the native KaldiFbank frontend.

    Args:
        wave: mono float32 samples in [-1, 1].
        sr: sample rate in Hz (must be 16000 for JS parity).

    Returns:
        np.ndarray of shape [nFrames, 80], float32.
    """
    wave = np.ascontiguousarray(np.asarray(wave, dtype=np.float32).reshape(-1))
    waveform = torch.from_numpy(wave).unsqueeze(0)
    kwargs = dict(KALDI_FBANK_KWARGS)
    kwargs["sample_frequency"] = float(sr)
    feats = torchaudio.compliance.kaldi.fbank(waveform, **kwargs)
    return feats.cpu().numpy().astype(np.float32, copy=False)
