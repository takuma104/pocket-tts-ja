"""DNSMOS P.835 (SIG / BAK / OVRL), as used by Emilia-Pipe for its quality filter.

Port of microsoft/DNS-Challenge DNSMOS/dnsmos_local.py (non-personalized polynomial fits)
for batches of 16 kHz clips. The model scores 9.01 s windows; shorter clips are tiled to
that length as in the reference script, longer clips are scored on windows with a 1 s hop
(at most `max_windows`, evenly spaced) and averaged.
Model file: data/models/dnsmos/sig_bak_ovr.onnx (DNS-Challenge repo, DNSMOS/DNSMOS/).
"""

from pathlib import Path

import numpy as np
import numpy.typing as npt
import onnxruntime as ort

SR = 16000
WINDOW = int(9.01 * SR)
_P_SIG = np.poly1d([-0.08397278, 1.22083953, 0.0052439])
_P_BAK = np.poly1d([-0.13166888, 1.60915514, -0.39604546])
_P_OVR = np.poly1d([-0.06766283, 1.11546468, 0.04602535])
DEFAULT_MODEL = Path(__file__).resolve().parents[2] / "data" / "models" / "dnsmos" / "sig_bak_ovr.onnx"


class DNSMOS:
    def __init__(self, model: Path = DEFAULT_MODEL, threads: int = 4, max_windows: int = 3):
        opts = ort.SessionOptions()
        opts.intra_op_num_threads = threads
        self.session = ort.InferenceSession(str(model), opts, providers=["CPUExecutionProvider"])
        self.max_windows = max_windows

    def _windows(self, wav: npt.NDArray[np.float32]) -> list[npt.NDArray[np.float32]]:
        while len(wav) < WINDOW:
            wav = np.concatenate([wav, wav])
        n_hops = int(np.floor(len(wav) / SR) - 9.01) + 1
        starts = np.linspace(0, max(0, n_hops - 1), min(n_hops, self.max_windows)).round().astype(int)
        return [wav[s * SR : s * SR + WINDOW] for s in starts if s * SR + WINDOW <= len(wav)]

    def __call__(self, wavs: list[npt.NDArray[np.float32]]) -> npt.NDArray[np.float32]:
        """[N] 16 kHz clips -> [N, 3] (sig, bak, ovrl) MOS."""
        wins, owner = [], []
        for i, w in enumerate(wavs):
            for seg in self._windows(np.asarray(w, dtype=np.float32)):
                wins.append(seg)
                owner.append(i)
        raw = self.session.run(None, {"input_1": np.stack(wins)})[0]
        mos = np.stack([_P_SIG(raw[:, 0]), _P_BAK(raw[:, 1]), _P_OVR(raw[:, 2])], axis=1)
        out = np.zeros((len(wavs), 3), dtype=np.float32)
        cnt = np.zeros(len(wavs))
        for m, i in zip(mos, owner, strict=True):
            out[i] += m
            cnt[i] += 1
        return out / np.maximum(cnt, 1)[:, None]
