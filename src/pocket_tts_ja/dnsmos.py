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
    def __init__(
        self, model: Path = DEFAULT_MODEL, threads: int = 4, max_windows: int = 3, batch: int = 2
    ):
        opts = ort.SessionOptions()
        opts.intra_op_num_threads = threads
        # The model needs ~0.4 GB of activations per 9 s window; without the arena the memory
        # goes back after each call instead of staying at the high-water mark.
        opts.enable_cpu_mem_arena = False
        self.batch = batch
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
        raw = np.concatenate(
            [
                self.session.run(None, {"input_1": np.stack(wins[i : i + self.batch])})[0]
                for i in range(0, len(wins), self.batch)
            ]
        )
        mos = np.stack([_P_SIG(raw[:, 0]), _P_BAK(raw[:, 1]), _P_OVR(raw[:, 2])], axis=1)
        out = np.zeros((len(wavs), 3), dtype=np.float32)
        cnt = np.zeros(len(wavs))
        for m, i in zip(mos, owner, strict=True):
            out[i] += m
            cnt[i] += 1
        return out / np.maximum(cnt, 1)[:, None]


class TorchDNSMOS(DNSMOS):
    """The same model converted with onnx2torch and run on a GPU (~2 ms per window on a
    RTX 5090 versus ~0.9 s per window on one CPU thread)."""

    def __init__(self, device: "torch.device", model: Path = DEFAULT_MODEL, max_windows: int = 3, batch: int = 16):
        import onnx
        import torch
        from onnx2torch import convert

        self.max_windows = max_windows
        self.batch = batch
        self.device = device
        self.net = convert(onnx.load(str(model))).eval().to(device)
        self._torch = torch

    def __call__(self, wavs: list[npt.NDArray[np.float32]]) -> npt.NDArray[np.float32]:
        torch = self._torch
        wins, owner = [], []
        for i, w in enumerate(wavs):
            for seg in self._windows(np.asarray(w, dtype=np.float32)):
                wins.append(seg)
                owner.append(i)
        outs = []
        with torch.no_grad():
            for i in range(0, len(wins), self.batch):
                x = torch.from_numpy(np.stack(wins[i : i + self.batch])).to(self.device)
                outs.append(self.net(x).float().cpu().numpy())
        raw = np.concatenate(outs)
        mos = np.stack([_P_SIG(raw[:, 0]), _P_BAK(raw[:, 1]), _P_OVR(raw[:, 2])], axis=1)
        out = np.zeros((len(wavs), 3), dtype=np.float32)
        cnt = np.zeros(len(wavs))
        for m, i in zip(mos, owner, strict=True):
            out[i] += m
            cnt[i] += 1
        return out / np.maximum(cnt, 1)[:, None]
