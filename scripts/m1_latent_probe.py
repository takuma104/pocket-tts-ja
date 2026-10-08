"""M1: is consonant identity still in the Mimi latent, or lost by the encoder?

The M0 round trip mostly adds place-of-articulation confusions (キ→チ, ニ⇄ミ, ガ→ダ, ...).
This probe classifies the consonant of each CV mora in JVS read speech (phoneme timings
from lab/mon) from three views of the same 240 ms window around the consonant onset:

  latent   3 Mimi latent frames (12.5 Hz, 32-dim) — what the FlowLM models
  orig_mel / orig_mel_fine    log-mel of the original recording, as three 80 ms block
                              means or at full 10 ms resolution (upper reference)
  recon_mel / recon_mel_fine  the same for the Mimi reconstruction — what the decoder renders

Train on speakers jvs001–080, test on jvs081–100 (unseen voices), same MLP for all views.
latent ≈ orig_mel > recon_mel  -> information survives encoding; the decoder loses it
latent ≈ recon_mel < orig_mel  -> the encoder discards it

    uv run python scripts/m1_latent_probe.py
"""

import argparse
import json
from pathlib import Path

import numpy as np
import soundfile as sf
import torch
import torchaudio
from tqdm import tqdm

from pocket_tts_ja.jvs import list_items, read_mono_lab
from pocket_tts_ja.mimi_utils import encode, load_mimi, load_pocket_config

CONSONANTS = ["k", "g", "t", "d", "n", "m", "b", "p", "h", "s", "sh", "ch", "j", "r", "w", "y", "z", "ts", "f"]
VOWELS = set("aiueo")
FRAME_RATE = 12.5
SR = 24000


def lab_segments(path: Path) -> list[tuple[float, float, str]]:
    out = []
    for line in path.read_text().splitlines():
        if line.strip():
            s, e, p = line.split()[:3]
            out.append((float(s), float(e), p))
    return out


def cv_onsets(segs: list[tuple[float, float, str]]) -> list[tuple[float, str, str]]:
    """(consonant start, consonant, vowel) for every consonant+voiced vowel pair."""
    res = []
    for (s, _, c), (_, _, v) in zip(segs, segs[1:]):
        if c in CONSONANTS and v in VOWELS:
            res.append((s, c, v))
    return res


def mel_window(mel: torch.Tensor, t: float, fine: bool = False) -> np.ndarray:
    """Log-mel over the 240 ms window starting 80 ms before the onset: three 80 ms block
    means (coarse, same time resolution as the latent) or all 24 frames at 10 ms (fine,
    keeps burst and formant-transition cues)."""
    hop = 0.01
    a = int(round((t - 0.08) / hop))
    a = max(0, min(a, mel.shape[1] - 24))
    win = mel[:, a : a + 24]
    if fine:
        return win[::2].reshape(-1).numpy()  # 40 mel bins x 24 frames
    return win.reshape(mel.shape[0], 3, 8).mean(dim=2).T.reshape(-1).numpy()


def latent_window(lat: torch.Tensor, t: float) -> np.ndarray:
    f = int(t * FRAME_RATE)
    idx = [max(0, min(lat.shape[0] - 1, f + k)) for k in (-1, 0, 1)]
    return lat[idx].reshape(-1).cpu().numpy()


def train_probe(x_tr: np.ndarray, y_tr: np.ndarray, x_te: np.ndarray, y_te: np.ndarray, n_cls: int) -> float:
    dev = torch.device("cuda")
    mu, sd = x_tr.mean(0), x_tr.std(0) + 1e-5
    xt = torch.tensor((x_tr - mu) / sd, dtype=torch.float32, device=dev)
    yt = torch.tensor(y_tr, device=dev)
    xv = torch.tensor((x_te - mu) / sd, dtype=torch.float32, device=dev)
    yv = torch.tensor(y_te, device=dev)
    torch.manual_seed(0)
    net = torch.nn.Sequential(
        torch.nn.Linear(xt.shape[1], 512), torch.nn.GELU(), torch.nn.Dropout(0.2),
        torch.nn.Linear(512, 512), torch.nn.GELU(), torch.nn.Dropout(0.2),
        torch.nn.Linear(512, n_cls),
    ).to(dev)
    opt = torch.optim.AdamW(net.parameters(), lr=1e-3, weight_decay=1e-2)
    for _ in range(40):
        perm = torch.randperm(len(xt), device=dev)
        net.train()
        for i in range(0, len(xt), 512):
            b = perm[i : i + 512]
            loss = torch.nn.functional.cross_entropy(net(xt[b]), yt[b])
            opt.zero_grad()
            loss.backward()
            opt.step()
    net.eval()
    with torch.no_grad():
        return (net(xv).argmax(-1) == yv).float().mean().item()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--jvs-root", type=Path, default=Path("data/jvs_ver1"))
    parser.add_argument("--recon-dir", type=Path, default=Path("exp/m0_mimi_recon/jvs/recon"))
    parser.add_argument("--out", type=Path, default=Path("exp/m1_latent_probe"))
    args = parser.parse_args()
    dev = torch.device("cuda")
    mimi = load_mimi(load_pocket_config(), dev)
    melspec = torchaudio.transforms.MelSpectrogram(SR, n_fft=1024, hop_length=240, n_mels=80)

    feats: dict[str, list[np.ndarray]] = {
        "latent": [], "orig_mel": [], "recon_mel": [], "orig_mel_fine": [], "recon_mel_fine": []
    }
    labels, vowels, speakers = [], [], []
    items = list_items(args.jvs_root, ("parallel100", "nonpara30"))
    for it in tqdm(items, desc="features"):
        lab = Path(it.wav).parent.parent / "lab" / "mon" / f"{it.utt_id}.lab"
        recon = args.recon_dir / f"{it.key}.wav"
        if not lab.exists() or not recon.exists():
            continue
        onsets = cv_onsets(lab_segments(lab))
        if not onsets:
            continue
        wav, _ = sf.read(it.wav, dtype="float32")
        rwav, _ = sf.read(str(recon), dtype="float32")
        lat = encode(mimi, wav, dev).float()
        mel_o = torch.log(melspec(torch.from_numpy(wav)) + 1e-5)
        mel_r = torch.log(melspec(torch.from_numpy(rwav)) + 1e-5)
        for t, c, v in onsets:
            feats["latent"].append(latent_window(lat, t))
            feats["orig_mel"].append(mel_window(mel_o, t))
            feats["recon_mel"].append(mel_window(mel_r, t))
            feats["orig_mel_fine"].append(mel_window(mel_o, t, fine=True))
            feats["recon_mel_fine"].append(mel_window(mel_r, t, fine=True))
            labels.append(CONSONANTS.index(c))
            vowels.append(v)
            speakers.append(int(it.speaker[3:]))
    y = np.array(labels)
    spk = np.array(speakers)
    tr, te = spk <= 80, spk > 80
    results = {"n_train": int(tr.sum()), "n_test": int(te.sum()), "classes": CONSONANTS}
    # majority baseline
    counts = np.bincount(y[tr], minlength=len(CONSONANTS))
    results["majority"] = float((y[te] == counts.argmax()).mean())
    for name, f in feats.items():
        x = np.stack(f)
        results[name] = train_probe(x[tr], y[tr], x[te], y[te], len(CONSONANTS))
        print(name, f"{results[name]:.4f}", flush=True)
    # place-of-articulation subset that M0 found confused: k/t/p, g/d/b, n/m
    sub = np.isin(y, [CONSONANTS.index(c) for c in ("k", "t", "p", "g", "d", "b", "n", "m")])
    for name, f in feats.items():
        x = np.stack(f)
        results[f"{name}_stops_nasals"] = train_probe(
            x[tr & sub], y[tr & sub], x[te & sub], y[te & sub], len(CONSONANTS)
        )
        print(f"{name} (k t p g d b n m)", f"{results[name + '_stops_nasals']:.4f}", flush=True)
    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / "results.json").write_text(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
