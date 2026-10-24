"""Noise probe: the principal axes of a student's normalised latent frames.

The flow head turns N(0, temp) noise into the next frame in the normalised space
((latent - emb_mean) / emb_std, the student's own statistics). student_synth.py --pca-temps sets a
temperature per axis of that space, largest variance first, to see which directions carry the
timing (pauses, readings) and which the harmonics. Frames are the speech of rows of a training
jsonl (up to each row's speech end).

    uv run python scripts/eval/latent_pca.py exp/irodori_distill/latent_pca_p3m.npz \
        --rows data/irodori_distill/mix/P3h_all_train.jsonl \
        --weights runs/irodori_distill/P3m_6l/snapshots/step_00020000.safetensors

--loudness-from TARGETS (decode_targets.py output: the teacher's latents with their DACVAE audio,
960 samples per frame): the first axis is instead the direction of the frames' loudness (a least
squares fit of each frame's log RMS from its normalised latent), the others the principal axes of
what is orthogonal to it: a temperature for the axis that moves a frame toward silence or speech.

Writes axes ([32, 32], columns largest first), variances and the mean of the normalised frames.
"""

import argparse
from pathlib import Path

import numpy as np
import safetensors.torch

from pocket_tts_ja.dv_loader import _Corpus


def loudness_axis(targets: Path, mean: np.ndarray, std: np.ndarray, clips: int, seed: int) -> np.ndarray:
    """Unit direction of the least-squares fit of the frames' log RMS (dB) from their latents."""
    files = sorted((targets / "wav").glob("*.npy"))
    files = [files[i] for i in np.random.default_rng(seed).choice(len(files), size=min(clips, len(files)), replace=False)]
    xs, ys = [], []
    for f in files:
        lat = np.load(targets / "lat" / f.name).astype(np.float32)
        wav = np.load(f).astype(np.float32) / 32768
        n = min(len(lat), len(wav) // 960)
        rms = np.sqrt(np.square(wav[: n * 960].reshape(n, 960)).mean(1) + 1e-10)
        xs.append((lat[:n] - mean) / std)
        ys.append(20 * np.log10(rms))
    x, y = np.concatenate(xs), np.concatenate(ys)
    a = np.concatenate([x, np.ones((len(x), 1))], axis=1)
    coef = np.linalg.lstsq(a, y, rcond=None)[0]
    r2 = 1 - np.var(y - a @ coef) / np.var(y)
    w = coef[:-1]
    print(f"loudness from {len(x)} frames: R^2 {r2:.3f}, {np.linalg.norm(w):.1f} dB per unit along the axis")
    return w / np.linalg.norm(w)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("out", type=Path)
    parser.add_argument("--rows", type=Path, required=True)
    parser.add_argument("--weights", type=Path, required=True, help="student snapshot (flow_lm.emb_mean / emb_std)")
    parser.add_argument("--clips", type=int, default=4000)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--loudness-from", type=Path, default=None, help="decode_targets.py output")
    args = parser.parse_args()

    state = safetensors.torch.load_file(str(args.weights))
    mean, std = state["flow_lm.emb_mean"].float().numpy(), state["flow_lm.emb_std"].float().numpy()
    corpus = _Corpus(str(args.rows))
    rng = np.random.default_rng(args.seed)
    idx = rng.choice(len(corpus.speech_end), size=min(args.clips, len(corpus.speech_end)), replace=False)
    x = np.concatenate([np.asarray(corpus.latents(int(i), int(corpus.speech_end[i])), dtype=np.float32) for i in idx])
    x = (x - mean) / std
    first = loudness_axis(args.loudness_from, mean, std, args.clips, args.seed) if args.loudness_from else None
    rest = x if first is None else x - np.outer(x @ first, first)
    var, axes = np.linalg.eigh(np.cov(rest.T))
    order = np.argsort(var)[::-1]
    var, axes = var[order], axes[:, order]
    if first is not None:  # the null direction of rest (first) is the last of eigh's order
        axes = np.concatenate([first[:, None], axes[:, :-1]], axis=1)
        var = np.concatenate([[float(np.var(x @ first))], var[:-1]])
    np.savez(args.out, axes=axes.astype(np.float32), variances=var.astype(np.float32), mean=x.mean(0).astype(np.float32))
    share = np.cumsum(var) / var.sum()
    print(f"{len(x)} frames of {len(idx)} clips; total variance {var.sum():.1f}")
    for k in (1, 2, 4, 8, 16, 24, 32):
        print(f"  first {k:2d} axes: {share[k - 1]:.1%} of the variance (axis {k}: {var[k - 1]:.3f})")


if __name__ == "__main__":
    main()
