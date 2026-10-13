"""M2 listening page that swaps frequency bands between two decoders of the same latents.

Where does an artifact of decoder B live? Every clip is split at --cross (zero-phase FIR, the two
parts add up to the clip again) and recombined:

    A, B                    the two decoders as they are
    low B + high A          B's artifact gone? then it is in B's high band
    low A + high B          the artifact with A's low band: it follows the high band
    low B + high recording  what B would be with a perfect high band (only with --ref)
    B, high band -X dB      is it the level of the high band, or what is in it?

A and B must decode the same latents (reconstruction, or TTS with the same seeds), so that they
are aligned sample by sample; the page reports the correlation of their low bands as a check.
--spectrograms adds a narrowband spectrogram of each version (needs matplotlib:
`uv run --with matplotlib python ...`). JVS audio is not redistributable: keep the page local.

    uv run --with matplotlib python scripts/m2/make_bandswap_page.py --out exp/listening/m2_bandswap_recon \
        --ref "exp/listening/m2_recon/audio/orig_{key}.wav" --a "exp/listening/m2_recon/audio/mimi_{key}.wav" \
        --b "exp/listening/m2_recon/audio/gs2_80k_{key}.wav" --a-label "元の Mimi" --b-label gs2_80k \
        --items exp/listening/m2_recon/items.jsonl --keys jvs010_VOICEACTRESS100_034 --spectrograms
"""

import argparse
import html
import json
from pathlib import Path

import numpy as np
import soundfile as sf
from scipy.signal import firwin, stft


def split(x: np.ndarray, sr: int, cross: float) -> tuple[np.ndarray, np.ndarray]:
    """Low and high part of x; the filter is symmetric, so low + high == x and nothing shifts."""
    low = np.convolve(x, firwin(1023, cross, fs=sr), "same").astype(np.float32)
    return low, x - low


def spectrogram(paths: list[tuple[str, Path]], out: Path, seconds: float) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(len(paths), 1, figsize=(13, 2.6 * len(paths)), squeeze=False)
    for ax, (label, path) in zip(axes[:, 0], paths):
        x, sr = sf.read(str(path), dtype="float32")
        x = x[: int(seconds * sr)]
        f, _, z = stft(x, fs=sr, nperseg=1024, noverlap=1024 - 64, window="hann", padded=False)
        db = 20 * np.log10(np.abs(z) + 1e-7)
        ax.imshow(db, origin="lower", aspect="auto", extent=(0, len(x) / sr, 0, f[-1] / 1000),
                  vmin=db.max() - 85, vmax=db.max(), cmap="magma", interpolation="nearest")
        ax.set_ylabel(f"{label}\nkHz", fontsize=8)
        ax.tick_params(labelsize=7)
    fig.tight_layout()
    fig.savefig(out, dpi=60)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--a", required=True, help="path pattern with {key}: the reference decoder")
    parser.add_argument("--b", required=True, help="path pattern with {key}: the decoder under test")
    parser.add_argument("--ref", default=None, help="path pattern with {key}: the recording, if there is one")
    parser.add_argument("--a-label", default="A")
    parser.add_argument("--b-label", default="B")
    parser.add_argument("--keys", nargs="+", required=True)
    parser.add_argument("--items", type=Path, default=None, help="items .jsonl / .json with key and text")
    parser.add_argument("--cross", type=float, nargs="+", default=[4000.0], help="crossover(s) in Hz")
    parser.add_argument("--atten", type=float, nargs="*", default=[6.0, 12.0], help="dB off B's high band")
    parser.add_argument("--spectrograms", action="store_true")
    parser.add_argument("--spectrogram-sec", type=float, default=4.0)
    parser.add_argument("--title", default="M2: bands swapped between two decoders")
    args = parser.parse_args()

    texts: dict[str, str] = {}
    if args.items is not None:
        raw = args.items.read_text(encoding="utf-8")
        rows = json.loads(raw) if raw.lstrip().startswith("[") else [json.loads(line) for line in raw.splitlines() if line]
        texts = {r["key"]: r["text"] for r in rows}
    audio = args.out / "audio"
    audio.mkdir(parents=True, exist_ok=True)
    a_label, b_label = args.a_label, args.b_label
    k0 = args.cross[0] / 1000

    blocks = []
    for key in args.keys:
        xa, sr = sf.read(args.a.format(key=key), dtype="float32")
        xb, _ = sf.read(args.b.format(key=key), dtype="float32")
        n = min(len(xa), len(xb))
        xa, xb = xa[:n], xb[:n]
        versions: list[tuple[str, np.ndarray]] = []
        if args.ref is not None:
            xr, _ = sf.read(args.ref.format(key=key), dtype="float32")
            xr = np.pad(xr, (0, max(0, n - len(xr))))[:n]
            versions.append(("録音", xr))
        versions += [(a_label, xa), (b_label, xb)]
        corr = 0.0
        for i, cross in enumerate(args.cross):
            la, ha = split(xa, sr, cross)
            lb, hb = split(xb, sr, cross)
            k = cross / 1000
            if i == 0:
                corr = float(np.dot(la, lb) / (np.linalg.norm(la) * np.linalg.norm(lb) + 1e-9))
            versions.append((f"低域 {b_label} + 高域 {a_label}（{k:g}kHz）", lb + ha))
            if i == 0:
                versions.append((f"低域 {a_label} + 高域 {b_label}（{k:g}kHz）", la + hb))
                if args.ref is not None:
                    versions.append((f"低域 {b_label} + 高域 録音（{k:g}kHz）", lb + split(xr, sr, cross)[1]))
                for db in args.atten:
                    versions.append((f"{b_label}、{k:g}kHz 以上を −{db:g}dB", lb + hb * 10 ** (-db / 20)))
        cells, shown = [], []
        for j, (label, x) in enumerate(versions):
            path = audio / f"{key}_{j:02d}.wav"
            sf.write(str(path), np.clip(x, -1, 1), sr, subtype="PCM_16")
            cells.append(f"<td><div class='lab'>{html.escape(label)}</div>"
                         f"<audio controls preload='none' src='audio/{path.name}'></audio></td>")
            shown.append((label, path))
        image = ""
        if args.spectrograms:
            png = args.out / "audio" / f"{key}.png"
            pick = shown[: (4 if args.ref is not None else 3)]  # recording, A, B, low B + high A
            spectrogram(pick, png, args.spectrogram_sec)
            image = f"<details><summary>スペクトログラム（上から {' / '.join(html.escape(s[0]) for s in pick)}）</summary><img src='audio/{png.name}'></details>"
        blocks.append(f"<h3>{html.escape(key)}</h3><p>{html.escape(texts.get(key, ''))} "
                      f"<small>低域（{k0:g}kHz 以下）どうしの相関 {corr:.3f}</small></p>"
                      f"<table><tr>{''.join(cells)}</tr></table>{image}")
        print(key, f"low-band correlation {corr:.3f}", len(versions), "versions")

    page = f"""<!doctype html><html lang="ja"><head><meta charset="utf-8"><title>{html.escape(args.title)}</title>
<style>body{{font-family:sans-serif;margin:1.5em}}table{{border-collapse:collapse}}td{{border:1px solid #ccc;padding:4px 6px;vertical-align:top}}
.lab{{font-size:12px;max-width:190px;min-height:2.6em}}audio{{width:190px;height:32px}}small{{color:#666}}img{{max-width:100%}}h3{{margin-bottom:.2em}}</style></head><body>
<h1>{html.escape(args.title)}</h1>
<p>{html.escape(a_label)} と {html.escape(b_label)} は同じ潜在表現を復号している。クリップを境目の周波数で低域と高域に分け（足すと元に戻る）、組み合わせを替えた。</p>
{''.join(blocks)}</body></html>"""
    (args.out / "index.html").write_text(page, encoding="utf-8")
    print(args.out / "index.html")


if __name__ == "__main__":
    main()
