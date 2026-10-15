"""M2 high band: which scale of the decoder's 4 kHz+ error is the mechanical sound? Fix one scale from the recording.

Listening (2026-10-16, m2h_noise): the mechanical sound is above 4 kHz, in the decoder's STFT magnitude (not its
phase), and the decoder with its excess over the recording removed sounds fine. The excess is spread over all
loud frames and over the whole band (per frame and 1 kHz band the level is off by about 4.7 dB rms, against
3 dB at 2-4 kHz), so the numbers cannot say which scale of it is heard. Each row here takes the decoder's
4 kHz+ part (zero-phase FIR split) and replaces one scale of it with the recording's:

    envelope in time   the 4 kHz+ level every ~10 ms (time domain: power smoothed with a 10 ms Hann window)
    spectral shape     the shape over 4-12 kHz in 500 Hz steps per ~20 ms frame (STFT 512), the frame's level kept
    both               the 500 Hz x ~20 ms envelope, level and shape
    fine structure     what is finer than 500 Hz (the harmonics) (STFT 1024), the envelope kept

plus, for reference, the decoder with its excess removed (as on m2h_noise), and a change that needs no
recording: the decoder's mean 80 ms grid pattern divided out (as frame_modulation.py measures it).
The decoder's phase is kept everywhere. JVS audio is not redistributable: keep the page local.

    uv run python scripts/m2/make_scale_page.py exp/m2/recon --dec M2H_H2g --also M2H_H2gh \
        --keys jvs010_VOICEACTRESS100_081 jvs002_VOICEACTRESS100_059 --out exp/listening/m2h_noise2
"""

import argparse
import html
import json
from pathlib import Path

import numpy as np
import numpy.typing as npt
import soundfile as sf
from make_noise_page import above, grid_component, grid_profiles, lowpass
from scipy.ndimage import uniform_filter1d
from scipy.signal import istft, stft

Audio = npt.NDArray[np.float64]
Spec = npt.NDArray[np.complex128]
CROSS = 4000.0
FIXES = ("noexcess", "time", "shape", "both", "fine", "demod80")


def spec(x: Audio, sr: int, nfft: int) -> Spec:
    return stft(x, fs=sr, window="hann", nperseg=nfft, noverlap=nfft - nfft // 4)[2]


def unspec(z: Spec, sr: int, nfft: int, n: int) -> Audio:
    y = istft(z, fs=sr, window="hann", nperseg=nfft, noverlap=nfft - nfft // 4)[1]
    return np.pad(y, (0, max(0, n - len(y))))[:n]


def band_envelope(p: npt.NDArray[np.float64], sr: int, nfft: int) -> npt.NDArray[np.float64]:
    """Power smoothed over 500 Hz along frequency."""
    return uniform_filter1d(p, max(1, round(500 / (sr / nfft))), axis=0, mode="nearest") + 1e-20


def time_envelope(hd: Audio, hr: Audio) -> Audio:
    win = np.hanning(241)
    win /= win.sum()
    gain = np.sqrt((np.convolve(hr**2, win, "same") + 1e-12) / (np.convolve(hd**2, win, "same") + 1e-12))
    return hd * np.clip(gain, 0.1, 10)


def spectral_envelope(hd: Audio, hr: Audio, sr: int, keep_level: bool) -> Audio:
    nfft = 512
    zd, zr = spec(hd, sr, nfft), spec(hr, sr, nfft)
    pd = np.abs(zd) ** 2
    gain2 = band_envelope(np.abs(zr) ** 2, sr, nfft) / band_envelope(pd, sr, nfft)
    freqs = np.fft.rfftfreq(nfft, 1 / sr)
    gain2[freqs < CROSS] = 1
    if keep_level:  # the frame's 4 kHz+ power stays the decoder's
        sel = freqs >= CROSS
        gain2[sel] *= (pd[sel].sum(0) + 1e-20) / ((pd * gain2)[sel].sum(0) + 1e-20)
    return unspec(zd * np.sqrt(np.clip(gain2, 1e-2, 1e2)), sr, nfft, len(hd))


def fine_structure(hd: Audio, hr: Audio, sr: int) -> Audio:
    nfft = 1024
    zd, zr = spec(hd, sr, nfft), spec(hr, sr, nfft)
    pd, pr = np.abs(zd) ** 2, np.abs(zr) ** 2
    gain2 = (pr / band_envelope(pr, sr, nfft)) / (pd / band_envelope(pd, sr, nfft) + 1e-20)
    freqs = np.fft.rfftfreq(nfft, 1 / sr)
    gain2[freqs < CROSS] = 1
    return unspec(zd * np.sqrt(np.clip(gain2, 1e-2, 1e2)), sr, nfft, len(hd))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("recon", type=Path, help="directory written by recon_jvs.py")
    parser.add_argument("--dec", required=True, help="the decoder whose 4 kHz+ part is fixed")
    parser.add_argument(
        "--also", nargs="*", default=[], help="decoders played as they are, for reference (NAME or NAME=label)"
    )
    parser.add_argument("--keys", nargs="+", required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument(
        "--fixes",
        nargs="*",
        default=list(FIXES),
        choices=FIXES,
        help="the rows that fix one scale of --dec (default: all of them)",
    )
    parser.add_argument("--profile-items", type=int, default=150)
    parser.add_argument("--title", default="M2: 4 kHz 以上のどの細かさの違いが機械音か")
    parser.add_argument("--intro", default=None, help="the page's opening paragraph (HTML)")
    args = parser.parse_args()
    also = dict(a.split("=", 1) if "=" in a else (a, f"{a} (参考)") for a in args.also)

    with open(args.recon / "items.jsonl", encoding="utf-8") as f:
        items = {it["key"]: it for it in map(json.loads, f)}
    audio = args.out / "audio"
    audio.mkdir(parents=True, exist_ok=True)
    keys_all = list(items)
    sr = sf.info(args.recon / "orig" / f"{args.keys[0]}.wav").samplerate
    profile_keys = keys_all[:: max(1, len(keys_all) // args.profile_items)][: args.profile_items]
    prof = grid_profiles(args.recon, args.dec, profile_keys, sr)
    d = args.dec
    rows: list[tuple[str, str]] = [
        ("orig", "録音"),
        (d, f"{d}"),
        ("noexcess", f"{d}: 多い成分を除く (前回、ほぼ問題なしだったもの)"),
        ("time", f"{d}: 4 kHz 以上の、時間の包絡 (約 10 ms ごとの音量) だけを録音に"),
        ("shape", f"{d}: 4 kHz 以上の、スペクトルの形 (500 Hz ごと、約 20 ms ごと) だけを録音に (各フレームの音量は {d} のまま)"),
        ("both", f"{d}: 4 kHz 以上の、包絡 (500 Hz × 約 20 ms の音量) を録音に (上の 2 つを合わせたもの)"),
        ("fine", f"{d}: 4 kHz 以上の、500 Hz より細かい構造 (倍音) だけを録音に (包絡は {d} のまま)"),
        ("demod80", f"{d}: 80 ms 格子の平均の模様を割って除く (録音を使わない、推論でもできる処理)"),
        *also.items(),
    ]
    rows = [(tag, text) for tag, text in rows if tag not in FIXES or tag in args.fixes]
    cells: dict[str, list[str]] = {tag: [] for tag, _ in rows}
    heads = []
    for key in args.keys:
        r, _ = sf.read(args.recon / "orig" / f"{key}.wav", dtype="float64")
        xs = {name: sf.read(args.recon / name / f"{key}.wav", dtype="float64")[0] for name in [d, *also]}
        n = min(len(r), *(len(x) for x in xs.values()))
        r, x = r[:n], xs[d][:n]
        lr, ld = lowpass(r, sr, CROSS), lowpass(x, sr, CROSS)
        hr, hd = r - lr, x - ld
        zx, zr = spec(x, sr, 1024), spec(r, sr, 1024)
        out = {
            "orig": r,
            d: x,
            "noexcess": unspec(zx * (1 - above(zx, zr)), sr, 1024, n),
            "time": ld + time_envelope(hd, hr),
            "shape": ld + spectral_envelope(hd, hr, sr, keep_level=True),
            "both": ld + spectral_envelope(hd, hr, sr, keep_level=False),
            "fine": ld + fine_structure(hd, hr, sr),
            "demod80": x - grid_component(x, sr, prof, "80ms"),
            **{a: xs[a][:n] for a in also},
        }
        for tag, y in out.items():
            if tag not in cells:
                continue
            peak = np.abs(y).max()
            y = y * 0.99 / peak if peak > 0.99 else y
            path = audio / f"{tag}_{key}.wav"
            sf.write(path, y.astype(np.float32), sr)
            cells[tag].append(f'<td><audio controls preload="none" src="audio/{path.name}"></audio></td>')
        it = items[key]
        heads.append(f'<th>{html.escape(key)} ({it["gender"]})<div class="t">{html.escape(it["text"])}</div></th>')
        print(key, flush=True)
    intro = args.intro or (
        "前回の結果: 機械音は 4 kHz 以上にあり、位相ではなく振幅 (スペクトル) にあり、「多い成分を除く」とほぼ消える。"
        f"ここでは {html.escape(d)} の 4 kHz 以上のうち、1 つの細かさだけを録音のものに入れ替えています"
        f" (4 kHz 未満と位相は {html.escape(d)} のまま)。機械音が消える行が、その細かさに原因があることを示します。"
    )
    body = "".join(f'<tr><th class="w">{html.escape(text)}</th>{"".join(cells[tag])}</tr>' for tag, text in rows)
    page = f"""<!doctype html><html lang="ja"><head><meta charset="utf-8"><title>{html.escape(args.title)}</title>
<style>body{{font-family:sans-serif;margin:1.5em}}table{{border-collapse:collapse}}td,th{{border:1px solid #ccc;padding:4px 8px;vertical-align:top;text-align:left}}
th.w{{font-weight:normal;width:26em}}.t{{font-weight:normal;font-size:80%;color:#555;max-width:16em}}audio{{width:200px}}</style>
</head><body><h1>{html.escape(args.title)}</h1>
<p>{intro}</p>
<table><tr><th></th>{"".join(heads)}</tr>{body}</table></body></html>"""
    (args.out / "index.html").write_text(page, encoding="utf-8")
    print("written", args.out / "index.html")


if __name__ == "__main__":
    main()
