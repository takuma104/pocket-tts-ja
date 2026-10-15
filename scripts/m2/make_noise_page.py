"""M2 high band: a decoder's differences from the recording taken apart, for finding the mechanical sound by ear.

H2 and H2g sound slightly mechanical (listening, 2026-10-15). The page lets the listener find what carries
that sound, by placing it and by hearing candidate components alone:

    where      the recording with one band taken from the decoder (zero-phase FIR split, the parts add up
               to the clip again): the mechanical sound shows up in the version whose band carries it
    excess     what the decoder has more of than the recording: per STFT cell, the decoder's part above the
               recording's magnitude (power smoothed over 3 bins x 3 frames), also above 4 kHz alone (most of
               it is below 2 kHz); deficit: the reverse
    error      decoder - recording below 2 kHz, where the two waveforms still line up (above 3 kHz they do
               not correlate at all, so the plain difference is meaningless there)
    grid       the amplitude pattern the decoder repeats on its 5 ms frame grid / 80 ms latent grid (folded
               relative band envelopes, as in frame_modulation.py and the grid loss): x - demodulated x
    harmonics  above 4 kHz, the cells at the harmonics of the recording's F0 (raised-cosine weight, 1 on a
               harmonic, 0 half-way to the next one) and the rest between them, of the recording and the decoders
    magnitude  the decoder's STFT magnitude with the recording's phase, and the reverse: is it the spectrum
    / phase    (magnitude) or the fine timing (phase) that sounds mechanical?
    exaggerate H2 with its grid pattern x4; the decoder with its excess x3, and without it

A component alone is played at a common level; its own level relative to the decoder's output is in the label.
The masks of excess / deficit and harmonics change the signal per cell, which can itself sound a little
watery: compare the decoders' versions of the same component rather than judging one alone.
JVS audio is not redistributable: keep the page local.

    uv run python scripts/m2/make_noise_page.py exp/m2/recon --dec M2H_H2g --also M2H_H2 keep4k \
        --keys jvs002_VOICEACTRESS100_059 jvs010_VOICEACTRESS100_081 --out exp/listening/m2h_noise
"""

import argparse
import html
import json
from pathlib import Path

import librosa
import numpy as np
import numpy.typing as npt
import soundfile as sf
from scipy.ndimage import uniform_filter
from scipy.signal import firwin, istft, stft

Audio = npt.NDArray[np.float64]
NFFT = 1024
HOP = 256
LEVEL = 0.05  # rms of a component played alone
GRID5 = ((2000.0, 4000.0), (4000.0, 6000.0), (6000.0, 8000.0), (8000.0, 11000.0))
GRID80 = ((1000.0, 2000.0), (2000.0, 4000.0), (4000.0, 8000.0), (8000.0, 11000.0))
LABELS = {"orig": "録音", "mimi": "元の Mimi"}


def lowpass(x: Audio, sr: int, f: float) -> Audio:
    return np.convolve(x, firwin(1023, f, fs=sr), "same")


def band_part(x: Audio, sr: int, lo: float, hi: float | None) -> Audio:
    """The zero-phase FIR part of x between lo and hi (0 / None: open ends)."""
    low = lowpass(x, sr, hi) if hi else x
    return low - lowpass(x, sr, lo) if lo else low


def spec(x: Audio, sr: int) -> npt.NDArray[np.complex128]:
    return stft(x, fs=sr, window="hann", nperseg=NFFT, noverlap=NFFT - HOP)[2]


def unspec(z: npt.NDArray[np.complex128], sr: int, n: int) -> Audio:
    y = istft(z, fs=sr, window="hann", nperseg=NFFT, noverlap=NFFT - HOP)[1]
    return np.pad(y, (0, max(0, n - len(y))))[:n]


def above(z: npt.NDArray[np.complex128], z_ref: npt.NDArray[np.complex128]) -> npt.NDArray[np.float64]:
    """Per cell, the share of z's magnitude above z_ref's (smoothed powers): 0 where z_ref is as loud or louder."""
    p = uniform_filter(np.abs(z) ** 2, size=3)
    p_ref = uniform_filter(np.abs(z_ref) ** 2, size=3)
    return np.clip(1 - np.sqrt(p_ref / (p + 1e-20)), 0, 1)


def bandsig(x: Audio, sr: int, lo: float, hi: float) -> Audio:
    s = np.fft.rfft(x)
    f = np.fft.rfftfreq(len(x), 1 / sr)
    return np.fft.irfft(s * ((f >= lo) & (f < hi)), len(x))


def envelope(x: Audio) -> Audio:
    n = len(x)
    h = np.zeros(n)
    h[0] = 1
    h[1 : (n + 1) // 2] = 2
    if n % 2 == 0:
        h[n // 2] = 1
    return np.abs(np.fft.ifft(np.fft.fft(x) * h))


def smooth(x: Audio, n: int) -> Audio:
    c = np.cumsum(np.concatenate([np.zeros(1), x]))
    y = (c[n:] - c[:-n]) / n
    return np.concatenate([np.full(n // 2, y[0]), y, np.full(len(x) - len(y) - n // 2, y[-1])])


def grid_profile(sigs: list[Audio], refs: list[Audio], period: int, sub: int) -> Audio:
    """Relative band envelope folded on the grid (period x sub samples), weighted by the recording's level."""
    acc = np.zeros(period)
    wsum = np.zeros(period)
    for x, r in zip(sigs, refs, strict=True):
        e, er = envelope(x), envelope(r)
        if sub > 1:
            n = len(e) // sub * sub
            e, er = e[:n].reshape(-1, sub).mean(1), er[:n].reshape(-1, sub).mean(1)
        n = len(e) // period * period
        rel = e[:n] / (smooth(e[:n], 2 * period) + 1e-9) - 1
        w = smooth(er[:n], 2 * period)
        w[: 2 * period] = 0
        w[-2 * period :] = 0
        acc += (rel * w).reshape(-1, period).sum(0)
        wsum += w.reshape(-1, period).sum(0)
    p = acc / np.maximum(wsum, 1e-9)
    return p - p.mean()


GridProfiles = dict[str, list[Audio]]


def grid_profiles(root: Path, name: str, keys: list[str], sr: int) -> GridProfiles:
    pairs = []
    for k in keys:
        x, _ = sf.read(root / name / f"{k}.wav", dtype="float64")
        r, _ = sf.read(root / "orig" / f"{k}.wav", dtype="float64")
        n = min(len(x), len(r))
        pairs.append((x[:n], r[:n]))
    out: GridProfiles = {"5ms": [], "80ms": []}
    for tag, bands, period, sub in (("5ms", GRID5, 120, 1), ("80ms", GRID80, 16, 120)):
        for lo, hi in bands:
            sigs = [bandsig(x, sr, lo, hi) for x, _ in pairs]
            refs = [bandsig(r, sr, lo, hi) for _, r in pairs]
            out[tag].append(grid_profile(sigs, refs, period, sub))
    return out


def grid_component(x: Audio, sr: int, prof: GridProfiles, tag: str) -> Audio:
    """The part of x that the fixed grid pattern adds: x - x with the pattern divided out (per band)."""
    bands, period, sub = (GRID5, 120, 1) if tag == "5ms" else (GRID80, 16, 120)
    comp = np.zeros_like(x)
    idx = (np.arange(len(x)) // sub) % period
    for (lo, hi), p in zip(bands, prof[tag], strict=True):
        s = bandsig(x, sr, lo, hi)
        g = 1 / (1 + p[idx])
        if sub > 1:
            g = smooth(g, sub)
        comp -= s * (g - 1)
    return comp


def harmonic_weight(f0: npt.NDArray[np.float64], freqs: npt.NDArray[np.float64], lo: float) -> npt.NDArray[np.float64]:
    """Per cell (freq x frame): 1 on a harmonic of F0, falling (raised cosine) to 0 half-way to the next; 0 unvoiced."""
    w = np.zeros((len(freqs), len(f0)))
    for t, f in enumerate(f0):
        if not np.isfinite(f):
            continue
        h = freqs / f
        d = np.abs(h - np.round(h))  # 0 on a harmonic, 0.5 half-way
        w[:, t] = 0.5 * (1 + np.cos(2 * np.pi * d))
    w[freqs < lo] = 0
    return w


def rms(x: Audio) -> float:
    return float(np.sqrt(np.mean(x**2)) + 1e-12)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("recon", type=Path, help="directory written by recon_jvs.py")
    parser.add_argument("--dec", required=True, help="the decoder under test")
    parser.add_argument("--also", nargs="*", default=[], help="decoders to compare it with")
    parser.add_argument("--keys", nargs="+", required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--profile-items", type=int, default=150, help="utterances the grid patterns are averaged over")
    parser.add_argument("--note", action="append", default=[], help="NAME=TEXT: what a decoder is")
    parser.add_argument("--title", default="M2: 機械音の候補を取り出して聴く")
    args = parser.parse_args()

    with open(args.recon / "items.jsonl", encoding="utf-8") as f:
        items = {it["key"]: it for it in map(json.loads, f)}
    decs = [args.dec, *args.also]
    notes = dict(n.split("=", 1) for n in args.note)

    def label(name: str) -> str:
        return LABELS.get(name, name)

    audio = args.out / "audio"
    audio.mkdir(parents=True, exist_ok=True)
    keys_all = list(items)
    profile_keys = keys_all[:: max(1, len(keys_all) // args.profile_items)][: args.profile_items]
    sr = sf.info(args.recon / "orig" / f"{args.keys[0]}.wav").samplerate
    profiles = {d: grid_profiles(args.recon, d, profile_keys, sr) for d in decs}

    def put(key: str, tag: str, y: Audio, alone_of: Audio | None = None) -> str:
        """Write a clip; alone_of: the decoder output it was taken from, to play it at LEVEL and say its level."""
        note = ""
        if alone_of is not None:
            note = f"{20 * np.log10(rms(y) / rms(alone_of)):+.0f} dB"
            y = y * LEVEL / rms(y)
        peak = np.abs(y).max()
        if peak > 0.99:
            y = y * 0.99 / peak
        path = audio / f"{tag}_{key}.wav"
        sf.write(path, y.astype(np.float32), sr)
        cap = f'<div class="lv">{note}</div>' if note else ""
        return f'<audio controls preload="none" src="audio/{path.name}"></audio>{cap}'

    sections = []
    for key in args.keys:
        r, _ = sf.read(args.recon / "orig" / f"{key}.wav", dtype="float64")
        xs = {}
        for d in decs:
            x, _ = sf.read(args.recon / d / f"{key}.wav", dtype="float64")
            xs[d] = x
        n = min(len(r), *(len(x) for x in xs.values()))
        r = r[:n]
        xs = {d: x[:n] for d, x in xs.items()}
        zr = spec(r, sr)
        zs = {d: spec(x, sr) for d, x in xs.items()}
        freqs = np.fft.rfftfreq(NFFT, 1 / sr)
        f0 = librosa.pyin(r, fmin=65, fmax=500, sr=sr, frame_length=2048, hop_length=HOP, center=True)[0]
        f0 = np.pad(f0, (0, max(0, zr.shape[1] - len(f0))), constant_values=np.nan)[: zr.shape[1]]
        wh = harmonic_weight(f0, freqs, 4000.0)
        hp4 = (freqs >= 4000)[:, None]
        rows: list[tuple[str, str, list[str], list[str]]] = []

        def row(group: str, what: str, cols: list[str], cells: list[str], rows: list = rows) -> None:
            rows.append((group, what, cols, cells))

        row("基準", "そのまま", ["録音", *map(label, decs)], [put(key, "orig", r)] + [put(key, d, xs[d]) for d in decs])
        for lo, hi, text in ((0, 4000, "0–4 kHz"), (4000, None, "4 kHz–"), (0, 2000, "0–2 kHz"), (2000, 4000, "2–4 kHz"),
                             (4000, 8000, "4–8 kHz"), (8000, None, "8 kHz–")):
            tag = f"{lo // 1000}-{hi // 1000 if hi else ''}k"
            cells = [put(key, f"rec_{d}{tag}", r - band_part(r, sr, lo, hi) + band_part(xs[d], sr, lo, hi)) for d in decs]
            row("どこに", f"録音の {text} だけをデコーダの音に", list(map(label, decs)), cells)
        cells = [put(key, f"excess_{d}", unspec(zs[d] * above(zs[d], zr), sr, n), xs[d]) for d in decs]
        row("成分だけ", "多い成分: デコーダが録音より強く出している分", list(map(label, decs)), cells)
        cells = [put(key, f"excess4k_{d}", unspec(zs[d] * above(zs[d], zr) * hp4, sr, n), xs[d]) for d in decs]
        row("成分だけ", "多い成分の 4 kHz 以上だけ (上の行は 0–2 kHz が主)", list(map(label, decs)), cells)
        cells = [put(key, f"deficit_{d}", unspec(zr * above(zr, zs[d]), sr, n), xs[d]) for d in decs]
        row("成分だけ", "足りない成分: 録音にあってデコーダで弱い分", list(map(label, decs)), cells)
        cells = [put(key, f"error2k_{d}", band_part(xs[d] - r, sr, 0, 2000), xs[d]) for d in decs]
        row("成分だけ", "0–2 kHz の波形の誤差 (デコーダ − 録音)", list(map(label, decs)), cells)
        for tag, text in (("5ms", "5 ms 格子の振幅パターン (2–11 kHz)"), ("80ms", "80 ms 格子の振幅パターン (1–11 kHz)")):
            cells = [put(key, f"grid{tag}_{d}", grid_component(xs[d], sr, profiles[d], tag), xs[d]) for d in decs]
            row("成分だけ", text, list(map(label, decs)), cells)
        harm = {"orig": (zr, r), **{d: (zs[d], xs[d]) for d in decs}}
        for part, text in (("harm", "4 kHz 以上の倍音の位置"), ("inter", "4 kHz 以上の倍音の間 (+無声音)")):
            w = wh if part == "harm" else hp4 * (1 - wh)
            cells = [put(key, f"{part}_{name}", unspec(z * w, sr, n), src) for name, (z, src) in harm.items()]
            row("倍音", text, list(map(label, harm)), cells)
        cells = [put(key, f"magdec_{d}", unspec(np.abs(zs[d]) * np.exp(1j * np.angle(zr)), sr, n)) for d in decs]
        row("振幅と位相", "デコーダの振幅 + 録音の位相", list(map(label, decs)), cells)
        cells = [put(key, f"phasedec_{d}", unspec(np.abs(zr) * np.exp(1j * np.angle(zs[d])), sr, n)) for d in decs]
        row("振幅と位相", "録音の振幅 + デコーダの位相", list(map(label, decs)), cells)
        cells, cols = [], []
        for d in decs:
            grid = grid_component(xs[d], sr, profiles[d], "5ms") + grid_component(xs[d], sr, profiles[d], "80ms")
            cells.append(put(key, f"gridx4_{d}", xs[d] + 3 * grid))
            cols.append(label(d))
        row("誇張と除去", "格子パターンを 4 倍に", cols, cells)
        cells = [put(key, f"excessx3_{d}", unspec(zs[d] * (1 + 2 * above(zs[d], zr)), sr, n)) for d in decs]
        row("誇張と除去", "多い成分を 3 倍に", list(map(label, decs)), cells)
        cells = [put(key, f"noexcess_{d}", unspec(zs[d] * (1 - above(zs[d], zr)), sr, n)) for d in decs]
        row("誇張と除去", "多い成分を除く", list(map(label, decs)), cells)

        it = items[key]
        body = []
        group = None
        for g, what, cols, cells in rows:
            if g != group:
                group = g
                body.append(f'<tr class="g"><th colspan="{1 + len(decs) + 1}">{html.escape(g)}</th></tr>')
            body.append(f'<tr><th class="w">{html.escape(what)}</th>'
                        + "".join(f'<td><div class="c">{html.escape(c)}</div>{cell}</td>' for c, cell in zip(cols, cells, strict=True))
                        + "</tr>")
        voiced = np.isfinite(f0).mean()
        sections.append(f"<h2>{html.escape(key)} ({it['gender']}, 有声 {100 * voiced:.0f}%)</h2>"
                        f"<p>{html.escape(it['text'])}</p><table>{''.join(body)}</table>")
        print(key, flush=True)

    legend = "".join(f"<li><b>{html.escape(label(d))}</b>: {html.escape(notes[d])}</li>" for d in decs if d in notes)
    page = f"""<!doctype html><html lang="ja"><head><meta charset="utf-8"><title>{html.escape(args.title)}</title>
<style>body{{font-family:sans-serif;margin:1.5em;max-width:1400px}}table{{border-collapse:collapse;margin-bottom:2em}}
td,th{{border:1px solid #ccc;padding:4px 8px;vertical-align:top;text-align:left}}tr.g th{{background:#eee}}
th.w{{font-weight:normal;width:22em}}.c{{font-size:80%;color:#555}}.lv{{font-size:75%;color:#a33}}audio{{width:220px}}</style>
</head><body><h1>{html.escape(args.title)}</h1>
<ul>{legend}</ul>
<p>「成分だけ」「倍音」の行は、取り出した成分を聴きやすい音量にそろえて再生します。赤字は、その成分の元の音量 (デコーダ出力の全体に対して)。
「多い成分 / 足りない成分」と「倍音」は STFT のマスクで切り出しているので、切り出し自体のにじみ (水っぽさ) があります。1 つだけで判断せず、デコーダどうしで比べてください。</p>
<p>聴く順の提案: (1)「どこに」で機械音がどの帯域で出るか、(2)「成分だけ」と「倍音」でその音だけを含むものはどれか、(3)「振幅と位相」で、振幅 (スペクトル) と位相 (細かい時間構造) のどちらで出るか、(4)「誇張と除去」で、強めると機械音が増える / 除くと消えるか。</p>
{''.join(sections)}</body></html>"""
    (args.out / "index.html").write_text(page, encoding="utf-8")
    print("written", args.out / "index.html")


if __name__ == "__main__":
    main()
