"""P1-side: start the stock-wheel decoder for DACVAE latents where the stock decoder already is.

The decode side of pocket_tts_ja.wheel_codec keeps the stock decoder transformer and SEANet and
puts two fresh parts in front of them: the latent projection (quantizer.output_proj, 32 -> 512,
1x1, no bias) and the channel-wise transposed-conv upsampling (25 -> 200 Hz, kernel 16). Random,
they gave the first probe a buzz at the multiples of 200 Hz and 25 Hz (line contrast 4.7 dB after
11k GAN steps; stock Mimi 1.6, the DACVAE decode 0.65) that hid whatever the high band did.

Here the upsampling starts as linear interpolation (triangle kernel; channel-wise, so it commutes
with the projection) and the projection is the least-squares map from the interpolated DACVAE
latents to what the stock decoder transformer gets for the same audio: teacher latents decoded
with DACVAE, encoded with the stock Mimi, through the stock projection and upsampling. The lag
between the two 200 Hz grids is searched. The result decodes like a stock-Mimi round trip of the
DACVAE audio before any training; the report compares the two.

    uv run python scripts/wheel/bridge_init.py data/irodori_distill/teacher/s3/latents \
        --out runs/irodori_distill/bridge_init.safetensors
"""

import argparse
import random
import sys
from pathlib import Path

import numpy as np
import safetensors.torch
import soundfile as sf
import soxr
import torch
import torch.nn.functional as F

from pocket_tts.modules.stateful_module import init_states
from pocket_tts_ja.codec_losses import MultiScaleMelLoss, decode_window
from pocket_tts_ja.dacvae import DACVAE, SAMPLE_RATE as DV_SR
from pocket_tts_ja.mimi_utils import load_mimi, load_pocket_config
from pocket_tts_ja.wheel_codec import build_dv_mimi, decoder_state

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "eval"))
from tts_comb import comb_frames  # noqa: E402

SR = 24000
STRIDE = 8
MAX_LAG = 24  # 200 Hz frames (120 ms) each way


def triangle(stride: int) -> torch.Tensor:
    j = torch.arange(2 * stride, dtype=torch.float32)
    return 1 - (j + 0.5 - stride).abs() / stride


def interp(z: torch.Tensor) -> torch.Tensor:
    """[T, C] 25 Hz -> [T * 8, C] 200 Hz, as the triangle-initialised streaming upsampling does."""
    c = z.shape[1]
    w = triangle(STRIDE).to(z)[None, None].expand(c, 1, -1)
    return F.conv_transpose1d(z.T[None], w, stride=STRIDE, groups=c)[0, :, : z.shape[0] * STRIDE].T


@torch.no_grad()
def stock_features(mimi: torch.nn.Module, wav: np.ndarray, device: torch.device) -> torch.Tensor:
    """[T200, 512]: the stock decoder transformer's input for this 24 kHz waveform."""
    zm = mimi.encode_to_latent(torch.from_numpy(wav)[None, None].to(device))
    state = init_states(mimi, 1, (zm.shape[1] + 8) * 16)
    return mimi._to_encoder_framerate(mimi.quantizer(zm.transpose(-1, -2)), state)[0].T


def line_contrast(x: np.ndarray, period_hz: float = 200.0, lo: float = 2000, hi: float = 10000) -> float:
    n = 8192
    frames = [x[i : i + n] * np.hanning(n) for i in range(0, len(x) - n, n // 2)]
    db = 10 * np.log10(np.mean([np.abs(np.fft.rfft(f)) ** 2 for f in frames], axis=0) + 1e-12)
    step = SR / n
    k = np.arange(int(lo / period_hz), int(hi / period_hz))
    return float(np.mean(db[np.round(k * period_hz / step).astype(int)] - db[np.round((k + 0.5) * period_hz / step).astype(int)]))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("latents", type=Path, help="directory of teacher latents <id>.npy")
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--clips", type=int, default=400)
    parser.add_argument("--val-latents", type=Path, default=Path("exp/irodori_distill/eval/voices/teacher_latents"))
    parser.add_argument("--val-clips", type=int, default=20)
    args = parser.parse_args()
    device = torch.device("cuda")

    codec = DACVAE.load(str(device))
    stock = load_mimi(load_pocket_config(), device).eval()
    paths = sorted(args.latents.glob("*.npy"))
    random.Random(0).shuffle(paths)
    lags = range(-MAX_LAG, MAX_LAG + 1)
    xtx = {lag: torch.zeros(32, 32, dtype=torch.float64) for lag in lags}
    xty = {lag: torch.zeros(32, 512, dtype=torch.float64) for lag in lags}
    yty = {lag: torch.zeros((), dtype=torch.float64) for lag in lags}
    for p in paths[: args.clips]:
        z = torch.from_numpy(np.load(p).astype(np.float32)).to(device)
        with torch.no_grad():
            wav = soxr.resample(codec.decode(z[None])[0].cpu().numpy(), DV_SR, SR, quality="HQ").astype(np.float32)
        y = stock_features(stock, wav, device).double().cpu()
        x = interp(z).double().cpu()
        for lag in lags:  # y[t] against x[t + lag]
            a, b = max(0, -lag), min(len(y), len(x) - lag)
            if b - a < 50:
                continue
            xs, ys = x[a + lag : b + lag], y[a:b]
            xtx[lag] += xs.T @ xs
            xty[lag] += xs.T @ ys
            yty[lag] += (ys**2).sum()
    fits = {}
    for lag in lags:
        w = torch.linalg.solve(xtx[lag] + 1e-6 * torch.eye(32, dtype=torch.float64), xty[lag])
        resid = yty[lag] - 2 * (w * xty[lag]).sum() + (w * (xtx[lag] @ w)).sum()
        fits[lag] = (float(1 - resid / yty[lag]), w)
    best = max(fits, key=lambda k: fits[k][0])
    r2, w = fits[best]
    print(f"least squares over {args.clips} clips: best lag {best} frames at 200 Hz ({best * 5} ms), "
          f"R^2 {r2:.3f} (lag 0: {fits[0][0]:.3f})")

    mimi = build_dv_mimi().to(device)
    with torch.no_grad():
        mimi.quantizer.output_proj.weight.copy_(w.T.float()[:, :, None])
        mimi.upsample.convtr.convtr.weight.copy_(triangle(STRIDE)[None, None].expand(512, 1, -1))
    mimi.eval()
    args.out.parent.mkdir(parents=True, exist_ok=True)
    safetensors.torch.save_file(decoder_state(mimi), str(args.out))

    mel = MultiScaleMelLoss(SR).to(device)
    rows = {"bridged": [], "stock round trip": []}
    lines = {"target": [], "bridged": [], "stock round trip": []}
    combs = {k: [] for k in lines}
    for p in sorted(args.val_latents.glob("female_0[25]*.npy"))[: args.val_clips]:
        z = torch.from_numpy(np.load(p).astype(np.float32)).to(device)
        with torch.no_grad():
            target = soxr.resample(codec.decode(z[None])[0].cpu().numpy(), DV_SR, SR, quality="HQ").astype(np.float32)
            bridged = decode_window(mimi, z[None], 0)[0, 0].cpu().numpy()
            zm = stock.encode_to_latent(torch.from_numpy(target)[None, None].to(device))
            rt = decode_window(stock, zm, 0)[0, 0].cpu().numpy()
        # the bridged features at t are the stock ones at t - lag: its output is late by the lag
        if best > 0:
            bridged = bridged[best * 120 :]
        elif best < 0:
            target, rt = target[-best * 120 :], rt[-best * 120 :]
        n = min(len(target), len(bridged), len(rt))
        outs = {"target": target[:n], "bridged": bridged[:n], "stock round trip": rt[:n]}
        for k in rows:
            rows[k].append(float(mel(torch.from_numpy(outs[k])[None].to(device), torch.from_numpy(outs["target"])[None].to(device))))
        for k, v in outs.items():
            lines[k].append(line_contrast(v))
            tmp = args.out.with_suffix(f".{k.replace(' ', '_')}.wav")
            sf.write(str(tmp), v, SR)
            combs[k].extend(comb_frames(tmp))
            tmp.unlink()
    print(f"female_02/05 x JSUT, {len(lines['target'])} clips, before any training:")
    print("| | mel L1 vs target | 200 Hz line dB | comb 2.5-4k | comb 4-6k |")
    print("|---|---:|---:|---:|---:|")
    for k in lines:
        c = np.array(combs[k])
        m = f"{np.mean(rows[k]):.3f}" if k in rows else "-"
        print(f"| {k} | {m} | {np.mean(lines[k]):.2f} | {c[:, 1].mean():.1f} | {c[:, 2].mean():.1f} |")
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
