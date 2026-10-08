"""Speaker-label sanity check: listing of example speakers + embedding cosine similarity.

Reads audio by random access using (shard, audio_offset, audio_size) from emilia_ja_meta.parquet.
Embeddings: microsoft/wavlm-base-plus-sv (HF transformers) and speechbrain/spkrec-ecapa-voxceleb.
"""

from __future__ import annotations

import io
import json
from itertools import combinations
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import soundfile as sf
import soxr
import torch
from speechbrain.inference.speaker import EncoderClassifier
from transformers import AutoFeatureExtractor, WavLMForXVector

OUT = Path(__file__).resolve().parent
ROOT = Path("/mnt/ext11/Emilia-Dataset")
SUBDIR: dict[str, Path] = {"Emilia": ROOT / "Emilia" / "JA", "Emilia-YODAS": ROOT / "Emilia-YODAS" / "JA"}
N_SPK = 30
N_UTT = 5
MAX_SEC = 15.0
SEED = 1234
THRESH: dict[str, float] = {"wavlm": 0.86, "ecapa": 0.25}


def read_audio(subset: str, shard: str, offset: int, size: int) -> tuple[np.ndarray, int]:
    with open(SUBDIR[subset] / shard, "rb") as f:
        f.seek(offset)
        b = f.read(size)
    x, sr = sf.read(io.BytesIO(b), dtype="float32")
    if x.ndim > 1:
        x = x.mean(axis=1)
    return x, sr


class Embedders:
    def __init__(self) -> None:
        self.fe = AutoFeatureExtractor.from_pretrained("microsoft/wavlm-base-plus-sv")
        self.wavlm = WavLMForXVector.from_pretrained("microsoft/wavlm-base-plus-sv").cuda().eval()
        self.ecapa = EncoderClassifier.from_hparams(
            source="speechbrain/spkrec-ecapa-voxceleb", savedir=str(OUT / "models" / "ecapa"), run_opts={"device": "cuda:0"}
        )

    @torch.no_grad()
    def __call__(self, x16: np.ndarray) -> dict[str, np.ndarray]:
        inp = self.fe([x16], sampling_rate=16000, return_tensors="pt")
        e1 = self.wavlm(**{k: v.cuda() for k, v in inp.items()}).embeddings[0]
        e2 = self.ecapa.encode_batch(torch.from_numpy(x16)[None].cuda()).reshape(-1)
        return {
            "wavlm": torch.nn.functional.normalize(e1, dim=-1).cpu().numpy(),
            "ecapa": torch.nn.functional.normalize(e2, dim=-1).cpu().numpy(),
        }


def sample_speakers(df: pd.DataFrame, rng: np.random.Generator) -> pd.DataFrame:
    rows: list[pd.DataFrame] = []
    for subset in ["Emilia", "Emilia-YODAS"]:
        g = df[(df["subset"] == subset) & (df["duration"] >= 2.0)].drop_duplicates(["speaker", "dup_key"])
        cnt = g.groupby("speaker").size()
        elig = cnt[cnt >= N_UTT].index.to_numpy()
        spk = rng.choice(elig, size=N_SPK, replace=False)
        for s in spk:
            gs = g[g["speaker"] == s]
            take = gs.iloc[rng.choice(len(gs), size=N_UTT, replace=False)].copy()
            take["group"] = "random>=5"
            rows.append(take)
        # 3 largest speakers: 10 utts each
        for s in cnt.sort_values(ascending=False).index[:3]:
            gs = g[g["speaker"] == s]
            take = gs.iloc[rng.choice(len(gs), size=10, replace=False)].copy()
            take["group"] = "top3"
            rows.append(take)
        # 10 small speakers (2-4 utts): 2 utts each
        small = cnt[(cnt >= 2) & (cnt <= 4)].index.to_numpy()
        for s in rng.choice(small, size=10, replace=False):
            gs = g[g["speaker"] == s]
            take = gs.iloc[rng.choice(len(gs), size=2, replace=False)].copy()
            take["group"] = "small2-4"
            rows.append(take)
        if subset == "Emilia-YODAS":
            # videos with >=2 diarized speakers, each with >=3 utts: same-source different-speaker pairs
            g2 = g.copy()
            c2 = g2.groupby(["source", "speaker"]).size().reset_index(name="n")
            c2 = c2[c2["n"] >= 3]
            multi = c2.groupby("source").size()
            vids = rng.choice(multi[multi >= 2].index.to_numpy(), size=15, replace=False)
            for v in vids:
                for s in c2[c2["source"] == v]["speaker"].iloc[:2]:
                    gs = g[g["speaker"] == s]
                    take = gs.iloc[rng.choice(len(gs), size=3, replace=False)].copy()
                    take["group"] = "yodas_multispk_video"
                    rows.append(take)
    return pd.concat(rows, ignore_index=True)


def summarize(vals: list[float]) -> dict[str, float]:
    a = np.asarray(vals)
    if len(a) == 0:
        return {}
    return {"n": int(len(a)), "mean": float(a.mean()), **{f"p{p}": float(np.percentile(a, p)) for p in [1, 5, 10, 25, 50, 75, 90]}}


def eer(pos: np.ndarray, neg: np.ndarray) -> tuple[float, float]:
    ths = np.unique(np.concatenate([pos, neg]))
    best = (1.0, 0.0)
    for t in ths:
        frr = (pos < t).mean()
        far = (neg >= t).mean()
        if abs(frr - far) < best[0]:
            best = (abs(frr - far), float(t))
    t = best[1]
    return t, float(((pos < t).mean() + (neg >= t).mean()) / 2)


def main() -> None:
    rng = np.random.default_rng(SEED)
    df = pd.read_parquet(OUT / "emilia_ja_meta.parquet")
    samp = sample_speakers(df, rng).drop_duplicates(subset=["subset", "id"]).reset_index(drop=True)
    print(f"sampled {len(samp)} utts", flush=True)
    emb = Embedders()
    E: dict[str, list[np.ndarray]] = {"wavlm": [], "ecapa": []}
    for i, r in samp.iterrows():
        x, sr = read_audio(r["subset"], r["shard"], int(r["audio_offset"]), int(r["audio_size"]))
        x16 = soxr.resample(x, sr, 16000)[: int(MAX_SEC * 16000)].astype(np.float32)
        e = emb(x16)
        for k in E:
            E[k].append(e[k])
        if i % 50 == 0:
            print(i, flush=True)
    res: dict[str, Any] = {}
    pair_rows: list[dict[str, Any]] = []
    for model in E:
        M = np.stack(E[model])
        S = M @ M.T
        for subset in ["Emilia", "Emilia-YODAS"]:
            idx = samp.index[samp["subset"] == subset].to_numpy()
            within: dict[str, list[float]] = {}
            cross_any: list[float] = []
            cross_same_src: list[float] = []
            for a, b in combinations(idx, 2):
                ra, rb = samp.loc[a], samp.loc[b]
                s = float(S[a, b])
                if ra["speaker"] == rb["speaker"]:
                    within.setdefault(ra["group"], []).append(s)
                    pair_rows.append({"model": model, "subset": subset, "group": ra["group"], "speaker": ra["speaker"],
                                      "id_a": ra["id"], "id_b": rb["id"], "cos": s})
                else:
                    if ra["source"] == rb["source"]:
                        cross_same_src.append(s)
                    elif ra["group"] == "random>=5" and rb["group"] == "random>=5":
                        cross_any.append(s)
            th = THRESH[model]
            out: dict[str, Any] = {}
            for grp, v in within.items():
                arr = np.asarray(v)
                out[f"within[{grp}]"] = summarize(v) | {f"frac<{th}": float((arr < th).mean())}
            out["cross_diff_source"] = summarize(cross_any) | {f"frac>={th}": float((np.asarray(cross_any) >= th).mean())}
            if cross_same_src:
                out["cross_same_source_diff_spk"] = summarize(cross_same_src) | {
                    f"frac>={th}": float((np.asarray(cross_same_src) >= th).mean())}
            pos = np.asarray(within.get("random>=5", []))
            t, e = eer(pos, np.asarray(cross_any))
            out["eer_threshold_within_vs_cross"] = t
            out["eer"] = e
            out[f"frac_within_below_eer_th"] = float((pos < t).mean())
            # per-speaker: fraction of sampled speakers with any within pair below threshold
            pr = pd.DataFrame([p for p in pair_rows if p["model"] == model and p["subset"] == subset and p["group"] == "random>=5"])
            per = pr.groupby("speaker")["cos"].min()
            out[f"speakers_with_min_pair<{th}"] = f"{int((per < th).sum())}/{len(per)}"
            out["speakers_with_min_pair<eer_th"] = f"{int((per < t).sum())}/{len(per)}"
            res[f"{model}/{subset}"] = out
    (OUT / "spk_check_results.json").write_text(json.dumps(res, indent=2, ensure_ascii=False))
    pd.DataFrame(pair_rows).to_csv(OUT / "spk_check_pairs.tsv", sep="\t", index=False)
    samp.drop(columns=[c for c in samp.columns if c.startswith("audio")]).to_csv(OUT / "spk_check_utts.tsv", sep="\t", index=False)
    print(json.dumps(res, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
