from __future__ import annotations
from pathlib import Path
import pandas as pd

O = Path(__file__).resolve().parent
df = pd.read_parquet(O / "emilia_ja_meta.parquet")
for sub, g in df.groupby("subset"):
    g = g.copy()
    hours = g["duration"].sum() / 3600
    g["k_exact"] = g["text"] + "|" + g["duration"].round(3).astype(str)
    first = ~g.duplicated("k_exact", keep="first")
    n_dupe = (~first).sum()
    print(f"===== {sub}: utts={len(g):,} hours={hours:,.1f}")
    print(f"exact (text,duration) duplicates beyond first copy: {n_dupe:,} utts ({n_dupe / len(g):.1%}), "
          f"{g.loc[~first, 'duration'].sum() / 3600:,.1f} h; unique keys {first.sum():,}")
    # also text-only duplicates for texts >= 15 chars (likely same recording/re-upload or same script)
    long = g["text"].str.replace(r"\s", "", regex=True).str.len() >= 15
    ft = ~g[long].duplicated("text", keep="first")
    print(f"text-only duplicates (texts>=15 chars) beyond first: {(~ft).sum():,} utts ({(~ft).sum() / len(g):.1%}), "
          f"{g[long].loc[~ft, 'duration'].sum() / 3600:,.1f} h")
    # source-level: fraction of a source's utts whose exact key appears in another source
    cnt_src = g.groupby("k_exact")["source"].nunique()
    g["shared"] = g["k_exact"].map(cnt_src) > 1
    per = g.groupby("source").agg(n=("shared", "size"), frac=("shared", "mean"), h=("duration", lambda x: x.sum() / 3600))
    for th in [0.5, 0.8, 0.95]:
        sel = per[per["frac"] >= th]
        print(f"sources with >= {th:.0%} of utts exactly shared with another source: {len(sel):,} / {len(per):,} ({sel['h'].sum():,.1f} h)")
    # dedup estimate: keep first of each exact key
    print(f"hours after exact dedup: {g.loc[first, 'duration'].sum() / 3600:,.1f}")
    # cross-batch duplicates
    if sub == "Emilia":
        g["b"] = g["speaker"].str[:9]
        nb = g.groupby("k_exact")["b"].nunique()
        print(f"exact keys spanning >1 batch: {(nb > 1).sum():,}")
    # cross-subset duplicates
other = {s: set((g2["text"] + "|" + g2["duration"].round(2).astype(str))) for s, g2 in df.groupby("subset")}
print("exact (text,dur@10ms) keys shared between Emilia and YODAS:", len(other["Emilia"] & other["Emilia-YODAS"]))
