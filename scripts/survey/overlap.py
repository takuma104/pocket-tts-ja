from __future__ import annotations
from pathlib import Path
import pandas as pd

O = Path(__file__).resolve().parent
df = pd.read_parquet(O / "emilia_ja_meta.parquet")
df["k"] = df["text"] + "|" + df["duration"].round(2).astype(str) + "|" + df["audio_size"].astype(str)
e = df[df["subset"] == "Emilia"].copy()
y = df[df["subset"] == "Emilia-YODAS"].copy()
e["b"] = e["speaker"].str[:9]
ky = set(y["k"])
e["in_yodas"] = e["k"].isin(ky)
print("Emilia per batch: utts, hours, frac utts byte-identical in YODAS")
print(e.groupby("b").agg(utts=("k", "size"), hours=("duration", lambda x: round(x.sum() / 3600, 1)),
                         frac_in_yodas=("in_yodas", "mean"), bitrate_kbps=("audio_size", "size")).drop(columns="bitrate_kbps").round(3).to_string())
# source mapping
m = e[e["in_yodas"]][["source", "k"]].merge(y[["k", "source", "speaker"]].rename(columns={"source": "video", "speaker": "yspk"}), on="k")
pairs = m.groupby("source")["video"].nunique()
print(f"Emilia sources with overlap: {pairs.size:,}; mapping to #videos: {pairs.value_counts().sort_index().head(5).to_dict()}")
vids = m.groupby("video")["source"].nunique()
print(f"YODAS videos with overlap: {vids.size:,}; mapped from #Emilia sources: {vids.value_counts().sort_index().head(5).to_dict()}")
ysp = m.groupby("source")["yspk"].nunique()
print(f"#YODAS diarized speakers inside one Emilia source (overlapping sources): {ysp.value_counts().sort_index().head(8).to_dict()}")
m[["source", "video"]].drop_duplicates().to_csv(O / "emilia_source_to_yodas_video.tsv", sep="\t", index=False)
# combined dedup
both = df.drop_duplicates("k")
print(f"combined utts {len(df):,} -> unique {len(both):,}; hours {df['duration'].sum() / 3600:,.1f} -> {both['duration'].sum() / 3600:,.1f}")
# what about YODAS-only part
y["in_emilia"] = y["k"].isin(set(e["k"]))
print(f"YODAS utts NOT in Emilia: {(~y['in_emilia']).sum():,} ({y.loc[~y['in_emilia'], 'duration'].sum() / 3600:,.1f} h); per shard frac in Emilia:")
print(y.groupby("shard")["in_emilia"].mean().round(2).to_dict())
# save flags back to table
df["dup_key"] = df["k"]
df["is_first_copy"] = ~df.duplicated("k", keep="first")
df["emilia_in_yodas"] = df["k"].isin(ky) & (df["subset"] == "Emilia")
df["yodas_in_emilia"] = df["k"].isin(set(e["k"])) & (df["subset"] == "Emilia-YODAS")
df.drop(columns=["k"]).to_parquet(O / "emilia_ja_meta.parquet", index=False)
print("dedup unique (first copy) per subset hours:", df[df["is_first_copy"]].groupby("subset")["duration"].sum().div(3600).round(1).to_dict())
