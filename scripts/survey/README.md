# Emilia JA survey scripts (2026-10-09)

One-off scripts behind the numbers in `docs/reports/2026-10-09_precheck.md` §2. They were
run from a throwaway environment (pandas, speechbrain, transformers) and write their outputs
next to themselves (`OUT = Path(__file__).parent`); the results were copied to
`exp/emilia_survey/` (gitignored). `scan_meta.py` builds `emilia_ja_meta.parquet`, the
metadata index (with tar offsets) that the data preparation step starts from.
