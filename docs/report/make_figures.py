#!/usr/bin/env python3
"""Figures for the technical report (English and Japanese labels).

    uv run --with matplotlib python docs/report/make_figures.py --font-dir docs/report/fonts

Numbers: JKYB-mini accuracy of the teachers from runs/*/eval/step*/jkyb_mini_rotation_cfg2.0
(main line A5_4k / B1_4k, CC line BCC_4k / BCC_4k_tail / BCC_4k_peos), and the share of
JKYB-mini sentences with extra speech after the text (2 kana or more) from
docs/plans/ja_finetune_plan.md (B-CC section, 10/15-10/16).
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager

# (k steps, JKYB-mini accuracy %), CFG 2.0, voices in rotation
MAIN = [(10, 61.1), (15, 65.3), (20, 66.8), (30, 68.8), (40, 71.3), (50, 71.0), (60, 72.1), (70, 71.9),
        (80, 72.0), (90, 72.3), (100, 77.2), (110, 79.5), (120, 81.1), (130, 81.3), (140, 82.1), (150, 84.0),
        (160, 83.5), (170, 83.6), (180, 84.3), (190, 83.2), (200, 83.5), (210, 83.5), (220, 84.8), (230, 84.3),
        (240, 84.2), (250, 87.5), (260, 87.2), (290, 87.1), (300, 88.1), (310, 88.3), (320, 87.5), (330, 87.2),
        (340, 87.5), (350, 87.7), (360, 88.8), (370, 88.3), (380, 89.4), (390, 89.5), (400, 89.2), (410, 90.6),
        (420, 90.5), (430, 90.5), (440, 91.5), (450, 91.8), (460, 91.8), (470, 91.3), (480, 91.9), (490, 91.8),
        (500, 92.3), (510, 92.2), (520, 92.5), (530, 92.9), (540, 91.6), (550, 92.6), (560, 92.5), (570, 90.9),
        (580, 92.0), (590, 92.6), (600, 93.0), (610, 92.1), (620, 92.5), (630, 93.8), (640, 92.7), (650, 92.9),
        (660, 93.2), (670, 91.3), (680, 92.9), (690, 93.3), (700, 93.3)]
CC = [(10, 79.5), (15, 84.1), (20, 86.0), (30, 88.7), (40, 89.8), (50, 89.8), (60, 90.5), (70, 91.0), (80, 90.4),
      (90, 91.2), (100, 91.1), (110, 91.3), (120, 91.3), (130, 92.1), (140, 92.0), (150, 91.3), (160, 91.8),
      (170, 91.1), (180, 91.9), (190, 91.3), (200, 92.1), (210, 92.0), (220, 92.7), (230, 92.2), (240, 91.9),
      (250, 92.0), (260, 92.1), (270, 92.4), (280, 92.9), (290, 92.2), (300, 92.5), (310, 91.9), (400, 91.4),
      (420, 91.7), (440, 91.9), (460, 91.9), (480, 90.9), (500, 91.5), (505, 92.7), (510, 92.4), (520, 93.2),
      (530, 92.6), (540, 91.9), (550, 92.0), (555, 92.3), (560, 92.6), (580, 92.7), (590, 93.9), (600, 93.2),
      (620, 92.7)]
# (k steps, % of JKYB-mini sentences with 2+ kana of extra speech after the text)
EXTRA_MAIN = [(20, 7.7), (50, 2.7), (100, 1.7), (150, 2.5), (200, 3.7), (250, 2.4), (300, 0.4), (700, 1.1)]
EXTRA_CC = [(10, 13.9), (20, 9.1), (50, 6.7), (100, 7.7), (150, 9.1), (200, 8.3), (250, 10.6), (300, 15.2),
            (400, 14.0), (500, 34.4), (505, 8.4), (510, 11.1), (520, 10.5), (530, 11.0), (535, 9.9), (540, 8.8),
            (545, 8.5), (550, 10.9)]
EXTRA_CC_PEOS = [(550, 10.9), (555, 1.9), (560, 0.9), (580, 0.9), (590, 1.1), (600, 3.7), (620, 4.3)]
DECAYS_MAIN = [(600, 660), (660, 700)]
DECAYS_CC = [(400, 500), (560, 620)]

LABELS = {
    "en": {
        "main": "main line (Emilia, CC BY-NC)",
        "cc": "CC BY line",
        "acc": "JKYB-mini accuracy (%)",
        "extra": "extra speech (%)",
        "step": "training step (k)",
        "synth": "synthetic\nspeech added",
        "data": "new synthetic sets",
        "decay": "LR decay",
        "peos": "post-EOS\nsilence",
        "first_decay": "first decay",
        "teacher": "teacher",
        "cc_peos": "CC BY line, post-EOS silence",
    },
    "ja": {
        "main": "本線（Emilia、CC BY-NC）",
        "cc": "CC BY 版",
        "acc": "JKYB-mini の正解率（%）",
        "extra": "余計な発話（%）",
        "step": "学習ステップ（k）",
        "synth": "合成音声を\n追加",
        "data": "合成データの追加",
        "decay": "LR の減衰",
        "peos": "EOS の後の\n無音",
        "first_decay": "最初の減衰",
        "teacher": "教師",
        "cc_peos": "CC BY 版、EOS の後の無音",
    },
}
BLUE, ORANGE = "#1f77b4", "#e07b00"


def xy(points: list[tuple[float, float]]) -> tuple[list[float], list[float]]:
    return [p[0] for p in points], [p[1] for p in points]


def plot(lang: str, output: Path) -> None:
    lab = LABELS[lang]
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(3.4, 3.7), sharex=True, gridspec_kw={"height_ratios": [1.25, 1]})
    for ax in (ax1, ax2):
        for lo, hi in DECAYS_MAIN:
            ax.axvspan(lo, hi, color=BLUE, alpha=0.08, lw=0, zorder=0)
        for lo, hi in DECAYS_CC:
            ax.axvspan(lo, hi, color=ORANGE, alpha=0.10, lw=0, zorder=0)
        ax.tick_params(labelsize=6.5)
    x, y = xy(MAIN)
    ax1.plot(x, y, color=BLUE, lw=1.1, label=lab["main"])
    x, y = xy(CC)
    ax1.plot(x, y, color=ORANGE, lw=1.1, label=lab["cc"])
    ax1.axvline(93, color=BLUE, ls=":", lw=0.8)
    ax1.text(100, 64.5, lab["synth"], fontsize=5.8, color=BLUE, va="bottom", linespacing=0.95)
    for s in (375, 400, 457, 521):
        ax1.plot([s], [86.2], marker="v", ms=3, color=BLUE)
    ax1.text(448, 84.4, lab["data"], fontsize=5.8, color=BLUE, ha="center", va="top")
    ax1.annotate(lab["teacher"], (580, 92.7), xytext=(468, 95.2), fontsize=5.8, color=ORANGE,
                 arrowprops={"arrowstyle": "-", "lw": 0.6, "color": ORANGE})
    ax1.set_ylim(60, 97.5)
    ax1.set_ylabel(lab["acc"], fontsize=6.5)
    ax1.legend(fontsize=6, loc="lower right", frameon=False)
    ax1.text(630, 61.0, lab["decay"], fontsize=5.8, color="#555555", ha="center", va="bottom")

    x, y = xy(EXTRA_MAIN)
    ax2.plot(x, y, color=BLUE, lw=1.0, marker="o", ms=2.2)
    x, y = xy(EXTRA_CC)
    ax2.plot(x, y, color=ORANGE, lw=1.0, marker="o", ms=2.2)
    x, y = xy(EXTRA_CC_PEOS)
    ax2.plot(x, y, color=ORANGE, lw=1.0, marker="s", ms=2.2, ls="--", label=lab["cc_peos"])
    ax2.annotate(lab["first_decay"], (500, 34.4), xytext=(330, 30.5), fontsize=5.8, color=ORANGE,
                 arrowprops={"arrowstyle": "-", "lw": 0.6, "color": ORANGE})
    ax2.annotate(lab["peos"], (556, 2.0), xytext=(585, 14.0), fontsize=5.8, color=ORANGE, linespacing=0.95,
                 arrowprops={"arrowstyle": "-", "lw": 0.6, "color": ORANGE})
    ax2.set_ylim(0, 37)
    ax2.set_ylabel(lab["extra"], fontsize=6.5)
    ax2.set_xlabel(lab["step"], fontsize=6.5)
    ax2.set_xlim(0, 705)
    ax2.legend(fontsize=6, loc="upper left", frameon=False)
    fig.tight_layout(h_pad=0.5)
    fig.savefig(output)
    plt.close(fig)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--font-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, default=Path(__file__).parent)
    args = parser.parse_args()
    for path in args.font_dir.glob("NotoSansJP-*.otf"):
        font_manager.fontManager.addfont(str(path))
    plt.rcParams["pdf.fonttype"] = 42
    plt.rcParams["font.family"] = "DejaVu Sans"
    plot("en", args.output_dir / "fig_training_en.pdf")
    # Noto Sans JP is CFF-based; embed it as Type 3 (Type 42 needs TrueType outlines).
    plt.rcParams["pdf.fonttype"] = 3
    plt.rcParams["font.family"] = "Noto Sans JP"
    plot("ja", args.output_dir / "fig_training_ja.pdf")
    print("written fig_training_{en,ja}.pdf")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
