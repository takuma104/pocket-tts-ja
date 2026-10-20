#!/usr/bin/env bash
# Build the technical report PDFs (English and Japanese) with Tectonic.
#
#   bash docs/report/build.sh
#
# Needs `tectonic` (https://tectonic-typesetting.github.io; set TECTONIC to its
# path if it is not on PATH, e.g. TECTONIC=~/bin/tectonic). The Noto Serif/Sans JP
# fonts are downloaded into docs/report/fonts/ on the first run (not tracked). The
# figures are made with `make_figures.py` (matplotlib).
set -euo pipefail
cd "$(dirname "$0")"
TECTONIC=${TECTONIC:-tectonic}

mkdir -p fonts
for f in Serif/SubsetOTF/JP/NotoSerifJP-Regular.otf Serif/SubsetOTF/JP/NotoSerifJP-Bold.otf \
  Sans/SubsetOTF/JP/NotoSansJP-Regular.otf Sans/SubsetOTF/JP/NotoSansJP-Bold.otf; do
  [ -f "fonts/$(basename "$f")" ] || curl -sfL -o "fonts/$(basename "$f")" \
    "https://github.com/notofonts/noto-cjk/raw/main/$f"
done

if [ ! -f fig_training_en.pdf ] || [ ! -f fig_training_ja.pdf ] || [ make_figures.py -nt fig_training_en.pdf ]; then
  (cd ../.. && uv run --with matplotlib python docs/report/make_figures.py --font-dir docs/report/fonts)
fi
for lang in en ja; do
  "$TECTONIC" --keep-logs "report_$lang.tex" > "build_$lang.log" 2>&1 || {
    tail -20 "build_$lang.log"
    exit 1
  }
done
mv report_en.pdf pocket_tts_ja_report_en.pdf
mv report_ja.pdf pocket_tts_ja_report_ja.pdf
echo "built pocket_tts_ja_report_{en,ja}.pdf"
