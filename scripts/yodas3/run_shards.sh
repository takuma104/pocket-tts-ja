#!/usr/bin/env bash
# Y1 over every downloaded YODAS3 ja shard that has not been processed yet, one at a time.
# Usage: scripts/yodas3/run_shards.sh OUT_DIR [extra process_shard.py args...]
set -u
cd "$(dirname "$0")/../.."
OUT=$1; shift
UV=${UV:-$HOME/.local/bin/uv}
ROOT=/mnt/artifacts/yodas3/data/ja
while true; do
  todo=""
  for tar in $(ls $ROOT/audio/*.tar 2>/dev/null | sort); do
    s=$(basename $tar .tar)
    [ -f $OUT/utts/$s.jsonl ] && continue
    [ -f $OUT/.claim_$s ] && continue
    todo=$s; break
  done
  if [ -z "$todo" ]; then
    # nothing left that is downloaded; stop once all 108 metadata shards are done
    [ $(ls $OUT/utts/*.jsonl 2>/dev/null | wc -l) -ge $(ls $ROOT/metadata/*.parquet | wc -l) ] && exit 0
    sleep 300; continue
  fi
  touch $OUT/.claim_$todo
  $UV run python scripts/yodas3/process_shard.py --shard $todo --out $OUT "$@" > $OUT/process_$todo.log 2>&1
  rm -f $OUT/.claim_$todo
done
