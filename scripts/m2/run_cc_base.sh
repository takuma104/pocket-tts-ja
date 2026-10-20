#!/usr/bin/env bash
# usage: scripts/m2/run_cc_base.sh NAME [train_mimi_decoder.py options for both steps...]
# The CC line's base decoder (docs/plans/mimi_ft_cc_plan.md, C2): the chain stock Mimi -> M2_gan_spec(2) -> keep4k
# redone with CC BY audio only, discriminators included:
#   1. runs/${NAME}_gs: 60k steps from the stock decoder, mel + MPD / MS-STFT discriminators + the shared-spectrum loss
#      over 300 Hz-12 kHz (M2_gan_spec with M2_gan_spec2's band from the start), on the CC voices' Irodori synthesis
#      (s3_cc) and YODAS3 (extract_yodas3.py) half and half, where the chain had s3_full and Emilia HQ (both NC)
#   2. runs/$NAME: 10k more steps (its checkpoint, discriminators and all) with the band above 4 kHz held at the stock
#      decoder's, as run_gan_keep.sh made keep4k
# then the JVS reconstructions of both (as ${NAME}_gs and $NAME), MOS, Kana-CER and similarity, as run_hf.sh.
# YODAS3 clips longer than --max-seconds are kept with their first 8 s (the trainer drops longer rows otherwise);
# rows without a wav (failed extraction) and the videos of scripts/m2/yodas3_exclude.txt are left out.
cd "$(dirname "$0")/../.." || exit 1
NAME=$1; shift
OLD=${OLD:-/home/takuma/co/pocket-tts-ja}
UV=$(command -v uv || echo ~/.local/bin/uv)
JVS="jvs=data/jvs_ver1/jvs0[0-9][0-9]/parallel100/wav24kHz16bit/VOICEACTRESS100_00[1-5].wav"
Y3=${Y3:-data/m2/yodas3/manifest.jsonl}
S3="irodori=$OLD/data/synth/s3_cc/manifest.jsonl:$OLD/data/synth/s3_cc/audio:0.5"
echo $$ > "runs/run_cc_base_$NAME.pid"
python3 -c "
import json, os, sys
excluded = {line.split('#')[0].strip() for line in open('scripts/m2/yodas3_exclude.txt')} - {''}
with open(sys.argv[2], 'w') as f:
    for line in open(sys.argv[1]):
        row = json.loads(line)
        if os.path.exists(f'data/m2/yodas3/audio/{row[\"id\"]}.wav') and row['video'] not in excluded:
            f.write(json.dumps({'id': row['id'], 'duration': min(row['duration'], 7.99)}) + '\n')
" "$Y3" "runs/${NAME}_yodas3.jsonl" || exit 1
Y="yodas3=runs/${NAME}_yodas3.jsonl:data/m2/yodas3/audio:0.5"
train() {  # RUN LAST_STEP options...
  local run=$1 last=$2; shift 2
  local dec=runs/$run/decoder_step_$(printf %08d "$last").safetensors
  for try in 1 2 3 4 5 6 7 8; do
    [ -f "$dec" ] && return 0
    PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True $UV run python scripts/m2/train_mimi_decoder.py "runs/$run" \
      --data "$S3" --data "$Y" --valid-glob "$JVS" --max-steps "$last" --max-seconds 8 --no-wandb "$@" \
      >> "runs/$run.log" 2>&1 < /dev/null
    [ -f "$dec" ] && return 0
    echo "$(date) $run stopped (try $try): $(grep -a 'Error' "runs/$run.log" | tail -1 | cut -c1-200)"; sleep 120
  done
  return 1
}
train "${NAME}_gs" 60000 --lr 3e-5 --save-every 10000 --valid-every 1000 --gpu-memory-gb 14 \
  --w-spec 200 --spec-band 300 12000 "$@" || { echo "$(date) CC_BASE_FAILED"; exit 1; }
echo "$(date) step 1: $(grep -a '^valid @' "runs/${NAME}_gs.log" | tail -1 | cut -c1-200)"
mkdir -p "runs/$NAME"
[ -f "runs/$NAME/checkpoint.pt" ] || cp "runs/${NAME}_gs/checkpoint.pt" "runs/$NAME/checkpoint.pt" || exit 1
train "$NAME" 70000 --lr 8e-5 --lr-d 4e-4 --batch-size 8 --save-every 2500 --valid-every 500 --gpu-memory-gb 9 \
  --w-spec 200 --spec-band 300 4000 --high-cross 4000 --high-mode stock --w-high 15 --gan-lowpass 4000 "$@" \
  || { echo "$(date) CC_BASE_FAILED"; exit 1; }
echo "$(date) step 2: $(grep -a '^valid @' "runs/$NAME.log" | tail -1 | cut -c1-200)"
GS="${NAME}_gs"
for try in 1 2 3 4; do
  $UV run python scripts/m2/recon_jvs.py --out exp/m2/recon --decoder "$GS=runs/$GS/decoder_step_00060000.safetensors" \
    --decoder "$NAME=runs/$NAME/decoder_step_00070000.safetensors" > "runs/cc_recon_$NAME.log" 2>&1 && break
  sleep 30
done
echo "$(date) CC_AUDIO_READY"
flock runs/.mos.lock $UV run python scripts/m2/mos_eval.py --highpass 40 --workers 12 exp/m2/recon orig mimi gs2_80k keep4k \
  "$GS" "$NAME" --ref keep4k > "runs/cc_mos_$NAME.txt" 2>&1
echo "$(date) CC_MOS_DONE"
for try in 1 2 3; do
  $UV run --project third-party/Joyo-Kanji-Yomi-Benchmark-Parakeet-Edition --extra asr \
    python scripts/asr_transcribe.py "exp/m2/recon/$GS" "exp/m2/recon/$NAME" --skip-text > "runs/cc_asr_$NAME.log" 2>&1 && break
  sleep 60
done
$UV run python scripts/analyze_first_kana.py --dataset exp/m2/recon/items.jsonl --reference text \
  orig=exp/m2/recon/asr_orig/kana.jsonl mimi=exp/m2/recon/asr_mimi/kana.jsonl gs2_80k=exp/m2/recon/asr_gs2_80k/kana.jsonl keep4k=exp/m2/recon/asr_keep4k/kana.jsonl \
  "$GS=exp/m2/recon/asr_$GS/kana.jsonl" "$NAME=exp/m2/recon/asr_$NAME/kana.jsonl" > "runs/cc_kana_$NAME.txt" 2>&1
$UV run python scripts/m2/recon_similarity.py exp/m2/recon mimi gs2_80k keep4k "$GS" "$NAME" > "runs/cc_similarity_$NAME.txt" 2>&1
echo "$(date) CC_BASE_DONE"
