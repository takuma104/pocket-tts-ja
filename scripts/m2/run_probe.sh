#!/usr/bin/env bash
# usage: scripts/m2/run_probe.sh NAME STEPS [train_mimi_decoder.py options...]
# A short mel-only probe of the Mimi decoder fine-tune on galleria2, from the repo root, for
# listening tests of one change at a time:
#   1. train runs/NAME for STEPS steps (Irodori S3 + Emilia HQ, clips up to 8 s, no discriminators)
#   2. decode the 1,000 JVS reconstruction clips into exp/m2/recon/NAME
#   3. the released student (D1b 120k) on jvs_bal and short with the decoder (tag suffix NAME)
#   4. Kana-CER of the reconstruction next to the recording, the stock Mimi and gs2_80k
# 2 and 3 are what a listening page needs; PROBE_AUDIO_READY is printed after them.
#   nohup setsid scripts/m2/run_probe.sh M2_fine4k 10000 --high-cross 4000 --high-mode fine \
#     > runs/run_probe_M2_fine4k.out 2>&1 < /dev/null &
cd "$(dirname "$0")/../.." || exit 1
NAME=$1; STEPS=$2; shift 2
echo $$ > "runs/run_probe_$NAME.pid"
UV=$(command -v uv || echo ~/.local/bin/uv)
JVS="jvs=data/jvs_ver1/jvs0[0-9][0-9]/parallel100/wav24kHz16bit/VOICEACTRESS100_00[1-5].wav"
DEC=runs/$NAME/decoder_step_$(printf %08d "$STEPS").safetensors
for try in 1 2 3 4; do
  [ -f "$DEC" ] && break  # trained already: only the steps after it are repeated
  PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True $UV run python scripts/m2/train_mimi_decoder.py "runs/$NAME" \
    --data irodori=data/synth/s3_full/manifest.jsonl:data/synth/s3_full/audio:0.5 \
    --data emilia=data/m2/emilia_hq/manifest.jsonl:data/m2/emilia_hq/audio:0.5 \
    --valid-glob "$JVS" --max-steps "$STEPS" --lr 5e-5 --w-adv 0 --w-fm 0 --save-every 2500 --max-seconds 8 \
    --gpu-memory-gb 8 --no-wandb "$@" >> "runs/$NAME.log" 2>&1 < /dev/null
  [ -f "$DEC" ] && break
  echo "$(date) training stopped (try $try): $(grep -a 'Error' "runs/$NAME.log" | tail -1 | cut -c1-200)"; sleep 60
done
[ -f "$DEC" ] || { echo "$(date) PROBE_FAILED"; exit 1; }
echo "$(date) trained: $(grep -a '^valid @' "runs/$NAME.log" | tail -1 | cut -c1-120)"
for try in 1 2 3 4 5 6; do  # the GPU is shared: the decoding fails when it is full
  $UV run python scripts/m2/recon_jvs.py --out exp/m2/recon --decoder "$NAME=$DEC" > "runs/probe_recon_$NAME.log" 2>&1 && break
  sleep 30
done
for s in jvs_bal short; do
  for try in 1 2 3; do
    $UV run python scripts/eval_tts.py runs/D1_6l_b --snapshot 120000 --set $s --voice rotation --cfg 1.0 \
      --mimi-decoder "$DEC" --tag-suffix "$NAME" --no-wandb > "runs/probe_tts_${NAME}_$s.log" 2>&1 < /dev/null && break
    sleep 60
  done
  echo "$(date) student $s: $(tr -d '\n ' < "runs/D1_6l_b/eval/step00120000/${s}_rotation_cfg1.0_$NAME/metrics.json" | cut -c1-260)"
done
echo "$(date) PROBE_AUDIO_READY"
$UV run --project third-party/Joyo-Kanji-Yomi-Benchmark-Parakeet-Edition --extra asr \
  python scripts/asr_transcribe.py "exp/m2/recon/$NAME" --skip-text > "runs/probe_asr_$NAME.log" 2>&1
$UV run python scripts/analyze_first_kana.py --dataset exp/m2/recon/items.jsonl --reference text \
  orig=exp/m2/recon/asr_orig/kana.jsonl mimi=exp/m2/recon/asr_mimi/kana.jsonl gs2_80k=exp/m2/recon/asr_gs2_80k/kana.jsonl \
  "$NAME=exp/m2/recon/asr_$NAME/kana.jsonl" 2>&1 | grep -a "^|" | cut -c1-200
echo "$(date) PROBE_DONE"
