#!/usr/bin/env bash
# usage: scripts/m2/run_gan_keep.sh NAME FROM_RUN LAST_STEP [train_mimi_decoder.py options...]
# galleria2, from the repo root: continue an adversarial decoder run (FROM_RUN's checkpoint, with
# its discriminators) to LAST_STEP while the band above 4 kHz is held at the stock decoder's
# (plan M2, 10/14: a decoder that raises that band to the recording's level fills it with noise
# in voiced speech). The mel loss against the recording, the shared-spectrum loss and what the
# discriminators see are all limited to the band below 4 kHz. Batch 8: the GPU is shared.
# Then the checks of scripts/m2/run_probe.sh plus JKYB-mini, the scoring jobs under the lock.
#   nohup setsid scripts/m2/run_gan_keep.sh M2_gs2gan_keep4k M2_gan_spec2 90000 \
#     > runs/run_gan_keep_M2_gs2gan_keep4k.out 2>&1 < /dev/null &
cd "$(dirname "$0")/../.." || exit 1
NAME=$1; FROM=$2; STEPS=$3; shift 3
echo $$ > "runs/run_gan_keep_$NAME.pid"
UV=$(command -v uv || echo ~/.local/bin/uv)
LOCK="flock runs/extra_gpu.lock"
JVS="jvs=data/jvs_ver1/jvs0[0-9][0-9]/parallel100/wav24kHz16bit/VOICEACTRESS100_00[1-5].wav"
DEC=runs/$NAME/decoder_step_$(printf %08d "$STEPS").safetensors
mkdir -p "runs/$NAME"
[ -f "runs/$NAME/checkpoint.pt" ] || cp "runs/$FROM/checkpoint.pt" "runs/$NAME/checkpoint.pt" || exit 1
for try in 1 2 3 4 5 6 7 8; do
  [ -f "$DEC" ] && break
  PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True $UV run python scripts/m2/train_mimi_decoder.py "runs/$NAME" \
    --data irodori=data/synth/s3_full/manifest.jsonl:data/synth/s3_full/audio:0.5 \
    --data emilia=data/m2/emilia_hq/manifest.jsonl:data/m2/emilia_hq/audio:0.5 \
    --valid-glob "$JVS" --max-steps "$STEPS" --lr 8e-5 --lr-d 4e-4 --batch-size 8 --save-every 2500 --valid-every 500 \
    --max-seconds 8 --gpu-memory-gb 9 --w-spec 200 --spec-band 300 4000 \
    --high-cross 4000 --high-mode stock --w-high 15 --gan-lowpass 4000 --no-wandb "$@" >> "runs/$NAME.log" 2>&1 < /dev/null
  [ -f "$DEC" ] && break
  echo "$(date) training stopped (try $try): $(tr '\r' '\n' < "runs/$NAME.log" | grep -a '^step' | tail -1 | cut -c1-40) / $(grep -a 'Error' "runs/$NAME.log" | tail -1 | cut -c1-160)"
  sleep 120
done
[ -f "$DEC" ] || { echo "$(date) GAN_KEEP_FAILED"; exit 1; }
echo "$(date) trained: $(grep -a '^valid @' "runs/$NAME.log" | tail -1 | cut -c1-120)"
for try in 1 2 3 4 5 6; do
  $UV run python scripts/m2/recon_jvs.py --out exp/m2/recon --decoder "$NAME=$DEC" > "runs/probe_recon_$NAME.log" 2>&1 && break
  sleep 60
done
$UV run python scripts/m2/hf_structure.py exp/m2/recon orig mimi gs2_80k M2_gs2keep4k "$NAME" --per-speaker 6 2>&1 | grep -a "^|\|frames" | cut -c1-200
for s in jvs_bal short jkyb_mini; do
  for try in 1 2 3 4; do
    $LOCK $UV run python scripts/eval_tts.py runs/D1_6l_b --snapshot 120000 --set $s --voice rotation --cfg 1.0 \
      --mimi-decoder "$DEC" --tag-suffix "$NAME" --no-wandb > "runs/probe_tts_${NAME}_$s.log" 2>&1 < /dev/null && break
    sleep 120
  done
  echo "$(date) student $s: $(tr -d '\n ' < "runs/D1_6l_b/eval/step00120000/${s}_rotation_cfg1.0_$NAME/metrics.json" | cut -c1-260)"
done
echo "$(date) GAN_KEEP_AUDIO_READY"
for try in 1 2 3 4; do
  $LOCK $UV run --project third-party/Joyo-Kanji-Yomi-Benchmark-Parakeet-Edition --extra asr \
    python scripts/asr_transcribe.py "exp/m2/recon/$NAME" --skip-text > "runs/probe_asr_$NAME.log" 2>&1 && break
  sleep 120
done
$UV run python scripts/analyze_first_kana.py --dataset exp/m2/recon/items.jsonl --reference text \
  orig=exp/m2/recon/asr_orig/kana.jsonl mimi=exp/m2/recon/asr_mimi/kana.jsonl gs2_80k=exp/m2/recon/asr_gs2_80k/kana.jsonl \
  M2_gs2keep4k=exp/m2/recon/asr_M2_gs2keep4k/kana.jsonl "$NAME=exp/m2/recon/asr_$NAME/kana.jsonl" 2>&1 | grep -a "^|" | cut -c1-200
$LOCK $UV run python scripts/m2/recon_similarity.py exp/m2/recon mimi gs2_80k M2_gs2keep4k "$NAME" 2>&1 | grep -a "^|"
echo "$(date) GAN_KEEP_DONE"
