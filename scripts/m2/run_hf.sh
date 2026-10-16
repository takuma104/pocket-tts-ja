#!/usr/bin/env bash
# usage: scripts/m2/run_hf.sh NAME [train_mimi_decoder.py options...]
# Plan M2 high band (docs/plans/mimi_ft_hf_plan.md), from the repo root: one Phase 1 run and its checks.
#   1. train runs/NAME for 10k steps from the adopted decoder (M2_gs2gan_keep4k) and its discriminators,
#      on D-nat (Emilia HQ 50%, Hi-Fi-CAPTAIN F 25 / M 15, JVNV 10); the high-band options come from the caller
#   2. decode the 1,000 JVS reconstruction clips with the half-way and final decoders (exp/m2/recon/NAME_5k, NAME;
#      the half-way one is named by its step: STEPS=20000 gives NAME_10k)
#   3. high-band structure (hf_structure.py --own-f0), fixed-frequency lines (periodicity.py) and
#      amplitude modulation locked to the 5 ms frame grid (frame_modulation.py)
#   4. neural MOS (UTMOS22 strong, UTMOSv2; mos_eval.py, every file high-passed at 40 Hz first)
#   5. Kana-CER and first-kana errors (kana-whisper), speaker similarity to the recordings
# QUICK=1 skips step 3.
#     nohup setsid scripts/m2/run_hf.sh M2H_H1 --high-cross 4000 --high-mode fine --gan-lowpass 4000 \
#       --spec-band 300 4000 > runs/run_hf_M2H_H1.out 2>&1 < /dev/null &
# OLD: the earlier working folder of this machine (the adopted run's checkpoint with its discriminators).
cd "$(dirname "$0")/../.." || exit 1
NAME=$1; shift
OLD=${OLD:-/home/takuma/co/pocket-tts-ja}
STEPS=${STEPS:-10000}
# the sources' shares of the batches (W_EMILIA=0.1 W_HFC_F=0.45 ... to change them)
W_EMILIA=${W_EMILIA:-0.5} W_HFC_F=${W_HFC_F:-0.25} W_HFC_M=${W_HFC_M:-0.15} W_JVNV=${W_JVNV:-0.1}
echo $$ > "runs/run_hf_$NAME.pid"
UV=$(command -v uv || echo ~/.local/bin/uv)
JVS="jvs=data/jvs_ver1/jvs0[0-9][0-9]/parallel100/wav24kHz16bit/VOICEACTRESS100_00[1-5].wav"
INIT=runs/M2_gs2gan_keep4k/decoder_step_00090000.safetensors
DEC=runs/$NAME/decoder_step_$(printf %08d "$STEPS").safetensors
HALF=runs/$NAME/decoder_step_$(printf %08d $((STEPS / 2))).safetensors
H="${NAME}_$((STEPS / 2000))k"  # the half-way decoder
for try in 1 2 3 4; do
  [ -f "$DEC" ] && break
  PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True $UV run python scripts/m2/train_mimi_decoder.py "runs/$NAME" \
    --data "emilia=data/m2/emilia_hq/manifest.jsonl:data/m2/emilia_hq/audio:$W_EMILIA" \
    --data "hfc_f=data/m2/hfc_f/manifest.jsonl:data/m2/hfc_f/audio:$W_HFC_F" \
    --data "hfc_m=data/m2/hfc_m/manifest.jsonl:data/m2/hfc_m/audio:$W_HFC_M" \
    --data "jvnv=data/m2/jvnv/manifest.jsonl:data/m2/jvnv/audio:$W_JVNV" \
    --valid-glob "$JVS" --init "$INIT" --init-disc "$OLD/runs/M2_gs2gan_keep4k/checkpoint.pt" \
    --max-steps "$STEPS" --d-warmup 1000 --warmup 500 --lr 5e-5 --lr-d 2e-4 --batch-size 16 --max-seconds 8 \
    --save-every 2500 --valid-every 500 --w-spec 200 --no-wandb "$@" >> "runs/$NAME.log" 2>&1 < /dev/null
  [ -f "$DEC" ] && break
  echo "$(date) training stopped (try $try): $(grep -a 'Error' "runs/$NAME.log" | tail -1 | cut -c1-200)"; sleep 60
done
[ -f "$DEC" ] || { echo "$(date) HF_FAILED"; exit 1; }
echo "$(date) trained: $(grep -a '^valid @' "runs/$NAME.log" | tail -1 | cut -c1-300)"
for try in 1 2 3 4; do
  $UV run python scripts/m2/recon_jvs.py --out exp/m2/recon --decoder "$H=$HALF" --decoder "$NAME=$DEC" \
    > "runs/hf_recon_$NAME.log" 2>&1 && break
  sleep 30
done
if [ -z "$QUICK" ]; then
  $UV run python scripts/m2/hf_structure.py exp/m2/recon orig mimi keep4k "$H" "$NAME" --per-speaker 6 --own-f0 \
    > "runs/hf_structure_$NAME.txt" 2>&1
  $UV run python scripts/m2/periodicity.py exp/m2/recon orig mimi keep4k "$NAME" > "runs/hf_periodicity_$NAME.txt" 2>&1
  $UV run python scripts/m2/frame_modulation.py exp/m2/recon orig keep4k M2H_H2 "$H" "$NAME" --gender F > "runs/hf_modulation_$NAME.txt" 2>&1
fi
echo "$(date) HF_AUDIO_READY"
# one MOS job at a time: each takes ~50 GB of RAM with its data loader workers (two at once ran the machine out of memory)
flock runs/.mos.lock $UV run python scripts/m2/mos_eval.py --highpass 40 --workers 12 exp/m2/recon orig keep4k M2H_H2gh \
  "$H" "$NAME" --ref keep4k > "runs/hf_mos_$NAME.txt" 2>&1
echo "$(date) HF_MOS_DONE"
for try in 1 2 3; do
  $UV run --project third-party/Joyo-Kanji-Yomi-Benchmark-Parakeet-Edition --extra asr \
    python scripts/asr_transcribe.py "exp/m2/recon/$H" "exp/m2/recon/$NAME" --skip-text > "runs/hf_asr_$NAME.log" 2>&1 && break
  sleep 60
done
$UV run python scripts/analyze_first_kana.py --dataset exp/m2/recon/items.jsonl --reference text \
  orig=exp/m2/recon/asr_orig/kana.jsonl mimi=exp/m2/recon/asr_mimi/kana.jsonl keep4k=exp/m2/recon/asr_keep4k/kana.jsonl \
  "$H=exp/m2/recon/asr_$H/kana.jsonl" "$NAME=exp/m2/recon/asr_$NAME/kana.jsonl" > "runs/hf_kana_$NAME.txt" 2>&1
$UV run python scripts/m2/recon_similarity.py exp/m2/recon mimi keep4k "$H" "$NAME" > "runs/hf_similarity_$NAME.txt" 2>&1
echo "$(date) HF_DONE"
