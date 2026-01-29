#!/usr/bin/env bash
set -euo pipefail

# Default paths (can be overridden by env or CLI args)
ROOT="/data/flux_kontext"
WORKDIR="${ROOT}/matching_anything"
TXT_FILE_DEFAULT="${ROOT}/dataloaders/result_no_border.txt"
CKPT_OUT_DEFAULT="${WORKDIR}/roma_sar_opt.pth"

# Allow overrides via env
TXT_FILE="${TXT_FILE:-$TXT_FILE_DEFAULT}"
CKPT_OUT="${CKPT_OUT:-$CKPT_OUT_DEFAULT}"
BATCH_SIZE="${BATCH_SIZE:-2}"
EPOCHS="${EPOCHS:-1}"
STEPS_PER_EPOCH="${STEPS_PER_EPOCH:-100}"
SIZE="${SIZE:-512}"
NUM_WORKERS="${NUM_WORKERS:-4}"
DEVICE="${DEVICE:-cuda}"
AMP_FLAG="${AMP_FLAG:---amp}"              # set to empty to disable AMP
STRETCH_FLAG="${STRETCH_FLAG:---resize_by_stretch}"  # set to empty to disable stretch

cd "$WORKDIR"

python "${WORKDIR}/matching_train_demo.py" \
  --txt_file "${TXT_FILE}" \
  --batch_size "${BATCH_SIZE}" \
  --epochs "${EPOCHS}" \
  --steps_per_epoch "${STEPS_PER_EPOCH}" \
  --size "${SIZE}" \
  --num_workers "${NUM_WORKERS}" \
  --device "${DEVICE}" \
  ${AMP_FLAG} \
  ${STRETCH_FLAG} \
  --ckpt_out "${CKPT_OUT}"

