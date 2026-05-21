#!/bin/bash
#SBATCH --job-name=ar_s2_bbox_dl
#SBATCH --array=0-4%2
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=48:00:00
#SBATCH --output=/home/yikebe/research/AR_Crop_Identification/logs/%x-%A_%a.out

set -euo pipefail

cd /home/yikebe/research/AR_Crop_Identification
mkdir -p logs

PYTHON_BIN="${PYTHON_BIN:-$HOME/.conda/envs/ar_download/bin/python}"
if [[ ! -x "$PYTHON_BIN" ]]; then
  echo "Python not found: $PYTHON_BIN" >&2
  exit 127
fi

export ROI="-92.299544,33.003327,-89.894896,36.509871"
export S2_ROOT="$HOME/AR_sentinel_align_with_Planet"
export DOWNLOAD_WORKERS="${DOWNLOAD_WORKERS:-${SLURM_CPUS_PER_TASK:-8}}"

YEARS=(2020 2021 2022 2023 2024)
YEAR="${YEARS[$SLURM_ARRAY_TASK_ID]}"

echo "[info] job_id=${SLURM_JOB_ID:-na} task_id=${SLURM_ARRAY_TASK_ID:-na} year=${YEAR}"
echo "[info] workers=${DOWNLOAD_WORKERS} roi=${ROI} data_root=${S2_ROOT}"

case "$YEAR" in
  2020)
    "$PYTHON_BIN" ar_deploy.py download \
      --project ee-yukiikebe20010210 \
      --start-day 2020-03-11 \
      --end-day 2020-12-31 \
      --roi-bbox="$ROI" \
      --data-root "$S2_ROOT" \
      --download-workers "$DOWNLOAD_WORKERS" \
      --cloud-thresh 100
    ;;
  2024)
    "$PYTHON_BIN" ar_deploy.py download \
      --project ee-yukiikebe20010210 \
      --start-day 2024-01-01 \
      --end-day 2024-12-24 \
      --roi-bbox="$ROI" \
      --data-root "$S2_ROOT" \
      --download-workers "$DOWNLOAD_WORKERS" \
      --cloud-thresh 100
    ;;
  *)
    "$PYTHON_BIN" ar_deploy.py download \
      --project ee-yukiikebe20010210 \
      --year "$YEAR" \
      --whole-year \
      --roi-bbox="$ROI" \
      --data-root "$S2_ROOT" \
      --download-workers "$DOWNLOAD_WORKERS" \
      --cloud-thresh 100
    ;;
esac
