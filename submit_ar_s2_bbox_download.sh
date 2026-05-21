#!/usr/bin/env bash
set -euo pipefail

REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SLURM_SCRIPT="$REPO_DIR/submit_ar_s2_bbox_download.slurm"
TARGET="${TARGET:-auto}"
YEARS=(2020 2021 2022 2023 2024)

if ! command -v sbatch >/dev/null 2>&1; then
    echo "sbatch not found. Run this on the HPC login node." >&2
    exit 1
fi

submit_chain() {
    local partition="$1"
    local nodelist="${2:-}"
    local dependency=""
    local index year jobid

    for index in "${!YEARS[@]}"; do
        year="${YEARS[$index]}"
        echo "[info] queueing year=${year} on partition=${partition}${nodelist:+ node=${nodelist}}"

        if [[ -n "$nodelist" ]]; then
            if [[ -n "$dependency" ]]; then
                jobid="$(sbatch --parsable --partition="$partition" --nodelist="$nodelist" --dependency="afterok:${dependency}" --array="$index" "$SLURM_SCRIPT")"
            else
                jobid="$(sbatch --parsable --partition="$partition" --nodelist="$nodelist" --array="$index" "$SLURM_SCRIPT")"
            fi
        else
            if [[ -n "$dependency" ]]; then
                jobid="$(sbatch --parsable --partition="$partition" --dependency="afterok:${dependency}" --array="$index" "$SLURM_SCRIPT")"
            else
                jobid="$(sbatch --parsable --partition="$partition" --array="$index" "$SLURM_SCRIPT")"
            fi
        fi

        echo "[info] submitted year=${year} jobid=${jobid}"
        dependency="$jobid"
    done
}

submit_condo() {
    echo "[info] submitting sequential yearly jobs to condo on c3903"
    submit_chain condo c3903
}

submit_cloud() {
    echo "[info] submitting sequential yearly jobs to cloud partition"
    submit_chain cloud
}

case "$TARGET" in
    condo)
        submit_condo
        ;;
    cloud)
        submit_cloud
        ;;
    auto)
        state="$(sinfo -h -n c3903 -o '%T' 2>/dev/null | head -n 1 || true)"
        case "$state" in
            idle|mix|mixed)
                submit_condo
                ;;
            *)
                echo "[info] c3903 state is '${state:-unknown}', falling back to cloud"
                submit_cloud
                ;;
        esac
        ;;
    *)
        echo "Unsupported TARGET='$TARGET'. Use auto, condo, or cloud." >&2
        exit 1
        ;;
esac
