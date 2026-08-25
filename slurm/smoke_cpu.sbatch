#!/usr/bin/env bash
#SBATCH --job-name=cro2o-smoke
#SBATCH --partition=cpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=00:30:00
#SBATCH --requeue
#SBATCH --open-mode=append
#SBATCH --output=slurm-smoke-%j.out
#SBATCH --error=slurm-smoke-%j.out

set -euo pipefail

repo_root="${CRO2O_REPO_ROOT:-${SLURM_SUBMIT_DIR:-}}"
if [[ -z "$repo_root" || ! -f "$repo_root/run_experiment.py" ]]; then
    echo "Submit this job from the repository root, or export CRO2O_REPO_ROOT." >&2
    exit 2
fi
repo_root="$(cd "$repo_root" && pwd -P)"
cd "$repo_root"

# shellcheck source=cpu_env.sh
source "$repo_root/slurm/cpu_env.sh"
cro2o_activate_cpu_environment

echo "SMOKE_HOST=$(hostname)"
echo "SMOKE_PARTITION=$SLURM_JOB_PARTITION"

python scripts/download_d4rl_datasets.py \
    --env-name hopper-medium-replay-v2 \
    --dataset-dir "$CRO2O_DATASET_DIR"

comparison_name="slurm-smoke-${SLURM_JOB_ID}"
output_dir="${CRO2O_SMOKE_OUTPUT_DIR:-$repo_root/results/slurm-smoke}"
python run_experiment.py \
    --algorithm rpex \
    --env-name hopper-medium-replay-v2 \
    --corruption clean \
    --stage both \
    --device cpu \
    --dataset-dir "$CRO2O_DATASET_DIR" \
    --output-dir "$output_dir" \
    --comparison-name "$comparison_name" \
    --offline-steps 2 \
    --online-steps 12 \
    --initial-collection-steps 4 \
    --warmup-steps 4 \
    --batch-size 8 \
    --replay-size 64 \
    --eval-period 1000 \
    --eval-episodes 1 \
    --checkpoint-period 0 \
    --train-log-period 1 \
    --hidden-dim 32 \
    --hidden-layers 1 \
    --seed 0

comparison_dir="$output_dir/comparisons/hopper-medium-replay-v2/clean/none/$comparison_name"
python - "$comparison_dir" <<'PY'
import json
import sys
from pathlib import Path

comparison_dir = Path(sys.argv[1])
summaries = list(comparison_dir.glob("runs/**/summary.json"))
if len(summaries) != 1:
    raise SystemExit(f"expected one smoke summary, found {len(summaries)}")

run_dir = summaries[0].parent
summary = json.loads(summaries[0].read_text(encoding="utf-8"))
assert summary["status"] == "completed", summary
assert (run_dir / "metrics.csv").is_file()
assert (run_dir / "checkpoints/offline/final.pt").is_file()
assert (run_dir / "checkpoints/online/final.pt").is_file()
print("TRAINING_SMOKE_RUN_DIR=" + str(run_dir))
print("END_TO_END_SMOKE_COMPLETE=1")
PY
