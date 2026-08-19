#!/usr/bin/env bash
#SBATCH --job-name=rpex-smoke
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

readonly EXPECTED_D4RL_COMMIT="d842aa194b416e564e54b0730d9f934e3e32f854"

repo_root="${RPEX_REPO_ROOT:-${SLURM_SUBMIT_DIR:-}}"
if [[ -z "$repo_root" || ! -f "$repo_root/run_experiment.py" ]]; then
    echo "Submit this job from the repository root, or export RPEX_REPO_ROOT." >&2
    exit 2
fi
repo_root="$(cd "$repo_root" && pwd -P)"
cd "$repo_root"

# shellcheck source=cpu_env.sh
source "$repo_root/slurm/cpu_env.sh"
rpex_activate_cpu_environment

echo "SMOKE_HOST=$(hostname)"
echo "SMOKE_PARTITION=$SLURM_JOB_PARTITION"

# First verify the exact legacy environment, one MuJoCo step, and the real D4RL
# dataset. The setup job has already serialized the initial dataset download.
python scripts/smoke_rpex_d4rl_v2.py \
    --env-name hopper-medium-replay-v2 \
    --seed 0 \
    --dataset-dir "$RPEX_DATASET_DIR"

# Then exercise both real training phases: 2 offline optimizer steps, 12 online
# environment steps, and 5 online replay updates once the 8-sample buffer fills.
comparison_name="slurm-smoke-${SLURM_JOB_ID}"
output_dir="${RPEX_SMOKE_OUTPUT_DIR:-$repo_root/results/slurm-smoke}"
python run_experiment.py \
    --algorithm rpex \
    --env-name hopper-medium-replay-v2 \
    --corruption clean \
    --stage both \
    --protocol rpex_d4rl_v2_legacy \
    --device cpu \
    --dataset-dir "$RPEX_DATASET_DIR" \
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
    --max-episode-steps 25 \
    --checkpoint-period 0 \
    --train-log-period 1 \
    --hidden-dim 32 \
    --hidden-layers 1 \
    --num-critics 3

comparison_dir="$output_dir/comparisons/hopper-medium-replay-v2/clean/none/$comparison_name"
python - "$comparison_dir" "$EXPECTED_D4RL_COMMIT" <<'PY'
import json
import sys
from pathlib import Path

comparison_dir = Path(sys.argv[1])
expected_commit = sys.argv[2]
summaries = list(comparison_dir.glob("runs/**/summary.json"))
if len(summaries) != 1:
    raise SystemExit(f"expected one smoke summary, found {len(summaries)}")

run_dir = summaries[0].parent
summary = json.loads((run_dir / "summary.json").read_text())
config = json.loads((run_dir / "config.json").read_text())
train_rows = [
    json.loads(line)
    for line in (run_dir / "train_metrics.jsonl").read_text().splitlines()
    if line.strip()
]

assert summary["status"] == "completed", summary
assert config["resolved_device"] == "cpu", config["resolved_device"]
assert config["protocol"] == "rpex_d4rl_v2_legacy", config["protocol"]
assert config["installed_d4rl_commit"] == expected_commit
assert {row["phase"] for row in train_rows} == {"offline", "online"}
assert max(row["updates"] for row in train_rows) == 7
assert (run_dir / "checkpoints/offline/final.pt").is_file()
assert (run_dir / "checkpoints/online/final.pt").is_file()

print("TRAINING_SMOKE_RUN_DIR=" + str(run_dir))
print("TRAINING_SMOKE_MAX_UPDATES=" + str(max(row["updates"] for row in train_rows)))
print("END_TO_END_SMOKE_COMPLETE=1")
PY
