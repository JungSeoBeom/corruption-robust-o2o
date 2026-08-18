#!/usr/bin/env bash
#SBATCH --job-name=rpex-cpu
#SBATCH --partition=cpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=24:00:00
#SBATCH --requeue
#SBATCH --open-mode=append
#SBATCH --output=slurm-%x-%j.out
#SBATCH --error=slurm-%x-%j.out

set -euo pipefail

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

if (( $# == 0 )); then
    set -- \
        --algorithm rpex \
        --env-name hopper-medium-replay-v2 \
        --corruption clean \
        --stage both \
        --offline-steps 500000 \
        --online-steps 500000 \
        --seed 0
fi

echo "JOB_ID=${SLURM_JOB_ID:-none}"
echo "HOST=$(hostname)"
echo "PARTITION=${SLURM_JOB_PARTITION:-none}"
echo "CPUS=${SLURM_CPUS_PER_TASK:-1}"
echo "COMMAND=python run_experiment.py $* --protocol rpex_d4rl_v2_legacy --device cpu --dataset-dir $RPEX_DATASET_DIR"

# Put the enforced options last so a submitted job cannot accidentally select a
# GPU device or the non-reproduction Gymnasium protocol.
exec python run_experiment.py "$@" \
    --protocol rpex_d4rl_v2_legacy \
    --device cpu \
    --dataset-dir "$RPEX_DATASET_DIR"
