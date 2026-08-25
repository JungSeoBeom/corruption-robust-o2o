#!/usr/bin/env bash
#SBATCH --job-name=cro2o-setup
#SBATCH --partition=cpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=01:00:00
#SBATCH --requeue
#SBATCH --open-mode=append
#SBATCH --output=slurm-setup-%j.out
#SBATCH --error=slurm-setup-%j.out

set -euo pipefail

repo_root="${CRO2O_REPO_ROOT:-${SLURM_SUBMIT_DIR:-}}"
if [[ -z "$repo_root" || ! -f "$repo_root/requirements.txt" ]]; then
    echo "Submit this job from the repository root, or export CRO2O_REPO_ROOT." >&2
    exit 2
fi
repo_root="$(cd "$repo_root" && pwd -P)"
cd "$repo_root"

# shellcheck source=cpu_env.sh
source "$repo_root/slurm/cpu_env.sh"
cro2o_configure_cpu_environment

mkdir -p "$CRO2O_MAMBA_ROOT_PREFIX"
exec 9>"$CRO2O_MAMBA_ROOT_PREFIX/cro2o-setup.lock"
if ! flock --exclusive --nonblock 9; then
    echo "Another shared environment setup is already running." >&2
    exit 2
fi

if [[ "${SLURM_JOB_PARTITION:-}" != "cpu" ]]; then
    echo "Refusing setup outside the cpu partition." >&2
    exit 2
fi

echo "SETUP_HOST=$(hostname)"
echo "SETUP_PARTITION=$SLURM_JOB_PARTITION"
echo "ENV_PREFIX=$CRO2O_ENV_PREFIX"
echo "DATASET_DIR=$CRO2O_DATASET_DIR"

if [[ ! -x "$CRO2O_MICROMAMBA_BIN" ]]; then
    echo "Installing micromamba..."
    tmp_mamba="$(mktemp -d)"
    trap 'if [[ -n "${tmp_mamba:-}" && -d "$tmp_mamba" ]]; then rm -rf -- "$tmp_mamba"; fi' EXIT
    curl -fsSL --retry 5 --retry-delay 2 \
        https://micro.mamba.pm/api/micromamba/linux-64/latest \
        | tar -xj -C "$tmp_mamba" bin/micromamba
    mkdir -p "$(dirname "$CRO2O_MICROMAMBA_BIN")"
    install -m 0755 "$tmp_mamba/bin/micromamba" "$CRO2O_MICROMAMBA_BIN"
fi
"$CRO2O_MICROMAMBA_BIN" --version

if [[ ! -x "$CRO2O_ENV_PREFIX/bin/python" ]]; then
    "$CRO2O_MICROMAMBA_BIN" create --yes --prefix "$CRO2O_ENV_PREFIX" \
        --channel conda-forge python=3.13 pip
else
    "$CRO2O_MICROMAMBA_BIN" install --yes --prefix "$CRO2O_ENV_PREFIX" \
        --channel conda-forge python=3.13 pip
fi

cro2o_activate_cpu_environment

# A reused project prefix must not retain the retired runtime packages.
python -m pip uninstall --yes gym d4rl mujoco-py pybullet
python -m pip install --upgrade pip
python -m pip install \
    --index-url https://download.pytorch.org/whl/cpu \
    "torch==2.13.0"
python -m pip install -r requirements.txt
python -m pip check

python scripts/download_d4rl_datasets.py \
    --env-name hopper-medium-replay-v2 \
    --dataset-dir "$CRO2O_DATASET_DIR"
python -m pytest -q

echo "SETUP_COMPLETE=1"
