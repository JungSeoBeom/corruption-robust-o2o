#!/usr/bin/env bash
#SBATCH --job-name=rpex-setup
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

readonly EXPECTED_MUJOCO_SHA256="a436ca2f4144c38b837205635bbd60ffe1162d5b44c87df22232795978d7d012"
readonly MUJOCO_URL="https://mujoco.org/download/mujoco210-linux-x86_64.tar.gz"

repo_root="${RPEX_REPO_ROOT:-${SLURM_SUBMIT_DIR:-}}"
if [[ -z "$repo_root" || ! -f "$repo_root/requirements-rpex-v2.txt" ]]; then
    echo "Submit this job from the repository root, or export RPEX_REPO_ROOT." >&2
    exit 2
fi
repo_root="$(cd "$repo_root" && pwd -P)"
cd "$repo_root"

# shellcheck source=cpu_env.sh
source "$repo_root/slurm/cpu_env.sh"
rpex_configure_cpu_environment

mkdir -p "$RPEX_MAMBA_ROOT_PREFIX"
exec 9>"$RPEX_MAMBA_ROOT_PREFIX/rpex-setup.lock"
if ! flock --exclusive --nonblock 9; then
    echo "Another shared RPEX environment setup is already running." >&2
    exit 2
fi

if [[ "${SLURM_JOB_PARTITION:-}" != "cpu" ]]; then
    echo "Refusing setup outside the cpu partition." >&2
    exit 2
fi
if [[ "$(uname -s)" != "Linux" || "$(uname -m)" != "x86_64" ]]; then
    echo "The strict mujoco_py environment requires Linux x86_64." >&2
    exit 2
fi

echo "SETUP_HOST=$(hostname)"
echo "SETUP_PARTITION=$SLURM_JOB_PARTITION"
echo "ENV_PREFIX=$RPEX_ENV_PREFIX"
echo "DATASET_DIR=$RPEX_DATASET_DIR"

if [[ ! -x "$RPEX_MICROMAMBA_BIN" ]]; then
    echo "Installing micromamba..."
    tmp_mamba="$(mktemp -d)"
    trap 'if [[ -n "${tmp_mamba:-}" && -d "$tmp_mamba" ]]; then rm -rf -- "$tmp_mamba"; fi' EXIT
    curl -fsSL --retry 5 --retry-delay 2 \
        https://micro.mamba.pm/api/micromamba/linux-64/latest \
        | tar -xj -C "$tmp_mamba" bin/micromamba
    mkdir -p "$(dirname "$RPEX_MICROMAMBA_BIN")"
    install -m 0755 "$tmp_mamba/bin/micromamba" "$RPEX_MICROMAMBA_BIN"
fi
"$RPEX_MICROMAMBA_BIN" --version

if [[ ! -f "$RPEX_MUJOCO_DIR/bin/libmujoco210.so" ]]; then
    echo "Installing MuJoCo 2.1..."
    mkdir -p "$(dirname "$RPEX_MUJOCO_DIR")"
    mujoco_archive="$(mktemp)"
    curl -fL --retry 5 --retry-delay 2 "$MUJOCO_URL" -o "$mujoco_archive"
    actual_sha256="$(sha256sum "$mujoco_archive" | awk '{print $1}')"
    if [[ "$actual_sha256" != "$EXPECTED_MUJOCO_SHA256" ]]; then
        echo "MuJoCo archive checksum mismatch: $actual_sha256" >&2
        exit 1
    fi
    tar -xzf "$mujoco_archive" -C "$(dirname "$RPEX_MUJOCO_DIR")"
fi
test -f "$RPEX_MUJOCO_DIR/bin/libmujoco210.so"

if [[ ! -x "$RPEX_ENV_PREFIX/bin/python" ]]; then
    echo "Creating the Python 3.10 environment..."
    "$RPEX_MICROMAMBA_BIN" create --yes --prefix "$RPEX_ENV_PREFIX" \
        --channel conda-forge \
        python=3.10 pip=23.3.2 git make patchelf \
        mesalib=21.2.5 mesa-libgl-devel-cos7-x86_64 \
        gcc_linux-64 gxx_linux-64
else
    echo "Reusing $RPEX_ENV_PREFIX"
    "$RPEX_MICROMAMBA_BIN" install --yes --prefix "$RPEX_ENV_PREFIX" \
        --channel conda-forge \
        python=3.10 pip=23.3.2 git make patchelf \
        mesalib=21.2.5 mesa-libgl-devel-cos7-x86_64 \
        gcc_linux-64 gxx_linux-64
fi

rpex_activate_cpu_environment

echo "Installing the CPU-only PyTorch wheel..."
python -m pip install --no-cache-dir \
    --index-url https://download.pytorch.org/whl/cpu \
    "torch==2.5.1"

echo "Installing the pinned RPEX/D4RL-v2 dependencies..."
python -m pip install --no-cache-dir -r requirements-rpex-v2.txt
python -m pip check

echo "Compiling and importing mujoco_py in the CPU allocation..."
MUJOCO_PY_FORCE_REBUILD=1 python - <<'PY'
import os
import mujoco_py

assert os.environ.get("MUJOCO_PY_FORCE_CPU") == "1"
print("MUJOCO_PY_VERSION=" + mujoco_py.__version__)
print("MUJOCO_PY_EXTENSION=" + mujoco_py.cymj.__file__)
PY

echo "Running the repository unit suite..."
python -m unittest discover -s tests -v

echo "Running the strict environment and dataset integration smoke test..."
python scripts/smoke_rpex_d4rl_v2.py \
    --env-name hopper-medium-replay-v2 \
    --seed 0 \
    --dataset-dir "$RPEX_DATASET_DIR"

echo "SETUP_COMPLETE=1"
