#!/usr/bin/env bash

# Shared configuration for CPU-only Slurm jobs. Source this file from a batch
# script; do not execute it directly.

cro2o_configure_cpu_environment() {
    export CRO2O_MICROMAMBA_BIN="${CRO2O_MICROMAMBA_BIN:-$HOME/.local/bin/micromamba}"
    export CRO2O_MAMBA_ROOT_PREFIX="${CRO2O_MAMBA_ROOT_PREFIX:-$HOME/.local/share/micromamba}"
    export CRO2O_ENV_PREFIX="${CRO2O_ENV_PREFIX:-$CRO2O_MAMBA_ROOT_PREFIX/envs/corruption-robust-o2o}"
    export CRO2O_DATASET_DIR="${CRO2O_DATASET_DIR:-$HOME/.d4rl/datasets}"

    export MAMBA_ROOT_PREFIX="$CRO2O_MAMBA_ROOT_PREFIX"
    export CUDA_VISIBLE_DEVICES=""
    export MPLBACKEND=Agg
    export PYTHONUNBUFFERED=1
    export TZ="${CRO2O_TIMEZONE:-Asia/Seoul}"

    local thread_count="${SLURM_CPUS_PER_TASK:-1}"
    export OMP_NUM_THREADS="$thread_count"
    export MKL_NUM_THREADS="$thread_count"
    export OPENBLAS_NUM_THREADS="$thread_count"
    export NUMEXPR_NUM_THREADS="$thread_count"

    local mpl_root="${TMPDIR:-/tmp}"
    export MPLCONFIGDIR="${MPLCONFIGDIR:-$mpl_root/cro2o-matplotlib-${SLURM_JOB_ID:-shell}}"
    mkdir -p "$CRO2O_DATASET_DIR" "$MPLCONFIGDIR"
}

cro2o_activate_cpu_environment() {
    cro2o_configure_cpu_environment

    if [[ -n "${SLURM_JOB_PARTITION:-}" && "$SLURM_JOB_PARTITION" != "cpu" ]]; then
        echo "Refusing to run on Slurm partition '$SLURM_JOB_PARTITION'; expected 'cpu'." >&2
        return 1
    fi
    if [[ ! -x "$CRO2O_MICROMAMBA_BIN" ]]; then
        echo "Missing $CRO2O_MICROMAMBA_BIN; submit slurm/setup_cpu.sbatch first." >&2
        return 1
    fi
    if [[ ! -x "$CRO2O_ENV_PREFIX/bin/python" ]]; then
        echo "Missing environment $CRO2O_ENV_PREFIX; submit slurm/setup_cpu.sbatch first." >&2
        return 1
    fi

    eval "$("$CRO2O_MICROMAMBA_BIN" shell hook --shell bash)"
    micromamba activate "$CRO2O_ENV_PREFIX"
}
