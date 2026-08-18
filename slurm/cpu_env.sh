#!/usr/bin/env bash

# Shared environment configuration for the KAIST RL Lab CPU-only Slurm jobs.
# Source this file; do not execute it directly.

rpex_configure_cpu_environment() {
    export RPEX_MICROMAMBA_BIN="${RPEX_MICROMAMBA_BIN:-$HOME/.local/bin/micromamba}"
    export RPEX_MAMBA_ROOT_PREFIX="${RPEX_MAMBA_ROOT_PREFIX:-$HOME/.local/share/micromamba}"
    export RPEX_ENV_PREFIX="${RPEX_ENV_PREFIX:-$RPEX_MAMBA_ROOT_PREFIX/envs/corruption-rpex-v2}"
    export RPEX_MUJOCO_DIR="${RPEX_MUJOCO_DIR:-$HOME/.mujoco/mujoco210}"
    export RPEX_DATASET_DIR="${RPEX_DATASET_DIR:-$HOME/.d4rl/datasets}"

    export MAMBA_ROOT_PREFIX="$RPEX_MAMBA_ROOT_PREFIX"
    export MUJOCO_PY_MUJOCO_PATH="$RPEX_MUJOCO_DIR"
    export MUJOCO_PY_FORCE_CPU=1
    export MUJOCO_GL=osmesa
    export CUDA_VISIBLE_DEVICES=""
    export MPLBACKEND=Agg
    export TZ="${RPEX_TIMEZONE:-Asia/Seoul}"

    local thread_count="${SLURM_CPUS_PER_TASK:-1}"
    export OMP_NUM_THREADS="$thread_count"
    export MKL_NUM_THREADS="$thread_count"
    export OPENBLAS_NUM_THREADS="$thread_count"
    export NUMEXPR_NUM_THREADS="$thread_count"

    local mpl_root="${TMPDIR:-/tmp}"
    export MPLCONFIGDIR="${MPLCONFIGDIR:-$mpl_root/rpex-matplotlib-${SLURM_JOB_ID:-shell}}"
    mkdir -p "$RPEX_DATASET_DIR" "$MPLCONFIGDIR"
}

rpex_activate_cpu_environment() {
    rpex_configure_cpu_environment

    if [[ -n "${SLURM_JOB_PARTITION:-}" && "$SLURM_JOB_PARTITION" != "cpu" ]]; then
        echo "Refusing to run on Slurm partition '$SLURM_JOB_PARTITION'; expected 'cpu'." >&2
        return 1
    fi
    if [[ ! -x "$RPEX_MICROMAMBA_BIN" ]]; then
        echo "Missing $RPEX_MICROMAMBA_BIN; submit slurm/setup_cpu.sbatch first." >&2
        return 1
    fi
    if [[ ! -x "$RPEX_ENV_PREFIX/bin/python" ]]; then
        echo "Missing environment $RPEX_ENV_PREFIX; submit slurm/setup_cpu.sbatch first." >&2
        return 1
    fi
    if [[ ! -f "$RPEX_MUJOCO_DIR/bin/libmujoco210.so" ]]; then
        echo "Missing MuJoCo 2.1 under $RPEX_MUJOCO_DIR; submit slurm/setup_cpu.sbatch first." >&2
        return 1
    fi

    # Activation supplies the Conda compiler/sysroot variables used to build the
    # legacy mujoco_py C extension on Rocky Linux.
    eval "$("$RPEX_MICROMAMBA_BIN" shell hook --shell bash)"
    micromamba activate "$RPEX_ENV_PREFIX"

    # mujoco_py 2.1.2.14's generated C is not accepted by current GCC 14+.
    # Rocky Linux 8's GCC 8.5 is compatible; the Conda environment still
    # supplies the headers and user-space libraries below.
    export CC="${RPEX_CC:-/usr/bin/gcc}"
    export CXX="${RPEX_CXX:-/usr/bin/g++}"
    if [[ ! -x "$CC" || ! -x "$CXX" ]]; then
        echo "Missing compatible system compilers: CC=$CC CXX=$CXX" >&2
        return 1
    fi

    # Both paths are needed while mujoco_py links and later loads its CPU extension.
    local compiler_sysroot="$RPEX_ENV_PREFIX/x86_64-conda-linux-gnu/sysroot/usr/lib64"
    export LD_LIBRARY_PATH="$RPEX_MUJOCO_DIR/bin:$RPEX_ENV_PREFIX/lib:$compiler_sysroot${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
    export LIBRARY_PATH="$RPEX_MUJOCO_DIR/bin:$RPEX_ENV_PREFIX/lib:$compiler_sysroot${LIBRARY_PATH:+:$LIBRARY_PATH}"
    export CPATH="$RPEX_ENV_PREFIX/include${CPATH:+:$CPATH}"
    export PATH="$RPEX_ENV_PREFIX/bin:$PATH"
}
