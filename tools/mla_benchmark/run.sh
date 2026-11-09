#!/usr/bin/env bash
set -euo pipefail
repo_dir=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)
python_bin=${PYTHON_BIN:-python}
result_dir=${1:?Usage: run.sh RESULT_DIR [benchmark arguments...]}
source_root=${MONAI_BENCH_SOURCE:?Set MONAI_BENCH_SOURCE to the source dataset directory}
data_root=${MONAI_BENCH_DATA_ROOT:?Set MONAI_BENCH_DATA_ROOT to the converted data directory}
mkdir -p "$result_dir"
export PYTHONPATH="$repo_dir${PYTHONPATH:+:$PYTHONPATH}"
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1
export MPLCONFIGDIR="$result_dir/matplotlib"
hostname > "$result_dir/host.txt"
mkdir -p "$data_root"
df -h "$data_root" > "$result_dir/disk_before.txt"
"$python_bin" -m pip freeze > "$result_dir/requirements.lock.txt"
"$python_bin" -u "$repo_dir/tools/mla_benchmark/benchmark.py" --output "$result_dir" --source "$source_root" --data-root "$data_root" "${@:2}"
"$python_bin" "$repo_dir/tools/mla_benchmark/report.py" "$result_dir"
du -sh "$data_root" > "$result_dir/data_size.txt"
