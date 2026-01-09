#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
OUT_DIR="${ROOT_DIR}/../torchlite/cbits/rocm_build"
mkdir -p "${OUT_DIR}"

hipcc -O3 -std=c++14 -c "${ROOT_DIR}/rocm_backend.cpp" -o "${OUT_DIR}/rocm_backend.o"
ar rcs "${OUT_DIR}/libtorchlite_rocm.a" "${OUT_DIR}/rocm_backend.o"
echo "Built ${OUT_DIR}/libtorchlite_rocm.a"
