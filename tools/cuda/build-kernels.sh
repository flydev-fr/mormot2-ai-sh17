#!/usr/bin/env bash
# LandrixAI - compiles tools/cuda/*.cu to PTX and writes them as Pascal
# constants into src/infer/mormot.ai.infer.cuda.ptx.inc (checked in: the build
# and the run time need no CUDA compiler; the driver compiles the PTX for its
# GPU when the module is loaded)
# - needs nvcc of CUDA 13.4.92, without root from NVIDIA's wheel:
#     python3 -m venv ~/.venvs/cuda && ~/.venvs/cuda/bin/pip install nvidia-cuda-nvcc==13.4.92
#   (any platform: PTX is the same; NVCC=<path> to use another one)
# - target compute_75 (Turing and newer); the PTX of CUDA 13.4 (ISA 9.4) needs
#   a driver for CUDA 13.4 or newer (PTX is excluded from the minor version
#   compatibility of CUDA 13)
set -eu
cd "$(dirname "$0")"
NVCC=${NVCC:-$(command -v nvcc || echo ~/.venvs/cuda/lib/python3.12/site-packages/nvidia/cu13/bin/nvcc)}
"$NVCC" --version | grep -q 'release 13.4, V13.4.92' ||
  { echo "nvcc 13.4.92 expected: $("$NVCC" --version | tail -1)"; exit 1; }
OUT=../../src/infer/mormot.ai.infer.cuda.ptx.inc
TMP=$(mktemp -d)
trap 'rm -rf "$TMP"' EXIT
ARGS=()
for f in *.cu; do
  n=${f%.cu}
  "$NVCC" -ptx -arch=compute_75 -O3 -o "$TMP/$n.ptx" "$f" > "$TMP/$n.log" 2>&1 ||
    { cat "$TMP/$n.log"; exit 1; }
  ARGS+=("PTX_$(echo "$n" | tr a-z A-Z)=$TMP/$n.ptx")
done
python3 ptx2pas.py "$OUT" "${ARGS[@]}"
echo "written: $OUT (${#ARGS[@]} modules)"
