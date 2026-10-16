#!/usr/bin/env bash
# LandrixAI - compiles tools/vk/*.comp to SPIR-V and writes them as Pascal
# constants into src/infer/mormot.ai.infer.vulkan.spv.inc (checked in: the
# build and the run time need no shader compiler)
# - needs glslangValidator (package glslang-tools; without root:
#   apt-get download glslang-tools && dpkg -x glslang-tools_*.deb ~/.local/opt/vk/root)
set -eu
cd "$(dirname "$0")"
GLSLANG=${GLSLANG:-$(command -v glslangValidator || echo ~/.local/opt/vk/root/usr/bin/glslangValidator)}
OUT=../../src/infer/mormot.ai.infer.vulkan.spv.inc
TMP=$(mktemp -d)
trap 'rm -rf "$TMP"' EXIT
ARGS=()
for f in *.comp; do
  n=${f%.comp}
  # Vulkan 1.0 SPIR-V: every driver takes it; -Os keeps the constants small
  "$GLSLANG" -V --target-env vulkan1.0 -Os -o "$TMP/$n.spv" "$f" > "$TMP/$n.log" ||
    { cat "$TMP/$n.log"; exit 1; }
  ARGS+=("SPV_$(echo "$n" | tr a-z A-Z)=$TMP/$n.spv")
done
python3 spv2pas.py "$OUT" "${ARGS[@]}"
echo "written: $OUT (${#ARGS[@]} shaders)"
