#!/usr/bin/env bash
set -euo pipefail

# Erzeugt die Embedding-Referenz (Schritt 1 aus docs/native-embeddings.md): bettet
# den deutschen Testkorpus mit dem ORAKEL sqlite-lembed + bge-m3 ein und schreibt
# tests/data/embed-de/reference/ (manifest.json, docs.f32, queries.f32).
#
#   embed-reference.sh                 # Default-Korpus tests/data/embed-de
#   embed-reference.sh <datadir>       # anderer Korpus im selben Format
#   LEMBED_MODEL=<gguf> embed-reference.sh
#
# Braucht die gitignorten Laufzeit-Artefakte unter vendor/ (sqlite-ext/<arch>-linux,
# models/bge-m3-q8_0.gguf). Die erzeugte Referenz wird eingecheckt, damit spaetere
# Paritaets-/Qualitaetstests OHNE lembed laufen. Nur neu erzeugen, wenn sich Korpus,
# Modell oder lembed-Build aendern - und dann die Metriken im Commit nennen.

LAI="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
source "$LAI/scripts/fpc-env.sh"
ARCH="${ARCH:-$("$FPC" -iTP)}"

"$LAI/scripts/build-demo.sh" rag/embed-reference.lpr

export SQLITE_EXT_DIR="${SQLITE_EXT_DIR:-$LAI/vendor/sqlite-ext/$ARCH-linux}"
export LEMBED_MODEL="${LEMBED_MODEL:-$LAI/vendor/models/bge-m3-q8_0.gguf}"
export LD_LIBRARY_PATH="$SQLITE_EXT_DIR${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
[ -f "$SQLITE_EXT_DIR/lembed0.so" ] || { echo "FAIL: $SQLITE_EXT_DIR/lembed0.so fehlt" >&2; exit 2; }
[ -f "$LEMBED_MODEL" ] || { echo "FAIL: Modell $LEMBED_MODEL fehlt" >&2; exit 2; }

cd "$LAI"
# llama.cpp schreibt seinen Lade-Log auf stderr - nur die eigenen Zeilen zeigen
"$LAI/bin/fpc/demos/embed-reference" "${1:-tests/data/embed-de}" 2> "$LAI/bin/fpc/demos/embed-reference.stderr.log"
