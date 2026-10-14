# infer-ref — Referenzdaten der nativen Inferenz

Erzeugt von [tools/infer-ref/make_reference.py](../../../tools/infer-ref/make_reference.py)
(Umgebung und Festlegungen: [tools/infer-ref/README.md](../../../tools/infer-ref/README.md)).
Gelesen von der Suite `infer.tests` (`tests/test.infer.ref.pas`). Keine dieser Dateien wird
von Hand geändert.

## Format

Jeder Datensatz ist ein Ordner mit `manifest.json` und `tensors.bin`:
- `tensors.bin` enthält die Tensoren als rohe Little-Endian-Werte, hintereinander,
  4-Byte-ausgerichtet;
- `manifest.json` beschreibt jeden Tensor mit `name`, `dtype` (`f32` oder `i32`),
  `shape` (zeilenweise, C-Reihenfolge), `offset`, `bytes` und `sha256`. Dazu kommen die
  Hashes der ganzen Datei, Modellrevision, Umgebung und Pipeline, außerdem der Zeitpunkt der
  Erzeugung. Die Blobs sind reproduzierbar, die Manifeste wegen dieses Zeitstempels nicht.

Der Lader prüft beim Öffnen den Hash der Datei, Lage und Länge jedes Tensors und das
Produkt der Form, bevor er irgendetwas liest. Ein Rechenergebnis wird nie über einen Hash
verglichen, sondern elementweise: `|a − r| ≤ atol + rtol·|r|`, dazu Kosinus und Normen.

## Datensätze

| Ordner | Inhalt | Größe |
|---|---|---|
| `ops/` | Operator-Fixtures: Linear (`x·Wᵀ + b`), LayerNorm (nicht-triviale γ/β, fast konstante und konstante Zeile), maskiertes Softmax (großer Logit), exaktes GELU (erf), L2 (Zeile mit Norm ~1e-20). Eingaben f32, Erwartung in float64 gerechnet. | 5 KiB |
| `tiny-xlmr/` | XLM-R mit 2 Layern, Breite 32, 4 Köpfe, Vokabular 64, BGE-M3-Konventionen (pad 1, bos 0, eos 2, eps 1e-5, exaktes GELU). Vertrag sind `model.safetensors` und `config.json`, nicht der Seed. Drei Sequenzen mit allen Zuständen, dazu derselbe Inhalt als rechts gepaddeter Batch. | 106 KiB |
| `bge-m3/` | Sechs kurze Fälle (≤ 16 Tokens inkl. `<s>`/`</s>`): leer, nur Leerzeichen, Handwerkstext, Unicode, mehrsprachig, Sondertoken-Literal. Je Fall IDs, Maske, Positionen, alle 25 Zustände (0 = Ausgang des Embedding-Moduls nach LayerNorm, 1–24 = Ausgang von Layer k), CLS und normierter Vektor. Dazu ein rechts gepaddeter Batch aus zwei Texten, jeder auch als Einzellauf. Prüfung der Sondertoken-Literale unter `tokenizer_checks`. | 5,6 MiB |
| `bge-m3-embed-de/` | FP32-Vektoren des Korpus [embed-de](../embed-de/) (262 Dokumente, 58 Anfragen, Zeile = Dateizeile). Sie ersetzen weder das lembed-Q8-Orakel in `embed-de/reference/` noch das nDCG-Gate von Phase 1b. | 1,25 MiB |

Gemessen am 2026-10-14:
- Beim Tiny-Modell sind die Batch-Zeilen bitgleich mit den Einzelläufen. Bei BGE-M3 weicht
  die gepaddete Zeile um höchstens 1,3e-7 ab (Summationsreihenfolge), die ungepaddete gar
  nicht.
- Die FP32-Vektoren liegen beim Kosinus mindestens bei 0,99868 zum lembed-Q8-Orakel
  (Mittel 0,99929).
