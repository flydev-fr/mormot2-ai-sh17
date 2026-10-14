# infer-ref — Referenzdaten für die native Inferenz

`make_reference.py` erzeugt die Referenzdaten, gegen die die native Engine
([docs/native-inference.md](../../docs/native-inference.md)) geprüft wird. Das Werkzeug
läuft nur bei der Bereitstellung, nie zur Laufzeit und nie im Test. Die Tests lesen die
eingecheckten Ergebnisse unter [tests/data/infer-ref/](../../tests/data/infer-ref/).

## Referenzumgebung

Verbindlich ist **CPU auf WSL aarch64**: Python 3.12.13, torch 2.14.1+cpu,
transformers 5.19.0. Die vollständige Paketauflösung steht in
[requirements.txt](requirements.txt). Die Umgebung liegt außerhalb des Repos:

```bash
uv venv -p 3.12 ~/.venvs/lai-infer-ref
VIRTUAL_ENV=~/.venvs/lai-infer-ref uv pip install \
  --index-url https://download.pytorch.org/whl/cpu torch==2.14.1+cpu
VIRTUAL_ENV=~/.venvs/lai-infer-ref uv pip install -r tools/infer-ref/requirements.txt
~/.venvs/lai-infer-ref/bin/python tools/infer-ref/make_reference.py            # alles
~/.venvs/lai-infer-ref/bin/python tools/infer-ref/make_reference.py ops tiny   # ohne Modell
```

Mit dem Argument `bge-m3` lädt das Skript BAAI/bge-m3 in der festen Revision
`5617a9f61b028005a4858fdac845db406aefb181` (rund 2,3 GB in den Hugging-Face-Cache).

## Was das Skript festlegt

- **FP32, eval(), ein Thread, deterministische Algorithmen, Attention „eager".** Jeder
  Fall läuft zweimal und muss bitgleich herauskommen. Zwei Läufe des Skripts schreiben
  bytegleiche Blobs und ein bytegleiches `model.safetensors`; nur der Zeitstempel in den
  Manifesten ändert sich.
- **Pooling aus der Revision selbst:** `1_Pooling/config.json` muss nur CLS verlangen und
  `modules.json` ein `Normalize` enthalten, sonst bricht das Skript ab.
- **Gewichte als safetensors.** Die Revision enthält nur `pytorch_model.bin` (Pickle). Das
  Skript lädt es mit `weights_only=True` und schreibt ein F32-`model.safetensors` nach
  `vendor/models/bge-m3/` (gitignored, mit `LAI_MODEL_DIR` umlenkbar), zusammen mit
  Tokenizer und Konfiguration. Danach lädt es die Datei erneut und vergleicht Tensor für
  Tensor. Außerdem muss der Forward aus beiden Quellen bitgleich sein. Fehlende Gewichte
  brechen ab; ungenutzt bleibt nur der Pooler (`pooler.dense.*`), weil BGE-M3 dense den
  CLS-Zustand nimmt, nicht `pooler_output`.
- **Token-IDs aus SentencePiece**, nicht aus dem HF-Tokenizer. SentencePiece erkennt seine
  Steuersymbole (`<s>`, `</s>`, `<pad>`) nie im Text. Das ist `parse_special=false`, und
  so muss der native Tokenizer arbeiten. Der HF-Tokenizer erkennt `<s>` dagegen auch mit
  `split_special_tokens` oder `encode_special_tokens`, weil `<s>` ein Eintrag seines
  Unigram-Vokabulars ist (gemessen, siehe `tokenizer_checks` im Manifest). Für jeden Text
  ohne solches Literal prüft das Skript, dass beide dieselben IDs liefern. Abbildung auf
  XLM-R: SentencePiece-ID + 1, außer `<unk>` 0→3, `<s>` 1→0, `</s>` 2→2.

Neu erzeugen nur, wenn sich Modell, Umgebung oder Fälle ändern. Dann alle Datensätze
gemeinsam neu schreiben, `requirements.txt` und [UPSTREAM_BASE](../../UPSTREAM_BASE)
nachziehen und im Commit nennen, was sich geändert hat.
