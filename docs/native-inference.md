# Native Inferenz in Pascal — Plan

> **Status (2026-10-14):** Phase 0, 1a und 1b erledigt: BGE-M3 läuft nativ von Text bis
> Vektor, in FP32 gegen PyTorch und aus demselben Q8-GGUF wie lembed gegen das Orakel
> (Token-IDs, Kosinus, nDCG@10 - siehe dort). 1c auf aarch64 und x86_64 (AVX2) erledigt:
> auf x86_64 1,6-mal (Linux) bis doppelt (Windows) so schnell wie lembed, auf aarch64 gleichauf bis schneller, jeweils
> mit weniger Speicher; unter Linux x86_64 (die Produktion) alle Suiten und das
> Korpus-Gate grün, Soak-Test über 90 Minuten bestanden; im Backend der empfohlene
> Provider (`AI_EMBED_PROVIDER=native`). Phase 1c ist damit abgeschlossen.
> Phase 2 (Decoder Qwen3) bis 2c erledigt: Tokenizer und Chat-Template treffen
> transformers in jedem Referenzfall, und die Greedy-Ausgabe von Qwen3-0.6B in FP32 ist
> für alle sieben Referenz-Prompts tokengleich - das Gate von Phase 2. Offen ist 2d, die
> Anbindung hinter `ILlmClient`.
> Grundlage ist der Entwurf des Nutzers
> („Plan: Native Embedding- und LLM-Inferenz in Pascal", 2026-10-14). Er wurde mit dem
> bestehenden Plan für Embeddings ([native-embeddings.md](native-embeddings.md),
> „Option D") zusammengeführt und mit Codex abgestimmt, das gegen Code, Modelldatei und
> Referenzdaten geprüft hat. Die Plattformbefunde unten sind am 2026-10-14 gemessen.
> Einordnung: [CONCEPT.md §5](../CONCEPT.md).

## Ziel

Eine eigene Inferenz-Engine in reinem Object Pascal für alle Plattformen, die mORMot2
unterstützt (FPC und Delphi). Sie beginnt mit **BGE-M3** (Embeddings, ersetzt
`lembed0` + `libllama`/`ggml`) und danach kommen **Decoder-LLMs** wie Qwen3. Zur Laufzeit
gibt es keine Abhängigkeit von fremden Repositories und keine Copyleft-Vorlagen.

**Was sich damit ändert:** Bisher war neural-api das strategische Backend für native
Inferenz (CONCEPT.md §5). Das entfällt. Das Beitragsangebot
[neural-api#225](https://github.com/joaopauloschuler/neural-api/issues/225) ist seit
2026-10-14 als „not planned" geschlossen. Die eigene Engine übernimmt beides: Embedder
**und** generatives `ILlmClient`. „Option D" aus native-embeddings.md geht als Phase 1
in diesen Plan ein.

## Architektur

Jede Schicht hat eine schmale, nicht-generische Schnittstelle. Neue Formate, Backends,
Tokenizer oder Modelle kommen dazu, ohne dass Bestehendes geändert wird.

Stil: wie der Rest der Extension **im Stil von mORMot2** — Regeln in
[CONCEPT.md §1](../CONCEPT.md). Für die Engine heißt das vor allem: keine Generics und
keine anonymen Methoden (Delphi 7), mORMot-Bausteine für Threads und Puffer, keine
Heap-Allokation in den Rechenschleifen.

Ablage (entschieden): im Unterordner **`src/infer/`** dieses Repos. Das ist eine klare
Grenze, ohne zweites Repo und ohne tiefe Verzeichnishierarchie. mORMot2 selbst ordnet
ebenso nach `src/<bereich>/`. Die Build-Skripte müssen den Pfad zusätzlich aufnehmen;
heute kennen sie nur `src/`.

| Unit | Inhalt |
|---|---|
| `mormot.ai.infer.tensor` | Tensor, Datentypen (F32, F16, BF16, Q8_0, Q4_0 …), Formen |
| `mormot.ai.infer.memory` | Memory-Mapping (Fenster für Win32 erst bei Modellen über 1,5 GB), Scratch-Puffer |
| `mormot.ai.infer.loader.safetensors` | safetensors-Leser |
| `mormot.ai.infer.loader.gguf` | GGUF-v3-Leser, robust gegen beschädigte Dateien |
| `mormot.ai.infer.backend` | Backend-Schnittstelle, Fähigkeitsabfrage, Fallback |
| `mormot.ai.infer.backend.cpu` | Pascal-Referenz + SIMD-Kernel (NEON, AVX2) |
| `mormot.ai.infer.tokenizer.ugm` | Unigram/SentencePiece inkl. `precompiled_charsmap` |
| `mormot.ai.infer.tokenizer.bpe` | Byte-Level-BPE (Qwen) |
| `mormot.ai.infer.model.registry` | Modelle nach `model_type` |
| `mormot.ai.infer.model.xlmr` | XLM-RoBERTa / BERT-Encoder (BGE-M3) |
| `mormot.ai.infer.model.qwen` | Qwen-Decoder |
| `mormot.ai.infer.pipeline.embed` | Pooling (CLS / Mean / Last-Token), Normierung |
| `mormot.ai.infer.pipeline.generate` | KV-Cache, Sampler-Kette, Stop-Behandlung |

Öffentliche Adapter liegen neben den bestehenden in `src/`:
- `mormot.ai.embed.native` mit `TNativeEmbedder: IEmbedder`. Das Interface passt
  unverändert.
- `mormot.ai.llm.native` mit `TNativeLlmClient: ILlmClient`. Completion und Streaming
  einschließlich Tool-Call-Daten passen. Die Engine muss aber das Chat-Template, das
  Parsen von Tool-Argumenten, Stop-Tokens und das Puffern unvollständiger UTF-8-Sequenzen
  selbst leisten. Inferenz-Einstellungen kommen über eine optionale Zusatzschnittstelle,
  **nicht** über die HTTP-Felder von `TLlmProviderConfig`.

**Die native Engine ist eine Implementierung unter mehreren.** Hinter denselben
Schnittstellen `ILlmClient` und `IEmbedder` gibt es heute schon:

| Weg | Chat (`ILlmClient`) | Embeddings (`IEmbedder`) |
|---|---|---|
| OpenAI-kompatibel: OpenAI, **IONOS AI Model Hub**, **Ollama**, LiteLLM, vLLM … | `TLlmClient` mit `OpenAIConfig`/`OllamaConfig`/`LiteLLMConfig`/`LlmConfigFromEnv` (beliebige `BaseUrl` + Key) | `TProviderEmbedder` über `/embeddings` |
| Anthropic | `TAnthropicClient` | – |
| lokal mit C-Bibliothek | – | `TLembedEmbedder` |
| **nativ (dieser Plan)** | `TNativeLlmClient` | `TNativeEmbedder` |

Welcher Weg gilt, entscheidet die Konfiguration des Konsumenten, nie ein `{$ifdef}`. Ein
Vektorraum bleibt dabei an **ein** Modell gebunden (Referenzvertrag): bge-m3 über IONOS
und bge-m3 lokal sind nur dann austauschbar, wenn ihre Vektoren den Paritätstest gegen
die Referenz bestehen. Sonst muss neu indiziert werden. Im Konsumenten fehlt heute noch
die Auswahl `provider` für Embeddings (die Backend-Fabrik kennt nur `none`/`lembed`).

### Was die Engine hinter `ILlmClient` leisten muss

Alles, was heute auf `ILlmClient` und `IEmbedder` aufsetzt, bleibt unverändert und
funktioniert mit der Engine, **sobald** sie die jeweilige Fähigkeit liefert:
- Agent-Loop `TLlmAgent`;
- MCP-Brücke `mormot.ai.agent.mcp`;
- Structured Output `ChatStructured`;
- RAG `TLlmRag` und das MCP-Tool `search_docs`;
- nativer Store.

Die Fähigkeit selbst muss die Engine erbringen. Ein Cloud-Anbieter liefert sie über sein
API mit:

| Fähigkeit | über einen Anbieter | nativ |
|---|---|---|
| Chat + Streaming | ✓ | Phase 2: Tokens gehen direkt aus der Dekodier-Schleife an den Streaming-Callback |
| Tool-Calls (Agent-Loop, MCP-Tools) | ✓ | Phase 2. Was die Engine dafür leisten muss:<br>• das Tool-Schema ins Chat-Template legen<br>• die Tool-Call-Ausgabe des Modells (bei Qwen3 JSON in `<tool_call>`) in `TLlmToolCall` übersetzen<br>Der Agent-Loop bleibt gleich. |
| Structured Output | ✓ (`response_format`) | erst Prompt + Validierung. **Garantiert** nur mit eingeschränktem Dekodieren (JSON-Schema → Grammatik → Maske über die Logits), eigener Schritt nach Phase 2. |
| Vision | ✓ | nicht geplant (bräuchte einen Bild-Encoder, etwa Qwen-VL) |
| Embeddings für RAG | ✓ (`TProviderEmbedder`) | Phase 1 (`TNativeEmbedder`) |

### Designentscheidungen

**Konfigurierbare Blöcke statt einer Klasse je Modell.** Ein Transformer-Block wird aus
der Modellkonfiguration zusammengesetzt:

| Eigenschaft | Varianten |
|---|---|
| Normalisierung | LayerNorm, RMSNorm |
| Position der Norm | Pre-Norm, Post-Norm |
| FFN | GELU, SwiGLU |
| Positionen | absolut, RoPE |
| Attention | MHA, GQA; optional Q/K-Norm |
| Kopfgröße | aus der Konfiguration (`head_dim`), **nicht** `hidden / heads` |
| Maske | bidirektional, kausal |
| Bias | an/aus je Projektion |

Qwen3 ist trotzdem **nicht** nur „llama2.c plus Konfiguration". Es braucht Q/K-Norm, und
Qwen3-0.6B hat `head_dim = 128`, obwohl 1024 / 16 = 64 wäre. Die Registry entsteht erst
mit dem zweiten Modell: kein Ausbau auf Vorrat.

**Backend ohne Generics und ohne anonyme Methoden.** Delphi 7 kennt beides nicht,
FPC 3.2.2 keine anonymen Methoden. Der Aufbau:
- **eine** Schnittstelle mit typisierten Operationen (MatMul, Add, LayerNorm, RMSNorm,
  RoPE, Softmax, GELU, SiLU, Attention), umgesetzt von einer virtuellen Basisklasse;
- intern Tabellen von Prozedurzeigern für die Skalar- und SIMD-Kernel, **einmal** beim
  Start per CPU-Erkennung von mORMot2 gewählt;
- **kein** Interface-Objekt je Operation.

**Fallback je Operation.** Jedes Backend meldet seine Unterstützung je Operation **und**
Datentyp/Layout/Form. Auf die CPU-Referenz wird nur bei „nicht unterstützt" ausgewichen,
nie nach einem Laufzeitfehler. Für die GPU-Phase gehören Gerätezuordnung, Pufferresidenz
und Scratch-Speicher vorher in den Entwurf: Ein stiller CPU-Fallback mitten im Graphen
kostet sonst Kopien und Synchronisation, die niemand sieht.

**CPU-Referenz ist Pflicht.** Jede Operation existiert zuerst in reinem Pascal. Sie läuft
überall, auch unter Delphi 7, und ist der Maßstab für alle optimierten Pfade.

**Threads.** `TSynParallelProcess` passt für grobe Zeilenblöcke, hat aber zwei
Eigenschaften, die man kennen muss: Aufrufe auf **einem** Pool laufen nacheinander, und
die Ausführung ist synchron. Deshalb:
- ein eigener Executor mit expliziter Zuteilung;
- Schwellen, unter denen nicht parallelisiert wird;
- nie denselben Pool rekursiv aufrufen.

**Tokenizer.** Ein Kern mit Adaptern für `tokenizer.json` (Hugging Face) und die
GGUF-Metadaten. Gleiche Charsmap-Bytes beweisen noch nicht dasselbe Verhalten:
Whitespace, Sondertokens und Nachbearbeitung werden einzeln gegen die Referenz geprüft.
Nutzertext wird mit `parse_special = false` tokenisiert (Referenzvertrag v1).

**Positionen bei BGE-M3.** Die Originalkonfiguration hat 8194 Positionen, im GGUF stehen
8192, weil der XLM-R-Offset (+2) beim Konvertieren entfernt wurde. Diese Behandlung
bleibt je Format ausdrücklich im Code und wird nie „passend gerechnet".

## Phasen

Reihenfolge (mit Codex abgestimmt): Im FP32-Pfad lassen sich Fehler am leichtesten
finden. Der nutzbare Ersatz für lembed (GGUF/Q8) kommt **vor** dem Decoder, damit der
erste Gewinn nicht hinter dem größten Umfang wartet.

### Phase 0 — Fundament

- ~~Lizenz festlegen~~ — **entschieden 2026-10-14**: mORMot-Dreifachlizenz
  (MPL 1.1 / GPL 2.0 / LGPL 2.1) für das ganze Repo, auch für `src/infer/`.
- Python-Referenzskript mit `transformers`, eingecheckt unter `tools/`. Folgendes wird
  festgelegt: Python- und Paketversionen, Modell-Revision und Hashes, FP32,
  `eval()`-Modus, Attention-Implementierung, Tokenizer-Einstellungen. Die verbindliche
  Referenzumgebung ist **CPU auf WSL aarch64**; GPU-Ausgaben dienen später nur als
  Vergleich.
- Eingecheckt (`tests/data/infer-ref/`):
  - Manifest;
  - Token-IDs, Masken, Positionen;
  - End-Embeddings;
  - **vollständige** Zwischentensoren für wenige kurze Fälle (16 Tokens × 24 Layer ≈
    1,5 MiB je Fall).

  Hashes oder Stichwerte genügen nicht: Sie lokalisieren keinen numerischen Fehler und
  ändern sich schon bei harmlosen Rundungsunterschieden.
- **Nicht** eingecheckt (gitignored, bei Bedarf erzeugt): Aktivierungen über den ganzen
  Korpus, Attention-Matrizen, Modelldateien.
- Winzige deterministische Modell- und Operator-Fixtures, damit die tägliche Testsuite
  ohne die 2-GB-Gewichte läuft.
- Testrahmen: mORMot `TSynTestCase` wie die bestehenden Suiten. Vergleiche über maximale
  absolute und relative Abweichung, Kosinus **und** Norm. Kosinus allein übersieht
  Skalierungsfehler.

**Fertig, wenn:** die Referenzdaten für BGE-M3 vorliegen und ein Test gegen die winzigen
Fixtures läuft.

**Erledigt (2026-10-14).** Ergebnis:
- Werkzeug [tools/infer-ref/](../tools/infer-ref/) mit fester Umgebung (Python 3.12.13,
  torch 2.14.1+cpu, transformers 5.19.0); Daten in
  [tests/data/infer-ref/](../tests/data/infer-ref/) (rund 7 MiB): Operator-Fixtures,
  Tiny-XLM-R, sechs BGE-M3-Fälle mit allen 25 Zuständen plus gepaddetem Batch,
  FP32-Vektoren des Korpus embed-de. Modellrevision, Datei-Hashes und Umgebung stehen in
  den Manifesten und in [UPSTREAM_BASE](../UPSTREAM_BASE).
- Zwei Läufe erzeugen bytegleiche Blobs und ein bytegleiches `model.safetensors`.
- Suite `infer.tests` (`scripts/run-fpc-infer-tests.sh`, `-Suite infer` in beiden
  Delphi-Skripten), 149 Assertions grün unter FPC 3.2.4 und 3.2.2 (aarch64-linux),
  Delphi 13 (Win64) und Delphi 7 (Win32). Sie prüft den **Testrahmen**, noch keinen
  Pascal-Forward:
  - Unversehrtheit (Hashes);
  - Abweisen beschädigter Daten;
  - Gegenproben für `InferCompare` (Skalierung, Ausreißer, NaN/Inf);
  - innere Konsistenz der Referenzen.

  Gegenproben: Eine ausgebaute Normierung, ein gekipptes Byte, eine ausgebaute
  Überlaufsperre im Lader und eine überzogene Kosinus-Schwelle lassen den Lauf
  fehlschlagen.

Befunde, die in Phase 1 eingehen:
- **Kein safetensors bei BGE-M3.** Die Revision enthält nur `pytorch_model.bin`
  (Pickle). Das Werkzeug konvertiert nach F32-safetensors (`vendor/models/bge-m3/`,
  gitignored) und prüft Tensoren und Forward gegen die Quelle. Phase 1a liest diese
  Datei.
- **Sondertokens im Text.** `tokenizer.json` führt `<s>` als Eintrag des
  Unigram-Vokabulars. Der HF-Tokenizer erkennt es deshalb auch mit
  `split_special_tokens` oder `encode_special_tokens` im Text (gemessen, Manifest
  `tokenizer_checks`). Die Referenz-IDs kommen
  aus SentencePiece (`sentencepiece.bpe.model`), das Steuersymbole nie im Text erkennt,
  mit fairseq-Offset. Für alle Texte ohne solches Literal sind beide gleich (geprüft).
  Der native Tokenizer muss die Steuer-Einträge beim Zerlegen auslassen.
- **FP32 gegen lembed-Q8:** Kosinus mindestens 0,99868, im Mittel 0,99929 über die 262
  Korpusdokumente.
- **Batch gegen Einzellauf:** Beim Tiny-Modell bitgleich. Bei BGE-M3 weicht die
  gepaddete Zeile um höchstens 1,3e-7 ab (Summationsreihenfolge). Die Engine-Tests
  vergleichen deshalb mit Toleranz, nicht bitgenau.

### Phase 1 — BGE-M3 als Ersatz für lembed

**1a. FP32 auf der CPU**
- Tensor-Kern, safetensors-Leser mit Memory-Mapping, naive MatMul in FP32.
- XLM-RoBERTa-Forward (nachgeschlagen in candle und `modeling_xlm_roberta.py`).
- Unigram-Tokenizer aus `tokenizer.json`, CLS-Pooling und L2-Normalisierung. Steuer-
  Einträge (`<s>`, `</s>`, `<pad>`, `<unk>`, `<mask>`) werden im Text nicht erkannt
  (Befund aus Phase 0). Verhalten wie SentencePiece, nicht wie der HF-Tokenizer: Der
  weicht auch bei C1-Steuerzeichen, Hangul-Jamo und kombinierenden Zeichen ab.

**Fertig, wenn:**
- die Token-IDs identisch sind;
- jeder Layer innerhalb fester absoluter und relativer Toleranzen liegt;
- das End-Embedding Kosinus ≥ 0,9999 und eine Norm von 1 ± 1e-5 hat;
- Batch und Einzelaufruf dasselbe liefern;
- Padding die Ergebnisse nicht verändert.

**Erledigt (2026-10-14).** Units unter `src/infer/`: `tensor`, `memory`,
`loader.safetensors`, `backend`, `backend.cpu`, `model.xlmr`, `tokenizer.ugm`,
`pipeline.embed` (`TInferEmbedder`). Ergebnis gegen die Referenz:
- Tokenizer: alle 1348 Fälle von `bge-m3-tokens` mit identischen IDs und identischem
  normalisierten Text (320 Korpustexte, 128 Grenzfälle, 900 Unicode-Mischungen), auch
  unter Delphi 7/Win32. Portiert aus llama.cpp (MIT, Herkunft in `UPSTREAM_BASE`).
- Encoder: alle 25 Zustände der sechs Fälle und des gepaddeten Batchs innerhalb
  atol 1e-4 / rtol 1e-3; gemessen höchstens 1,7e-5 absolut in den Zuständen und
  1,8e-7 im End-Embedding (Kosinus 0,999999999999).
- Text bis Vektor in einem gepaddeten Batch: Kosinus ≥ 0,9999 je Text; mehr als 512
  Tokens werden abgewiesen, nie gekürzt.
- Suite `infer.tests`: 752 Assertions grün unter FPC 3.2.4/3.2.2 und Delphi 13; unter
  Delphi 7 laufen alle Teile außer dem echten Modell (SKIP: die 2,3-GB-F32-Datei passt
  nicht ganz in einen 32-Bit-Prozess, das kommt mit Q8 in 1b).
- Die Referenzrechnung summiert durchgehend in double (jeder Faktor wird vor der
  Multiplikation erweitert), damit x87, SSE und NEON dieselbe Referenz liefern. Ein
  späterer SIMD-Pfad in FP32 wird gegen diese Referenz geprüft.
- Tempo (skalar, ein Thread, aarch64): rund 35 s für die sechs Fälle und den Batch.
  Geschwindigkeit ist Thema von 1c.

**1b. GGUF/Q8_0 — dasselbe Asset wie lembed** (vorher „Schritte 3 und 4" in
native-embeddings.md)
- GGUF-Leser, Q8_0-Dequantisierung bzw. Q8_0×F32-Kernel.
- Tokenizer-Adapter für die GGUF-Metadaten.

**Fertig, wenn:**
- die Token-IDs auf mindestens 1000 Unicode-Fällen identisch zu lembed sind. Wörtliche
  Sondertoken-Zeichenfolgen (`<s>`, `</s>`, `<pad>`) im Text werden **getrennt** gegen das
  gewollte Verhalten `parse_special = false` geprüft, nicht gegen lembed (das sie als
  Steuertokens liest);
- beschädigte Dateien sicher abgelehnt werden;
- der Kosinus gegen die Referenzvektoren aus `tests/data/embed-de` (lembed) ≥ 0,999 ist;
- **nDCG@10 ≥ 0,8863** erreicht wird, also höchstens 1 Prozentpunkt unter 0,8963.

Der Referenzvertrag v1 ist an **genau dieses** Q8-GGUF, 512 Tokens und das Verhalten der
Sondertokens gebunden. Ein bestandener FP32-Test (1a) beweist deshalb noch **keine**
Verträglichkeit mit bereits gespeicherten Vektoren. Beide Referenzen behalten getrennte
Identitäten (Vertrag, Fingerprint).

**Erledigt (2026-10-14).** Neu: `loader.gguf`; GGUF-Konstruktoren in `model.xlmr`,
`tokenizer.ugm` und `pipeline.embed` (`TInferEmbedder.CreateFromGguf`); Q8_0 als
Blocktyp in `tensor`; `Linear` nimmt typisierte Gewichte (`TInferWeight`). Ergebnis:
- **Rechenweg, gemessen vor dem Bau** (PyTorch mit den Q8-Gewichten gegen die
  lembed-Vektoren der 320 Korpustexte): Q8-Gewichte mit F32-Aktivierungen erreichen
  nur Kosinus 0,99829 (ein Text unter 0,999). Erst wenn die Aktivierungen wie bei ggml
  vor jeder Matrixmultiplikation auf Q8_0 quantisiert werden, liegt jeder Text darüber
  (min 0,99949). Die FP16-Tanh-GELU-Tabelle von ggml war schlechter (min 0,99883),
  daher bleibt die exakte GELU. Die Engine rechnet also wie ggml-cpu auf aarch64:
  Skala `max|x|/127` in FP16, `x·(1/d)` auf die nächste gerade Zahl gerundet
  (NEON und AVX runden so, die C-Referenz von ggml nicht), ganzzahlige Blocksummen.
  Der AVX-Pfad bildet den Kehrwert als `127/max|x|` und kann einen Grenzfall um eine
  Stufe verschieben: bytegleich zum aarch64-Orakel, nicht zu jedem ggml-Build.
- **Positionen:** Der Konverter von llama.cpp lässt die ersten pad + 1 Zeilen weg
  (8192 statt 8194). Die Regel steht je Format im Code (`FirstPosition`: pad + 1 bei
  safetensors, 0 bei GGUF) und ist mit Gegenprobe getestet.
- **Token-IDs:** alle 1318 Fälle ohne Sondertoken-Literal identisch zu lembed, alle
  1348 identisch zu SentencePiece. Die 30 Literal-Fälle prüft der Test getrennt: lembed
  liest dort in 24 Fällen Steuertokens, die Engine Text (`parse_special = false`).
  Stücke vom Typ UNUSED (`[PAD250000]`) erkennt auch lembed nie im Text (Extra-Fälle).
- **Vektoren und Gate** (ganzer Korpus, `LAI_INFER_GATE=1`): Kosinus gegen lembed
  mindestens 0,99953 (alle 320 Texte über 0,999), **nDCG@10 0,8989**
  (Gate 0,8863, lembed 0,8963).
- **Beschädigte Dateien:** rund 30 gezielte Defekte (Zählungen, Typen, NUL, Schachtelung,
  Ausrichtung, Überlappung, Lücken, Reihenfolge), jede Kürzung der Tiny-Datei und 4096
  gekippte Header-Bytes enden in `EInferError`, nie in einem anderen Fehler.
- **Fixtures aus ggml selbst** (`q8`, `tiny-gguf`, `lembed-tokens`): FP16-Rundung,
  Aktivierungsquantisierung byte-genau, Q8_0-Skalarprodukt, ein Tiny-XLM-R als GGUF
  mit 64-Stück-Tokenizer.
- **Plattformen:** `infer.tests` grün unter FPC 3.2.4/3.2.2, Delphi 13 und Delphi 7.
  Unter Delphi 7/Win32 läuft jetzt auch das echte Modell: Das Q8-GGUF (634 MB) passt
  ganz in den Adressraum, Mapping in Fenstern braucht erst ein größeres Modell.
- **Tempo** (skalar, ein Thread, aarch64): der ganze Korpus in rund 51 min;
  lembed braucht rund 72 s. Das ist Thema von 1c.

**1c. Tempo, Messung, Umschalten** (vorher „Schritte 5 und 6")
- Minimal-SIMD für das Q8_0-Skalarprodukt (NEON, AVX2) und Threads über Zeilenblöcke.

**Fertig, wenn:**
- der Durchsatz mindestens 80 % von lembed erreicht, bei höchstens 125 % Peak-RSS;
- der Durchsatz unter Dauerlast stabil ist: Soak-Test über 90 Minuten unter Linux x86_64,
  gedrosselt auf 16 Threads, jeder Vektor über alle Durchgänge bitgleich (`embed-bench …
  soak90 16`). Die Erstindizierung von 2 Mio Texten wird daraus hochgerechnet, nicht
  abgewartet: 28 h Volllast wären für den Rechner unnötige Last (geändert 2026-10-14, mit
  dem Nutzer abgestimmt);
- der Build unter FPC 3.2.2/3.2.4 und Delphi 7 bis 13 grün ist (Delphi 7 ohne
  Leistungsziel).

Danach wird **nativ** zum Default. lembed bleibt als expliziter Fallback und als
unabhängiges Orakel erhalten, bis die nativen Gates auf allen Zielplattformen stehen.

**Stand 2026-10-14: Teil a (aarch64) erledigt.** Neu: `kernels` (NEON, Befehle als
Kodierung, weil FPC 3.2 keine NEON-Registerlisten assembliert), `threads` (eigener Pool),
`TInferCpuFastBackend` in `backend.cpu`, der Adapter `mormot.ai.embed.native`
(`TNativeEmbedder`, `SharedNativeEmbedder`) und `demos/infer/embed-bench.lpr`.
- **Zwei Rechenverträge** (mit Codex entschieden): `Exact = true` rechnet bitgleich zur
  skalaren double-Referenz, nur auf allen Kernen. Der Default rechnet wie ggml in
  float-Spuren (Q8_0-Linear: SDOT-Teilsummen, je Block mit dw·dx skaliert, am Zeilenende
  reduziert; Attention in float; GELU über eine kubische Hermite-Tabelle, Fehler < 1e-9)
  und wird mit Toleranz gegen die Referenz geprüft. Exakt bleiben in beiden Modi die
  Quantisierung der Aktivierungen (byte-gleich zu ggml, NEON wie Pascal) und die
  Integer-Blocksummen. Ein Aufrufer, der den Pool belegt findet, rechnet mit denselben
  Kernen auf seinem Thread: dieselben Bits, unabhängig von der Last.
- **erf neu:** eigene Polynomanpassung (`tools/infer-ref/fit_erf.py`, mpmath),
  relativer Fehler < 1e-14, rund 20 ns statt bis zu 200 ns je Wert.
- **Qualität (schneller Pfad, ganzer Korpus):** nDCG@10 **0,9004** (lembed 0,8963,
  Gate 0,8863), Kosinus gegen lembed mindestens 0,99949; der Korpus in 34 s statt
  51 min auf der Referenz.
- **Durchsatz** (Snapdragon X, WSL aarch64, ganzer Korpus, im Wechsel gemessen, Modell
  auf dem Linux-Dateisystem - über `/mnt/d` kostet das Mapping rund 25 %):

  | | lembed | nativ | Anteil |
  |---|---|---|---|
  | 4 Kerne (taskset, je 4 Threads), Runde 1 / 2 | 267 / 216 Tokens/s | 284 / 280 | 106 % / 130 % |
  | ganze Maschine, je Default, Runde 1 / 2 | 251 / 222 | 285 / 238 | 113 % / 107 % |
  | 1 Kern, 40 Texte | 114 | 96 | 84 % |
  | Peak-RSS | 760 MB | 480 MB | 63 % |

  Default sind zwei Drittel der Kerne: Alle Kerne spinnen zu lassen ist in der VM
  langsamer. Unter Dauerlast drosselt der Laptop; deshalb wechseln die Läufe ab.
- **Hochrechnung Erstindizierung:** 5,6 Texte/s bei ⌀ 51 Tokens ergibt für 2 Mio Texte
  rund 100 h auf diesem Laptop (Hochrechnung; das Gate ist der Soak-Test, siehe oben).
- **Reviews:** Codex und Claude-Finder fanden u. a. einen ABA-Fehler im Pool (Aufträge
  mit 8, 2, 8 Einträgen konnten Count und Base mischen - jetzt Seqlock), einen
  Überlauf des Ticketzählers (jetzt Neustart bei 2^30 im Ruhezustand), den verlorenen
  `Exact`-Modus im seriellen Attention-Pfad, einen Delphi-7-Startfehler (signalisierendes
  NaN durch die x87-FPU) und Tests, die NaN durchließen. Alle behoben, mit Gegenprobe.

**Stand 2026-10-14: Teil b (x86_64, AVX2) erledigt.**
- **Kernel** in `kernels` unter `{$ifdef CPUX64}{$ifdef ASMX64AVX1}`, Auswahl zur Laufzeit
  bei AVX2 + FMA + F16C: der FP32-Kernel 4×2 (vpsignb/vpmaddubsw/vpmaddwd wie ggml, FMA
  mit dw·dx), der Quantisierer (byte-gleich zu Pascal), die exakten Integer-Blocksummen
  (vpmovsxbw + vpmaddwd, auch für -128) und die beiden float-Attention-Kernel.
- **Eine ABI für alle:** Jeder Kernel bekommt genau einen Record mit seinen Parametern
  und Platz für die Register, die die ABI zurückverlangt (unter Win64 auch xmm6 bis
  xmm15). Der Asm-Körper ist damit für Win64 und System V, FPC und Delphi derselbe; eine
  Pascal-Hülle mit der gewohnten Signatur füllt den Record.
- **F16C als Bytes:** FPC 3.2.2 kennt `vcvtph2ps`/`vcvtps2ph` nicht; per objdump geprüft.
- **Vertrag des FP32-Kernels:** Aktivierungen liegen in -127..127 (der Quantisierer
  schreibt nie -128). Der vpsignb-Trick von ggml verlangt das; der Test hält sich daran.
- **LayerNorm auf allen Kernen**, zeilenweise, je Zeile die Referenzarithmetik: bitgleich
  in beiden Modi. Auf dem i9 fiel sie damit von 26 % auf 3 % der Zeit.
- **Messung** (i9-14900K, Windows 11, FPC 3.2.2 win64; Zahlen in der README): nativ
  1039 Tokens/s gegen lembed 524 (×1,98), 377 MB gegen 514 MB Spitzenspeicher, Kosinus
  ≥ 0,9991. Default (22 Threads) ist zugleich der beste Wert.
- **Windows drosselt fensterlose Prozesse** (Power Throttling/EcoQoS, z. B. per SSH oder
  als Dienst gestartet): Beide Engines verlieren dann etwa die Hälfte (lembed 187, nativ
  417). `embed-bench` schaltet das für sich ab; ein Windows-Dienst, der viel einbettet,
  sollte es ebenso tun.
- **Erstindizierung, hochgerechnet:** 20 Texte/s bei ⌀ 51 Tokens ergeben für 2 Mio Texte
  rund 28 h auf dem i9.

- **Linux x86_64** (Ubuntu 24.04 unter WSL2 auf dem i9, FPC 3.2.2 wie im Docker-Build,
  2026-10-14): MCP 1257, LLM 656, Inferenz 828 mit Korpus-Gate (nDCG@10 0,8994, Kosinus
  ≥ 0,99918), der System-V-Pfad damit erstmals auf Hardware gelaufen. Durchsatz nativ
  861 gegen lembed 544 Tokens/s (×1,58), 439 gegen 755 MB.
- **Fund dabei:** Die Fixture `tests/data/infer-ref/tiny-gguf/model.gguf` fiel unter
  `*.gguf` in `.gitignore` und fehlte in jedem frischen Klon (26 Fehlschläge). Jetzt mit
  Ausnahme eingecheckt.

- **Soak-Test bestanden** (2026-10-14, Linux x86_64 unter WSL2 auf dem i9, FPC 3.2.2,
  16 Threads, `embed-bench … soak90 16`): 90 Minuten, 248 volle Durchgänge über den
  Korpus plus 268 Texte, zusammen 79.628 Texte und 4,05 Mio Tokens. Durchsatz über alle
  neun 10-Minuten-Intervalle 744 bis 752 Tokens/s, RSS durchgehend 437 MB (Spitze 442):
  kein Einbruch und kein Wachstum beobachtet. **0 Vektoren** weichen vom ersten Durchgang
  ab (79.308 bitgenaue Vergleiche). CPU bei 88 bis 90 °C, Lüfter 100 %; Microcode 0x12F
  (die Raptor-Lake-Korrektur ab 0x12B ist drauf). Nach Überschlag (rund 25.000
  Pool-Tickets je Text, zusammen rund 2,0 Mrd) ist der Ticket-Neustart bei 2^30 dabei
  einmal gelaufen; gezählt hat das Programm ihn nicht.
- **Erstindizierung, hochgerechnet:** aus dem Soak 14,7 Texte/s mit 16 Threads, für
  2 Mio Texte rund 38 h unter Linux; aus der Einzelmessung mit dem Default von 22
  Threads (861 Tokens/s) rund 33 h, über Dauer nicht gemessen.

- **Umschalten im Backend** (2026-10-14): Provider `native` in
  `landrix.server.ai.Embedder` - braucht nur `AI_EMBED_MODEL_PATH`, `AI_EMBED_THREADS`
  optional; `.env.sample`, Docker-Doku und Entrypoint führen ihn als empfohlenen Weg,
  lembed bleibt Rückfall und Orakel. Der Feature-Schalter bleibt per Default `none`
  (kein Modell im RAM, solange niemand AI einschaltet). Ein echter Warmup-Test lädt
  dabei die 32-KB-Fixture `tiny-gguf`.
- **Load Governor** (`mormot.ai.governor`, im Backend `GetBulkEmbedder` mit
  `AI_BULK_*`): Tastgrad, Temperaturschwelle über Linux-hwmon, Durchsatz-Heuristik.
  Review-Fund dabei: sysfs meldet für jede Datei 4096 Bytes, `StringFromFile` lieferte
  deshalb leere Strings - die Temperatur war nie lesbar; jetzt `StringFromFileNoSize`,
  mit Regressionstest gegen das echte `/sys` und Gegenprobe.

**Phase 1c ist abgeschlossen.** Offen bleibt nur der erste Aufrufer des Governors (ein
Indizier-Job im Backend) und ein Lauf auf echter Sensor-Hardware.

### Phase 2 — Decoder

- Gerüst nach llama2.c (MIT), umgebaut auf konfigurierbare Blöcke.
- Erstes Modell **Qwen3-0.6B**: Q/K-Norm, `head_dim` aus der Konfiguration, GQA.
- Byte-Level-BPE, KV-Cache, Greedy-Sampling.
- Chat-Template der unterstützten Modelle fest in Pascal.
- `mormot.ai.llm.native` hinter `ILlmClient`, mit Streaming und Tool-Calls.

**Fertig, wenn:** Greedy-Sampling in FP32 für einen festen Prompt-Satz tokengleich zu
`transformers` ist.

**Stand 2026-10-14:** 2-0 bis 2c erledigt, das Gate ist erfüllt. Offen ist 2d.

- **2-0, Referenzdaten** (`tools/infer-ref/make_reference_qwen3.py`, Qwen/Qwen3-0.6B in
  der Revision `c1899de289a04d12100db370d81485cdf75e47ca`):
  - `qwen3-tokens`: 2416 Fälle für den Tokenizer;
  - `qwen3-chat`: 21 Gespräche für das Chat-Template, mit Tools, Tool-Aufrufen und
    Thinking;
  - `tiny-qwen3`: ein winziges Qwen3 mit 2 Layern, BF16-Gewichten und `head_dim` 32
    statt 64/4, mit allen Zuständen;
  - `qwen3`: sieben Prompts des echten Modells in FP32, mit Zuständen, Logits und je
    48 Greedy-Tokens samt dem Abstand der zwei besten Logits pro Schritt.
- **2a, Tokenizer** (`mormot.ai.infer.unicode`, `mormot.ai.infer.tokenizer.bpe`):
  - NFC und die Zeichenklassen des Split-Regex (`\p{L}`, `\p{N}`, `\s`) stehen als
    erzeugte Tabellen im Code. Sie sind am HF-Tokenizer selbst gemessen und über alle
    Codepoints dagegen geprüft.
  - Befund dabei: Der NFC des HF-Tokenizers beruht auf älteren Unicode-Daten als Python
    (15.0), mit 1129 Abweichungen. Die Tabellen folgen deshalb dem Tokenizer, nicht der
    neuesten Unicode-Version.
  - Das Regex läuft nicht über eine Regex-Engine, sondern als handgeschriebener Abgleich
    seiner sieben Alternativen. Ein Fuzzing mit 200.000 Fällen gegen den HF-Pre-Tokenizer
    ergab keine Abweichung.
  - BPE arbeitet mit derselben Merge-Reihenfolge wie die tokenizers-Bibliothek: Rang,
    dann Position, veraltete Einträge werden übersprungen.
- **2a, Chat-Template** (`mormot.ai.infer.chat`): Das Jinja-Template von Qwen3 ist in
  Pascal nachgebaut. Tools werden wie `json.dumps` von Python serialisiert
  (`PythonJson`).
  - Eine Abweichung ist gewollt: Steuertokens kommen nur aus dem Template, Nachrichtentext
    bleibt immer Text. Ein `<|im_end|>` im Text eines Nutzers schließt also nicht seinen
    Turn.
  - Floats mit mehr als 15 signifikanten Stellen weist `PythonJson` ab, statt die
    binäre Rundung zu raten.
- **2b, Forward** (`mormot.ai.infer.model.qwen3`):
  - Aufbau: RMSNorm, Q/K-Norm je Kopf, RoPE, GQA, kausale Attention, SwiGLU, verbundener
    oder eigener LM-Head.
  - BF16-Gewichte werden beim Laden exakt auf F32 erweitert.
  - Abweichung von der Referenz: Zustände und Logits weichen höchstens um 9e-5 ab
    (Kosinus 0,99999999998).
  - Rotationswinkel: Sie folgen den float32-Schritten der Referenz.
- **2c, KV-Cache und Greedy:**
  - Logits über den Cache sind bitgleich zu einem vollen Durchlauf.
  - Alle sieben Prompts sind tokengleich, auch einer mit nur 0,0046 Abstand zwischen den
    zwei besten Tokens und der Tool-Aufruf.
  - Gegenprobe: Schlüssel an Position 0 statt an ihrer Cache-Position machen den
    Cache-Test und das Greedy-Gate rot.
  - Zeiten auf dem Snapdragon X (WSL, FP32, Pascal-Skalarcode): 4 + 48 Tokens in
    13 s, 193 + 20 Tokens in 46 s. Phase 2 ist bewusst auf Korrektheit gebaut,
    Geschwindigkeit ist Phase 3.
- **2d, offen:** `mormot.ai.llm.native` hinter `ILlmClient`. Dazu gehören:
  - Streaming aus der Dekodier-Schleife;
  - Stop-Tokens;
  - das Übersetzen von `<tool_call>` in `TLlmToolCall`.

Qwen3-Embedding ist **kein** Nebenprodukt: Es braucht einen eigenen, separat trainierten
Checkpoint und eigene Eingabekonventionen, nicht nur Last-Token-Pooling. Es gehört zu
„Danach".

### Phase 3 — Quantisierung und Geschwindigkeit

- Q4_0, später K-Quants (ggml, MIT, ausgewählt übernommen).
- Volle SIMD-Kernel (AVX2, NEON) und Multithreading.
- Sampler: Temperature, Top-k, Top-p, Stop-Tokens.
- Benchmarks je Plattform.

**Fertig, wenn:** quantisierte Modelle die Referenztests mit festgelegter Toleranz
bestehen und die Benchmarks dokumentiert sind.

### Phase 4 — GPU

- CUDA über die Driver API, dynamisch geladen.
- Kernel als eingebettetes PTX (Vorlagen: calm, llama.cpp, beide MIT).
- Reihenfolge: MatMul → Norm/RoPE/Softmax → Attention.
- Optional Vulkan Compute für herstellerübergreifenden Einsatz.
- Zielhardware: der RTX-Rechner, Windows 11 x64 mit RTX 5090 (32 GB VRAM, ca. 1,8 TB/s;
  mit 4 Bit Modelle bis ca. 30B).
- Zur Laufzeit genügt der Treiber (`nvcuda.dll` bzw. `libcuda.so`). Das PTX entsteht beim
  Bauen und wird als Ressource eingecheckt. Dafür braucht der Bau-Rechner das
  CUDA-Toolkit (`nvcc`), das heute noch fehlt.

**Fertig, wenn:** das GPU-Backend dieselben Referenztests besteht wie die CPU.

### Danach — nach Bedarf

Weitere Architekturen (Llama 3, Mistral, Gemma), Reranker (BGE-Reranker),
Sparse-/ColBERT-Ausgabe von BGE-M3, Qwen3-Embedding, MoE.

Eine eigene Spur daneben ist die Verteilung großer Jobs auf ungenutzte Rechner:
[distributed-execution.md](distributed-execution.md). Ihr Executor ist diese Engine.

## Plattformen und Tests

### Messung am 2026-10-14

| Plattform | Compiler | Ergebnis |
|---|---|---|
| WSL Linux aarch64 | FPC 3.2.4 (Baseline) + 3.2.2 | grün (MCP 1257, LLM 656, Inferenz 1215, alle Demos) |
| WSL2 Linux x86_64 auf dem RTX-Rechner (Ubuntu 24.04) | FPC 3.2.2 (wie Docker) | grün aus einem sauberen `git archive`: MCP 1257, LLM 656, Inferenz 828 mit Korpus-Gate |
| macOS M2, aarch64-darwin | FPC 3.2.4-rc2 per SSH | grün: MCP 1257, LLM 647 (der vec0-Test wird mangels Extension übersprungen) |
| Windows ARM, aarch64-win64 | FPC 3.3.1 (fpcupdeluxe, `D:\bin\fpc`) | grün: MCP 1257 und LLM, **aber** nur mit einer ARM64-`sqlite3-64.dll` (sqlite.org 3.51.2) neben dem Binary. mORMot2 hat für dieses Ziel kein statisches SQLite. Ein Skript gibt es noch nicht. |
| Windows x64 | Delphi 13 | grün (MCP 1257, LLM 659, Inferenz 1205) |
| Windows x64 unter Prism | FPC 3.2.2 | nur Kompatibilitätsprobe, nicht für Leistungsaussagen |
| Windows 11 x64, i9-14900K, 96 GB, RTX 5090 (32 GB, Treiber 616.64), per SSH (`ai-rtx5090`) | FPC 3.2.2 win64 (lokal übersetzt, nativ ausgeführt) | grün, Stand `bfd2507` + README-Umbau: MCP 1257, LLM 656 (mit vec0), Inferenz 821 (ohne konvertiertes FP32-Modell, dessen Tests überspringen); das Delphi-13-Testbinary dort ebenfalls grün (Inferenz 818 vor den letzten Tests). Delphi 13 ist installiert, CUDA-Toolkit (`nvcc`) nicht. |
| Windows x86 | **Delphi 7** | grün nach dem Umbau (MCP 1253, LLM 647, Inferenz 669; vec0 übersprungen, es gibt kein 32-Bit-Bundle), siehe unten |

### Rollen im Testplan

Compiler-Verträglichkeit, numerische Korrektheit und Leistung werden **getrennt**
geprüft. Grüne Ergebnisse aus übersprungenen Tests darf es nicht geben:
- Pflichttests ohne Modell sind von modellabhängigen Läufen getrennt.
- Übersprungenes wird ausdrücklich gemeldet.
- Ungetestete mORMot-Ziele gelten als „ungeprüft".

| Plattform | Rolle |
|---|---|
| WSL aarch64, FPC 3.2.4 + 3.2.2 | Hauptkorrektheit; Differenztests Skalar ↔ NEON |
| macOS M2 | Mapping, Dateisystem, NEON-Korrektheit; native Benchmarks |
| Windows x64 (RTX-Rechner) | AVX2-Korrektheit und native x64-Leistung (auch POPCNT für den Store); CUDA-Differenztests und -Leistung |
| Linux x64 | Deploy-Ziel (Docker); auf dem RTX-Rechner über WSL2 (Ubuntu 24.04, FPC 3.2.2, Konto `rtx5090`), seit 2026-10-14 eingerichtet |
| Delphi 13 Win64 | Compiler/ABI und Laufzeitkorrektheit |
| Delphi 7 Win32 | Build plus kleine Tests zu RTTI, Unicode, Numerik und Speichergrenzen; **kein** Leistungs-Gate |
| Windows ARM, FPC 3.3.1 | experimentell, bis Skript und SQLite-DLL geregelt sind |
| Windows x64 unter Prism | Kompatibilitätsprobe |

## Delphi 7

> **Erledigt (2026-10-14):** alle 22 Units bauen, beide Suiten laufen unter Delphi 7 grün
> (`scripts/run-delphi7-tests.ps1`), ebenso weiter unter FPC 3.2.4/3.2.2, Delphi 13, macOS
> und Windows ARM. Umgesetzt wie unten geplant; dazu ein Fund: der stdio-Transport hätte
> unter Delphi 7 eingehendes UTF-8 doppelt kodiert (`StringToUtf8` auf einem AnsiString) —
> behoben in `McpReadLine`, abgesichert durch `StdioReadsUtf8Unchanged`, das erstmals den
> echten Lesepfad prüft. Vorher dazugekommen: `RegistriesKeepTheirContract` und
> `TypedToolCarriesManagedFields` als Netz für den Umbau, danach
> `GenericShellMatchesRecordTool`. Fallen für die Engine: siehe „Praxis" unten.

**Befund vor dem Umbau (2026-10-14, dcc32 gegen mORMot2 master):**
- mORMot2-Kern, `mormot.db.raw.sqlite3.static` und `mormot.net.client` bauen.
- Von den 22 landrixai-Units bauen **11** unverändert: `llm`, `llm.types`, `llm.openai`,
  `llm.anthropic`, `llm.sse`, `embeddings`, `embed.provider`, `vectorstore`, `rag`,
  `sqlite`, `mcp.tools`.

Die Hindernisse:

| Ursache | Stellen | Folge |
|---|---|---|
| `TMcpToolBase<T: record>` (generische Klasse, `mcp.pas:1068`) | 1, dazu die Spezialisierung `rag.tool.pas:75`; im Konsumenten rund 200 Nutzungen | `mcp`, `mcp.stdio`, `mcp.legacy`, `mcp.server`, `rag.tool`, `llm.structured`, `agent.mcp` |
| generische Registries `IKeyValue<>`/`TPair<>` aus `mormot.core.collections` (`mcp.pas:1153 ff.`, `:3386 ff.`) | 12 | MCP-Kern |
| `exit(wert)` | ~37 (`mcp.server` 21, `mcp` 12, `agent`, `agent.mcp`, `vectorstore.native`) | u. a. `agent` |
| Record mit Methoden (`TTopK`, `vectorstore.native.pas:267`) | 1 | `vectorstore.native` |
| `for … in` | 4 (`mcp.pas`) | MCP-Kern |
| UTF-8-BOM | `mcp.server.pas` | `mcp.server` |
| fehlendes `uses mormot.core.unicode` (`Utf8ToString`) | 2 | `embed.lembed`, `vectorstore.sqlitevec` |

Hinter den ersten Fehlern können weitere liegen.

**Vorgehen (mit Codex abgestimmt):**
1. **Mechanisch:**
   - `exit(x)` ersetzen;
   - BOM entfernen und `uses` ergänzen;
   - `TTopK` nach mORMot-Muster umstellen
     (`{$ifdef USERECORDWITHMETHODS}record{$else}object{$endif}`);
   - `for … in` ersetzen.
2. **Registries:** für **alle** Compiler auf eine nicht-generische mORMot-Struktur
   (`TSynDictionary`) umstellen. Ein Codepfad ist besser als `{$ifdef}`-Zweige.
3. **Typisierte Tools:** eine nicht-generische Basis, die einen RTTI-Deskriptor nimmt.
   - `TypeInfo` allein liefert unter Delphi 7 **keine** Record-Feld-Metadaten. Die
     Basis nimmt deshalb registrierte `TRttiCustom`-Beschreibungen
     (`Rtti.RegisterFromText`, mit dessen Layout-Regeln).
   - Initialisierung und Freigabe verwalteter Felder (Strings, dynamische Arrays) der
     Argument-Records müssen dabei erhalten bleiben.
   - `TMcpToolBase<T>` bleibt unter `HASGENERICS` als dünne Hülle darüber, sodass der
     Konsument unverändert bleibt.
4. **Tests:**
   - eigenes Skript `run-delphi7-tests.ps1` (Win32, dcc32, explizite Pfade; die
     installierte `dcc32.cfg` zeigt auf einen falschen Pfad);
   - Programme, die Generics brauchen, werden ausgeklammert;
   - Pflicht sind Laufzeittests zu Schema-Erzeugung, Serialisierung, Unicode, Store und
     Numerik, nicht nur „baut".
5. Das Backend des Konsumenten bleibt FPC-only und ist **nicht** Teil dieser Portierung.

Hauptrisiko: das Ganze als reine Syntaxarbeit zu behandeln, während Schema-Erzeugung,
Serialisierung oder die Lebensdauer verwalteter Felder zur Laufzeit brechen. Aufwand:
einige Tage.

**Praxis (beim Umbau gelernt, gilt auch für die Engine):**
- Keine Array-Konstruktoren (`TFoo.Create(1, 2)`), kein `Default()`, kein `as` vom
  Interface auf die Klasse (`ObjectFromInterface`), kein `TThread.Start` (`TSynThread`),
  keine Konstante `-2147483648` (`Low(integer)`), kein `UnicodeString` ohne
  `HASVARUSTRING`, keine globale `FormatSettings` (vor Delphi XE).
- Kein UTF-8-BOM und keine Nicht-ASCII-Zeichen in String-Literalen: Delphi liest eine
  Datei ohne BOM als ANSI.
- Nach dem ersten Fehler meldet dcc32 oft Folgefehler in fremden Units (auch in mORMot) —
  immer den ersten Fehler beheben.
- Delphi 7 kürzt den Programmnamen am ersten Punkt (`mcp.tests` → `mcp.exe`) und braucht
  FastMM4 als eigene Bibliothek.

**Engine unter Delphi 7 / Win32:**
- Reine Pascal-Referenz, keine SIMD-Pflicht. Assembler wäre möglich, ist aber kein Ziel.
- Große Gewichte als Q8_0/F16; Normparameter und Ähnliches bleiben F32. Akkumuliert wird
  in FP32.
- Kontext, Batch, Scratch und KV-Cache werden begrenzt. Das Modell wird nie komplett auf
  FP32 entpackt.
- Adressraum: 2 GB, mit LAA auf 64-Bit-Windows 4 GB. Der Mapping-Helfer von mORMot lehnt
  auf CPU32 Ansichten über `MaxInt` ab, deshalb mappt `mormot.ai.infer.memory` in Fenstern.
- Gate: nur Korrektheit, keine Leistung.

## Vorlagen und Lizenzen

| Bereich | Projekt | Lizenz | Nutzung |
|---|---|---|---|
| Decoder-Gerüst | llama2.c (Karpathy) | MIT | portieren |
| Modelldetails | candle (Hugging Face) | MIT / Apache 2.0 | nachschlagen; Code nur aus dem MIT-Zweig |
| Spezifikation | transformers (Hugging Face) | Apache 2.0 | **nur** nachschlagen |
| Tokenizer | tokenizers (Hugging Face) | Apache 2.0 | **nur** nachschlagen (Clean-Room) |
| Quantisierung, GGUF, CPU-Kernel | llama.cpp / ggml | MIT | ausgewählt portieren |
| CUDA-Kernel | calm (zeux), llama.cpp | MIT | ausgewählt portieren |

**Nicht verwenden:** PasLLM (AGPL 3.0) und neural-api (LGPL 2.1) als Vorlage.

**Pflichten:**
- Bei Portierung den Copyright-Hinweis der Vorlage mitführen und die Herkunft je Datei
  festhalten.
- Apache-2.0-Code wird **nicht** übernommen: Er ist mit GPLv2, einem Zweig der
  Dreifachlizenz, unverträglich und lässt sich nicht einfach umetikettieren.
- Vor jeder Übernahme die aktuelle LICENSE-Datei der Vorlage prüfen. (Keine
  Rechtsberatung.)

**Modelle:** BGE-M3 (MIT), Qwen3 (Apache 2.0, nur als Gewichte genutzt, nicht als Code).

## Richtwerte

| | |
|---|---|
| BGE-M3 | 24 Layer, Hidden 1024, 16 Köpfe, FFN 4096, ca. 568M Parameter |
| BGE-M3 Speicher | FP32 ca. 2,2 GB, Q8_0 ca. 600 MB |
| Generierung | durch die Speicherbandbreite begrenzt: Tokens/s ≈ Bandbreite / Modellgröße |
| lembed heute (Snapdragon X, WSL) | ~25 Texte/s kurz, ~3,6–5,7 Texte/s bei ⌀ 51 Tokens |

## Offene Entscheidungen

1. ~~Lizenz~~ — **entschieden 2026-10-14**: mORMot-Dreifachlizenz für das ganze Repo,
   auch für die Engine (`LICENSE`).
2. ~~Zeitpunkt der Delphi-7-Portierung~~ — **erledigt 2026-10-14**, vor dem ersten
   Engine-Code: die Engine entsteht von Anfang an unter `run-delphi7-tests.ps1`.
3. **Maximale Kontextlänge** für Embeddings: 512 Tokens wie im Vertrag v1, oder mehr? Das
   bestimmt den Speicherbedarf der Attention.
