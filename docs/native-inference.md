# Native Inferenz in Pascal — Plan

> **Status (2026-10-14):** Plan, noch kein Code. Grundlage ist der Entwurf des Nutzers
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
| `mormot.ai.infer.memory` | Memory-Mapping (auch in Fenstern für Win32), Scratch-Puffer |
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

### Phase 1 — BGE-M3 als Ersatz für lembed

**1a. FP32 auf der CPU**
- Tensor-Kern, safetensors-Leser mit Memory-Mapping, naive MatMul in FP32.
- XLM-RoBERTa-Forward (nachgeschlagen in candle und `modeling_xlm_roberta.py`).
- Unigram-Tokenizer aus `tokenizer.json`, CLS-Pooling und L2-Normalisierung.

**Fertig, wenn:**
- die Token-IDs identisch sind;
- jeder Layer innerhalb fester absoluter und relativer Toleranzen liegt;
- das End-Embedding Kosinus ≥ 0,9999 und eine Norm von 1 ± 1e-5 hat;
- Batch und Einzelaufruf dasselbe liefern;
- Padding die Ergebnisse nicht verändert.

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

**1c. Tempo, Messung, Umschalten** (vorher „Schritte 5 und 6")
- Minimal-SIMD für das Q8_0-Skalarprodukt (NEON, AVX2) und Threads über Zeilenblöcke.

**Fertig, wenn:**
- der Durchsatz mindestens 80 % von lembed erreicht, bei höchstens 125 % Peak-RSS;
- der Durchsatz für die Erstindizierung von 2 Mio Texten gemessen ist;
- der Build unter FPC 3.2.2/3.2.4 und Delphi 7 bis 13 grün ist (Delphi 7 ohne
  Leistungsziel).

Danach wird **nativ** zum Default. lembed bleibt als expliziter Fallback und als
unabhängiges Orakel erhalten, bis die nativen Gates auf allen Zielplattformen stehen.

### Phase 2 — Decoder

- Gerüst nach llama2.c (MIT), umgebaut auf konfigurierbare Blöcke.
- Erstes Modell **Qwen3-0.6B**: Q/K-Norm, `head_dim` aus der Konfiguration, GQA.
- Byte-Level-BPE, KV-Cache, Greedy-Sampling.
- Chat-Template der unterstützten Modelle fest in Pascal.
- `mormot.ai.llm.native` hinter `ILlmClient`, mit Streaming und Tool-Calls.

**Fertig, wenn:** Greedy-Sampling in FP32 für einen festen Prompt-Satz tokengleich zu
`transformers` ist.

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

## Plattformen und Tests

### Messung am 2026-10-14 (heutiger Code, ohne Engine)

| Plattform | Compiler | Ergebnis |
|---|---|---|
| WSL Linux aarch64 | FPC 3.2.4 (Baseline) + 3.2.2 | grün (MCP 1257, LLM 656, alle Demos) |
| macOS M2, aarch64-darwin | FPC 3.2.4-rc2 per SSH | grün: MCP 1257, LLM 647 (der vec0-Test wird mangels Extension übersprungen) |
| Windows ARM, aarch64-win64 | FPC 3.3.1 (fpcupdeluxe, `D:\bin\fpc`) | grün: MCP 1257 und LLM, **aber** nur mit einer ARM64-`sqlite3-64.dll` (sqlite.org 3.51.2) neben dem Binary. mORMot2 hat für dieses Ziel kein statisches SQLite. Ein Skript gibt es noch nicht. |
| Windows x64 | Delphi 13 | grün (MCP 1257, LLM 659) |
| Windows x64 unter Prism | FPC 3.2.2 | nur Kompatibilitätsprobe, nicht für Leistungsaussagen |
| Windows 11 x64, i9-14900K, 96 GB, RTX 5090 (32 GB, Treiber 616.64), per SSH (`rtx5090`) | FPC 3.2.2 (fpcupdeluxe) | grün **vor** dem Delphi-7-Umbau: MCP 1012, LLM 647, **nativ** (ohne Prism); danach nicht erneut gemessen (Rechner aus). Delphi 13 ist installiert, CUDA-Toolkit (`nvcc`) nicht. |
| Windows x86 | **Delphi 7** | grün nach dem Umbau (MCP 1253, LLM 647; vec0 übersprungen, es gibt kein 32-Bit-Bundle), siehe unten |

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
| Linux x64 | Deploy-Ziel (Docker); auf dem RTX-Rechner über WSL2 möglich (noch nicht eingerichtet) |
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
