# Backlog — native Inferenz und Umfeld

Offene Themen, jedes so beschrieben, dass ein Agent ohne Vorwissen aus dieser Session
loslegen kann. Reihenfolge = grobe Priorität. Stand 2026-10-16 (landrixai `9a58adb`).

**Für jeden Eintrag gilt** (Regeln aus `CONCEPT.md` §1 und `docs/native-inference.md`):

- Stil wie mORMot2: nicht-generischer Kern, kein `exit(wert)`, kein `with`, keine
  anonymen Methoden/Inline-Variablen, ASCII-Literale, Parameter `a…`, Felder `f…`.
  Kommentare im Code Englisch, Doku Deutsch.
- Compiler: FPC 3.2.2 und 3.2.4 (Referenz: WSL, `scripts/run-fpc-infer-tests.sh`), Delphi 13
  (`scripts/run-delphi-tests.ps1 -Suite infer`) und Delphi 7
  (`scripts/run-delphi7-tests.ps1 -Suite infer -FastMM4 'D:\Projekte\src\utils'`).
  FPC zuerst; die x64-Delphi-Suite läuft auf dem ARM-Laptop nur unter Emulation und
  braucht dort Stunden — nativ auf dem RTX-Rechner (`ssh ai-rtx5090`, Exe + `tests/data`
  + Modelle nach `D:\Temp\ai-infer`, Daten werden vom Exe-Pfad aufwärts gesucht).
- Jeder Kernel bekommt Gegenproben (absichtlich eingebauter Fehler → Test muss
  fehlschlagen) und ein Review (Claude-Finder + Codex).
- Vorlagen nur mit verträglicher Lizenz (Tabelle „Vorlagen und Lizenzen" in
  `native-inference.md`); PasLLM (AGPL) und neural-api (LGPL) nicht als Vorlage.

---

## B1 — K-Quant-Prefill: fusionierter Mehrzeilen-Kernel

**Ziel:** Q4_K_M-Prefill näher an Q8_0 bringen. Heute (Snapdragon X, 200 Tokens):
Q4_K_M 290 Tokens/s, Q8_0 545 Tokens/s; i9-14900K: 455 gegen 757.

**Ausgangslage:** `LinearTileWork` in `src/infer/mormot.ai.infer.backend.cpu.pas` rechnet
K-Quants je Ausgabe und Tokenzeile einzeln über `DotK` mit dem fusionierten Kernel
`InferQ4KSums` / `InferQ6KSums` (`src/infer/mormot.ai.infer.kernels.pas`, NEON als
`.long`-Opcodes, AVX2 als Inline-Asm). Q8_0 dagegen nutzt `InferQ8Linear4x2`
(4 Gewichtszeilen × 2 Tokenzeilen, jede geladene Zeile zweimal verwendet).

**Verworfen (2026-10-16, gemessen):** Q4_K-Kachel einmal in int8-Blöcke zu 32 auspacken,
dann `InferQ8BlockDots4` und Skalen/Minima in Pascal. Bitgleich, aber langsamer (205 statt
270 Tokens/s im A/B-Lauf): Das Auspacken der Nibbles kostet im fusionierten Kernel fast
nichts, die Pascal-Skalenrechnung pro Zeile dafür viel.

**Vorgehen:**
1. Neuer Kernel-Typ, z. B. `TInferKSums2` (eine Gewichtszeile gegen zwei Q8_K-Tokenzeilen,
   Ausgabe je Block 2 × (isum, msum)), Pascal-Fassung als Maßstab.
2. NEON: aus `Q4KSumsNeonDot` ableiten — Gewichts-Nibbles und Skalen einmal laden, SDOT
   gegen beide Zeilen. Opcodes mit `aarch64-linux-gnu-as` erzeugen (siehe vorhandene
   Kommentare je `.long`), AAPCS64: v8..v15 nicht anfassen.
3. AVX2: aus `Q4KSumsAvx2Asm` ableiten, nur ymm0..ymm5 (Win64-ABI), Scratch im X-Bereich
   von `TInferAvx2Args`; FPC 3.2.2 kennt `vmovq`/`vpmovzxbw` mit 64-Bit-Speicheroperand
   nicht → über GPR laden.
4. `LinearTileWork`: Tokenzeilen paarweise, Rest einzeln; Skalierung exakt wie `DotK`
   (Reihenfolge der Double-Operationen!), damit `Exact` bitgleich bleibt.
5. Danach dasselbe für Q6_K (Gruppen zu 16, `scales` als int8).

**Fertig, wenn:** `TTestInferFast.LinearKMatchesReference` (bitgleich, auch 513 Blöcke)
mit einer Form „gerade/ungerade Zeilenzahl" grün; neuer Test Kernel = Pascal; Gegenproben
schlagen an; `demos/infer/llm-bench vendor/models/qwen3-0.6b-q4_k_m.gguf 200 32` im
A/B-Lauf (alte und neue Exe abwechselnd, das Notebook schwankt um bis zu 25 %) schneller.

---

## B2 — Q4_0-Kernel und ARM-Repacking

**Ziel:** Q4_0 beim Dekodieren mindestens so schnell wie Q8_0 (heute halb so schnell:
31 gegen 63 Tokens/s, Snapdragon).

**Ausgangslage:** Q4_0 ist korrekt (byte-gleich ggml, `InferQuantizeRowQ4_0`) aber
experimentell: `LinearTileWork` entpackt jede Kachel per `InferQ4ToQ8` nach Q8_0 und
nutzt dann die Q8-Kernel. Qualität auf Qwen3-0.6B schwach (cos 0,76–0,96), sinnvoll eher
für 4B/8B-Modelle.

**Vorgehen:** fusionierter Q4_0×Q8_0-Kernel wie ggml (`ggml_vec_dot_q4_0_q8_0`, MIT):
NEON — Nibbles maskieren, −8, SDOT; AVX2 — `maddubs` mit Vorzeichentrick. Optional
ARM-Repacking (block_q4_0 4×4/8×4 verschränkt, GEMV über 4/8 Zeilen).

**Fertig, wenn:** `TTestInferFast.LinearQ4MatchesReference` grün (exakt bitgleich),
Gegenproben, Decode-Tempo im A/B-Lauf ≥ Q8_0.

---

## B3 — CUDA-Backend (zweites GPU-Backend nach Vulkan)

**Ziel:** NVIDIA-Karten über die Driver API (`nvcuda.dll` / `libcuda.so`, zur Laufzeit
geladen, **keine** mitgelieferte DLL), Kernel als eingebettetes PTX.

**Voraussetzung:** Vulkan-Backend (Phase 4) steht; dessen Backend-Schnittstelle und
Differenztests werden wiederverwendet. Bau-Rechner braucht `nvcc` (fehlt heute überall).

**Vorlagen:** calm (zeux, MIT), llama.cpp CUDA (MIT).

**Fertig, wenn:** dieselben Referenztests wie die CPU grün auf dem RTX-Rechner (RTX 5090).

---

## B4 — Weitere Modelle

Je Modell ein eigener Eintrag, sobald gebraucht: Llama 3, Mistral, Gemma, Reranker
(BGE-Reranker), Qwen3-Embedding, BGE-M3 Sparse-/ColBERT-Ausgabe, MoE.

**Muster:** wie Qwen3 (`src/infer/mormot.ai.infer.model.qwen3.pas`): Referenzdaten per
`tools/make_reference*.py` aus `transformers` erzeugen (nur nachschlagen, Apache 2.0),
Tiny-Modell für schnelle Tests, echtes Modell hinter `LAI_REQUIRE_MODEL`, Laden aus
safetensors und GGUF, Gate gegen die Referenz.

---

## B5 — Plattform-Läufe nachziehen

Die README-Tabelle „Where it runs" hat veraltete Zellen:

- Windows x64 FPC 3.2.2 und Linux x86_64 (WSL auf dem RTX-Rechner): Inferenz von 821/828
  — vor den Qwen3-Tests (Fußnote ⁵). Rezept: `docs/native-inference.md` „Plattformen und
  Tests" bzw. Notiz zum RTX-Rechner (WSL-Distribution gehört dort Konto `sh`).
- Windows ARM64 FPC 3.3.1: Inferenz „not yet run" (braucht `sqlite3-64.dll` für ARM64
  neben dem Binary).
- LLM-Spalte macOS/Windows ARM64 vor den Load-Governor-Tests (Fußnote ⁴).

**Fertig, wenn:** die Zellen aktuelle Zahlen tragen und die Fußnoten entfallen.

---

## B6 — Delphi-Build-Skript für die Demos

`demos/infer/llm-bench.lpr` und `embed-bench.lpr` werden nur mit FPC gebaut. Ein
`scripts/build-delphi-demos.ps1` (Delphi 13 Win64, analog `run-delphi-tests.ps1`) fehlt.

---

## B7 — Offene Entscheidung: Kontextlänge für Embeddings

512 Tokens wie im Vertrag v1 oder mehr? Bestimmt den Speicherbedarf der Attention
(`native-inference.md`, „Offene Entscheidungen" Nr. 3). Braucht eine Entscheidung des
Nutzers, keine Umsetzung vorab.

---

## B8 — Vulkan auf macOS (MoltenVK)

**Ziel:** `TInferVulkan` auch auf dem Mac (M2 Pro), über MoltenVK.

**Ausgangslage:** `src/infer/mormot.ai.infer.vulkan.pas` fordert die Instanz-Erweiterung
`VK_KHR_portability_enumeration` nicht an und setzt nicht
`VK_INSTANCE_CREATE_ENUMERATE_PORTABILITY_BIT_KHR`; MoltenVK meldet sein Gerät dann
nicht, `Available` ist false (Tests: SKIP). Auf dem Mac ist kein Vulkan-Loader
installiert (`libvulkan.1.dylib` fehlt; Vulkan SDK oder Homebrew `molten-vk` +
`vulkan-loader`).

**Vorgehen:** Erweiterung per `vkEnumerateInstanceExtensionProperties` prüfen und nur
dann anfordern; auf dem Gerät `VK_KHR_portability_subset` aktivieren, wenn gemeldet.
Werte und Strukturen aus dem gepinnten `vulkan_core.h` (Tag in `UPSTREAM_BASE`).

**Fertig, wenn:** `TTestInferVulkan` auf dem Mac mit `LAI_REQUIRE_VULKAN=1` grün.

---

## B9 — Vulkan-Attention: Test für Kopfdimensionen über 128

**Ziel:** Der Zweig für die Dimensionen `t + 128` in `tools/vk/attention.comp` (`acc1`,
Kopfdimension 129 bis 256) läuft in einem Test.

**Ausgangslage:** Das Tiny-Modell hat die Kopfdimension 32, Qwen3-0.6B hat 128. Der Zweig ist
gebaut und von Claude und Codex gelesen, aber nie ausgeführt worden. Modelle mit der
Kopfdimension 256 (etwa Gemma) würden ihn brauchen.

**Vorgehen:** ein Tiny-Qwen3 mit `head_dim` 256 (oder 160) erzeugen wie `tests/data/infer-ref/tiny-qwen3`
und durch `AttentionRangesMatchCpu` schicken. Gegenprobe: `acc1` nicht neu skalieren, der
Test muss scheitern.

**Fertig, wenn:** der Test grün ist und die Gegenprobe greift.

---

## Nicht mehr verfolgen

- SQLite-Extensions `vec0` / `lembed`: fliegen langfristig raus (nativer Store und native
  Engine ersetzen sie) — kein Ausbau mehr.
