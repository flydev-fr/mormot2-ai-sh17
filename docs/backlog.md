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

**Erledigt (2026-10-16): Schritte 1, 2 und 4 für Q4_K.**
- *Kernel:* `TInferKSums2`, `InferQ4KSums2Pascal`, `Q4KSums2NeonDot` (Opcodes aus
  `q4k2.s` mit GNU as 2.42, Quelltext als Kommentar je `.long`).
- *Rechnung:* `DotQ4K2` und die Paarung in `LinearTileWork`.
- *Test:* `LinearKMatchesReference` prüft Kernel = Pascal, auch über 513 Blöcke in zwei
  Zeilen. Die Gegenprobe (vertauschte Ausgaben) lässt 16 Assertions scheitern.
- *Messung:* Q4_K_M-Prefill bei 200 Tokens auf dem Snapdragon 275 → 299 Tokens/s, auf dem
  M2 Pro 271 → 294.
- *Schritt 5 (2026-10-17):* `InferQ6KSums2Pascal`, `Q6KSums2NeonDot` (aus `q6k2.s`),
  `DotQ4K2` zu `DotK2` für beide K-Typen verallgemeinert. Gegenprobe (vertauschte
  Ausgaben) lässt 16 Assertions scheitern. Q4_K_M-Prefill bei 200 Tokens: Snapdragon
  291 → 311 Tokens/s, M2 Pro 291 → 305.
- *Schritt 3 (2026-10-17):* `Q4KSums2Avx2Asm`, `Q6KSums2Avx2Asm` (erzeugt, ymm0..ymm7,
  xmm6/7 unter Win64 im X-Bereich gesichert, Skalen-Zwischenwerte in `A[5..6]`). Grün
  unter FPC 3.2.2 und Delphi 13 auf dem i9-14900K; Gegenprobe (Zeilen vertauscht) lässt
  32 Assertions scheitern. Q4_K_M-Prefill bei 200 Tokens: 462 → 579 Tokens/s.
- *Offen:* nur der Lauf des SysV-Einsprungs (Linux x86_64), statisch geprüft.

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

**Erledigt auf ARM (2026-10-17), ohne fusionierten Kernel:** Der Engpass war das
byteweise Entpacken nach Q8_0.
- *Pascal:* `InferQ4ToQ8` entpackt acht Nibbles je Int64 bitweise (`Q4Minus8`) auf x86,
  x64 und aarch64; andere CPUs bleiben byteweise (unausgerichtete Int64 bei Offset 2).
- *NEON:* `Q4UnpackNeon` hinter `InferQ4Unpack` (jedes aarch64).
- *Test:* `LinearQ4MatchesReference` prüft `InferQ4ToQ8` über alle 256 Bytewerte gegen
  die Definition (die Referenz teilt die Funktion, fast = ref allein fände dort nichts)
  und den Kernel = Pascal. Gegenproben: ohne `or (t shl 4)` scheitert der 256-Byte-Test,
  ein `nop` statt des zweiten NEON-`sub` lässt 13 Assertions scheitern.
- *Messung (Decode, Qwen3-0.6B Q4_0, 32 Tokens):* Snapdragon X 30 → 70–78 Tokens/s
  (Q8_0 58–66), M2 Pro 28,5 → 84.
- *x64:* Schon das Pascal-Entpacken erfüllt das Ziel (i9-14900K: 64 → 79 Tokens/s, Q8_0
  71–73); ein AVX2-`InferQ4Unpack` ist zurückgestellt (höchstens etwa 1 ms je Token).
  Fusionierter Kernel und Repacking nur, wenn größere Modelle es verlangen.

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

## B8 — Vulkan auf macOS: Leistung unter MoltenVK

**Ziel:** Auf dem M2 Pro soll die GPU schneller sein als der NEON-Pfad der CPU, auch bei
kurzem Kontext. Ein natives Metal-Backend folgt nur, wenn MoltenVK selbst bremst
(Roadmap: „Apple-GPU“ in `docs/native-inference.md`).

**Ausgangslage:** Seit 2026-10-16 läuft das Vulkan-Backend auf dem Mac über MoltenVK.
- **Korrektheit:** Die Suite ist grün mit `LAI_REQUIRE_VULKAN=1`.
- **Prefill:** Qwen3-0.6B Q8_0 schafft auf der GPU 305 Tokens/s bei 200 Tokens, die CPU 563.
  Bei 4000 Tokens ist die GPU mit 194 gegen 145 vorn.
- **Decode:** 63 Tokens/s auf der GPU gegen 73 auf der CPU bei 200 Tokens, 34 gegen 21 bei
  4000 Tokens.
- **Profil:** `Profile` misst unter MoltenVK nichts, weil alle Zeitstempel eines Batches
  gleich sind.

**Erledigt (2026-10-16):**
- Eine Wanduhr-Sonde misst je Kernel.
- MoltenVK läuft ohne Metal-Argumentpuffer. Barrieren kosten dann 5 statt 25 µs, der Decode
  gewinnt 5,6 %.
- Die Sonde zeigt die Kernel als Engpass, nicht MoltenVK:
  - Decode-Linears erreichen 40 bis 57 GB/s von rund 200 GB/s.
  - Der Prefill-Linear erreicht rund 400 GFLOPS von 6,8 TFLOPS.

**Vorgehen:**
1. **Decode-Matvec, erledigt 2026-10-16:** `matvec8.comp` auf Apple-GPUs, für Q8_0, Q4_K und
   Q6_K, je Typ ein eigener Build. Der Decode bei 200 Tokens steigt bei Q8_0 von 65,8 auf
   98,3 Tokens/s, bei Q4_K_M von 56 auf 83,6. Die CPU schafft 73 bzw. 75.
2. **Prefill-GEMM, erledigt 2026-10-16:** `gemm8.comp` auf allen Geräten. Der Prefill steigt
   bei 200 Tokens auf dem M2 Pro von 303 auf 1125 Tokens/s, auf der RTX 5090 von 10 000
   auf 13 600.
3. **Prefill-Attention, erledigt 2026-10-16:** `flash.comp` auf Apple-GPUs. Der Prefill bei 4000
   Tokens steigt von 381 auf 708 Tokens/s; auf der RTX ist der Kernel langsamer.
4. **Offen:** die Decode-Attention bei langem Kontext. Der Decode fällt auf dem M2 Pro von
   90 Tokens/s (200 Tokens Kontext) auf 40 (4000). `attention.comp` erreicht dort 63 GFLOPS.
5. Erst danach über ein Metal-Backend entscheiden.

Mac-Lauf: Homebrew `molten-vk` und `vulkan-loader` sind auf `macos-tests` installiert;
Vorgehen wie in `docs/native-inference.md` (Plattformtabelle).

**Fertig, wenn:** Prefill und Decode bei 200 Tokens auf der GPU schneller sind als auf
der CPU, oder belegt ist, dass MoltenVK die Grenze setzt.

---

## B9 — Vulkan-Attention: Test für Kopfdimensionen über 128 (erledigt 2026-10-16)

Der Zweig für die Dimensionen `t + 128` in `tools/vk/attention.comp` (`acc1`, Kopfdimension
129 bis 256) läuft jetzt in `AttentionRangesMatchCpu`, mit `tests/data/infer-ref/tiny-qwen3-hd256`:
- Das Modell ist ein Tiny-Qwen3 mit 2 Köpfen zu 256 über 1 KV-Kopf.
- Embedding, Normen und MLP stammen aus `tiny-qwen3`.
- Die Attention-Gewichte sind geseedet. Sie kommen aus `tools/infer-ref/make_tiny_headdim.py`,
  das nur numpy braucht, kein torch: Der Test vergleicht das Gerät mit dem CPU-Backend.

Prompt und Decode sind mit und ohne Splits grün. Die Gegenprobe (`acc1` nicht neu skaliert)
lässt 6 Assertions scheitern. `flash.comp` nimmt nur Kopfdimensionen bis 128; hier läuft
deshalb `attention.comp`.

---

## Nicht mehr verfolgen

- SQLite-Extensions `vec0` / `lembed`: fliegen langfristig raus (nativer Store und native
  Engine ersetzen sie) — kein Ausbau mehr.
