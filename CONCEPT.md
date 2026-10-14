# LandrixAI — Konzept (`mormot.ai.*`)

> **Zweck dieses Dokuments.** Architektur-Leitplanke für das LandrixAI-Paket:
> *was* gehört in welche Schicht und *warum*. Single Source of Truth für die
> Schichtgrenze zwischen **generischem mORMot-Code**, **LandrixAI-Paket** und
> **Landrix-Backend** — sowie für den anstehenden **Merge** mit
> [flydevs `mormot.ai`](https://github.com/flydev-fr/mormot.ai) (§4), die Wahl des
> **Inferenz-Backends** (§5) und den **MCP-Sprung auf 2026-07-28** (§6).
>
> Die ursprüngliche Brainstorming-Fassung (ChatGPT-Dialog) ist nach
> [docs/Feature-LandrixAI-Agent.md](../../../docs/Feature-LandrixAI-Agent.md)
> verdichtet — dort steht der Agent als Backend-Feature.

## 1. Was LandrixAI ist (und was nicht)

LandrixAI ist eine **mORMot-basierte AI-/LLM-Toolbox** im Namespace `mormot.ai.*`,
**FPC primär** (Linux/Windows/macOS, x64/aarch64) und **Delphi 7 bis 13** (Win32/Win64),
veröffentlicht als **eigenständige Extension in einem eigenen Repository**.

- **Eigenes Repository statt mORMot-PR** (Entscheidung 2026-07-29). Die Extension geht
  nicht als Pull Request an Synopse. mORMot2 bleibt die **Grundlage** (Basis-Units, HTTP,
  JSON, SQLite, Crypto), wir liefern nur nicht dorthin ab. Gründe: eigenes Tempo, eigene
  Plattformziele (CUDA, Engine) und MCP 2026-07-28 ohne Rücksicht auf ältere
  Protokollstände. Der ursprüngliche Grund, neural-api als FPC-only-Backend nutzen zu
  können, ist mit der eigenen Engine entfallen (§5).
- **Compiler: FPC und Delphi 7 bis 13**, dieselbe Breite wie mORMot2.
  - **FPC** ist Referenz und Gate: `scripts/run-fpc-*.sh`, fest auf FPC 3.2.4 (seit
    2026-09-30, vorher 3.2.2; die Skripte brechen bei einer anderen Version ab, siehe
    `scripts/fpc-env.sh`). FPC 3.2.2 muss ebenfalls bauen (Windows und Docker des
    Konsumenten).
  - **Delphi** wird an beiden Enden gebaut und getestet: Delphi 13 (Win64,
    `scripts/run-delphi-tests.ps1`) und Delphi 7 (Win32, `scripts/run-delphi7-tests.ps1`).
    Die Versionen dazwischen unterstützt mORMot2; wir bauen sie nicht eigens. Ein Bruch
    dort gilt als Fehler, ist aber durch keinen Lauf abgesichert.
  - Was nur unter FPC geht (der POSIX-Prozessstart in `vendor/mormot.ext.os`), bleibt per
    `{$ifdef FPC}` abgegrenzt und fehlt unter Delphi. Keine Doppelimplementierung
    erzwingen.
  - Verlauf: FPC-only (2026-07-29), dazu Delphi 13 (2026-09-10), dazu Delphi 7
    (2026-10-14, Umbau in [docs/native-inference.md](docs/native-inference.md#delphi-7)).
  - Die Delphi-Breite gilt nur für die Extension. Beim Konsumenten (Landrix-Plattform)
    bleiben Backend und Werkzeuge FPC-only; dessen geteilter Client baut eigenständig mit
    FPC und Delphi 13.
- **Code im Stil von mORMot2** (verbindlich, 2026-10-14): Die Extension soll sich lesen
  und bauen wie eine mORMot2-Unit, damit sie neben mORMot nicht als Fremdkörper wirkt und
  dieselben Compiler trägt. Zwei Teile:

  **Wie mORMot2** (bevorzugte Muster, wie sie mORMot selbst überwiegend nutzt):
  - **Unit-Kopf**: `/// Titel`, dann `// - …`-Zeilen mit der Lizenzzeile
    `// - licensed under a MPL/GPL/LGPL three license - see LICENSE`, danach
    `{$I mormot.defines.inc}`. Längere Units gliedern sich in `{ ************ Abschnitt }`
    und nennen die Abschnitte in einem Übersichtsblock unter der `unit`-Zeile.
  - **Typen und Bausteine von mORMot** vor den RTL-Pendants:
    - `RawUtf8` statt `string`, `PtrInt` für Indizes;
    - JSON über `TDocVariantData`/`_ObjFast`;
    - Text mit `FormatUtf8`;
    - `TSynDictionary`, `TSynThread`, `TSynTempBuffer`, `TLightLock`.
  - **Ausnahmen**: eigene Klassen von `ESynException` abgeleitet (Projektstandard), mit
    `CreateUtf8`/`RaiseUtf8` oder `CreateU` für feste Meldungen.
  - **Benennung**: Felder `f…`, Klassen `T…`, Zeiger `P…`, Ausnahmen `E…`;
    `result`/`exit` klein, zwei Leerzeichen Einzug, `begin` in eigener Zeile.
  - **Doku**: `///` vor öffentlichen Symbolen, Details als `// - …`-Aufzählung,
    Kommentare englisch.

  **Strenger als mORMot2** (landrixai-Entscheidungen, wegen Delphi 7 und Lesbarkeit):
  - **Nicht-generischer Kern**: keine Generics in der Bibliothek, außer als optionale
    typisierte Hülle unter `HASGENERICS` über einem nicht-generischen Kern (Beispiel
    `TMcpToolBase<T>` über `TMcpRecordTool`).
  - Keine anonymen Methoden, keine Inline-Variablen, keine Array-Konstruktoren.
  - Kein `exit(wert)` (Delphi 7). mORMot nutzt es vereinzelt.
  - Kein `with` (Lesbarkeit: jeder Bezeichner zeigt, wem er gehört). mORMot selbst nutzt
    `with` oft, im Trunk über 200-mal.
  - Parameter immer mit `a…`. Ausgenommen sind von außen vorgegebene Signaturen
    (mORMots `Ctxt` und Server-Konstruktoren, `TStream`-Overrides, SQLite-Callbacks)
    und kleine Helfer im `implementation`-Teil mit kurzen Namen (`c`, `v`, `q`).
    mORMot schreibt Parameter meist ohne Präfix (`Data`, `Len`), nur gut jeder zehnte
    trägt es; fest ist dort nur `f` für Felder. Wir halten es durchgängig, damit
    Parameter, Felder und lokale Variablen auseinanderzuhalten sind.
  - Records mit Methoden als `{$ifdef USERECORDWITHMETHODS}record{$else}object{$endif}`.
  - Quelltext ohne BOM, nur ASCII in String-Literalen.
  - **Leistung**: in heißen Schleifen keine Allokation je Element; amortisiert
    wachsende Puffer (wie in `TTopK`) sind erlaubt.

  Nachschlagen: Skill `landrix-mormot2` und die mORMot2-Quellen selbst.

  **Stand 2026-10-14**:
  - Beide Teile sind erfüllt: Delphi 7 baut, jede Unit hat Kopf, Lizenzzeile und
    Übersichtsblock, und `TTopK` indiziert mit `PtrInt`.
  - Zeilenenden sind im Repo einheitlich LF (mORMot speichert CRLF). Das ist bewusst so,
    weil die Extension primär unter Linux gebaut wird.
- **Die eiserne Regel bleibt**: **kein Landrix-Domänenwissen im `mormot.ai.*`-Code.**
  Sie begründet sich jetzt aus der **Repo-Grenze** (die Extension muss ohne Landrix
  brauchbar sein), nicht mehr aus der Upstream-Fähigkeit.
- **Es ist nicht „OpenClaw in Pascal".** Wir bauen einen schlanken, sicheren
  **Agent-Kernel** + Provider-neutrale Naht (LLM/Embeddings/VectorStore/RAG/MCP).
  Alles Landrix-Spezifische lebt im Backend (Schicht B unten).

> **Namespace `mormot.ai.*` bleibt** — trotz fehlender Upstream-Absicht **und trotz
> der Kollision mit flydev** (der denselben Namespace nutzt): er ist in der
> mORMot-Extension-Szene für AI etabliert und steht überall in unserem Code. Da wir
> von flydev **nur Code übernehmen** und kein gemeinsames Repo betreiben (§4), ist die
> Kollision folgenlos. **Lizenz bleibt MPL/GPL/LGPL**
> (wie mORMot): die adoptierten MCP-Units verlangen es, und es ist mit MIT
> (flydev-pgvector) wie mit LGPL-2.1+Linking-Exception verträglich. **Entschieden
> (2026-10-14):** Die Dreifachlizenz gilt endgültig für das ganze Repo, auch für die
> native Inferenz-Engine; `LICENSE` ist entsprechend gefasst. Portierter Fremdcode
> kommt nur aus MIT-Quellen ([docs/native-inference.md](docs/native-inference.md#vorlagen-und-lizenzen)).

## 2. Schichtenmodell (die Kernregel)

**Zwei Schichten** — die Grenze ist jetzt die **Repo-Grenze**. (Früher waren es drei:
„M — mORMot-Upstream-Kandidat" und „A — Inkubation" unterschieden sich nur durch die
Frage *„reif genug für einen Synopse-PR?"*. Ohne Upstream-Ziel (§1) fällt diese
Unterscheidung weg — beide wandern in **dasselbe** Extension-Repo. **A** und **B**
bleiben als Bezeichner; „Schicht M" gibt es nicht mehr.)

> **Nennt die Unit eine konkrete Landrix-Tabelle, einen Guard, eine Rolle oder
> ein Fach-Konzept?** → Schicht B (Backend). **Sonst** → Schicht A (Extension).

| Schicht | Namespace | Repo | Inhalt | Landrix-abhängig? |
|---|---|---|---|---|
| **A — Extension** | `mormot.ai.*`, `mormot.db.*` | **eigenes Repo** (§7), FPC und Delphi 7 bis 13 | Alles Generische: LLM-Wire-Treiber, Embeddings, VectorStore + pgvector-Binding, RAG-Engine, MCP, SSE, Structured Output, Agent-Loop, Skill-/Memory-/Session-**Interfaces**, CLI | **nein** |
| **B — Backend** | `landrix.server.ai.*` | landrix-platform | Landrix-Bindungen: Memory/Sessions als `TOrm` in `TLandrixDatabase`, guard-/scope-geprüfte Tools, Audit-Anbindung, Domänen-Skills, REST-Endpunkte | **ja** |

**Konsequenz für Interfaces vs. Implementierungen — „Interface tief, Bindung hoch":**

- **Interfaces** (`ILlmClient`, `IEmbedder`, `IVectorStore`, künftig `IMemoryStore`,
  `ISessionStore`, `IToolbox`) werden in **A** definiert.
- **Landrix-Implementierungen** (ORM-Memory, guard-geprüfte Tools) leben in **B**
  und werden zur Laufzeit injiziert. So bleibt die Extension **ohne Landrix
  brauchbar** (Voraussetzung für ein eigenes Repo), und das Backend erbt nur die
  Naht, nicht die Engine.

## 3. Persistenz- & Vektor-Abstraktion (sqlite-vec ↔ pgvector)

**Status heute ✓ (Interface/Impl-Split umgesetzt):** Die Abstraktion existiert nicht
nur, sie ist jetzt auch **sauber getrennt**. `IVectorStore` und `IEmbedder` liegen in
reinen Interface-Units ohne Backend-Abhängigkeiten; die Engine (`TLlmRag`,
`TRagSearchTool`) hängt **nur** an ihnen und zieht damit **keine** statische SQLite
mehr. Ein zweites Backend (pgvector) ist eine **neue Implementierung**, kein Umbau.

**Vorher das Problem (behoben):** Interface *und* sqlite-vec-Implementierung steckten
in **einer** Unit, die `mormot.db.raw.sqlite3.static` zog — wer pgvector (oder gar
kein SQLite) wollte, schleppte die statische SQLite-Abhängigkeit mit.

**Ist-Struktur (umgesetzt):**

```
mormot.ai.vectorstore           ✓ NUR IVectorStore/IFilteredVectorStore + TRagHit + Vektor-Helfer (SQLite-frei)
mormot.ai.sqlite                ✓ RollbackIfOpen + Extension-Loader (geteilt von vec0, lembed, nativ)
mormot.ai.vectorstore.sqlitevec ✓ TVec0Store (sqlite-vec) [static SQLite hier isoliert]
mormot.ai.vectorstore.native    ✓ TNativeVectorStore: BLOB je Vektorraum + Pascal-Suche, filterbar, ohne Extension — exakt (2a) + 1-Bit-RAM-Index mit exakter Nachbewertung (2b); IVF nur bei Bedarf (§5)
mormot.ai.vectorstore.pgvector  ○ TPgVectorStore (IVectorStore) auf mormot.db.pgvector.* (Merge §4)
mormot.ai.embeddings            ✓ NUR IEmbedder (zieht nicht mehr den LLM-Client)
mormot.ai.embed.provider        ✓ TProviderEmbedder (OpenAI-Wire)
mormot.ai.embed.ollama          ○ TOllamaEmbedder (aus Merge §4)
mormot.ai.embed.lembed          ✓ TLembedEmbedder (sqlite-lembed, lokal; eigene :memory:-Connection) + SharedLembedEmbedder()
mormot.ai.embed.native          ○ TNativeEmbedder (eigener Pascal-Encoder auf dem GGUF, Option D in §5)
```
(✓ = gebaut, Demos grün · ○ = offen — pgvector/Ollama kommen mit dem Merge §4, die
nativen Units mit §5)

**Embedder = Prozess-Service, nicht pro DB.** Embedding-Erzeugung (`text → vector`)
ist orthogonal zur Vektor-**Speicherung** (`IVectorStore`, pro DB). Ein lokales
GGUF-Modell (lembed) wird deshalb **einmal pro Prozess** geladen und von *N* Stores/
DBs geteilt — **nie** an eine Store-Connection gekoppelt (sonst lädt jede DB ihr
eigenes Modell in den RAM). `TLembedEmbedder` besitzt dazu eine **eigene, dedizierte
`:memory:`-Connection** (hostet nur lembed0 + das Modell); `SharedLembedEmbedder()`
gibt den **prozessweiten Singleton je logischem Modellnamen** zurück. Der
lembed-Context ist nicht re-entrant → Embed/EmbedBatch serialisieren unter einem
`TOSLightLock`; wenn Nebenläufigkeit dominiert, skaliert man über ein zweites
`IEmbedder`-Backend (Provider/IONOS, Ollama), nicht über mehrere Modell-Loads.

Regel (gilt jetzt durchgängig): **Die RAG-/Agent-Engine kennt nur die Interfaces.**
Welches Backend (sqlite-vec im Single-Binary-Edge-Fall, pgvector im
Server-/Multi-Tenant-Fall) zum Einsatz kommt, entscheidet die Komposition in
Schicht A/B per DI — nie ein `{$ifdef}` in der Engine.

`mormot.db.pgvector.*` (Binding/ORM/Core) bleibt bewusst unter `mormot.db.*` statt
unter `mormot.ai.*`: es ist eine **pure DB-Schicht ohne AI-Bezug** und für sich
allein nutzbar. Die Namensgebung folgt mORMots `mormot.db.*`-Familie — auch ohne
Upstream-Absicht (§1) ist das die richtige Einordnung im eigenen Repo.

## 4. Merge mit flydevs `mormot.ai` (seit 2026-07-28 öffentlich)

Das „parallele Repo" ist identifiziert und **öffentlich**:
[flydev-fr/mormot.ai](https://github.com/flydev-fr/mormot.ai), Doku
[mormot-ai.fsb.dev](https://mormot-ai.fsb.dev/). Die Unit-Liste deckt sich exakt mit
dem, was hier bisher als „Parallel-Repo" geführt wurde.

**Ziel: Vollintegration** („das große Ganze") — nicht nur Rosinen. *(Revidiert
2026-10-14: nur noch selektiv, pgvector erst bei Bedarf — siehe §9 Punkt 10.)* Es sind aber
**zwei getrennte Libraries mit unterschiedlicher Lizenz**, und genau daran hängt der
Zeitplan:

| Library | Inhalt | Lizenz | Status für uns |
|---|---|---|---|
| `mormot2-pgvector` | `mormot.ai.chunk`, `mormot.ai.embed.{base,ollama}`, `mormot.ai.llm.{base,ollama}`, `mormot.ai.rag.{history,service,stream}`, `mormot.db.pgvector.{binding,orm,base}` | **MIT** | ✅ sofort mergefähig |
| `mormot-ai-sdk` | LiteLLM-Port: `mormot.ai.litellm.{client,router,cost,proxy,streaming,handler,types,callbacks}` + `provider.{openai,anthropic,gemini}` (Router/Fallback, Cost-Tracking) | **keine** (Repo-Metadaten `license: null`, keine LICENSE im SDK-Teil) | ⛔ *all rights reserved* — kein Nutzungs-/Ableitungsrecht; **Lizenz ist angefragt** |

> **Rechtliche Leitplanke:** Ohne Lizenz ist Code **nicht** frei verwendbar — auch
> nicht bei öffentlichem Repo. Der SDK-Teil bleibt **gesperrt**, bis flydev eine
> Lizenz setzt. Der MIT-Teil ist mit Attribution mergefähig (MIT ist mit unserem
> MPL/GPL/LGPL-Ziel verträglich; Copyright-Notice erhalten, siehe [NOTICE](NOTICE)).

**Namens-/Konzept-Kollisionen** (jetzt schärfer als zuvor — teils **gleicher
Unit-Name bei anderem Inhalt**):

| Thema | Hier (LandrixAI) | flydev | Entscheidung |
|---|---|---|---|
| Embeddings | `mormot.ai.embeddings` (Interface) | `mormot.ai.embed.base` + `.ollama` | Interface in `mormot.ai.embeddings`, Treiber als `mormot.ai.embed.*` |
| SSE | `mormot.ai.llm.sse` | `mormot.ai.http.sse` | **`mormot.ai.http.sse`** — SSE ist generisch HTTP |
| LLM-Naht | `mormot.ai.llm` clean-room (OpenAI-Wire = Lingua franca) + nativer `.anthropic` | **zwei** konkurrierende: `mormot.ai.llm.base`/`.ollama` (pgvector-Lib) **und** der LiteLLM-Port (SDK) | **Unsere Naht bleibt der Kern** (clean-room, lizenzrein, Anthropic-nativ, MCP-Bridge, Structured). flydevs Ollama-Treiber als weiteren `ILlmClient` einhängen; LiteLLM-Router/Cost-Tracking erst **nach** Lizenzklärung als optionale Zusatzschicht |
| RAG | monolithisch `mormot.ai.rag` | `rag` + `.service` + `.stream` + `.history` | **Zerlegung übernehmen** |
| Chunking | `ChunkText` in `mormot.ai.rag` | eigenes `mormot.ai.chunk` | **Extrahieren** nach `mormot.ai.chunk` |
| Vektor-Store | sqlite-vec (`TVec0Store`) | `mormot.db.pgvector.*` | **Beide** als `IVectorStore`-Backends (§3) |
| Hybrid-Suche | — | cosine + `tsvector` (Postgres) | Konzeptionell übernehmen; unser SQLite-Pendant ist FTS5 + vec (RRF, s. Feature-Adressverwaltung) |
| MCP | `mormot.ai.mcp.*` (Server+Client) | — | bleibt **unser** Beitrag (§6) |

**Merge-Reihenfolge:** (1) MIT-Teil integrieren (pgvector-Backend, Ollama-Embedder,
RAG-Zerlegung, chunk) — blockiert nichts; (2) SDK-Teil erst nach Lizenzzusage.

### Modus: Code-Übernahme, kein gemeinsames Repo (entschieden)

**Wir übernehmen nur den Code** — kein Fork, kein geteiltes Repository, keine
Upstream-PRs. Unser **Namespace `mormot.ai.*` bleibt unverändert**, auch dort, wo er
mit flydevs kollidiert; adoptierte Units werden in unseren Baum eingepasst/umbenannt.
Das hält uns unabhängig (eigene Compiler-Strategie, eigene Inferenz-Engine, MCP 2026-07-28) — Punkte, die flydev
nicht mitträgt.

**Preis dieser Freiheit: manuelles Nachziehen.** Damit spätere flydev-Änderungen
einarbeitbar bleiben, wird der **übernommene Commit-Stand gepinnt** — in
[UPSTREAM_BASE](UPSTREAM_BASE), im selben Muster wie schon für die adoptierten
MCP-Units. Baseline steht dort bereits (`9ab26e88…`, `main`, 2026-07-28).

```bash
# Update-Runde: was hat sich seit unserem Stand getan?
gh api repos/flydev-fr/mormot.ai/commits/HEAD --jq .sha
git -C <klon> diff <gepinnter-sha>..<neuer-sha> -- mormot2-pgvector
# Relevantes portieren, dann Commit/Date in UPSTREAM_BASE im selben Commit bumpen.
```

**Regel:** Wer aus flydev portiert, **bumpt UPSTREAM_BASE im selben Commit** — sonst
ist die Diff-Basis verloren und die nächste Runde muss raten.

## 5. Inferenz-Backends: lembed ↔ Provider ↔ eigene native Engine

Die zweite Backend-Achse neben der Persistenz (§3): **wo läuft das Modell?** Auch
hier gilt „Interface tief, Bindung hoch" — die Engine kennt nur `IEmbedder` /
`ILlmClient`, das Backend ist eine Kompositions-Entscheidung.

| Backend | Wie | Fremdabhängigkeit | Status |
|---|---|---|---|
| **lembed** (heute) | GGUF-Modell in SQLite via `lembed0` | `lembed0` + `libllama`/`ggml` **je Arch** (`.so`/`.dll`), `load_extension` | ✓ gebaut, prozessweiter Singleton |
| **Provider** | OpenAI-Wire `/embeddings` | Netz + API-Key | ✓ gebaut |
| **Ollama** | HTTP an lokalen Ollama-Server | **externer Prozess** | ○ kommt mit §4 |
| **native** | eigene Pascal-Engine (`src/infer/`): Embedder **und** `ILlmClient` | **keine** | ○ geplant (2026-10-14), [docs/native-inference.md](docs/native-inference.md) |
| ~~neural-api~~ | native Pascal-Inferenz (LGPL) | – | ✗ **nicht weiter verfolgt** (2026-10-14) |

> **Stand 2026-10-13:** Die Ablösung von `vec0` **und** `lembed0` durch nativen Pascal-Code
> ist in [docs/native-embeddings.md](docs/native-embeddings.md) analysiert (Entwurf zur
> Diskussion, gegen den Code von neural-api `b172061` geprüft). Kernergebnisse unten; die
> Details, der Plan mit Prüfpunkten und die offenen Fragen stehen dort.

### Eigene Engine statt neural-api (Entscheidung 2026-10-14)

Die native Inferenz wird eine **eigene Engine in reinem Object Pascal**: zuerst BGE-M3
als Ersatz für lembed, danach Decoder-LLMs (Qwen3) hinter `ILlmClient`, später CUDA.
Sie läuft auf allen mORMot2-Compilern (FPC und Delphi 7 bis 13; Delphi 7 als reine
Korrektheitsstufe ohne Leistungsziel) und nutzt nur MIT-Vorlagen als Code (llama2.c, ggml, calm);
Apache-2.0-Quellen (transformers, tokenizers) dienen nur zum Nachschlagen. Plan, Phasen,
Gates und Plattformmatrix: [docs/native-inference.md](docs/native-inference.md).

**neural-api** war bis hierher das strategische Ziel-Backend. Es wird nicht weiter
verfolgt:
- Für bge-m3 fehlten Charsmap-Normalisierung, BERT-GGUF, NEON und Reentranz (Analyse in
  [docs/native-embeddings.md](docs/native-embeddings.md), Stand `b172061`).
- Es ist LGPL und FPC-only.
- Das Beitragsangebot [#225](https://github.com/joaopauloschuler/neural-api/issues/225)
  ist geschlossen („not planned", 2026-10-14).

Die Naht bleibt unverändert: Umgeschaltet wird per Config (`AI_EMBED_PROVIDER`), nie per
`{$ifdef}`, und Provider-/Ollama-/lembed-Backends bleiben sinnvoll (Cloud-Modelle,
bestehende Deployments, Fallback, Orakel).

**Vektorsuche gehört dazu:** `vec0` wird unabhängig davon durch einen nativen Store
ersetzt (`mormot.ai.vectorstore.native`: float32-`embedding BLOB` je Vektorraum (`<space>_items`) als Wahrheit, 1-Bit-Index im RAM + exakte Nachbewertung, IVF nur bei Bedarf — ausgelegt auf 2 Mio+ Vektoren je DB,
mORMot bindet `create_function_v2`/`create_module_v2` auch statisch). vec0 rechnete mit
**L2** (keine Metrik im Schema); der native Store rechnet **Cosinus** auf Einheitsvektoren —
das Ranking ist identisch (Referenzvertrag v1 in docs/native-embeddings.md).

**Reihenfolge:**
1. Referenzvertrag und deutscher Testkorpus ✓
2. nativer Store ✓ (ersetzt vec0)
3. Delphi-7-Portierung des bestehenden Codes (§9 Punkt 9) ✓ (2026-10-14);
   danach Engine Phase 0 ✓ (2026-10-14), Python-Referenz:
   - festgelegte FP32-Referenz pro Layer (`tests/data/infer-ref/`, `tools/infer-ref/`);
   - winzige Fixtures;
   - Suite `infer.tests` unter FPC 3.2.4/3.2.2, Delphi 13 und Delphi 7 grün.
4. Phase 1a, BGE-M3 FP32 ✓ (2026-10-14)
5. Phase 1b, das vorhandene Q8-GGUF ✓ (2026-10-14):
   - Token-IDs identisch zu lembed (wörtliche Sondertokens getrennt gegen
     `parse_special = false`);
   - nDCG@10 0,8989 ≥ 0,8863.
6. Phase 1c, Messung und Umschaltung ✓ (2026-10-14: aarch64 und x86_64, Linux x86_64
   und Soak-Test grün):
   - „native" ist der empfohlene Provider im Backend (`AI_EMBED_PROVIDER=native`);
   - lembed bleibt Fallback und Orakel.
7. Phase 2, Decoder Qwen3 (2-0 bis 2c ✓ 2026-10-14: Tokenizer, Chat-Template, Forward,
   KV-Cache; Greedy in FP32 tokengleich zu transformers; offen 2d hinter `ILlmClient`)
8. Phase 3, Quantisierung und SIMD
9. Phase 4, CUDA

Details stehen in [docs/native-inference.md](docs/native-inference.md).

## 6. MCP-Protokoll: Sprung auf 2026-07-28 (Greenfield, kein Legacy)

Die Spec **2026-07-28** ist die größte Änderung seit MCP-Launch
([Changelog](https://modelcontextprotocol.io/specification/2026-07-28/changelog)).
**Projektentscheidung: wir sprechen ausschließlich 2026-07-28.** Was die neue Spec
entfernt, wird bei uns **gelöscht** — keine Multi-Version-Negotiation, keine
Legacy-Pfade, kein Deprecation-Ballast (Greenfield-Regel des Projekts).

> **Ausnahme seit 2026-09-10 — Legacy-Brücke für stdio, zum Rückbau vorgesehen**
> (Marker `LEGACY-ERA` im Code). `src/mormot.ai.mcp.legacy.pas` beantwortet auf **stdio**
> zusätzlich den `initialize`-Handshake (Lifecycle 2025-11-25, dazu `ping` und
> `notifications/initialized`) und reicht danach Requests ohne `_meta` mit eingesetztem
> `_meta` an den **unveränderten** 2026-07-28-Kern weiter. Die Spec erlaubt genau das
> (`docs/specs/mcp-2026-07-28/basic/versioning.mdx`: ein Server **MAY** beide Eras
> bedienen, `initialize` wählt die Legacy-Semantik für den stdio-Prozess).
> **Warum:** Claude Code (2.1.267) öffnet stdio standardmäßig mit `initialize`; selbst mit
> aktivierter Negotiation (`MCP_PROTOCOL_NEGOTIATION=auto` + `MCP_SDK_GENERATION=v2`) gibt
> seine `server/discover`-Probe nach 3 s auf, ein per `wsl` gestarteter Server braucht kalt
> rund 7 s — ohne Brücke verbindet sich z. B. codenav nicht zuverlässig.
> **Grenzen:** nur stdio (HTTP bleibt modern-only); kein `logging/setLevel`, keine
> server-initiierten Requests, kein `resources/subscribe`; ein `input_required`-Ergebnis
> wird zum JSON-RPC-Fehler. Requests **mit** `_meta` laufen weiter rein modern.
> **Rückbau, sobald** die Ziel-Clients stdio standardmäßig per `server/discover` öffnen
> **und** ihre Probe einen langsamen Prozessstart übersteht. Dann: die Unit löschen, die mit
> `LEGACY-ERA` markierten Zeilen in `mormot.ai.mcp.stdio.pas` und `mcp.tests`-Tests
> (`StdioTransportLegacyInitialize`, `LegacyBridgeTranslation`) entfernen, diesen Absatz
> und den README-Abschnitt „One temporary exception … stdio legacy bridge“ streichen.
> Vorab abschaltbar per `TMcpStdioTransport.AcceptLegacyInitialize := false`.

**RAUS (ersatzlos löschen):**
- `initialize` / `notifications/initialized` **und die gesamte Versions-Negotiation**
  (2024-11-05 / 2025-03-26 / 2025-06-18 / 2025-11-25) — Protokoll ist stateless.
  *(Einzige Ausnahme: die Legacy-Brücke für stdio oben.)*
- **Protokoll-Sessions + `Mcp-Session-Id`** im Streamable HTTP → damit auch unsere
  Session-Registry, Ablauf-Logik und das FSafe-Re-Resolve.
- **SSE-Resumability**: `Last-Event-ID`, SSE-Event-IDs, `NextSessionEventId`.
- **HTTP-GET-Endpunkt** + `resources/subscribe`/`unsubscribe`.
- `ping`, `logging/setLevel`, `notifications/roots/list_changed`.
- **Legacy-HTTP+SSE-Transport** (`TMcpSseTransport`) — jetzt offiziell *Deprecated*;
  die offene Frage „härten oder entfernen" ist damit **entschieden: entfernen**.
- **Roots / Sampling / Logging** (deprecated) — gar nicht erst bauen.

**REIN:**
- **`server/discover`** — Pflicht-RPC (unterstützte Versionen, Capabilities, Identität).
- **`_meta` pro Request**: `io.modelcontextprotocol/protocolVersion`,
  `clientCapabilities`, `clientInfo`, `logLevel`; `serverInfo` in jedem Result.
- **`subscriptions/listen`** — ein langlebiger POST-Response-Stream, Opt-in je Typ
  (`toolsListChanged`, …), Notifications getaggt mit `subscriptionId`.
  Request-scoped (`notifications/progress`, `…/message`) bleiben auf dem
  Response-Stream ihres Requests.
- **`resultType`** auf allen Results (`complete` | `input_required`).
- **MRTR** statt server-initiierter Requests: `InputRequiredResult` mit
  `inputRequests`; Client antwortet per Retry mit `inputResponses` (+ `requestState`).
- **`CacheableResult`**: `ttlMs` + `cacheScope` auf `tools/list`, `prompts/list`,
  `resources/list`, `resources/read`, `resources/templates/list`; Tools in
  **deterministischer** Reihenfolge.
- **Header** `Mcp-Method` / `Mcp-Name` auf POST; `x-mcp-header` für Tool-Parameter.
- **Extensions-Framework** (`extensions` in Client/ServerCapabilities); **Tasks** nur
  noch als Extension `io.modelcontextprotocol/tasks` (Polling `tasks/get`,
  `tasks/update`, kein `tasks/list`).
- **Error-Codes**: `-32020` HeaderMismatch, `-32021` MissingRequiredClientCapability,
  `-32022` UnsupportedProtocolVersion; resource-not-found `-32002` → **`-32602`**.
- **Auth-Härtung**: `iss` prüfen (RFC 9207), `application_type` bei DCR, Credentials
  an den Issuer gebunden; **Client ID Metadata Documents** statt DCR.

> **Cross-Call-State**: „stateless" heißt nicht zustandslos für Tools — wer Zustand
> über Aufrufe braucht, nutzt **server-minted Handles als normale Tool-Argumente**
> (SEP-2567). Für unsere Agent-/RAG-Tools ist das der vorgesehene Weg.

**Aufwandseinschätzung: netto Vereinfachung.** Der größte Teil des Umbaus ist
**Löschen** — und er erledigt zwei offene Roadmap-Punkte gleich mit (Parallel-
Stresstest der Session-Logik, Härtung des Legacy-SSE-Transports), weil die
zugehörige Mechanik im Zielprotokoll nicht mehr existiert. **Bestehen bleibt** der
offene Punkt **Auth-Resolver** — durch die Auth-Härtung der neuen Spec eher größer.

## 7. Auslösung in ein eigenes Repository — **vollzogen (2026-07-29)**

Aus §1 folgt: der `mormot.ai.*`-Code verlässt das Landrix-Monorepo. Landrix ist
seither **Konsument** der Extension — genau wie schon bei mORMot2 selbst.

> **Stand:** Repo [`landrix/mormot2-ai`](https://github.com/landrix/mormot2-ai)
> (public), erzeugt per `git subtree split --prefix=shared/delphi/landrixai`
> **mit vollständiger Historie** (33 Commits). Landrix bindet es als Submodul unter
> `shared/delphi/landrixai` ein — der Pfad bleibt gleich, deshalb ändern sich die
> `-Fu`-Build-Pfade der Konsumenten **nicht**. `codenav/` wurde vorher aus dem
> Prefix nach `shared/delphi/codenav/` gehoben (s. u.).
>
> Der Repo-Name weicht bewusst von flydevs `mormot.ai` ab; der **Unit**-Namespace
> `mormot.ai.*` bleibt (§4, Modus „Code-Übernahme").

### Was geht, was bleibt

| Geht ins Extension-Repo | Bleibt in landrix-platform |
|---|---|
| `src/` (alle `mormot.ai.*`-Units) | `backend/src/landrix.server.ai.*` (Schicht B) |
| `tests/`, `demos/`, `scripts/` | `docs/Feature-LandrixAI-Agent.md` (Backend-Feature) |
| `CONCEPT.md`, `DESIGN.md`, `README.md`, `LICENSE`, `NOTICE`, `UPSTREAM_BASE` | `AI_EMBED_*`-Config, Docker-Verdrahtung, Modell-/Extension-Bereitstellung |
| `vendor/`-**Struktur** (Binaries bleiben gitignored) | `deploy/docker-native/*` (Entrypoint, `pack-ai-dist.sh`) |
| — | **`codenav/`** (Landrix-Werkzeug, s. u.) |

**`codenav/` bleibt außen vor** (entschieden): Es ist ein **Landrix-Entwickler-
Werkzeug**, kein Bestandteil der Extension — es wandert **nicht** mit. Konsequenz:
Es nutzt die `mormot.ai.mcp.*`-Units dann über das **Submodul**, wie jeder andere
Konsument auch (Build-Pfade entsprechend anpassen).

### Einbindung: Submodul (etabliertes Muster)

Landrix bindet die Extension als **git submodule** ein — dasselbe Muster wie
`shared/delphi/libs/_git_Synopse2`. Kein Paketmanager, kein Vendoring-Copy.

### Extraktion **mit** Historie

Die Historie ist wertvoll (Review-Härtungen, Design-Entscheidungen) — **nicht**
per Copy-Paste in ein leeres Repo:

So ist es gelaufen (zur Reproduktion bei weiteren Auslösungen):

```bash
# 1. codenav VOR dem Split aus dem Prefix heben - sonst landet das
#    Landrix-Werkzeug in der Extension und kollidiert mit dem Submodul-Mount
git mv shared/delphi/landrixai/codenav shared/delphi/codenav

# 2. Historie herausschneiden und in das leere Repo pushen
git subtree split --prefix=shared/delphi/landrixai -b split/mormot2-ai
git push https://github.com/landrix/mormot2-ai.git split/mormot2-ai:main

# 3. im Monorepo den Pfad durch das Submodul ersetzen
git rm -r shared/delphi/landrixai
git submodule add https://github.com/landrix/mormot2-ai.git shared/delphi/landrixai
```

**Fallstrick beim Ersetzen**: Unter `shared/delphi/landrixai` liegen ~2 GB
**gitignorierte** lokale Daten (`vendor/models` mit GGUF, `vendor/sqlite-ext`,
`_eval/`, `bin/`, `_cleanroom/`). `git rm -r` fasst sie nicht an, aber
`git submodule add` verweigert ein nicht-leeres Zielverzeichnis — also die
ignorierten Ordner wegschieben, Submodul klonen, zurückschieben. Neu heruntergeladen
werden müssten sonst mehrere GB.

### Nachzuziehen (Checkliste — hier hängt der Build dran)

1. **Build-Pfade** `-Fu` auf `shared/delphi/landrixai/src` in **drei** Skripten:
   `backend/scripts/run-fpc-tests.sh`, `build-server-linux.sh`, `build-server-win.ps1`.
2. **Docker/CI — der kritische Punkt**: `deploy/docker-native/Dockerfile` kopiert
   `shared/delphi/landrixai/src`. Als Submodul muss der CI-Checkout **`submodules:
   recursive`** setzen, sonst ist das Verzeichnis im Build-Kontext **leer** und der
   FPC-Build bricht ab. (Dasselbe Muster hat schon bei den Synopse2-Static-Libs
   zugeschlagen → `deploy/docker-native/static/`.)
3. **codenav-Indizierung** + CLAUDE.md-Pfade (`shared/delphi/{landrixai,client,dto}`).
4. `.gitignore`-Einträge (`vendor/models`, `vendor/sqlite-ext`, `bin/`) wandern mit.
5. `deploy/docker-native/pack-ai-dist.sh` (greift auf `landrixai/vendor/…`).
6. Skills mit landrixai-Bezug (`landrix-mormot2`).

### Zeitpunkt: **früh, vor den großen Umbauten**

Empfehlung: **jetzt splitten**, nicht nach Merge/MCP-Umbau. Begründung:
- Die Kopplung ans Backend ist **bereits dünn** (Schicht B nutzt nur `IEmbedder` +
  `GetSharedEmbedder`) — der Schnitt ist heute billig und wird mit jedem weiteren
  Feature teurer.
- Die drei großen anstehenden Arbeiten (flydev-Merge §4, MCP-Neubau §6,
  native Inferenz §5) finden **vollständig innerhalb** der Extension statt. Sie im
  Zielrepo zu machen erspart einen zweiten Umzug.
- Der flydev-Merge bedeutet ohnehin ein neues Repo-Layout (zwei Libraries
  zusammenführen) — das gehört nicht ins Landrix-Monorepo.

**Repo-Modell (entschieden):** **eigenes, unabhängiges Repo** — kein gemeinsames mit
flydev, kein Fork. Von dort wird **nur Code übernommen** (§4, Modus „Code-Übernahme"),
der Namespace `mormot.ai.*` bleibt trotz Kollision. Der Repo-**Name** ist damit frei
wählbar (`mormot.ai` ist bei flydev belegt) — z. B. `landrix/mormot-ai-ext`.

## 8. Ist-Stand-Inventar (Schicht A, gebaut)

Alles unten **gebaut + review-gehärtet + grün**; aktuelle Zahlen je Plattform (Stand
2026-10-14: MCP 1012; LLM je nach Plattform und vec0-Test 647–659) in
[docs/native-inference.md](docs/native-inference.md#plattformen-und-tests). Streaming/Tool-Loop/RAG/Vision auch
**live** verifiziert (OpenAI/Ollama/Anthropic). Aufbau-Historie: [DESIGN.md](DESIGN.md).

- **LLM-Client**: `mormot.ai.llm` (OpenAI-Wire = Lingua franca, deckt OpenAI/LiteLLM/
  Ollama) + nativer `mormot.ai.llm.anthropic` (Messages-API hinter derselben
  `ILlmClient`-Naht); Typen `mormot.ai.llm.types`; SSE-Streaming
  (`mormot.ai.llm.sse`, Basis + OpenAI-/Anthropic-Subklassen); Vision/Multimodal
  (`TLlmMessage.Images`, beide Wires).
- **Agent-Loop**: `TLlmAgent` + `ILlmToolbox` ([mormot.ai.agent.pas](src/mormot.ai.agent.pas)).
- **MCP**: Server+Client, stdio/HTTP/Streamable HTTP/in-process, RTTI-Tool-Schema
  (`mormot.ai.mcp.*`), MCP-Bridge in den Agenten (`mormot.ai.agent.mcp`). Spricht
  ausschließlich **2026-07-28** (§6, zustandslos); Legacy-SSE ist entfernt, nur eine
  stdio-Brücke für den alten `initialize`-Handshake (`mormot.ai.mcp.legacy`) ist noch da
  und zum Entfernen vorgesehen.
- **Structured Output**: `mormot.ai.llm.structured` (`ChatStructured` provider-agnostisch);
  OpenAI `response_format` ↔ Anthropic `output_config.format` (Adapter übersetzt + injiziert
  `additionalProperties:false`), beide live verifiziert.
- **Embeddings/RAG** (Interface/Impl getrennt, §3): Interfaces `IEmbedder`
  (`mormot.ai.embeddings`) + `IVectorStore` (`mormot.ai.vectorstore`, SQLite-frei);
  Impls `TProviderEmbedder` (`mormot.ai.embed.provider`), `TLembedEmbedder`
  (`mormot.ai.embed.lembed`), `TVec0Store` (`mormot.ai.vectorstore.sqlitevec`) und
  `TNativeVectorStore` (`mormot.ai.vectorstore.native`: ohne C-Extension, Vertrag,
  Fachfilter `SearchWhere`, 1-Bit-Index + exakte Nachbewertung, 2 Mio × 1024 gemessen);
  Engine `TLlmRag` (`mormot.ai.rag`) + **agentic RAG** als `search_docs`-MCP-Tool
  (`mormot.ai.rag.tool`, dasselbe Tool extern wie in-process).
- **codenav**: MCP-Tools für Code-Navigation (`codenav/`).

## 9. Offene Punkte / Roadmap

Die Engine-Funktionsfläche (Phase D) steht, die **Repo-Auslösung** (§7) ist erledigt,
der **MCP-Sprung auf 2026-07-28** (§6) ist durchgezogen und mit der
Konsumenten-Bindung (Phase B) auch praktisch belegt; offen sind der **flydev-Merge**
(§4) und die **native Inferenz** (§5, eigene Engine). Diese Liste ist die konsolidierte
Roadmap — DESIGN.md verweist hierher.

0. **Repo-Auslösung (§7)** ✓ **erledigt (2026-07-29)** — `landrix/mormot2-ai`, Historie
   erhalten, Landrix zieht per Submodul. Alle weiteren Punkte finden **hier** statt.

1. **Refactor §3** ✓ **umgesetzt** (Interface/Impl-Units getrennt: `vectorstore` +
   `vectorstore.sqlitevec`, `embeddings` + `embed.provider` + `embed.lembed`; Engine
   SQLite-frei, Demos grün). Offen bleibt nur das **neue** pgvector-Backend (Punkt 2).
2. **Merge §4** — *seit 2026-10-14 selektiv statt Vollintegration (Punkt 10)*;
   ursprünglich gestaffelt nach Lizenz:
   - **(a) jetzt**: `mormot2-pgvector` (**MIT**) — pgvector als zweites
     `IVectorStore`-Backend `mormot.ai.vectorstore.pgvector` (die Naht steht),
     Ollama-Embedder als `mormot.ai.embed.ollama`, RAG-Zerlegung
     (`rag.service`/`.stream`/`.history`), Chunking → `mormot.ai.chunk`,
     SSE → `mormot.ai.http.sse`. Attribution/NOTICE mitführen.
   - **(b) blockiert**: `mormot-ai-sdk` (LiteLLM-Port) — **keine Lizenz**,
     Anfrage läuft. Erst nach Zusage anfassen; unsere clean-room-LLM-Naht bleibt
     bis dahin (und voraussichtlich dauerhaft) der Kern.
3. **Memory-/Session-Interfaces** (Schicht A) definieren — Implementierung im
   Backend (Schicht B), Details in
   [docs/Feature-LandrixAI-Agent.md](../../../docs/Feature-LandrixAI-Agent.md).
4. **x64-linux-`.so`** für sqlite-vec/lembed ✓ **vorhanden + verifiziert** — alle vier
   Arch-Builds da (aarch64/x86_64 × linux/win64); neues vec0 `v0.1.10-alpha.4` +
   lembed0 laufzeit-geprüft auf **beiden** Deploy-Arches: aarch64 nativ (rag-spike/
   chat/agent) und x86_64 gegen mORMots **statisches** SQLite im amd64-Container
   (`scripts/x64-ext-verify.sh`). Offen bleibt nur die Verdrahtung ins Docker-Image.
5. **Anthropic-Restfläche** ✓ **erledigt**: Structured Output über `output_config.format`
   (Adapter übersetzt das neutrale `ResponseFormat` und injiziert das von Anthropic
   geforderte `additionalProperties:false`) — live gegen `claude-opus-4-8` (typisierter
   Record extrahiert). Anthropic-**Streaming** live verifiziert (SSE-Parser inkl.
   message_start/delta-usage end-to-end). Beide Demos (`llm-structured`, `llm-chat`) jetzt
   provider-agnostisch via `LLM_PROVIDER`.
6. **MCP-Sprung auf 2026-07-28** (§6) ✓ **erledigt** — ersetzt den früheren Punkt
   „Transport-Produktionsreife". Der Umbau ist überwiegend **Löschen**; mehrere
   Alt-Befunde erledigen sich dadurch. Stand: Branch `feat/mcp-2026-07-28` in `main`
   gemerged, Protokoll-Restlücken am 2026-08-01 geschlossen, **829 Assertions** grün
   (Belege im Einzelnen in [DESIGN.md](DESIGN.md), Abschnitte „Protokoll" und
   „Protokoll-Restlücken geschlossen"):
   - ~~Parallel-Stresstest der Streamable-Sessions~~ **entfällt** — Protokoll-Sessions
     (und damit Registry, Ablauf, FSafe-Re-Resolve, `NextSessionEventId`) existieren
     im Zielprotokoll nicht mehr.
   - ~~Legacy-SSE-Transport härten oder entfernen~~ → **entfernen** (offiziell
     *Deprecated*), Entscheidung getroffen.
   - **Auth** ✓ **erledigt** (Mechanik **und** Resolver). Der Server ist
     OAuth-2.1-**Resource-Server** (RFC-9728-Metadata, Token-Prüfung vor dem Dispatch,
     Audience-Bindung, 401/403-Challenges — Details in [DESIGN.md](DESIGN.md)). Die
     Naht ist `IMcpTokenVerifier`; ihre Implementierung steht seit 2026-08-01 im
     Konsumenten (`backend/src/landrix.server.ai.McpAuth.pas`) samt
     Scope→Permission-Abbildung, dazu erste Tools über die Registry — das ist
     **Phase B** und damit der Beleg, dass Registry, Auth-Context und RTTI-Schema
     an fremdem Code tragen. Der Konsument ist inzwischen **auch** Authorization
     Server (Selbstregistrierung, PKCE, Refresh-Rotation, Widerruf) — das ist
     ausdrücklich **nicht** Sache dieses Repos, siehe
     [Feature-MCP-Auth.md](../../../docs/Feature-MCP-Auth.md).
     **Korrektur zur früheren Fassung**: `iss`/RFC 9207, Client ID Metadata Documents
     und Issuer-Bindung sind **Client-/Authorization-Server-Pflichten**, nicht die des
     MCP-Servers — sie stehen hier nicht mehr an.
   - **Gebaut** (Details in [DESIGN.md](DESIGN.md), Abschnitt „Protokoll"):
     `server/discover`, `_meta`-Transport der Protokollversion, `subscriptions/listen`,
     `resultType`, **MRTR**, `ttlMs`/`cacheScope`, neue Header + Error-Codes.
   - ~~**Neu zu bauen**: `x-mcp-header`, Extensions-Framework, JSON Schema 2020-12 im
     `inputSchema`, OTel-`_meta`-Keys, deterministische `tools/list`-Reihenfolge,
     Prompts/Completion/Pagination~~ — **alles gebaut** (2026-08-01), dazu
     Progress-Notifications und der Template-Resolver. **Bewusst nicht gebaut**:
     `notifications/message` (Logging) ist in 2026-07-28 *deprecated* — hier unter §6
     schon entschieden, die frühere Nennung als Lücke war ein Doku-Fehler.
   - **Demo-Härtung** ✓ (bleibt gültig): Streamable-Demo bindet **Loopback**,
     `ask_claude` standardmäßig **deaktiviert** (`MCP_ENABLE_ASK_CLAUDE=1` als Opt-in).
7. **Native Embeddings + Vektorsuche + Inferenz** (§5):
   - Store ✓, 2026-10-13: `mormot.ai.vectorstore.native` ersetzt `vec0`. Analyse und
     Gate-5-Messung in `docs/native-embeddings.md`; 2c IVF nur bei Bedarf.
   - Engine ○: eigene Pascal-Inferenz nach
     [docs/native-inference.md](docs/native-inference.md). Zuerst BGE-M3 (ersetzt lembed),
     dann Qwen3 hinter `ILlmClient`, dann CUDA. Phase 0 (Referenzdaten, Testrahmen) ✓
     und Phase 1a (BGE-M3 FP32 von Text bis Vektor, trifft PyTorch) ✓ 2026-10-14,
     Phase 1b (dasselbe Q8-GGUF wie lembed, trifft das Orakel, nDCG@10 0,8989) ✓
     2026-10-14; 1c ✓ 2026-10-14: NEON und AVX2, auf x86_64 1,6- bis 2-mal so schnell
     wie lembed, Linux x86_64 grün, Soak-Test über 90 Minuten bestanden, im Backend der
     empfohlene Provider; dazu der Load Governor (Punkt 10.7). Phase 2 bis 2c ✓
     2026-10-14: Qwen3-0.6B in FP32, Greedy tokengleich zu transformers; offen 2d
     (`mormot.ai.llm.native` hinter `ILlmClient`).
   - Ziel: Alle nativen Fremd-`.so` und der Docker-Bundle-Download entfallen. vec0 und
     lembed gehen **unabhängig** voneinander, jeweils nach ihrem Gate. lembed bleibt bis
     dahin Fallback und Orakel.
8. **Tests**: TVec0Store hat einen **env-gated Real-vec0-Test** (Keyed-Ops
   `Upsert`/`Delete`/`Search`-Key, `test.llm.rag`, läuft bei gesetztem `SQLITE_EXT_DIR`,
   sonst Skip). Offen: TLembedEmbedder-Realtest + RAG-Atomar-Rollback (brauchen die
   lembed-Runtime, s. Punkt 4).
9. **Plattformbreite wie mORMot2** (2026-10-14, Befunde in
   [docs/native-inference.md](docs/native-inference.md#plattformen-und-tests)):
   - **Delphi 7** (Win32) ✓ **erledigt 2026-10-14**: alle 22 Units, MCP- und LLM-Suite
     grün (`scripts/run-delphi7-tests.ps1`). MCP-Registries auf `TSynDictionary`,
     typisierte Tools auf `TMcpRecordTool` (`TMcpToolBase<T>` bleibt als Hülle unter
     `HASGENERICS`, der Konsument unverändert), `exit(x)` ersetzt, `object` statt Record
     mit Methoden. Gefunden und behoben: stdio hätte unter Delphi 7 UTF-8 doppelt
     kodiert (Test `StdioReadsUtf8Unchanged`).
   - **Windows ARM** (FPC 3.3.1): Die Suiten sind grün, sofern eine ARM64-`sqlite3-64.dll`
     danebenliegt; mORMot2 hat für `aarch64-win64` kein statisches SQLite. Es fehlen ein
     Testskript und die Beschaffung der DLL.
   - **macOS M2** (FPC 3.2.4-rc2): grün. Es fehlt ein wiederholbares Skript.
   - **RTX-Rechner** (Windows 11 x64, i9-14900K, RTX 5090): Die Suiten sind nativ unter
     FPC 3.2.2 grün. Rolle: AVX2, native x64-Leistung, CUDA. Das CUDA-Toolkit fehlt
     noch.
10. **Weitere Lücken**, nach unmittelbarem Nutzen geordnet (Codex-Abgleich 2026-10-14):
    1. Embedder-Auswahl im Konsumenten vervollständigen. Die Backend-Fabrik kennt nur
       `none`/`lembed` und lehnt `ollama` ab. Der OpenAI-kompatible
       `TProviderEmbedder` (z. B. IONOS) reicht als zweiter Weg.
    2. Agent-Loop:
       - Abbruch, Fristen, Kontextbudget, Schritt-Ereignisse;
       - danach Sitzungs-Persistenz (Punkt 3). Wiederholte Tool-Aufrufe mit
         Seiteneffekten müssen dabei ausgeschlossen sein.
    3. Ingestion:
       - tokenbasiertes Chunking (`ChunkText` zählt Bytes);
       - stabile Dokument- und Chunk-Identität;
       - Ersetzen, Löschen und Neuindizieren als Ablauf.
    4. flydev-Übernahmen (Punkt 2) nur selektiv; pgvector erst, wenn ein Deployment es
       braucht.
    5. Die Stale-Zähler der Assertions in README/DESIGN vereinheitlichen.
    6. Ein Build-Skript für die Demos unter Delphi (13 und 7). Heute baut nur
       `scripts/build-demo.sh` (FPC); die Delphi-Skripte decken nur die Testsuiten ab.
    7. **Load Governor** für Bulk-Embedding ✓ gebaut (2026-10-14, `mormot.ai.governor`,
       im Backend `GetBulkEmbedder` mit `AI_BULK_*`); offen nur sein erster Aufrufer, ein
       Indizier-Job, und ein Lauf auf echter Sensor-Hardware (Linux, nicht WSL2):
       - Threads begrenzt der native Embedder selbst (`AI_EMBED_THREADS`), der Governor
         setzt einen Tastgrad je Worker (Pausen außerhalb des Locks, `Abort` beim
         Herunterfahren);
       - Temperaturschwelle, wo ein Sensor lesbar ist (Linux: `/sys/class/hwmon`,
         im Docker-Container ohne root); unter WSL2 und Windows gibt es ohne Treiber
         keinen Sensor;
       - solange keine Schwelle aktiv ist, eine Heuristik über den Durchsatz gegen eine
         langsam alternde Referenz: Ein dauerhafter Einbruch senkt den Tastgrad, eine
         Erholung hebt ihn wieder. Eine Heuristik - auch andere Last oder eine andere
         Textmischung senkt den Durchsatz (so fiel die Windows-Drosselung auf, siehe
         docs/native-inference.md).
       Lüfter und Leistungsgrenzen (PL1/PL2) bleiben bei Firmware und BIOS; von außen
       begrenzt im Docker-Betrieb `--cpus`.

**In diesem Review-Pass bereits behoben** (Build + Tests grün: MCP 202, LLM 256
Assertions): JSON-RPC-Envelope-Validierung (`jsonrpc:"2.0"`, Params-Typ) + breiter
Exception→JSON-RPC-Error-Fang mit korrekten Codes (−32600/−32601/−32603);
fail-closed `IsAuthenticated` (Identität nur aus einem echten Resolver, nie
aus dem Transport — NICHT „ohne Verifier wird alles abgelehnt“, siehe DESIGN
„Review-Härtung Runde 5“); Streamable-UAF via FSafe; SSE-Map thread-safe + Leak-Fix;
RAG-Ingestion **strikt** (exakte Vektoranzahl) **+ atomar** (`IVectorStore.AddBatch`,
eine Transaktion); `build-demo.sh` baut wieder **alle** Demos (vorher nur 5 MCP-Demos
trotz „ALL DEMOS OK"); Demo-Loopback + `ask_claude`-Gate.
