# Native Embeddings + Vektorsuche in Pascal — Analyse

> **Status (2026-10-13):** Plan entschieden und in Umsetzung. **Gebaut + gemessen:**
> Schritt 1 (Referenzvertrag, Testkorpus, Basis-Metriken) und Schritt 2a (nativer Store,
> exakte Suche) — siehe „Plan mit Prüfpunkten". **Noch Schätzung:** alles zu den
> späteren Stufen (int8/IVF, nativer Encoder), insbesondere Aufwand und Performance.
> Ziel: die C-Abhängigkeiten `vec0` (sqlite-vec) und `lembed0` + `libllama`/`ggml`
> (sqlite-lembed) durch nativen Pascal-Code ersetzen — kein `load_extension`, keine
> Arch-Bundles je Plattform. Entstanden aus einer Design-Diskussion (Claude + Codex),
> gegen den echten Code geprüft; die Umsetzung ist mit Codex und Claude Fable reviewt.
>
> Geprüfte Stände: dieses Repo, mORMot2 (Submodul), neural-api `b172061`
> (2026-10-11), Modelldatei `vendor/models/bge-m3-q8_0.gguf`.
> Einordnung in die Gesamtstrategie: [CONCEPT.md §3/§5](../CONCEPT.md).
>
> **Nachtrag 2026-10-14:** Teil B geht in den Plan für eine **eigene Inferenz-Engine** über
> ([native-inference.md](native-inference.md)). Option D wird dort zu Phase 1; die Schritte
> 3–6 unten heißen dort Phasen 1a–1c. neural-api wird nicht weiter verfolgt (#225
> geschlossen). Teil B bleibt hier als Analyse stehen.

## Kurzfassung

Die Ablösung zerfällt in **zwei unabhängige Teile mit sehr unterschiedlichem Risiko**:

| Teil | ersetzt | Aufwand | Risiko |
|---|---|---|---|
| **A. Nativer Vektor-Store** | `vec0` | ~1 Woche | gering |
| **B. Nativer Embedder** | `lembed0` + `libllama`/`ggml` | mehrere Wochen | hoch (Tokenizer, Numerik, Performance) |

Teil A lohnt sich unabhängig davon, wie B ausgeht. Für B ist **neural-api kein
fertiger Weg für unser Modell** (bge-m3). Seit 2026-10-14 gilt: eine eigene Engine für
Embeddings **und** generative Inferenz, siehe [native-inference.md](native-inference.md).

---

## Teil A — Vektorsuche nativ (vec0 ersetzen)

### Ausgangslage

- `TVec0Store` (`src/mormot.ai.vectorstore.sqlitevec.pas`) lädt `vec0` per
  `load_extension`, Tabelle `vec_documents USING vec0(embedding float[N])`, KNN über
  `WHERE embedding MATCH ? AND k = ?`.
- Das Schema nennt **keine Metrik** → vec0 rechnet mit **L2** (Default). Wichtig für
  die Parität: Cosine ändert Distanzwerte und bei nicht normierten Vektoren auch das
  Ranking. Bei Einheitsvektoren ist das Ranking gleich, die Distanzwerte nicht.
- mORMot bindet `sqlite3.create_function_v2` und `create_module_v2` — auch in der
  **statischen** SQLite (`mormot.db.raw.sqlite3.static.pas`). Eigene SQL-Funktionen
  und Virtual Tables in Pascal sind also ohne C möglich.
- mORMot hat **kein** SIMD-Skalarprodukt für `single`-Vektoren.

### Mengengerüst (Stand 2026-10-13, vom Nutzer genannt)

Größte reale SQLite-DB: **2 Mio Artikel**. Alternativ ein Index über Vorgänge mit
Positionen: **30 000 Dokumente in 14 Jahren** mit je 10–100 (teils mehr) Positionen,
also grob 1,5 Mio Positionen (~107k/Jahr); in weiteren 10 Jahren linear ~2,6 Mio.

| Bestand (bge-m3, 1024 dim) | Vektoren | float32 | int8 | 1-bit |
|---|---|---|---|---|
| Artikel (größte DB) | 2 Mio | 7,6 GiB | 1,9 GiB | 244 MiB |
| Positionen heute | ~1,5 Mio | 5,7 GiB | 1,4 GiB | 183 MiB |
| Positionen in 10 Jahren | ~2,6 Mio | 9,9 GiB | 2,5 GiB | 317 MiB |
| Artikel + Positionen | ~4,6 Mio | ~17,5 GiB | ~4,4 GiB | ~560 MiB |

Folgerungen:
- **Brute-Force mit float32 im RAM scheidet aus.** Ein exakter Scan über 2 Mio × 1024 sind
  ~2 Mrd. Multiplikationen und ~2 GB Speicherlesen je Abfrage (auch int8) — grob
  ≥ 100 ms pro Abfrage und Kern (**Schätzung, ungemessen**); mit vier Workern zu langsam.
- **Quantisierung + Nachbewertung gehören in v1** (umgesetzt in 2b als 1-Bit-Index), ein
  **ANN-Index (IVF) nur bei Bedarf** (2c).
- **Artikel und Positionen sind getrennte Stores** (Nutzer, 2026-10-13: Positionen sind
  i. d. R. keine Artikelkopien und sollen nicht zusammengelegt werden) — je eigener
  Indexvertrag, kein Dedup über Bestände hinweg. Innerhalb eines Stores optional nur ein
  Embedding-Cache für **wortgleiche** Texte (gleicher Text = gleicher Vektor), kein
  semantisches Zusammenführen.
- **Vorfilter + hybride Suche**: Vorgänge sucht man fast immer eingeschränkt (Kunde,
  Projekt, Zeitraum, Warengruppe) — Filter vor dem Vektorvergleich verkleinern die Menge
  um Größenordnungen. Für Artikelnummern/EAN/Maße ist FTS5 ohnehin besser; robust ist
  **FTS5 + Vektor, zusammengeführt per Reciprocal Rank Fusion**.
- **Engpass ist die Embedding-Erzeugung, nicht die Suche**: bge-m3 hat 568 Mio Parameter;
  bei angenommenen 50 Texten/s dauert die Erstindizierung von 2 Mio Artikeln ~11 h, bei
  200/s ~3 h (**Durchsatz ungemessen**, Gate 5). Deshalb ein **kleineres Modell** mit auf
  den Prüfstand (z. B. multilingual-e5-small, 384 dim: Vektoren ⅓ so groß, Inferenz grob
  5× schneller — Qualität bei kurzen deutschen Artikeltexten am Testkorpus entscheiden).
  Erstindizierung als fortsetzbarer Hintergrund-Job, nie im Request-Handler.

**Präzisierung durch den Nutzer (2026-10-13):**
- **Was eingebettet wird**: zunächst nur die **Texte** von Artikeln bzw. Positionen.
  Zweck ist **Analyse** über aktuelle Positionen („was wurde genommen, was gesucht"),
  nicht nur Suche. Folgen:
  - **Fachfilter sind Pflicht** (Zeitraum, Kunde, Vorgangsart …): `IVectorStore.Search`
    braucht eine Filter-Erweiterung (erlaubte ID-Menge oder SQL-Join auf eine Fachtabelle),
    Filter **vor** Top-k.
  - **Clustering fällt mit IVF ab**: die k-Means-Zentroiden sind zugleich Themencluster
    (z. B. „welche Positionsgruppen 2025 häufiger als 2024") — ein weiterer Grund für IVF.
  - Für Analyseläufe ist ein exakter Scan über eine gefilterte Teilmenge eher tragbar als
    für interaktive Suche.
- **Hardware**: Consumer-Klasse, **32/64 GB RAM** zumutbar. ⇒ ursprünglich **int8 im RAM**
  als Default geplant (2 Mio × 1024 ≈ 1,9 GiB); umgesetzt ist in 2b der **1-Bit-Index**
  (244 MiB) mit exakter Nachbewertung, weil int8 ohne SIMD ~10× langsamer war; RAM-Budget je Store
  konfigurierbar (Richtwert 2–4 GiB). SIMD-Ziel **AVX2 + NEON** (AVX-512 ist bei Consumer-
  CPUs nicht verlässlich; ARM-Laptops/Macs kommen vor).
- **Modell**: bge-m3 war bei Deutsch gut; ein **externer Anbieter** ist möglich —
  **flexibel bleiben**. Daraus:
  - **Indexvertrag je Store** (Modell, Dimension, Metrik, Fingerprint); Modellwechsel =
    Neuindizierung, nie stilles Mischen von Vektorräumen.
  - **Neuindizierung ohne Ausfall**: neuen Index im Hintergrund parallel aufbauen, dann
    umschalten — der Store muss zwei Vektorräume nebeneinander halten können.
  - **bge-m3 auch extern** (sofern gehostet angeboten): lokale und externe Vektoren sind
    dann praktisch austauschbar (q8 vs. fp16 — am Testkorpus prüfen); lokal und extern
    mischbar ohne Neuindizierung. Ein anderes Modell beim Anbieter = Neuindizierung.
  - **Datenschutz**: Positionstexte enthalten Kundendaten — externer Anbieter braucht
    AV-Vertrag/DSGVO-Klärung; als Betreiber-Schalter vorsehen.
- **Externer Anbieter: IONOS AI Model Hub** bietet **BGE M3 (1024 dim)** an und soll
  DSGVO-konform sein (AV-Vertrag trotzdem nötig). Anbindung über den Provider-Weg
  (`TProviderEmbedder`, OpenAI-Wire) — Endpunkt/Kompatibilität beim Einbau prüfen.
  Zu testen: Austauschbarkeit lokaler (q8) und IONOS-Vektoren. Mahnendes Beispiel für den
  Indexvertrag: IONOS entfernt BGE Large 1.5 am 2026-10-22 („Embeddings from the two
  models are not interchangeable") — Abkündigungen erzwingen Neuindizierung.
- **Folge für die Reihenfolge**: Teil A (nativer Store) ist der Kern. Teil B (Option D)
  verliert an Dringlichkeit — bis dahin decken lembed und der Provider das Embedding ab;
  D bleibt das Ziel für reinen Offline-Betrieb ohne C-Abhängigkeiten.

### Anwendungsfall und Fachfilter (vom Nutzer bestätigt, 2026-10-13)

**Ziel**: neue Angebote aus reinem Text oder GAEB-Import analysieren und automatisch
aus alten Positionen aufbauen.

1. **Eingang**: GAEB-LV oder Freitext (Freitext zuerst in Positionen zerlegen, z. B. per
   LLM mit `mormot.ai.llm.structured`) → je Position Kurztext, Langtext, Menge, Einheit.
2. **Je neue Position**: einbetten → ähnliche **alte** Positionen suchen.
3. **Vorschlag**: Kalkulation (EP-Zerlegung), Artikel, Langtext übernehmen; der
   Bearbeiter bestätigt.

**Harte Filter** (schließen aus, **vor** Top-k — sonst bleiben nach dem Aussortieren keine
Treffer übrig):
- Mandant/Firma (Sicherheit);
- Positionsart: nur echte Leistungspositionen (Legacy `poaNormal`, `poaBedarfMitGB`); Text-/
  Ausführungsbeschreibungen, Alternativ, NichtVerarbeiten, Schatten fallen heraus.

**Ranking-Signale** (gewichten, schließen nicht aus):
- Einheit (m² vs. Stk vs. psch): gleiche Einheit wiegt mehr — bewusst **kein** harter
  Filter (Nutzer, 2026-10-13), weil gleiche Leistungen in verschiedenen Einheiten
  angeboten werden;
- „genommen": Positionen aus Angeboten, die Auftrag/Rechnung wurden, nach oben;
  abgelehnte bleiben als „angeboten, nicht genommen" sichtbar;
- Aktualität (neuere Preise wiegen mehr; kein harter Zeitfilter, sonst verschwinden seltene
  Leistungen);
- gleicher Kunde;
- exakte Treffer schlagen das Embedding: gleiche STLB-Leistungsnummer, Artikelnummer,
  FTS5-Treffer — zusammengeführt per Reciprocal Rank Fusion.

**Gruppieren statt Einzeltreffer** (der Analyse-Teil): fast gleiche alte Positionen zu
**einem** Vorschlag bündeln — „37× verwendet, zuletzt 2025, EP-Median X, Spanne Y–Z".

**Schnitt zwischen Extension und Fachschicht:**
- Extension (`IVectorStore`): `Search` mit Filter, der per **SQL-Join** auf die
  Positionstabelle derselben DB wirkt (geht nur, weil die Vektoren als schlichter BLOB in
  einer normalen Tabelle liegen — ein weiteres Argument gegen vec0). Bei sehr selektivem
  Filter entscheidet der Store selbst, ob ein exakter Scan der gefilterten Menge günstiger
  ist als IVF.
- Fachschicht (Backend): Ranking-Signale, Fusion mit FTS5/STLB, Gruppierung und
  Statistik — **nicht** in die generische Extension.

### Optionen

| | Ansatz | Bewertung |
|---|---|---|
| (a) | Normale Tabelle mit `embedding BLOB` + Brute-Force-KNN in Pascal über eine Matrix im Speicher | **empfohlen** |
| (b) | Pascal-SQL-Funktion `vec_distance_*(blob, blob)` + `ORDER BY … LIMIT k` | optional, für Diagnose/Ad-hoc-SQL |
| (c) | Eigene vec0-artige Virtual Table | Aufwand ohne Gegenwert |

### Empfohlener Entwurf: `mormot.ai.vectorstore.native`

- **Schema** (umgesetzt in 2a): je **Vektorraum** („space", z. B. `articles`,
  `positions`, `positions_v2`) zwei Tabellen `<space>_items(id, ext_id, content,
  embedding BLOB NOT NULL CHECK(Länge = dim·4))` und `<space>_meta` (Vertrag als JSON).
  Text, Schlüssel und Vektor sind **eine** atomare Zeile — die Zwei-Tabellen-
  Synchronisation von vec0 (`documents` ↔ `vec_documents`) entfällt. Mehrere Räume
  liegen nebeneinander in derselben Datei (Neuindizierung ohne Ausfall).
- **Validierung beim Schreiben**: exakte Länge (= Dimension), keine NaN/Inf, kein
  Nullvektor.
- **Store-Metadaten** (eigene Tabelle): Dimension, Metrik, Fingerprint von Modell und
  Tokenizer, Formatversion. Modellwechsel ⇒ erzwungene Neuindexierung.
- **Suche (gestuft, umgesetzt in 2b)**: jede Zeile speichert zusätzlich einen
  **1-Bit-Code** (Spalte `sign`, Bit i = Komponente i > 0; 128 Byte bei 1024 dim). Ab
  `IndexMinRows` (Default 20 000) hält der Store diese Codes im RAM (244 MiB bei 2 Mio),
  wählt per **Hamming-Distanz** (Popcount, Zählsortierung statt sortierter Liste)
  `Oversample` Kandidaten (Default 2000) und bewertet sie **exakt mit den float32-
  Originalen aus SQLite** nach — Kandidaten-Scan unter einem Read-Lock (parallele
  Anfragen, auch gefiltert), Nachbewertung in **einer** Anweisung
  (`id IN (SELECT value FROM json_each(?))`) und die Treffertexte danach — beides im
  **selben** Lese-Snapshot. Darunter, oder wenn
  k ≥ Zeilenzahl: exakter Scan (2a). Fachfilter maskieren die Index-Slots **vor** der
  Kandidatenwahl (Vorauswahl) und werden in der Nachbewertungs-Anweisung **erneut**
  angewandt — so passen Filter, Vektor und Text immer zum selben Snapshot, auch wenn eine
  andere Verbindung die Fachdaten dazwischen ändert.
  **Abweichung vom ursprünglichen Plan (int8 als Default):** gemessen ohne SIMD-Assembler
  (200k Vektoren, ein Kern, aarch64): exakt float32 160 ms, int8 + Nachbewertung 95 ms,
  **1 Bit + Nachbewertung 10 ms** (1000 Kandidaten, Recall@10 0,99) — hochgerechnet auf
  2 Mio wäre int8 rund 950 ms. int8 bleibt eine Option für eine spätere SIMD-Stufe.
- **Embedding-Cache (optional)**: wortgleiche Texte innerhalb **eines** Stores nur einmal
  einbetten; Artikel und Positionen bleiben getrennte Stores.
- **Fachfilter**: Erweiterung von `Search` um Filter (ID-Menge oder SQL-Join), angewandt
  vor Top-k.
- **Mehrere Vektorräume**: ein Store kann während einer Neuindizierung zwei Indexverträge
  nebeneinander halten; Umschalten erst nach vollständigem Aufbau.
- **v2 — IVF-Partitionierung**: k-Means-Zentroiden (≈ √N Cluster), Spalte `cluster_id`
  mit normalem Index; Suche prüft nur die nächsten `nprobe` Cluster. Einfügen = nächsten
  Zentroid zuordnen, Löschen = `DELETE`, periodisches Neutrainieren. Vollständig
  persistent in normalen Tabellen, kein RAM-Graph (deshalb IVF statt HNSW: HNSW braucht
  Graph + Vektoren im RAM und hat schwieriges Löschen).
- **Transaktionen**: `AddBatch` bleibt vollständig atomar; das bestehende
  `RollbackIfOpen`-Muster übernehmen (mORMots `Commit` löscht das Transaktionsflag
  auch bei Commit-Fehlern).
- **Cache-Invalidierung (umgesetzt in 2b)**: ein **persistenter** Generationszähler in
  `<space>_meta`, den Trigger bei jedem INSERT/UPDATE/DELETE auf `<space>_items` erhöhen —
  egal welche Verbindung schreibt. Eigene Schreibvorgänge lesen die Generation in ihrer
  IMMEDIATE-Transaktion vor und nach den Anweisungen und **patchen** den RAM-Index nach
  dem Commit, sofern er genau auf dem Vorher-Stand war; sonst (fremder Schreiber
  dazwischen) wird er verworfen. Jede Suche vergleicht die Generation und lädt den Index
  bei Abweichung neu (nur die Codes). Rollback/Commit-Fehler veröffentlichen nichts.
  Damit ist die frühere Einschränkung „ein koordinierter Schreibpfad je Datenbank"
  für die **Konsistenz** aufgehoben; WAL bleibt Voraussetzung (siehe unten).
  **Latenz-Grenze:** jede fremde Änderung erzwingt ein Voll-Neuladen der Codes (2,8–11 s
  bei 2 Mio, je nach Dateicache) unter der Verbindungssperre. Ein zweiter Prozess, der
  laufend schreibt, macht damit jede Suche langsam — Schreiben gehört in den Prozess, der
  auch sucht (dann wird der Index nur gepatcht), oder Massenimporte laufen vor dem Start.
- **Speicher**: float32 nur persistent (Wahrheit), im RAM nur die quantisierte Kopie —
  bei 2 Mio × 1024 sind float32 7,6 GiB, int8 1,9 GiB, 1-bit 244 MiB. Recall der
  Quantisierung gegen den exakten Scan messen. Cache mit Speicherlimit/LRU, nicht alle
  Stores dauerhaft geladen; vorladen vs. streamen per Konfiguration.
- **SIMD**: zuerst skalare Referenz (zugleich Test-Orakel), danach AVX2 (mit
  Laufzeitprüfung) und NEON. SIMD ist Optimierung, keine Voraussetzung. Für Stufe 2b
  beachten: `column_blob` liefert bei kleinen BLOBs innerhalb einer Seite **nicht
  ausgerichtete** Zeiger — Kernel mit ausgerichteten Loads müssen in einen eigenen,
  ausgerichteten Puffer kopieren (der RAM-Index ist das ohnehin).
- **Nebenläufigkeit / WAL** (umgesetzt in 2a, dokumentiert im Unit-Kopf): Der Store hat
  eine eigene Verbindung. Schreibt die Anwendung dieselbe Datei parallel, **muss** die
  Datei im WAL-Modus laufen — im Rollback-Journal hält ein Scan eine SHARED-Sperre, an
  der jedes COMMIT der Anwendung sofort mit SQLITE_BUSY scheitert (mORMot setzt dort
  keinen Busy-Timeout). `TNativeVectorStore.JournalMode` meldet den Modus beim Öffnen;
  die Anwendung setzt WAL und einen Busy-Timeout auf ihrer Seite. Der RAM-Index von 2b
  verkürzt die Lesesperre auf das Nachladen der Treffer.
- ~~**Extension-Lader**~~ — **erledigt**: `EnableSqliteExtensions`/`LoadSqliteExtension`
  liegen in `mormot.ai.sqlite`; lembed braucht sie, bis Teil B steht.

### Vergleich mit sqlite-vec — gleichwertig, besser oder schlechter?

Wir nutzen von vec0 nur einen kleinen Teil: flache `float[N]`-Tabelle, L2-Default,
`MATCH … AND k = ?` — **keine** ANN-Indizes, Metadaten-/Partitionsspalten oder
int8/Bit-Vektoren. Ausgeliefert ist `v0.1.10-alpha.4` (stabil: v0.1.9; ANN-Indizes
DiskANN/IVF/Rescore kamen mit 0.1.10-alpha, IVF noch experimentell, Deletes mit DiskANN
laut Release-Notes teuer).

**Besser:**
- keine Fremdabhängigkeit (kein `load_extension`, keine Arch-Bundles, kein SHA256-Download),
  Store auch unter Delphi baubar;
- keine Alpha-Version als Produktionsabhängigkeit;
- eine atomare Zeile statt zwei synchron zu haltender Tabellen; Upsert ohne
  Delete + Re-Insert;
- volle Kontrolle: Filter vor Top-k, deterministische Gleichstände, Eingabevalidierung,
  Modell-/Metrik-Vertrag in den Metadaten;
- voraussichtlich schnellere Suche (zusammenhängende Matrix im RAM statt Chunk-Lesen aus
  SQLite-Seiten je Abfrage) — **ungemessen**.

**Schlechter:**
- **RAM**: vec0 scannt von der Platte (Seiten-Cache), unser Entwurf hält die Matrix im
  Speicher (~391 MiB bei 100k × 1024) + Aufbaukosten nach Neustart/Invalidierung;
- **kein ANN**: ab ~10⁶ Vektoren skaliert vec0 (sobald stabil) besser;
- **kein KNN in SQL** (Joins in einer Abfrage), solange Option (b) nicht gebaut ist;
- **anfangs langsamer ohne SIMD** (vec0 bringt nach unserem Kenntnisstand AVX-/NEON-
  Distanzfunktionen mit — nicht am ausgelieferten Build geprüft);
- **Reife**: vec0 ist verbreitet und fuzz-getestet, unser Code neu.

**Einordnung** (geschrieben unter der Annahme ≤ 10⁵ Vektoren — durch das Mengengerüst
oben überholt; für 2 Mio+ gilt der gestufte Entwurf mit Quantisierung + IVF, gegen den vec0
im genutzten Flat-Modus ohnehin nicht mithält): Für ≤ 10⁵ Vektoren je Store gleichwertig bis besser — vec0 macht im von
uns genutzten Modus intern ebenfalls Brute-Force. Die eigentliche Designfrage ist
**Matrix im RAM** (schnell, speicherhungrig) vs. **je Abfrage aus SQLite streamen** (wie
vec0, sparsam, langsamer); sie hängt am Mengengerüst (offene Frage unten). Echter
ANN-Bedarf wird eine eigene Ausbaustufe hinter `IVectorStore` (HNSW/IVF in Pascal oder
pgvector im Serverfall).

### Referenz: sqliteai/sqlite-vector — bestätigt den Entwurf

[sqliteai/sqlite-vector](https://github.com/sqliteai/sqlite-vector) (geprüft: `0c2223a`,
2026-09-11, Version 1.1.2) setzt fast genau Option (a) um — in C:

- **Lizenz Apache 2.0** (laut Changelog früher eingeschränkt, später freigegeben; Firma
  sqliteai/SQLite Cloud dahinter → bei Nutzung Version pinnen).
- **Normale Tabellen mit BLOB-Spalten**, keine Virtual Table für die Daten (nur die
  Scan-Funktionen sind Table-valued Functions).
- Typen f32/f16/bf16/i8/u8/bit; Metriken L2, L2², L1, Cosine, Dot, Hamming.
- **Exakter Full-Scan** + **quantisierter Scan** (int8/uint8/1-bit/TurboQuant 2–4 bit),
  jeweils **vorgeladen** oder **aus SQLite gestreamt**; **kein ANN** (bewusst, „no need to
  preindex"). Filter über normales SQL.
- SIMD je Backend-Datei: AVX2, AVX512, SSE2, NEON, RISC-V Vector.
- Eigene Zahlen (Apple M5 Pro, dateibasierte DB): int8 vorgeladen 37,6 ms/Abfrage bei
  740 MB, int8 gestreamt 114 ms bei 30 MB, exakter float32-Scan 484 ms — genau der
  RAM-vs.-Streaming-Zielkonflikt aus der offenen Frage unten.

**Zukunftsfähigkeit** entsteht nicht durch die Bibliothekswahl, sondern durch zwei
Entscheidungen: (1) Vektoren als **schlichter float32-BLOB in der eigenen Tabelle** —
damit ist die Suchmaschine (unsere Pascal-Engine, sqlite-vector, später ANN) ohne
Datenmigration austauschbar, anders als bei vec0 mit seinen Schattentabellen; (2) die
Naht `IVectorStore`.

**Empfehlung:**
- Eigene Pascal-Engine nach diesem bewährten Design: Full-Scan (Referenz) +
  **quantisierter Scan mit float32-Rescoring**, vorladen/streamen per Konfiguration.
  (Umgesetzt in 2b als **1-Bit**-Index statt int8 — ohne SIMD war int8 rund 10× langsamer.)
  Ideen übernehmen, Code clean-room (Apache 2.0 ist mit der GPL-2-Variante der
  mORMot-Tri-Lizenz nicht kompatibel).
- **sqlite-vector als mögliches Austausch-Backend** (`TSqliteVectorStore`), falls unsere
  Engine nicht mithält — gleiches Datenformat, Umstieg per Konfiguration; dann aber wieder
  C (statisch in mORMots SQLite gelinkt statt `load_extension`, C-Build je Architektur).
- ANN erst bei echtem Bedarf (≫ 10⁶ Vektoren): HNSW in Pascal oder pgvector.
- Benchmark-Methodik (dateibasierte DB, gemessener Peak-RSS) für Gate 5 übernehmen.

---

## Teil B — Embeddings nativ (lembed ersetzen)

### Was das vorhandene Modell mitbringt

`vendor/models/bge-m3-q8_0.gguf` (GGUF v3) ist **selbsttragend**:

| Schlüssel | Wert |
|---|---|
| `general.architecture` | `bert` |
| `bert.block_count` / `embedding_length` / `feed_forward_length` / `head_count` | 24 / 1024 / 4096 / 16 |
| `bert.pooling_type` | 2 (CLS) |
| `bert.attention.layer_norm_epsilon` | 1e-5 |
| `tokenizer.ggml.model` | `t5` (= Unigram/UGM in llama.cpp) |
| Tokens | 250 002, bereits im XLM-R-ID-Raum (bos 0, pad 1, eos/sep 2, unk 3) — **kein** Fairseq-Remapping |
| `tokenizer.ggml.precompiled_charsmap` | 237 539 Byte |
| Tensoren | 145 × Q8_0, 244 × F32; Positionstabelle `[1024, 8192]` |

### neural-api — Stand `b172061`

**Vorhanden** (deutlich mehr als in CONCEPT.md bisher notiert):
`BuildXLMRobertaFromSafeTensors` (inkl. `pQuantizeInt8`, auch für die
Embedding-Tabelle), Unigram-/SentencePiece-Tokenizer, `PoolSentenceEmbedding` mit
CLS/Mean/LastToken, Instruction-Präfixe für E5/BGE, ein GGUF-Reader
(`neuralgguf.pas`) mit Q8_0/K-Quants, SafeTensors-Reader.

**Lücken für bge-m3:**

1. **Charsmap wird ignoriert** (`neuralhftokenizer.pas:334/2444`) → Normalisierung
   weicht bei deutschem/Unicode-Text ab. **Risiko Nr. 1.**
2. GGUF-Pfad lehnt `architecture=bert` und `tokenizer.model=t5` ab (nur
   llama/qwen2/gemma2) → für unser Asset wäre ein SafeTensors-Download + eigener
   Bereitstellungsschritt nötig.
3. **AArch64 ohne SIMD** (`NOTAVX`, rein skalar) — betrifft ARM-Entwicklungsrechner,
   Apple Silicon und ARM-Server.
4. Ein Netz ist **nicht reentrant** (`Compute` überschreibt interne Outputs) →
   serialisierte Queue.
5. Abhängigkeitsfläche: `neuralpretrained.pas` hat ~91k Zeilen und zieht Chat-,
   Audio-, Diffusion-, Torch-/GGUF-Units; `neuralthread` braucht Lazarus `UTF8Process`.
6. Paritätstests nur gegen ein Pico-Fixture (2 Layer, Hidden 8, Vokabular 11).
7. Die BERT-Satz-Helfer passen nicht direkt (`[CLS]/[SEP]`, Mean-Pooling, keine
   Padding-Maske) — BGE braucht eigene Tokenaufbereitung + `epCLS` + Normierung.

Lizenz: LGPL-2.1 mit Linking-Exception — für uns unkritisch. FPC-only ist laut
CONCEPT.md §1 erlaubt (Backend hinter der Naht).

### Option D — eigener schlanker Encoder auf dem vorhandenen GGUF

| Unit | Inhalt |
|---|---|
| `mormot.ai.gguf` | read-only GGUF-v3-Reader (Metadaten + gemappte Tensoren; F32/F16/Q8_0), robust gegen beschädigte Dateien |
| `mormot.ai.tokenizer.ugm` | Unigram/Viterbi-Tokenizer inkl. `precompiled_charsmap` (clean-room nach SentencePiece-Spezifikation) |
| `mormot.ai.encoder.bert` | BERT/XLM-R-Forward + CLS-Pooling + L2-Norm; Q8_0×F32-Kernel |
| `mormot.ai.embed.native` | `TNativeEmbedder: IEmbedder`, prozessweiter Dienst, eigene Queue |

**Vorteile:**
- Dieselben Gewichte und Tokenizer-Daten wie lembed → **lembed ist das Test-Orakel**.
- Keine Konvertierung, kein Python, das Modell-Asset bleibt unverändert.
- Baut auch unter Delphi 13 (Dual-Compiler-Linie der Extension).
- Kleine, kontrollierte Codebasis; die Charsmap-Arbeit kann upstream an neural-api.

**Was leicht unterschätzt wird** (aus dem Review):
- **Positionen**: die Tabelle hat 8192 Zeilen, der XLM-R-Offset (+2) ist beim
  Konvertieren schon abgeschnitten → bei Position 0 beginnen.
- **Charsmap**: binärer Darts/XCDA-Trie mit Ersatzstrings und längstem Präfixtreffer —
  ein allgemeines NFKC genügt nicht. Dazu Whitespace-Reihenfolge, UTF-8-Grenzen,
  Score-Gleichstände, Unknown-Zusammenfassung, Sondertokens.
- **Numerik**: Embedding-LayerNorm, Attention-Skalierung `1/√64`, stabile Softmax,
  Padding-Maske. ggml nutzt tanh-GELU und quantisiert im Q8_0-Pfad auch die
  Aktivierungen → **Parität nur annähernd**, nicht exakt. Den ausgelieferten
  lembed/llama-Build als Referenz pinnen.
- **Speicher/Tempo**: volle Attention bei 8192 Tokens × 16 Köpfe ≈ 4 GiB float32 →
  Kontextlänge kappen und/oder blockweise rechnen. Blocking, Pufferplanung und SIMD
  dominieren den Produktionsaufwand.
- Die anfängliche Schätzung „3–5k Zeilen" ist **nicht belastbar**; für den Encoder
  mehrere Wochen einplanen.

### Koexistenz und späterer Wechsel

Beide Wege passen unverändert hinter `IEmbedder`. Je Modell ein geteilter Dienst,
nie eine Modellkopie je Store. Für jeden Wechsel braucht es einen **Indexvertrag**
(Fingerprint von Modell + Tokenizer, Pooling, Normierung, Trunkierung, Metrik) —
`Model()` allein reicht nicht; Wechsel = kontrollierte Neuindexierung.
~~neural-api übernimmt den Embedder, sobald es diesen Vertrag erfüllt~~ — überholt
(2026-10-14): eigene Engine, siehe [native-inference.md](native-inference.md). Für E5-Modelle fehlt in
`IEmbedder` noch der Verwendungszweck (Query vs. Dokument, Präfixe `query:`/`passage:`).

Alternative zum Einstieg (Codex, Runde 1): **multilingual-e5-small** (384 dim,
Mean-Pooling, 512 Tokens) als kleinerer neural-api-Spike — löst aber das
Tokenizer-Problem nicht automatisch und erzwingt einen Modellwechsel.

---

## Referenzvertrag v1 (Schritt 1, 2026-10-13)

Der Vertrag beschreibt, **was ein Vektor in einem Store bedeutet**. Jeder Store speichert
ihn in seinen Metadaten; ein Embedder, der nicht exakt dazu passt, darf in diesen Store
weder schreiben noch mit ihm suchen. Änderung eines Feldes = neuer Vektorraum =
Neuindizierung.

| Feld | Wert v1 | Herkunft |
|---|---|---|
| Modell | `BAAI/bge-m3` | Nutzerentscheidung |
| Modelldatei | `bge-m3-q8_0.gguf`, **SHA-256 im Manifest** | `vendor/models/` |
| Laufzeit (Orakel) | sqlite-lembed `v0.0.1-alpha.8-landrix.1` (llama.cpp), SQLite 3.51.2 | gemessen |
| Dimension | 1024 | gemessen |
| Pooling | CLS | GGUF `bert.pooling_type = 2` |
| Normierung | L2 (Einheitsvektoren) | gemessen: Norm 1 ± 1e-6 |
| Metrik | **Cosinus** = Skalarprodukt auf Einheitsvektoren; Distanz `1 − dot` | Entscheidung |
| Max. Tokens | **512 inkl. `<s>`/`</s>`** | gemessen (llama.cpp-Default-Kontext) |
| Präfixe | keine (weder Query noch Dokument) | bge-m3-Modellkarte |
| Eingabetext | der Text unverändert (Kurztext + `\n` + Langtext) | Entscheidung |
| Sondertokens | `add_special = true`; `parse_special`: Orakel `true`, nativ **`false`** | gemessen / Entscheidung |

**Metrik**: vec0 rechnete bisher L2. Weil alle Vektoren Einheitslänge haben, ist das
Ranking nach L2 und nach Cosinus **identisch** (`‖a−b‖² = 2 − 2·a·b`); nur die Zahlenwerte
unterscheiden sich. Der native Store rechnet daher mit dem Skalarprodukt — das ist die
billigste Operation und passt direkt zu int8-Quantisierung.

### Befunde am Orakel (gemessen)

- **Lange Texte**: lembed **verweigert** Eingaben über 512 Tokens mit einem SQL-Fehler
  („Input too large: 548 tokens exceeds model context size of 512") statt zu kürzen. Wird
  `n_ctx` über `lembed_context_options('n_ctx', 8192)` angehoben, **bricht der Prozess ab**
  (`GGML_ASSERT(cparams.n_ubatch >= n_tokens && "encoder requires n_ubatch >= n_tokens")`) —
  lembed lässt `n_ubatch` nicht setzen. ⇒ Der Aufrufer muss **vorher** auf 512 Tokens
  kürzen; `n_ctx` darf in Produktion **nie** angehoben werden. Lange GAEB-Langtexte
  brauchen eine Kürzungsregel (v1: abschneiden; Chunking erst bei Bedarf).
- **Tokenizer**: mehrfache Leerzeichen werden zusammengefasst; Unicode-Normalisierung über
  die Charsmap ist aktiv (NFD-„Größe" ergibt dieselben IDs wie NFC, Vollbreite/Ligaturen
  werden gefaltet). `lembed_tokenize_json` liefert die Token-IDs — **Orakel für Schritt 3**.
- **Sondertokens im Nutzertext**: lembed tokenisiert mit `parse_special = true`, d. h.
  `</s>`, `<s>`, `<pad>` in einem Positionstext werden zu echten Steuertokens (IDs 2, 0, 1)
  und verfälschen das Embedding. Der native Tokenizer behandelt Nutzertext als Text
  (`parse_special = false`); der Testkorpus enthält diese Zeichenfolgen bewusst nicht.
- **Determinismus**: zweimal derselbe Text ⇒ Cosinus 1,0; zwei getrennte Generatorläufe
  liefern **bitgleiche** Vektordateien.
- **Durchsatz** (Snapdragon X, WSL aarch64, serialisiert, stark längenabhängig):
  **~25 Texte/s** bei ~15 Tokens, **~3,6–5,7 Texte/s** beim Korpus (⌀ 51 Tokens, max. 234;
  Streuung über vier Läufe, der Wert im Manifest ist der des erzeugenden Laufs).
  Hochgerechnet auf 2 Mio Artikel ≈ **22 h** (kurze Texte) bis **> 100 h** (Positionslänge)
  Erstindizierung — bestätigt: die Erzeugung ist der Engpass; IONOS oder ein schnellerer
  lokaler Pfad ist für Großbestände Pflicht. Auf x86_64 noch nicht gemessen.

### Referenzdaten

- Korpus: [`tests/data/embed-de/`](../tests/data/embed-de/) — selbst verfasste deutsche
  Handwerkspositionen und Artikel mit bewerteten Anfragen (Grad 2 = gleiche Leistung,
  1 = verwandt), inkl. harter Negative und Dimensionsvarianten.
- Generator: `demos/rag/embed-reference.lpr`, Aufruf `scripts/embed-reference.sh`
  (braucht die gitignorierten `vendor/`-Artefakte). Er prüft den Vertrag (Tokenlimit,
  Einheitsnorm, Dimension, bekannte IDs) und schreibt `reference/manifest.json`
  (Vertrag, SHA-256 von Modell und Korpus, Statistik, Metriken, exakte Top-20 je Anfrage)
  sowie `docs.f32`/`queries.f32` (float32 little-endian, Dateireihenfolge).
- Die Referenz wird **eingecheckt**: Paritäts- und Qualitätstests der folgenden Schritte
  laufen damit ohne lembed (Delphi, CI ohne `vendor/`).
- Korpusumfang: **262 Dokumente** (221 Positionen, 41 Artikel; 61 mit Langtext), **58
  Anfragen** (knapp, LV-Stil, umgangssprachlich), 164 Grad-2- und 104 Grad-1-Paare,
  ~55 Konzepte über 10 Gewerke, 11 harte Negative ohne eigene Anfrage.

### Basis-Metriken des Orakels (lembed + bge-m3 q8, Stand 2026-10-13)

| nDCG@10 | Recall@10 (Grad 2) | MRR (erstes Grad-2-Dokument, volle Rangliste) |
|---|---|---|
| **0,896** | **0,974** | **0,928** |

Das ist die Messlatte für Gate 4/5 (nativer Encoder: nDCG@10 höchstens 1 Prozentpunkt
darunter) und das Referenz-Ranking für Gate 2 (nativer Store im exakten Modus: identische
Top-20; Plätze, deren Referenz-Scores weniger als 1e-6 auseinanderliegen, gelten als
Gleichstand und dürfen tauschen — die Referenz hat echte Nachbarabstände ab ~2e-7, und
andere Rundungspfade (Delphi, FMA, SIMD/int8 in Stufe 2) dürfen dort nicht als Fehler
zählen). Gleiche Scores ordnen Generator und Store in Dateireihenfolge (= rowid). Die schwächsten Anfragen sind die erwartbar schweren:
umgangssprachliche Kundenmails („Bad soll neue Fliesen an der Wand bekommen") und Wand- vs.
Bodenfliesen — der Korpus trennt also, ist nicht trivial.

## Plan mit Prüfpunkten

| # | Schritt | Umfang | Freigabe nur wenn |
|---|---|---|---|
| 1 ✓ | Referenzvertrag: Modell-Hash, Runtime-Version, Token-Flags, Kontextgrenze, Metrik, **deutscher Testkorpus** — **erledigt 2026-10-13**, siehe „Referenzvertrag v1" | 1–2 Tage | festgelegt und dokumentiert |
| 2a ✓ | Nativer Store, exakte Suche: float32-BLOB je Vektorraum, Vertrag persistiert + erzwungen, Fachfilter (`IFilteredVectorStore.SearchWhere`, read-only SQL auf Fachtabellen derselben Datei, vor Top-k), mehrere Räume je Datei, Eingabeprüfung — **erledigt 2026-10-13** (`mormot.ai.vectorstore.native`, Tests `test.llm.vectorstore.native`) | – | **Parität: alle 58 Referenzanfragen liefern die Top-20 der Referenz (Score je Rang ± 1e-5, Dokument je Rang identisch außer bei Beinahe-Gleichstand < 1e-6)**; FTS5-Schlüsselabfragen (Hybridsuche) funktionieren; Atomarität, Fehlerfälle, Neustart, Raumtrennung getestet |
| 2b ✓ | Nativer Store, schnell: 1-Bit-Index im RAM + exakte float32-Nachbewertung, Generationszähler, parallele Anfragen, Benchmark `demos/rag/vector-bench.lpr` — **erledigt 2026-10-13** | – | über den Index mit voller Nachbewertung **identisch** zur Referenz; Recall@10 mit nur 20 von 262 Kandidaten **0,929** (echte bge-m3-Vektoren); Gate 5 s. u. |
| 2c | IVF-Partitionierung (nur bei Bedarf, s. Gate-5-Ergebnis) | ~1–2 Wochen | Recall@10 ≥ 0,95 gegen exakten Scan bei 2 Mio × 1024; Einfügen/Löschen ohne Neutraining korrekt |
| 3 → 1b | GGUF-Reader + UGM-Tokenizer (jetzt Phase 1b in [native-inference.md](native-inference.md), nach FP32-Phase 1a) | ~1 Woche | identische Token-IDs zu lembed auf ≥ 1000 Unicode-Fällen (wörtliche Sondertokens getrennt gegen `parse_special = false`); beschädigte Dateien sicher abgelehnt |
| 4 | Encoder: skalare Referenz, dann Blocking/SIMD | mehrere Wochen | Cosine ≥ 0,999 je Fall, korrekte Normen, Batch = Einzel; nDCG@10 höchstens 1 Prozentpunkt unter lembed |
| 5 | Messung auf Zielhardware | – | ≥ 80 % lembed-Durchsatz, ≤ 125 % Peak-RSS; Store-p95 ≤ 100 ms bei **2 Mio × 1024**, k = 10, vier Clients; Recall@10 ≥ 0,95 gegen exakten Scan; Store-RAM im konfigurierten Budget (Richtwert 2–4 GiB, Consumer-Hardware mit 32/64 GB); Erstindizierungs-Durchsatz für 2 Mio Texte ermittelt — sonst optimieren oder D stoppen |
| 6 | Integration: FPC 3.2.2/3.2.4 + Delphi 13; Extension-Download beim Konsumenten je Provider trennen | 2–3 Tage | danach Default „native", lembed bleibt expliziter Fallback |

Die Gate-Zahlen sind **Vorschläge**, noch nicht abgestimmt.

### Gate-5-Messung des Stores (2026-10-13)

2 Mio synthetische 1024-dim-Vektoren (aus den echten bge-m3-Referenzvektoren gemischt
+ Rauschen, normiert), Datei 9,2 GB, WAL, Snapdragon X unter WSL (aarch64 — **ohne**
Hardware-Popcount in mORMot; x86_64 mit POPCNT ist noch nicht gemessen):

| Kandidaten | Recall@10 vs. exakt | p95 1 Client | p95 4 Clients |
|---|---|---|---|
| 1000 | 0,94 | 81 ms | 89 ms |
| **2000 (Default)** | **0,97** | **88 ms** | **105 ms** |
| 4000 | 0,99 | 118 ms | 166 ms |

Methode: Recall über die ersten **10** Referenzanfragen (Auflösung 0,01); Latenz aus den
58 Referenzanfragen je Client (1 Client: 58 Proben, 4 Clients: 232), Cache warm, gleiche
Reihenfolge — die Werte schwanken zwischen Läufen um einige ms. Einfügen 7 473 Zeilen/s
(2 Mio in 4,5 min), Index laden 2,8–11 s je nach Dateicache (244 MiB; `WarmUp` beim
Serverstart aufrufen), exakter Scan zum Vergleich 5,4 s je Anfrage. Recall-Gate (≥ 0,95)
erfüllt, Latenz-Gate (p95 ≤ 100 ms bei 4 Clients) um wenige ms verfehlt — der
Hamming-Scan dominiert. Nächste Hebel, falls nötig: Scan einer Anfrage über mehrere Kerne, NEON-/POPCNT-
Kernel, IVF (2c) zur Verkleinerung der gescannten Menge.

## Offene Fragen (vor dem Start zu klären)

- ~~Metrik~~ — **entschieden**: Cosinus = Skalarprodukt auf Einheitsvektoren (Ranking
  identisch zu L2), siehe „Referenzvertrag v1".
- ~~Mengengerüst~~ — **beantwortet** (bis 2 Mio Artikel, Positionen ~2,6 Mio in 10
  Jahren, siehe „Mengengerüst"): Quantisierung in v1 (1-Bit-Index, 2b), IVF nur bei Bedarf (2c).
- ~~Zielhardware~~ — **beantwortet**: Consumer-Klasse, 32/64 GB RAM; 1-Bit-Index im RAM als Default (2b).
- ~~Was eingebettet wird~~ — **beantwortet**: Texte von Artikeln bzw. Positionen, getrennte
  Stores, Zweck Analyse.
- ~~Modell~~ — **beantwortet**: bge-m3 lokal + IONOS (BGE M3) als externer Anbieter;
  ein kleineres Modell nur, falls der lokale Durchsatz nicht reicht.
- ~~Fachfilter~~ — **beantwortet**, siehe „Anwendungsfall und Fachfilter".
- ~~neural-api-Beitrag~~ — **erledigt**: Issue
  [#225](https://github.com/joaopauloschuler/neural-api/issues/225) am 2026-10-14 ohne
  Antwort als „not planned" geschlossen; eigene Engine ([native-inference.md](native-inference.md)).
- Maximale Kontextlänge für Embeddings (512? 2048?) — bestimmt Speicherbedarf der
  Attention.
- ~~Option D oder direkt neural-api?~~ — **entschieden**: eigene Engine, Option D geht
  als Phase 1 darin auf.
- Wird der heutige `TVec0Store` nach Teil A gelöscht (Greenfield) oder als Fallback
  behalten?
