# Verteilte Ausführung auf ungenutzten Rechnern — Plan

> **Status (2026-10-15):** Plan, noch kein Code. Arbeitstitel **Rechenknecht**, der Name
> ist offen.
> Grundlage ist der Entwurf des Nutzers „Plan: Verteilte Ausführung auf ungenutzten
> Rechnern" vom 2026-10-15. Er ist hier an den Stand der Engine angepasst. Neu sind die
> Abschnitte „Was es schon gibt", „Schichten" und „Offene Entscheidungen".
> Inspiriert von Apples „Minions" (Quelle unten). Einordnung: [CONCEPT.md §9](../CONCEPT.md),
> Punkt 11.

## Ziel

Große Inferenz-Jobs sollen auf Rechner im Netz verteilt werden, die gerade niemand
nutzt. Zuerst geht es um Embedding-Batches, später um LLM-Completions. Grundlage ist
die eigene Pascal-Engine aus [native-inference.md](native-inference.md).

## Grundprinzip

- **Aufgaben, nicht das Modell, werden verteilt.** Ein großer Job wird in unabhängige
  Teilaufgaben zerlegt, etwa 1 Mio. Dokumente in 1000 Pakete zu je 1000 Dokumenten.
  - Jeder Rechner verarbeitet ganze Pakete mit einem vollständigen, lokal geladenen
    Modell.
  - Ein einzelnes Modell wird nicht über mehrere Rechner verteilt.
- **Pull statt Push.** Die Rechner fragen selbst nach Arbeit. Der Server muss keinen
  Rechner erreichen können, das funktioniert auch hinter NAT und Firewalls.
- **Gemischte Hardware.** Jeder Rechner, auf dem die Engine läuft, kann mitarbeiten:
  Windows, Linux und macOS, mit oder ohne GPU.

## Was es schon gibt

Seit dem Entwurf hat sich die Lage geändert. Phase 1 kann sofort beginnen.

| Baustein | Stand | Nutzen hier |
|---|---|---|
| **Native Embeddings** (`mormot.ai.embed.native`, BGE-M3 aus dem Q8_0-GGUF) | Phase 1 der Engine fertig; im Backend der empfohlene Provider | Die Rechenarbeit des Executors, ohne Fremd-`.so`. Eine Codebasis, je Plattform ein eigenes Binary. |
| **Kernel und Threads** (`TInferCpuFastBackend`, AVX2/NEON) | fertig, gemessen | Eine Fähigkeit, die der Executor melden kann (`avx2`, `neon`, `pascal`). |
| **Load Governor** (`mormot.ai.governor`) | gebaut, noch ohne Aufrufer | Bremse des Executors: Tastgrad, Temperaturschwelle, Durchsatz-Heuristik. Er pausiert **zwischen** Arbeitseinheiten. `Abort` beendet nur seine Wartezeiten, eine laufende Berechnung bricht er nicht ab. |
| **Determinismus** | gemessen (Soak-Test, ein Rechner) | Auf einem Rechner (i9, Linux x86_64) liefert dasselbe Binary über 90 Minuten bitgleiche Vektoren: 79.308 Vergleiche, 0 Abweichungen. Zwischen Rechnern ist das nicht gemessen. |
| **Exact-Modus** des Backends | fertig | Bitgleich zur skalaren Referenz **desselben Builds** (getestet). Ob zwei Plattformen im Exact-Modus bitgleich rechnen (gleiche double-Arithmetik, aber verschiedene RTL-Funktionen wie `Exp`), ist nicht gemessen. Das prüft Phase 1. |
| **Decoder Qwen3** | bis 2c fertig | LLM-Jobs (Phase 4 unten) brauchen noch 2d (`ILlmClient`) und Phase 3 der Engine (Quantisierung, Sampling). |

**Was fehlt:**
- die Leerlauf-Erkennung;
- Queue, Lease und REST-API;
- der Executor als eigenes Binary.

## Komponenten

### Server

- Job-Queue in SQLite (mORMot2).
- REST-API über den HTTP-Server von mORMot2 (`THttpAsyncServer` und `TUriRouter`).
- Vergibt Teilaufgaben mit einer **Leasezeit** statt fester Zuteilung.
- Sammelt die Ergebnisse und setzt sie zum Gesamtergebnis zusammen.
- Verteilt Modelldateien mit Hash-Prüfung.

### Executor (auf jedem Rechner)

- Das Pascal-Binary mit der Inferenz-Engine.
- Erkennt Leerlauf und holt nur dann Arbeit.
- Meldet seine Fähigkeiten: Backend (Pascal, AVX2, NEON, später CUDA oder Metal),
  Speicher, Threads und die Modelle im Cache.
- Sendet Heartbeats, solange er an einer Aufgabe arbeitet.
- Drosselt oder pausiert, sobald jemand den Rechner wieder nutzt:
  - Das Drosseln übernimmt der Load Governor (Tastgrad, Temperatur). Er pausiert aber
    nur zwischen Arbeitseinheiten.
  - Ein **sofortiger** Halt mitten in einem Paket ist neu zu bauen: eine
    Abbruchprüfung zwischen den Batches des Embedders. Die Lease geht dann zurück.
  - Auslöser ist die neue Leerlauf-Erkennung.

### Watchdog (im Server)

Ein eigener Dienst ist nicht nötig:
- Läuft eine Leasezeit ab, ohne dass ein Heartbeat oder ein Ergebnis kommt, geht die
  Teilaufgabe zurück in die Queue.
- Nach N Fehlversuchen wird sie als fehlerhaft markiert.

## Datenmodell

| Tabelle | Inhalt |
|---|---|
| `jobs` | ID, Typ (embed / generate), Modell und Hash, Quantisierung, Status, Fortschritt, Ersteller, Erstellt am |
| `tasks` | ID, Job-ID, Eingabe-Referenz, Status (offen / vergeben / fertig / fehlerhaft), Executor-ID, Lease bis, Versuche |
| `results` | Task-ID, Ergebnis-Referenz oder BLOB, Rechenzeit, Executor-ID, Backend |
| `executors` | ID, Name, Plattform, Backend, Speicher, Modelle im Cache, letzter Kontakt, Freigabegruppe, gesperrt |
| `models` | Name, Version, Quantisierung, Hash, Dateigröße, Download-Pfad |

## REST-API (Entwurf)

| Methode | Pfad | Zweck |
|---|---|---|
| POST | `/executor/register` | Executor meldet sich mit seinen Fähigkeiten an |
| POST | `/task/claim` | Nächste passende Teilaufgabe holen, mit Lease |
| POST | `/task/{id}/heartbeat` | Lease verlängern, Fortschritt melden |
| POST | `/task/{id}/result` | Ergebnis abliefern |
| POST | `/task/{id}/fail` | Fehler melden, Aufgabe freigeben |
| GET | `/model/{name}` | Modelldatei herunterladen |
| POST | `/job` | Neuen Job anlegen |
| GET | `/job/{id}` | Status und Fortschritt |

- **Passende Aufgaben:** `/task/claim` vergibt nur Aufgaben, deren Modell, Quantisierung
  und Speicherbedarf zum Executor passen.
- **Ergebnis idempotent:** Ein doppelt geliefertes Ergebnis darf nichts doppelt
  schreiben. Das passiert zum Beispiel, wenn ein Executor nach Ablauf der Lease doch
  noch liefert. Gültig ist nur das Ergebnis des aktuellen Lease-Inhabers, oder das
  erste, das ankommt.
- **Kurze Handler:** Alle Aufrufe sind kurz. Claim, Heartbeat und Ergebnis sind
  DB-Operationen, nie lange Wartezeiten. Der Server-Pool von mORMot ist klein und
  darf nicht blockieren.
- **Paketgröße:** mORMot puffert Request-Bodies per Default vollständig im RAM.
  - Pakete bleiben deshalb klein: 1000 Dokumente ergeben etwa 4 MB Vektoren
    (1000 × 1024 × 4 Byte), am besten binär statt als JSON-Zahlenliste.
  - Größere Ergebnisse lassen sich über `OnBodyDownload` als Stream annehmen (mORMot
    `THttpAsyncServer`). Dann bleibt der Speicher begrenzt.

## Leerlauf-Erkennung

| Plattform | Benutzereingaben | Zusätzlich |
|---|---|---|
| Windows | `GetLastInputInfo` | CPU- und GPU-Last, Akku- oder Netzbetrieb; EcoQoS abschalten (siehe unten) |
| macOS | IOKit `HIDIdleTime` | Last, Akku, Energiesparmodus |
| Linux | logind `IdleHint` bzw. X11/Wayland | Load Average; ein Server ohne Benutzer ist immer frei |

- **Pro Rechner einstellbar:** Mindest-Leerlaufzeit, erlaubte Uhrzeiten, nur im
  Netzbetrieb, maximale Last, ob die GPU genutzt werden darf.
- **Windows-Falle (gemessen 2026-10-14):** Auf dem i9 lief ein per SSH gestarteter
  Prozess durch EcoQoS nur mit etwa halber Leistung.
  - Für Dienste ist dasselbe zu erwarten, aber nicht gemessen.
  - Der Executor sollte die Drosselung per
    `SetProcessInformation(ProcessPowerThrottling)` abschalten. `demos/infer/embed-bench.lpr`
    versucht das schon (außer mit `LAI_THROTTLE`), und der Erfolg ist zu prüfen.

## Konsistenz

- **Gleiches Modell überall.** Modellversion und Quantisierung sind pro Job festgelegt
  und werden per Hash geprüft.
- **Gleiche Rechnerklasse:** Auf einem Rechner liefert dasselbe Binary bitgleiche
  Vektoren (Soak-Test). Ob zwei Rechner mit gleichem Binary und gleichem Backend
  bitgleich rechnen, ist wahrscheinlich, aber erst in Phase 1 zu belegen.
- **Verschiedene Backends weichen minimal ab.**
  - AVX2, NEON und Pascal liefern minimal abweichende Werte. Gemessen bleibt die
    Retrieval-Qualität gleich: nDCG@10 liegt bei 0,8994 auf AVX2 und 0,9004 auf NEON.
  - Executor und Backend werden pro Ergebnis protokolliert.
  - Wo bitgleiche Ergebnisse nötig sind, ist der Exact-Modus der Kandidat. Er ist
    bitgleich zur Referenz desselben Builds. Bitgleichheit über Plattformen hinweg ist
    noch zu messen.
- **Optional:** Stichproben doppelt rechnen lassen und vergleichen, bei derselben Klasse
  bitgleich, bei verschiedenen innerhalb der Toleranz. So entsteht nebenbei auch der
  Nachweis über Rechner hinweg.

## Sicherheit

- **Verbindung:** TLS für alle Verbindungen.
- **Executor-Identität:** Jeder Executor hat ein eigenes Token. Einzelne Rechner lassen
  sich jederzeit sperren.
- **Freigabegruppen:** Jobs mit sensiblen Daten gehen nur an Rechner der eigenen Gruppe.
- **Daten auf dem Executor:** Eingaben und Ergebnisse liegen nur im Speicher oder
  verschlüsselt im lokalen Cache und werden nach Abschluss gelöscht.
- **Modelle:** Modelldateien kommen nur vom eigenen Server, mit Hash-Prüfung.
- **Rechte:** Der Executor läuft mit eingeschränkten Rechten.

## Schichten

Die Regel aus CONCEPT.md §2 gilt auch hier: Generisches gehört in die Extension, die
Bindung an Landrix in den Konsumenten.
- **landrixai (generisch):**
  - Queue, Lease und Watchdog als Bibliothek;
  - die Executor-Schleife (Claim, Heartbeat, Ergebnis, Governor, Leerlauf);
  - die Leerlauf-Erkennung je Plattform;
  - das Executor-Binary.
  - Arbeitsname der Units: `mormot.ai.dist.*`.
- **Konsument (Landrix-Backend):** Er liefert:
  - Anmeldung und Rechte (wer darf Jobs anlegen, wer Executoren verwalten);
  - Mandanten;
  - Speicherort von Eingaben und Ergebnissen (Blob-Speicher);
  - das Einspielen fertiger Vektoren in den Vektor-Store.

## Phasen

### Phase 1 — Embeddings im LAN

- Server mit Queue, Lease und REST-API.
- Executor auf der CPU, Leerlauf-Erkennung für Windows und Linux, Drosseln über den
  Load Governor.
- Nur der Job-Typ „embed", ein Modell (BGE-M3, Q8_0-GGUF).

**Fertig, wenn:** Alle drei Bedingungen sind erfüllt:
- Ein großer Embedding-Job wird auf mehrere Rechner verteilt.
- Ein mittendrin abgeschalteter Rechner verliert keine Daten.
- Das Ergebnis gleicht einem Lauf auf einem einzelnen Rechner, innerhalb der Toleranz.
  Ob über gleiche Rechnerklassen oder den Exact-Modus sogar Bitgleichheit erreichbar
  ist, wird in dieser Phase gemessen.

### Phase 2 — Robustheit und Komfort

- Modellverteilung und Cache.
- Freigabegruppen, Tokens, Sperrliste.
- Status-Übersicht: Jobs, Fortschritt, aktive Rechner.
- macOS-Executor (IOKit `HIDIdleTime`).

### Phase 3 — GPU-Rechner

- Der Executor nutzt das CUDA- bzw. Metal- oder Vulkan-Backend der Engine. Das setzt
  Phase 4 der Engine voraus.
- Der Scheduler bevorzugt schnelle Rechner für große Pakete.

### Phase 4 — LLM-Jobs

- Job-Typ „generate" für Batch-Completions, z. B. die Klassifikation vieler Texte.
- Zuteilung nach verfügbarem Speicher.
- Setzt den Decoder hinter `ILlmClient` (Engine-Phase 2d) und die Quantisierung voraus
  (Engine-Phase 3). In FP32 braucht Qwen3-0.6B etwa 2,4 GB.

### Optional später

- Verbund über Standorte hinweg (Internet statt LAN).
- Priorisierung und Kontingente pro Benutzer.

## Start

1. Datenmodell und REST-API mit mORMot2 aufsetzen, zunächst mit einem
   Dummy-Executor, der nur wartet und zurückmeldet.
2. Lease-Ablauf und Wiedervergabe testen: den Executor absichtlich abstürzen lassen.
3. Leerlauf-Erkennung für Windows einbauen.
4. Die echte Embedding-Berechnung einhängen (`TNativeEmbedder`). Das geht sofort, weil
   Phase 1 der Engine fertig ist.

Der Verteilteil lässt sich damit parallel zur Engine entwickeln und testen. Die
Engine-Roadmap (Decoder hinter `ILlmClient`, Quantisierung, GPU) bleibt davon
unberührt.

## Offene Entscheidungen

1. **Name:** Rechenknecht oder ein anderer.
2. **Wo der Server läuft:**
   - als eigener Daemon aus der Extension;
   - oder als Teil des Landrix-Backends, mit dessen Anmeldung und Blob-Speicher.
3. **Wie Eingaben und Ergebnisse übertragen werden:**
   - inline (klein, einfach);
   - oder als Referenz in den Blob-Speicher (große Pakete, 4-MiB-Chunks).
4. **Gate von Phase 1:** Toleranz über gemischte Rechner, oder bitgleich. Bitgleich
   setzt voraus, dass die Messung den Exact-Modus über Plattformen hinweg bestätigt.
   Er ist langsamer.
5. **Executor-Anmeldung:** Wie das Token auf einen neuen Rechner kommt, etwa durch eine
   einmalige Kopplung per Code.

## Quelle

Minions: A Dynamic and Distributed Virtual GPU Cloud Built on Idle Apple Silicon
Devices — Apple Machine Learning Research, Oktober 2026 —
https://machinelearning.apple.com/research/distributed-gpu-cloud
