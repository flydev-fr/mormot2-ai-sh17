# embed-de – deutscher Retrieval-Testkorpus (Handwerk)

## Zweck

Qualitäts-Gate für Embedding-Modelle (Referenz: BAAI/bge-m3) im Handwerksbereich.
Geprüft wird, ob eine **neue Angebotsposition** (aus einem GAEB-LV, als Freitext oder
aus einer Kundenmail) per Vektorsuche die passenden **alten Positionen und Artikel**
findet, damit ein Angebot aus Bestandspositionen aufgebaut werden kann.

## Format

Beide Dateien sind JSON Lines (UTF-8, LF, kein BOM), ein Objekt je Zeile.

`corpus.jsonl` – Dokumente:

```json
{"id":"p0001","kind":"position","unit":"m","text":"Kurztext\nLangtext …"}
{"id":"a0001","kind":"article","unit":"m","text":"Bezeichnung\nBeschreibung …"}
```

- `kind`: `position` (Leistungsposition: Kurztext, oft mit Langtext nach einem
  Zeilenumbruch) oder `article` (Katalogartikel: Bezeichnung + kurze Beschreibung).
- `unit`: Abrechnungseinheit (m, lfm, m², m³, Stk, St, Stck, psch, h, kg, l).

`queries.jsonl` – Anfragen mit Relevanzurteilen:

```json
{"id":"q001","text":"…","rel":{"p0001":2,"p0002":2,"a0001":1}}
```

- `2` = gleiche Leistung bzw. gleicher Artikel (anders formuliert, ggf. andere
  Einheit, Abkürzung oder Wortstellung).
- `1` = verwandt bzw. teilweise passend (andere Dimension oder Ausführung, z. B.
  DN 20 statt DN 15, oder Zubehör; reiner Artikel zu einer „liefern und
  montieren“-Anfrage).
- nicht genannt = irrelevant. Jede Anfrage hat mindestens zwei Grad-2-Dokumente.

## Aufbau

- Gewerke: Sanitär, Heizung, Lüftung, Elektro, Fliesen, Maler, Trockenbau,
  Rohbau/Abbruch, Dach/Klempner, Außenanlagen, dazu Allgemeines (Regiestunden,
  Baustelleneinrichtung).
- Rund 55 Leistungs-/Artikel-Konzepte mit je 2–6 unterschiedlich formulierten
  Dokumenten (Synonyme, Abkürzungen wie DN, inkl., einschl., lfm, Stck, psch;
  mal nur Kurztext, mal mit Langtext, mal im Artikelstil).
- Dimensions- und Ausführungsvarianten (z. B. DN 15/20/25, Typ 22/33,
  60/80 cm, 3 x 1,5 / 5 x 2,5 mm², Q2/Q3/Q4) sind als Grad 1 bewertet.
- **Harte Negative**: Dokumente mit fast gleichem Wortlaut, aber anderer Leistung
  (Kupferrohr verlegen ↔ demontieren, Heizkörper montieren ↔ lackieren ↔
  ab-/anbauen, Wand verputzen ↔ Putz abschlagen, tapezieren ↔ Tapete entfernen,
  Pflaster verlegen ↔ aufnehmen, Dachrinne montieren ↔ reinigen, Schlitze fräsen ↔
  zuputzen u. a.). Sie sind nie Grad 2 zu der jeweils anderen Leistung; 11 von
  ihnen gehören zu keiner Anfrage und dienen nur als Störer.
- Anfragen: knapp im LV-Stil, als Kurztext + Langtext (4 Anfragen) oder
  umgangssprachlich wie aus einer Kundenmail.
- Langtexte höchstens ca. 100 Wörter (Modellgrenze 512 Tokens); 46 Positionen
  (ca. 21 %) haben einen Langtext über 80 Wörter.

## Zählung

| Größe | Anzahl |
|---|---|
| Dokumente gesamt | 262 |
| davon `position` | 221 |
| davon `article` | 41 |
| Anfragen | 58 |
| Relevanzpaare Grad 2 | 164 |
| Relevanzpaare Grad 1 | 104 |

## Herkunft

Selbst verfasst, synthetisch. Keine Texte aus STLB, Normen oder Herstellerkatalogen,
keine Markennamen, keine Personen- oder Kundendaten. Normen werden nur als Kürzel
genannt (z. B. „nach DIN 1988“). Maße, Preise und Leistungswerte sind realistisch
gewählt, aber fiktiv.

## Referenz (`reference/`)

Erzeugt mit dem Orakel sqlite-lembed + bge-m3 q8 über `scripts/embed-reference.sh`
(Generator `demos/rag/embed-reference.lpr`); eingecheckt, damit spätere Tests ohne die
lembed-Laufzeit auskommen:

- `manifest.json` – Referenzvertrag (Modell-SHA-256, Laufzeit, Dimension, Pooling,
  Normierung, Metrik, Tokenlimit, Sondertoken-Verhalten), SHA-256 dieser beiden
  `.jsonl`-Dateien, Statistik, Metriken (nDCG@10, Recall@10, MRR) und die exakte
  Top-20 je Anfrage (`[id, score, grade]`).
- `docs.f32`, `queries.f32` – Vektoren als float32 little-endian, Zeile für Zeile in
  Dateireihenfolge.

Wer Korpus, Modell oder lembed-Build ändert, erzeugt die Referenz neu und nennt die neuen
Metriken im Commit. Details: [docs/native-embeddings.md](../../../docs/native-embeddings.md),
Abschnitt „Referenzvertrag v1".
