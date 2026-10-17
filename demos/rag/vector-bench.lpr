// LandrixAI - benchmark of the native vector store (gate 5 of
// docs/native-embeddings.md)
// - fills a store with N synthetic 1024-dim vectors derived from the committed
//   bge-m3 reference (mix of two reference documents + noise, normalised - so
//   the data has the cluster structure of real embeddings), then measures:
//   insert throughput, index load (first search), query latency p50/p95 with 1
//   and with T concurrent clients, recall@10 of the index against the exact scan
// - the database is kept: a second run with the same N skips the fill
//
//   usage: vector-bench <N> <dbfile> [clients=4] [oversample=1000]
//   e.g.   vector-bench 2000000 /tmp/vbench.db
program vector.bench;

{$I mormot.defines.inc}

{$ifdef OSWINDOWS}
  {$apptype console}
{$endif OSWINDOWS}

uses
  {$I mormot.uses.inc}
  classes,
  sysutils,
  mormot.core.base,
  mormot.core.os,
  mormot.core.unicode,
  mormot.core.text,
  mormot.db.raw.sqlite3,
  mormot.db.raw.sqlite3.static,
  mormot.ai.llm.types,
  mormot.ai.vectorstore,
  mormot.ai.vectorstore.native;

const
  DIM = 1024;
  K = 10;
  BATCH = 1000;

var
  store: TNativeVectorStore;
  istore: IVectorStore;
  queries: TLlmEmbeddingDynArray;

function FindReferenceDir: TFileName;
var
  dir, cand: TFileName;
  i: integer;
begin
  result := '';
  dir := ExcludeTrailingPathDelimiter(Executable.ProgramFilePath);
  for i := 1 to 6 do
  begin
    cand := MakePath([dir, 'tests', 'data', 'embed-de', 'reference'], true);
    if FileExists(cand + 'manifest.json') then
    begin
      result := cand;
      exit;
    end;
    dir := ExtractFileDir(dir);
  end;
end;

function LoadVectors(const aFile: TFileName): TLlmEmbeddingDynArray;
var
  raw: RawByteString;
  i: integer;
begin
  raw := StringFromFile(aFile);
  SetLength(result, length(raw) div (DIM * SizeOf(single)));
  for i := 0 to high(result) do
  begin
    SetLength(result[i], DIM);
    MoveFast(PByteArray(raw)^[i * DIM * SizeOf(single)], result[i][0],
      DIM * SizeOf(single));
  end;
end;

function RandGauss: double;
var
  u, v: double;
begin
  repeat
    u := Random;
  until u > 1e-12;
  v := Random;
  result := sqrt(-2 * ln(u)) * cos(2 * pi * v);
end;

procedure Fill(n: integer; const topics: TLlmEmbeddingDynArray);
var
  texts: TRawUtf8DynArray;
  vecs: TLlmEmbeddingDynArray;
  i, j, b, ta, tb, done: integer;
  wa, wb: single;
  t0: Int64;
begin
  RandSeed := 42;
  SetLength(texts, BATCH);
  SetLength(vecs, BATCH);
  for i := 0 to BATCH - 1 do
    SetLength(vecs[i], DIM);
  done := 0;
  t0 := GetTickCount64;
  while done < n do
  begin
    b := MinPtrInt(BATCH, n - done);
    SetLength(texts, b);
    SetLength(vecs, b);
    for i := 0 to b - 1 do
    begin
      ta := Random(length(topics));
      tb := Random(length(topics));
      wa := 0.6 + Random * 0.4;
      wb := Random * 0.5;
      for j := 0 to DIM - 1 do
        vecs[i][j] := wa * topics[ta][j] + wb * topics[tb][j] + 0.012 * RandGauss;
      texts[i] := FormatUtf8('row %', [done + i]);
    end;
    istore.AddBatch(texts, vecs);
    inc(done, b);
    if done mod 100000 = 0 then
      ConsoleWrite(FormatUtf8('  % rows, % rows/s', [done,
        done * 1000 div (GetTickCount64 - t0 + 1)]));
  end;
  ConsoleWrite(FormatUtf8('fill: % rows in % s (% rows/s)', [n,
    (GetTickCount64 - t0) div 1000, n * 1000 div (GetTickCount64 - t0 + 1)]));
end;

function Percentile(var a: TInt64DynArray; p: integer): Int64;
begin
  QuickSortInt64(pointer(a), 0, high(a));
  result := a[MinPtrInt(high(a), (length(a) * p) div 100)];
end;

type
  TClient = class(TThread)
  public
    Lat: TInt64DynArray;
    procedure Execute; override;
  end;

procedure TClient.Execute;
var
  i: integer;
  t: Int64;
begin
  SetLength(Lat, length(queries));
  for i := 0 to high(queries) do
  begin
    t := GetTickCount64;
    istore.Search(queries[i], K);
    Lat[i] := Int64(GetTickCount64) - t;
  end;
end;

var
  n, clients, oversample, i, j, h, found: integer;
  dbfile, refdir: TFileName;
  topics: TLlmEmbeddingDynArray;
  app: TSqlDataBase;
  t0: Int64;
  lat, all: TInt64DynArray;
  pool: array of TClient;
  approx, exact: TRagHitDynArray;
  recall: double;
begin
  n := StrToIntDef(ParamStr(1), 100000);
  dbfile := ParamStr(2);
  clients := StrToIntDef(ParamStr(3), 4);
  oversample := StrToIntDef(ParamStr(4), 1000);
  if dbfile = '' then
  begin
    ConsoleWrite('usage: vector-bench <N> <dbfile> [clients] [oversample]');
    exit;
  end;
  refdir := FindReferenceDir;
  topics := LoadVectors(refdir + 'docs.f32');
  queries := LoadVectors(refdir + 'queries.f32');
  // the owner sets WAL (the store never changes the journal mode)
  app := TSqlDataBase.Create(dbfile, '');
  try
    app.Execute('PRAGMA journal_mode = WAL;');
  finally
    app.Free;
  end;
  store := TNativeVectorStore.Create(StringToUtf8(dbfile), 'bench',
    VectorContract('bench/bge-m3-like', '', DIM));
  istore := store;
  store.Oversample := oversample;
  ConsoleWrite(FormatUtf8('store: % rows, journal %, oversample %',
    [istore.Count, store.JournalMode, oversample]));
  if istore.Count <> n then
  begin
    if istore.Count <> 0 then
    begin
      ConsoleWrite('row count differs - use a new db file', ccLightRed);
      exit;
    end;
    Fill(n, topics);
  end;
  // a FRESH store instance: measures loading the index from the sign column
  istore := nil;
  store := TNativeVectorStore.Create(StringToUtf8(dbfile), 'bench',
    VectorContract('bench/bge-m3-like', '', DIM));
  istore := store;
  store.Oversample := oversample;
  t0 := GetTickCount64;
  istore.Search(queries[0], K);
  ConsoleWrite(FormatUtf8('first search incl. index load: % ms (% rows, ~% MiB ' +
    'sign codes)', [GetTickCount64 - t0, store.IndexedRows,
    store.IndexedRows * (DIM div 8) shr 20]));
  // 1 client
  SetLength(lat, length(queries));
  for i := 0 to high(queries) do
  begin
    t0 := GetTickCount64;
    istore.Search(queries[i], K);
    lat[i] := GetTickCount64 - t0;
  end;
  ConsoleWrite(FormatUtf8('1 client : p50 % ms, p95 % ms', [Percentile(lat, 50),
    Percentile(lat, 95)]));
  // T concurrent clients
  SetLength(pool, clients);
  for i := 0 to clients - 1 do
    pool[i] := TClient.Create(false);
  all := nil;
  for i := 0 to clients - 1 do
  begin
    pool[i].WaitFor;
    for j := 0 to high(pool[i].Lat) do
    begin
      SetLength(all, length(all) + 1);
      all[high(all)] := pool[i].Lat[j];
    end;
    pool[i].Free;
  end;
  ConsoleWrite(FormatUtf8('% clients: p50 % ms, p95 % ms', [clients,
    Percentile(all, 50), Percentile(all, 95)]));
  // recall@10 of the index against the exact streamed scan (first 10 queries)
  recall := 0;
  for i := 0 to MinPtrInt(10, length(queries)) - 1 do
  begin
    store.IndexMinRows := 20000;
    approx := istore.Search(queries[i], K);
    store.IndexMinRows := MaxInt; // force the exact scan
    t0 := GetTickCount64;
    exact := istore.Search(queries[i], K);
    if i = 0 then
      ConsoleWrite(FormatUtf8('exact scan: % ms per query', [GetTickCount64 - t0]));
    found := 0;
    for j := 0 to high(exact) do
      for h := 0 to high(approx) do
        if approx[h].DocId = exact[j].DocId then
        begin
          inc(found);
          break;
        end;
    recall := recall + found / MaxPtrInt(1, length(exact));
  end;
  ConsoleWrite(FormatUtf8('recall@10 vs exact (10 queries): %',
    [recall / MinPtrInt(10, length(queries))]));
end.
