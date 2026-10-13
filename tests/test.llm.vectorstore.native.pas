/// tests for mormot.ai.vectorstore.native (stage 2a: exact search)
// - keyed CRUD mirrors the vec0 test, plus what only the native store has:
//   persisted contract, input validation, atomic batches, filtered search,
//   several spaces in one file
// - Reference parity (the gate of step 2): the committed bge-m3 reference
//   (tests/data/embed-de/reference) is loaded and every query must return the
//   exact top-20 of the manifest - no lembed runtime needed
unit test.llm.vectorstore.native;

interface

{$I mormot.defines.inc}

uses
  sysutils,
  mormot.core.base,
  mormot.core.os,
  mormot.core.unicode,
  mormot.core.text,
  mormot.core.buffers,
  mormot.core.variants,
  mormot.core.test,
  mormot.db.raw.sqlite3,
  mormot.db.raw.sqlite3.static,
  mormot.ai.llm.types,
  mormot.ai.vectorstore,
  mormot.ai.vectorstore.native;

type
  TTestVectorStoreNative = class(TSynTestCase)
  protected
    fTempFiles: TFileNameDynArray;
    function TempDb: TFileName;
    procedure CleanUp; override;
  published
    procedure KeyedOps;
    procedure ContractIsPersistedAndEnforced;
    procedure InputValidation;
    procedure AddBatchIsAtomic;
    procedure AddBatchRollsBackMidTransaction;
    procedure ContractTextIsNotReinterpreted;
    procedure SearchWhereFiltersBeforeTopK;
    procedure SearchWhereRejectsUnsafeSql;
    procedure SpacesAreIsolated;
    procedure ReferenceParity;
  end;

implementation

function V4(a, b, c, d: single): TLlmEmbedding;
begin
  SetLength(result, 4);
  result[0] := a;
  result[1] := b;
  result[2] := c;
  result[3] := d;
end;

function C4: TVectorContract;
begin
  result := VectorContract('test/model-4d', '', 4);
end;

function HasKey(const aHits: TRagHitDynArray; const aKey: RawUtf8): boolean;
var
  i: PtrInt;
begin
  result := false;
  for i := 0 to high(aHits) do
    if aHits[i].Key = aKey then
      exit(true);
end;

{ TTestVectorStoreNative }

function TTestVectorStoreNative.TempDb: TFileName;
begin
  result := TemporaryFileName;
  DeleteFile(result); // SQLite creates it
  SetLength(fTempFiles, length(fTempFiles) + 1);
  fTempFiles[high(fTempFiles)] := result;
end;

procedure TTestVectorStoreNative.CleanUp;
var
  i: PtrInt;
begin
  for i := 0 to high(fTempFiles) do
  begin
    DeleteFile(fTempFiles[i]);
    DeleteFile(fTempFiles[i] + '-journal');
    DeleteFile(fTempFiles[i] + '-wal');
    DeleteFile(fTempFiles[i] + '-shm');
  end;
  fTempFiles := nil;
  inherited CleanUp;
end;

procedure TTestVectorStoreNative.KeyedOps;
var
  store: IVectorStore;
  hits: TRagHitDynArray;
begin
  store := TNativeVectorStore.Create(':memory:', 'kb', C4);
  store.Upsert('addr-1', 'mueller hamburg', V4(1, 0, 0, 0));
  store.Upsert('addr-2', 'schmidt berlin', V4(0, 1, 0, 0));
  store.Upsert('addr-3', 'meyer koeln', V4(0, 0, 1, 0));
  CheckEqual(store.Count, 3, 'three keyed rows');
  hits := store.Search(V4(1, 0, 0, 0), 3);
  CheckEqual(length(hits), 3, 'k hits');
  CheckEqual(hits[0].Key, 'addr-1', 'top hit key');
  CheckEqual(hits[0].Text, 'mueller hamburg', 'top hit text');
  CheckSame(hits[0].Distance, 0, 1e-6, 'identical direction = cosine distance 0');
  CheckSame(hits[1].Distance, 1, 1e-6, 'orthogonal = cosine distance 1');
  // replace in place: no new row, new vector + text visible
  store.Upsert('addr-1', 'mueller hamburg altona', V4(0, 0, 0, 1));
  CheckEqual(store.Count, 3, 'upsert replaces');
  hits := store.Search(V4(0, 0, 0, 1), 1);
  CheckEqual(hits[0].Key, 'addr-1', 'replaced vector moved');
  CheckEqual(hits[0].Text, 'mueller hamburg altona', 'replaced text');
  store.Delete('addr-2');
  CheckEqual(store.Count, 2, 'delete');
  store.Delete('does-not-exist');
  CheckEqual(store.Count, 2, 'absent delete is a no-op');
  Check(not HasKey(store.Search(V4(0, 1, 0, 0), 3), 'addr-2'), 'deleted is gone');
  // anonymous rows: empty key, DocId usable
  Check(store.Add('chunk', V4(0, 1, 0, 0)) > 0, 'Add returns an id');
  hits := store.Search(V4(0, 1, 0, 0), 1);
  CheckEqual(hits[0].Key, '', 'anonymous row has no key');
  CheckEqual(hits[0].Text, 'chunk', 'anonymous row text');
  // k larger than the store, non-positive k
  CheckEqual(length(store.Search(V4(1, 1, 1, 1), 50)), 3, 'k > count');
  CheckEqual(length(store.Search(V4(1, 1, 1, 1), 0)), 0, 'k = 0');
  CheckEqual(length(store.Search(V4(1, 1, 1, 1), -1)), 0, 'k < 0');
end;

procedure TTestVectorStoreNative.ContractIsPersistedAndEnforced;
var
  fn: TFileName;
  store: IVectorStore;
  other: TVectorContract;
begin
  fn := TempDb;
  store := TNativeVectorStore.Create(StringToUtf8(fn), 'pos', C4);
  store.Upsert('p1', 'eins', V4(1, 0, 0, 0));
  store := nil;
  // same contract: reopens and sees the data
  store := TNativeVectorStore.Create(StringToUtf8(fn), 'pos', C4);
  CheckEqual(store.Count, 1, 'data survives reopen');
  store := nil;
  // any field differing = another vector space -> refused
  other := C4;
  other.Model := 'test/other-model';
  try
    TNativeVectorStore.Create(StringToUtf8(fn), 'pos', other).Free;
    Check(false, 'other model refused');
  except
    on E: EVectorStore do
      Check(true, 'other model refused');
  end;
  other := C4;
  other.MaxTokens := 512;
  try
    TNativeVectorStore.Create(StringToUtf8(fn), 'pos', other).Free;
    Check(false, 'other token limit refused');
  except
    on E: EVectorStore do
      Check(true, 'other token limit refused');
  end;
  // invalid names and contracts
  try
    TNativeVectorStore.Create(':memory:', 'Pos; DROP', C4).Free;
    Check(false, 'bad space name');
  except
    on E: EVectorStore do
      Check(true, 'bad space name');
  end;
  try
    TNativeVectorStore.Create(':memory:', 'x', VectorContract('m', '', 0)).Free;
    Check(false, 'zero dim');
  except
    on E: EVectorStore do
      Check(true, 'zero dim');
  end;
  Check(IsValidVectorSpace('a1_b'), 'valid name');
  Check(not IsValidVectorSpace('1a'), 'must start with a letter');
  Check(not IsValidVectorSpace(''), 'empty');
end;

procedure TTestVectorStoreNative.InputValidation;
var
  store: IVectorStore;
  v: TLlmEmbedding;
  hits: TRagHitDynArray;
  nan: single;
begin
  store := TNativeVectorStore.Create(':memory:', 'iv', C4);
  try
    store.Upsert('a', 't', TLlmEmbedding.Create(1, 0, 0, 0, 0));
    Check(false, 'wrong dim');
  except
    on E: EVectorStore do
      Check(true, 'wrong dim');
  end;
  try
    store.Add('t', nil);
    Check(false, 'empty vector');
  except
    on E: EVectorStore do
      Check(true, 'empty vector');
  end;
  try
    store.Add('t', V4(0, 0, 0, 0));
    Check(false, 'zero vector');
  except
    on E: EVectorStore do
      Check(true, 'zero vector');
  end;
  // quiet NaN from its bit pattern: 0/0 would raise EInvalidOp (FPU masks)
  PCardinal(@nan)^ := $7FC00000;
  v := V4(1, 0, 0, 0);
  v[2] := nan;
  try
    store.Add('t', v);
    Check(false, 'NaN component');
  except
    on E: EVectorStore do
      Check(true, 'NaN component');
  end;
  try
    store.Upsert('', 't', V4(1, 0, 0, 0));
    Check(false, 'empty upsert id');
  except
    on E: EVectorStore do
      Check(true, 'empty upsert id');
  end;
  CheckEqual(store.Count, 0, 'nothing written by refused input');
  // non-unit input is normalised: same direction = distance 0
  store.Upsert('a', 'long', V4(3, 4, 0, 0));
  hits := store.Search(V4(0.6, 0.8, 0, 0), 1);
  CheckSame(hits[0].Distance, 0, 1e-6, 'stored vector was normalised');
  // extreme but finite magnitudes: squaring in single precision would overflow
  // (1e20) or underflow to a false zero vector (1e-30)
  store.Upsert('big', 'big', V4(1e20, 0, 0, 0));
  store.Upsert('tiny', 'tiny', V4(0, 0, 1e-30, 0));
  hits := store.Search(V4(1, 0, 0, 0), 1);
  CheckEqual(hits[0].Key, 'big', 'huge components are normalised, not overflowed');
  hits := store.Search(V4(0, 0, 1, 0), 1);
  CheckEqual(hits[0].Key, 'tiny', 'tiny components are normalised, not zeroed');
  CheckSame(hits[0].Distance, 0, 1e-6, 'tiny vector keeps its direction');
end;

procedure TTestVectorStoreNative.AddBatchIsAtomic;
var
  store: IVectorStore;
  texts: TRawUtf8DynArray;
  vecs: TLlmEmbeddingDynArray;
begin
  store := TNativeVectorStore.Create(':memory:', 'ab', C4);
  texts := TRawUtf8DynArray.Create('a', 'b', 'c');
  vecs := TLlmEmbeddingDynArray.Create(V4(1, 0, 0, 0), V4(0, 1, 0, 0), nil);
  try
    store.AddBatch(texts, vecs);
    Check(false, 'bad vector in batch');
  except
    on E: EVectorStore do
      Check(true, 'bad vector in batch');
  end;
  CheckEqual(store.Count, 0, 'failed batch wrote nothing');
  vecs[2] := V4(0, 0, 1, 0);
  CheckEqual(store.AddBatch(texts, vecs), 3, 'batch stored');
  CheckEqual(store.Count, 3, 'all rows');
  try
    store.AddBatch(texts, Copy(vecs, 0, 2));
    Check(false, 'length mismatch');
  except
    on E: EVectorStore do
      Check(true, 'length mismatch');
  end;
end;

procedure TTestVectorStoreNative.AddBatchRollsBackMidTransaction;
var
  fn: TFileName;
  app: TSqlDataBase;
  store: IVectorStore;
  texts: TRawUtf8DynArray;
  vecs: TLlmEmbeddingDynArray;
begin
  // every vector is valid, so the batch passes validation and opens the
  // transaction - a trigger then aborts the SECOND insert: the first one must
  // be rolled back with it
  fn := TempDb;
  store := TNativeVectorStore.Create(StringToUtf8(fn), 'rb', C4);
  app := TSqlDataBase.Create(fn, '');
  try
    app.Execute('CREATE TRIGGER rb_boom BEFORE INSERT ON rb_items ' +
      'WHEN NEW.content = ''boom'' BEGIN SELECT RAISE(ABORT, ''boom''); END;');
  finally
    app.Free;
  end;
  texts := TRawUtf8DynArray.Create('ok', 'boom', 'never');
  vecs := TLlmEmbeddingDynArray.Create(V4(1, 0, 0, 0), V4(0, 1, 0, 0),
    V4(0, 0, 1, 0));
  try
    store.AddBatch(texts, vecs);
    Check(false, 'trigger abort must surface');
  except
    on E: ESynException do
      Check(true, 'trigger abort surfaced');
  end;
  CheckEqual(store.Count, 0, 'first row rolled back with the failed batch');
  // and the connection is usable again (no transaction left open)
  texts := TRawUtf8DynArray.Create('ok');
  vecs := TLlmEmbeddingDynArray.Create(V4(1, 0, 0, 0));
  CheckEqual(store.AddBatch(texts, vecs), 1, 'next batch works');
  CheckEqual(store.Count, 1, 'and is stored');
end;

procedure TTestVectorStoreNative.ContractTextIsNotReinterpreted;
var
  fn: TFileName;
  c: TVectorContract;
  store: IVectorStore;
begin
  // '1' and ' 1' are different prefixes (different token ids, different
  // vectors) - they must not collapse into the same persisted contract
  fn := TempDb;
  c := C4;
  c.QueryPrefix := '1';
  store := TNativeVectorStore.Create(StringToUtf8(fn), 'ct', c);
  store := nil;
  c.QueryPrefix := ' 1';
  try
    TNativeVectorStore.Create(StringToUtf8(fn), 'ct', c).Free;
    Check(false, 'prefix " 1" is another contract than "1"');
  except
    on E: EVectorStore do
      Check(true, 'prefix " 1" is another contract than "1"');
  end;
  Check(PosEx('"query_prefix":"1"', VectorContractToJson(VectorContract('m', '', 4,
    0, '1'))) > 0, 'prefix persisted as a string');
  // a hand-built record with an upper-case SHA equals the factory-built one
  c := C4;
  c.ModelSha256 := 'ABCDEF';
  store := TNativeVectorStore.Create(StringToUtf8(fn), 'sha', c);
  store := nil;
  store := TNativeVectorStore.Create(StringToUtf8(fn), 'sha',
    VectorContract('test/model-4d', 'abcdef', 4));
  CheckEqual(store.Count, 0, 'SHA case does not create another contract');
  CheckEqual((store as TNativeVectorStore).Contract.ModelSha256, 'abcdef',
    'contract keeps the SHA lower-case');
  store := nil;
  store := TNativeVectorStore.Create(StringToUtf8(fn), 'sha', c); // upper-case
  CheckEqual((store as TNativeVectorStore).Contract.ModelSha256, 'abcdef',
    'constructor normalises a hand-built upper-case SHA');
end;

procedure TTestVectorStoreNative.SearchWhereFiltersBeforeTopK;
var
  fn: TFileName;
  app: TSqlDataBase;
  store: IFilteredVectorStore;
  hits: TRagHitDynArray;
begin
  fn := TempDb;
  store := TNativeVectorStore.Create(StringToUtf8(fn), 'pos', C4);
  // the application owns its domain table in the SAME file
  app := TSqlDataBase.Create(fn, '');
  try
    app.Execute('CREATE TABLE positions(id TEXT PRIMARY KEY, kind TEXT, tenant INT);');
    app.Execute('INSERT INTO positions VALUES (''p1'', ''normal'', 1), ' +
      '(''p2'', ''text'', 1), (''p3'', ''normal'', 2), (''p4'', ''normal'', 1);');
  finally
    app.Free;
  end;
  // p2 (a text position) is the closest to the query, p4 the farthest
  store.Upsert('p1', 'Kupferrohr verlegen', V4(1, 1, 0, 0));
  store.Upsert('p2', 'Hinweis Kupferrohr', V4(1, 0, 0, 0));
  store.Upsert('p3', 'Kupferrohr anderer Mandant', V4(1, 0.1, 0, 0));
  store.Upsert('p4', 'Fliesen', V4(0, 0, 1, 0));
  hits := store.Search(V4(1, 0, 0, 0), 1);
  CheckEqual(hits[0].Key, 'p2', 'unfiltered top hit is the text position');
  // filter BEFORE top-k: with k = 1 a post-filter would return nothing
  hits := store.SearchWhere(V4(1, 0, 0, 0), 1,
    'SELECT id FROM positions WHERE kind = ? AND tenant = ?', ['normal', 1]);
  CheckEqual(length(hits), 1, 'one hit after filtering');
  CheckEqual(hits[0].Key, 'p1', 'best allowed hit');
  hits := store.SearchWhere(V4(1, 0, 0, 0), 10,
    'SELECT id FROM positions WHERE tenant = ?', [1]);
  CheckEqual(length(hits), 3, 'only tenant 1');
  Check(not HasKey(hits, 'p3'), 'other tenant excluded');
  hits := store.SearchWhere(V4(1, 0, 0, 0), 10,
    'SELECT id FROM positions WHERE tenant = ?', [99]);
  CheckEqual(length(hits), 0, 'empty filter = no hits');
  // a trailing comment must not swallow the closing parenthesis
  hits := store.SearchWhere(V4(1, 0, 0, 0), 10,
    'SELECT id FROM positions WHERE tenant = ? -- only tenant 1', [1]);
  CheckEqual(length(hits), 3, 'trailing comment is harmless');
end;

procedure TTestVectorStoreNative.SearchWhereRejectsUnsafeSql;
var
  store: IFilteredVectorStore;

  procedure Refused(const aSql: RawUtf8; const aWhat: string);
  begin
    try
      store.SearchWhere(V4(1, 0, 0, 0), 5, aSql, []);
      Check(false, aWhat);
    except
      on E: EVectorStore do
        Check(true, aWhat);
    end;
  end;

begin
  store := TNativeVectorStore.Create(':memory:', 'sq', C4);
  store.Upsert('a', 't', V4(1, 0, 0, 0));
  store.Upsert('b', 'other', V4(0, 1, 0, 0));
  // a second statement after ';' (TSqlRequest.Prepare would silently drop it)
  Refused('SELECT ''a''; DELETE FROM sq_items', 'second statement refused');
  // closing IN (...) from inside the key query would widen the filter to all
  // rows - standalone it does not even compile
  Refused('SELECT ''a'') OR 1=1 --', 'parenthesis escape refused');
  Refused('DELETE FROM sq_items RETURNING ext_id', 'writing key query refused');
  Refused('SELECT ''a'', 1', 'two columns refused');
  Refused('SELECT ?', 'param count must match');
  Refused('  ', 'empty key query refused');
  Refused('SELEKT', 'non-compiling key query refused');
  Refused('SELECT ''a'';', 'trailing semicolon refused');
  Refused('SELECT ''a''' + #0 + ') OR 1=1 --', 'NUL byte refused');
  // statements that act while being COMPILED must be denied at compile time
  Refused('PRAGMA busy_timeout = 0', 'pragma refused');
  // valid SQL (one column) - only the authorizer can refuse it, and only if it
  // is still active at Step time, when SQLite prepares the inner PRAGMA
  Refused('SELECT ''a'' FROM pragma_table_info(''sq_items'')',
    'pragma table-valued function refused');
  Refused('ATTACH '':memory:'' AS x', 'attach refused');
  // a legitimate CTE still works (WITH / VALUES are read-only)
  CheckEqual(length(store.SearchWhere(V4(1, 0, 0, 0), 5,
    'WITH k(id) AS (VALUES (''a'')) SELECT id FROM k', [])), 1, 'CTE key query');
  CheckEqual(store.Count, 2, 'nothing was deleted');
end;

procedure TTestVectorStoreNative.SpacesAreIsolated;
var
  fn: TFileName;
  a, b: IVectorStore;
  c: TVectorContract;
begin
  fn := TempDb;
  a := TNativeVectorStore.Create(StringToUtf8(fn), 'articles', C4);
  // a second space with ANOTHER contract in the same file (re-index scenario)
  c := VectorContract('test/model-3d', '', 3);
  b := TNativeVectorStore.Create(StringToUtf8(fn), 'articles_v2', c);
  a.Upsert('x', 'alt', V4(1, 0, 0, 0));
  CheckEqual(a.Count, 1, 'space a');
  CheckEqual(b.Count, 0, 'space b untouched');
  b.Upsert('x', 'neu', TLlmEmbedding.Create(0, 1, 0));
  CheckEqual(b.Search(TLlmEmbedding.Create(0, 1, 0), 1)[0].Text, 'neu', 'space b');
  CheckEqual(a.Search(V4(1, 0, 0, 0), 1)[0].Text, 'alt', 'space a unchanged');
end;

/// find tests/data/embed-de/reference from the executable upwards
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
      exit(cand);
    dir := ExtractFileDir(dir);
  end;
end;

procedure TTestVectorStoreNative.ReferenceParity;
var
  refdir: TFileName;
  manifest, queries, top: TDocVariantData;
  corpusLines: TRawUtf8DynArray;
  ids: TRawUtf8DynArray;
  docs, qblob: RawByteString;
  store: IVectorStore;
  vec, q: TLlmEmbedding;
  hits: TRagHitDynArray;
  doc: TDocVariantData;
  dim, n, i, j, mism: integer;
  qid: RawUtf8;
  c: TVectorContract;
  expect, hit: PDocVariantData;
  score: double;
begin
  refdir := FindReferenceDir;
  if CheckFailed(refdir <> '', 'reference data tests/data/embed-de found') then
    exit;
  // JSON_FAST_FLOAT: plain JSON_FAST reads floats as currency (4 decimals)
  manifest.InitJsonFromFile(refdir + 'manifest.json', JSON_FAST_FLOAT);
  dim := manifest.O['contract']^.I['dim'];
  c := VectorContract(manifest.O['contract']^.U['model'],
    manifest.O['contract']^.U['model_sha256'], dim,
    manifest.O['contract']^.I['max_tokens']);
  // corpus ids in file order = row order of docs.f32
  CSVToRawUtf8DynArray(pointer(StringReplaceAll(
    RawUtf8FromFile(ExtractFileDir(ExcludeTrailingPathDelimiter(refdir)) +
      PathDelim + 'corpus.jsonl'), #13, '')), corpusLines, #10);
  ids := nil;
  for i := 0 to high(corpusLines) do
    if TrimU(corpusLines[i]) <> '' then
    begin
      doc.InitJson(corpusLines[i], JSON_FAST);
      SetLength(ids, length(ids) + 1);
      ids[high(ids)] := doc.U['id'];
      doc.Clear;
    end;
  docs := StringFromFile(refdir + 'docs.f32');
  qblob := StringFromFile(refdir + 'queries.f32');
  CheckEqual(length(docs), length(ids) * dim * SizeOf(single), 'docs.f32 size');
  store := TNativeVectorStore.Create(':memory:', 'ref', c);
  SetLength(vec, dim);
  for i := 0 to high(ids) do
  begin
    MoveFast(PByteArray(docs)^[i * dim * SizeOf(single)], vec[0], dim * SizeOf(single));
    store.Upsert(ids[i], ids[i], vec);
  end;
  CheckEqual(store.Count, length(ids), 'all reference docs stored');
  queries.InitCopy(variant(manifest.O['queries']^), JSON_FAST_FLOAT);
  CheckEqual(length(qblob), queries.Count * dim * SizeOf(single), 'queries.f32 size');
  SetLength(q, dim);
  mism := 0;
  for i := 0 to queries.Count - 1 do
  begin
    qid := queries.Names[i];
    MoveFast(PByteArray(qblob)^[i * dim * SizeOf(single)], q[0], dim * SizeOf(single));
    expect := queries.O[qid];
    top.InitCopy(variant(expect^.A['top']^), JSON_FAST_FLOAT);
    hits := store.Search(q, top.Count);
    CheckEqual(length(hits), top.Count, 'top-N length');
    n := MinPtrInt(length(hits), top.Count);
    for j := 0 to n - 1 do
    begin
      // same document at the same rank, same score (float32 rows, double sum)
      hit := _Safe(top.Values[j]);
      if not VariantToDouble(hit^.Values[1], score) or
         (hits[j].Key <> VariantToUtf8(hit^.Values[0])) or
         (abs((1 - hits[j].Distance) - score) > 1e-5) then
        inc(mism);
    end;
    top.Clear;
  end;
  CheckEqual(mism, 0, FormatUtf8('% queries: exact top-N identical to the ' +
    'reference (rank + score)', [queries.Count]));
end;

end.
