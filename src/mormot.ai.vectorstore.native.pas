/// LandrixAI - native IVectorStore: plain SQLite tables + search in Pascal
// - part of the mormot.ai.* extension (LandrixAI)
// - replaces sqlite-vec (vec0) without any C extension: vectors are stored as a
//   float32 BLOB column in an ordinary table (the single source of truth, one
//   atomic row with text and key), and the nearest-neighbour search runs in
//   Pascal over that column. Because the data is a plain table, the search
//   engine stays replaceable (quantised in-memory index, IVF, sqlite-vector)
//   without migrating data - see docs/native-embeddings.md
// - this unit is stage 2a: EXACT search, streamed from SQLite (no in-memory
//   index yet) - the correctness reference for the faster stages that follow
// - every store is bound to a TVectorContract (model, dimension, metric ...),
//   persisted next to its table; opening it with a different contract fails, so
//   vectors of two models can never be mixed silently. Several "spaces" (one per
//   contract or purpose) can live side by side in one database file - that is how
//   a re-index runs in the background while the old space keeps serving
// - metric: cosine. The store normalises every vector to unit length on write
//   and on query, so cosine similarity is the dot product and TRagHit.Distance is
//   the cosine distance 1 - dot
// - CONCURRENCY: the store opens its own connection. When the application writes
//   domain tables of the SAME file through another connection, the file must use
//   WAL (PRAGMA journal_mode=WAL, set by the owner - this unit never changes it):
//   in rollback-journal mode a running scan holds a SHARED lock that makes every
//   application COMMIT fail with SQLITE_BUSY (mORMot sets no busy timeout on the
//   application side), and an application read blocks the store's writes.
//   JournalMode reports the mode found at open
// - clean-room; target license MPL/GPL/LGPL
unit mormot.ai.vectorstore.native;

interface

{$I mormot.defines.inc}

uses
  sysutils,
  mormot.core.base,
  mormot.core.os,
  mormot.core.unicode,
  mormot.core.text,
  mormot.core.variants,
  mormot.lib.static,      // SetFpuFlags/ResetFpuFlags
  mormot.db.raw.sqlite3,
  mormot.db.raw.sqlite3.static,
  mormot.ai.llm.types,
  mormot.ai.sqlite,
  mormot.ai.vectorstore;

type
  /// raised on contract violations and invalid input of a native vector store
  EVectorStore = class(ESynException);

  /// what a vector in a store MEANS - two stores (or a store and an embedder)
  // are compatible only if their contracts are equal field by field
  // - metric (cosine) and normalisation (L2) are fixed by this implementation
  // and part of the persisted contract text, so a future change is detected too
  TVectorContract = record
    /// the embedding model identity, e.g. 'BAAI/bge-m3'
    Model: RawUtf8;
    /// SHA-256 of the exact model file, '' when not pinned (e.g. a hosted API)
    ModelSha256: RawUtf8;
    /// the vector dimension
    Dim: integer;
    /// token limit the texts were cut to before embedding (0 = not recorded)
    MaxTokens: integer;
    /// instruction prefixes applied before embedding ('' for bge-m3)
    QueryPrefix: RawUtf8;
    DocumentPrefix: RawUtf8;
  end;

  /// a native, filterable vector store over plain SQLite tables
  // - owns its connection; aDbPath may be the application database itself, so
  //   SearchWhere filters can join domain tables of that file
  // - does NOT change the journal mode of the file - that stays the owner's call
  // - thread-safe: every access holds the connection lock (like TVec0Store)
  TNativeVectorStore = class(TInterfacedObject, IVectorStore, IFilteredVectorStore)
  protected
    fDB: TSqlDataBase;
    fSpace: RawUtf8;
    fItems: RawUtf8;
    fMeta: RawUtf8;
    fContract: TVectorContract;
    fBlobSize: integer;
    fJournalMode: RawUtf8;
    /// validate (dimension, finite, non-zero) and scale to unit length
    function UnitVector(const aVec: TLlmEmbedding; const aWhat: RawUtf8): TLlmEmbedding;
    function RowIdOfKey(const aId: RawUtf8): Int64;
    procedure OpenSpace;
    /// the key query of SearchWhere must be one read-only, one-column statement
    procedure CheckKeySql(const aKeySql: RawUtf8);
    /// the exact scan shared by Search and SearchWhere
    function Scan(const aQuery: TLlmEmbedding; aTopK: integer;
      const aKeySql: RawUtf8; const aParams: array of const): TRagHitDynArray;
  public
    /// open or create the vector space aSpace in aDbPath (':memory:' allowed)
    // - aSpace: 1..32 chars [a-z0-9_], starting with a letter; it prefixes the
    //   two tables <space>_items and <space>_meta
    // - raises EVectorStore if the space exists with a different contract
    constructor Create(const aDbPath, aSpace: RawUtf8;
      const aContract: TVectorContract); reintroduce;
    destructor Destroy; override;
    // IVectorStore
    function Add(const aText: RawUtf8; const aVector: TLlmEmbedding): Int64;
    function AddBatch(const aTexts: TRawUtf8DynArray;
      const aVectors: TLlmEmbeddingDynArray): integer;
    procedure Upsert(const aId, aText: RawUtf8; const aVector: TLlmEmbedding);
    procedure Delete(const aId: RawUtf8);
    function Search(const aQuery: TLlmEmbedding; aTopK: integer): TRagHitDynArray;
    function Count: Int64;
    // IFilteredVectorStore
    function SearchWhere(const aQuery: TLlmEmbedding; aTopK: integer;
      const aKeySql: RawUtf8; const aParams: array of const): TRagHitDynArray;
    /// the contract this store was opened with (= the persisted one)
    property Contract: TVectorContract read fContract;
    /// the space name (table prefix)
    property Space: RawUtf8 read fSpace;
    /// the journal mode of the file at open, lower-case ('wal', 'delete',
    // 'memory' ...) - anything but 'wal' (or 'memory') is unsafe when another
    // connection writes the same file concurrently, see the unit header
    property JournalMode: RawUtf8 read fJournalMode;
  end;

/// build a contract value
function VectorContract(const aModel, aModelSha256: RawUtf8; aDim: integer;
  aMaxTokens: integer = 0; const aQueryPrefix: RawUtf8 = '';
  const aDocumentPrefix: RawUtf8 = ''): TVectorContract;

/// the canonical persisted form of a contract (stable field order)
function VectorContractToJson(const aContract: TVectorContract): RawUtf8;

/// true if aSpace is a valid space name (1..32 of [a-z0-9_], first a letter)
function IsValidVectorSpace(const aSpace: RawUtf8): boolean;


implementation

const
  /// version of the on-disk layout and of the contract text
  NATIVE_FORMAT = 1;
  /// largest accepted dimension - far above any real embedding model, but keeps
  // the BLOB size check and the per-row arithmetic away from overflow
  NATIVE_MAX_DIM = 65536;
  /// IEEE-754 single: all exponent bits set = NaN or +-Inf
  SINGLE_EXP_MASK = $7F800000;
  /// IEEE-754 double: the same test for a computed score
  DOUBLE_EXP_MASK = Int64($7FF0000000000000);

function VectorContract(const aModel, aModelSha256: RawUtf8; aDim: integer;
  aMaxTokens: integer; const aQueryPrefix, aDocumentPrefix: RawUtf8): TVectorContract;
begin
  result.Model := aModel;
  result.ModelSha256 := LowerCase(aModelSha256);
  result.Dim := aDim;
  result.MaxTokens := aMaxTokens;
  result.QueryPrefix := aQueryPrefix;
  result.DocumentPrefix := aDocumentPrefix;
end;

function VectorContractToJson(const aContract: TVectorContract): RawUtf8;
var
  doc: TDocVariantData;
begin
  doc.InitFast;
  doc.AddValue('format', NATIVE_FORMAT);
  // AddValueText, never AddValueFromText: the latter parses '1' or 'true' into a
  // number/boolean, so two different prefixes could serialise identically
  doc.AddValueText('model', aContract.Model);
  doc.AddValueText('model_sha256', LowerCase(aContract.ModelSha256));
  doc.AddValue('dim', aContract.Dim);
  doc.AddValue('metric', 'cosine');
  doc.AddValue('normalize', 'l2');
  doc.AddValue('max_tokens', aContract.MaxTokens);
  doc.AddValueText('query_prefix', aContract.QueryPrefix);
  doc.AddValueText('document_prefix', aContract.DocumentPrefix);
  result := doc.ToJson;
end;

function IsValidVectorSpace(const aSpace: RawUtf8): boolean;
var
  i: PtrInt;
begin
  result := false;
  if (aSpace = '') or (length(aSpace) > 32) or
     not (aSpace[1] in ['a'..'z']) then
    exit;
  for i := 2 to length(aSpace) do
    if not (aSpace[i] in ['a'..'z', '0'..'9', '_']) then
      exit;
  result := true;
end;

/// cosine similarity of a unit query against a stored unit float32 row
function DotRow(q: PSingleArray; row: PSingleArray; n: PtrInt): double;
var
  i: PtrInt;
begin
  result := 0;
  for i := 0 to n - 1 do
    result := result + q^[i] * row^[i];
end;


{ top-k collector: kept sorted best-first, k is small (tens), so insertion
  into a fixed array beats any heap here }

type
  TTopK = record
    Score: array of double;
    RowId: array of Int64;
    Count, Capacity: integer;
    procedure Init(aCapacity: integer);
    /// a ranks before b: higher score, then lower rowid (deterministic ties)
    function Before(aScore: double; aRowId: Int64; aIndex: integer): boolean;
    procedure Offer(aScore: double; aRowId: Int64);
  end;

procedure TTopK.Init(aCapacity: integer);
begin
  Capacity := aCapacity;
  Count := 0;
  // grow on demand: a caller asking for k = MaxInt must not allocate 16 GB
  // up front when the store holds a few hundred rows
  SetLength(Score, MinPtrInt(aCapacity, 256));
  SetLength(RowId, length(Score));
end;

function TTopK.Before(aScore: double; aRowId: Int64; aIndex: integer): boolean;
begin
  result := (aScore > Score[aIndex]) or
            ((aScore = Score[aIndex]) and (aRowId < RowId[aIndex]));
end;

procedure TTopK.Offer(aScore: double; aRowId: Int64);
var
  i: integer;
begin
  if Count = Capacity then
    if not Before(aScore, aRowId, Count - 1) then
      exit // not better than the current worst
    else
      dec(Count); // drop the worst, make room
  if Count = length(Score) then
  begin
    SetLength(Score, MinPtrInt(Capacity, length(Score) * 2));
    SetLength(RowId, length(Score));
  end;
  i := Count;
  while (i > 0) and Before(aScore, aRowId, i - 1) do
  begin
    Score[i] := Score[i - 1];
    RowId[i] := RowId[i - 1];
    dec(i);
  end;
  Score[i] := aScore;
  RowId[i] := aRowId;
  inc(Count);
end;


{ TNativeVectorStore }

constructor TNativeVectorStore.Create(const aDbPath, aSpace: RawUtf8;
  const aContract: TVectorContract);
begin
  inherited Create;
  if not IsValidVectorSpace(aSpace) then
    EVectorStore.RaiseUtf8('%: invalid space name "%" (1..32 of [a-z0-9_], ' +
      'starting with a letter)', [self, aSpace]);
  if (aContract.Model = '') or (aContract.Dim <= 0) or
     (aContract.Dim > NATIVE_MAX_DIM) then
    EVectorStore.RaiseUtf8('%: contract needs a model and a dimension in 1..%',
      [self, NATIVE_MAX_DIM]);
  fSpace := aSpace;
  fItems := aSpace + '_items';
  fMeta := aSpace + '_meta';
  fContract := aContract;
  // a record assigned directly bypasses VectorContract(): normalise here too
  fContract.ModelSha256 := LowerCase(aContract.ModelSha256);
  fBlobSize := aContract.Dim * SizeOf(single);
  fDB := TSqlDataBase.Create(Utf8ToString(aDbPath), '');
  // mORMot sets no busy timeout: a lock held by another connection (the
  // application writing its domain tables) would fail at once with SQLITE_BUSY
  fDB.BusyTimeout := 5000;
  fJournalMode := LowerCase(fDB.ExecuteNoExceptionUtf8('PRAGMA journal_mode;'));
  OpenSpace;
end;

destructor TNativeVectorStore.Destroy;
begin
  fDB.Free;
  inherited Destroy;
end;

procedure TNativeVectorStore.OpenSpace;
var
  r: TSqlRequest;
  stored, expected: RawUtf8;
  hasItems: boolean;
begin
  expected := VectorContractToJson(fContract);
  fDB.Lock;
  try
    // IMMEDIATE: take the write lock up front - a deferred transaction that
    // reads first and upgrades later can fail with SQLITE_BUSY at once (SQLite
    // skips the busy handler to avoid the upgrade deadlock), and two processes
    // creating the same space must not both see "no contract yet"
    fDB.TransactionBegin(tbImmediate);
    try
      fDB.Execute(FormatUtf8('CREATE TABLE IF NOT EXISTS %(' +
        'key TEXT PRIMARY KEY, value TEXT NOT NULL);', [fMeta]));
      stored := '';
      r.Prepare(fDB.DB, FormatUtf8('SELECT value FROM % WHERE key = ''contract'';',
        [fMeta]));
      try
        if r.Step = SQLITE_ROW then
          r.FieldUtf8(0, stored);
      finally
        r.Close;
      end;
      if stored = '' then
      begin
        // a contract-less items table means someone created the space by hand
        // (or a half-written older layout): never adopt vectors of unknown meaning
        r.Prepare(fDB.DB,
          'SELECT 1 FROM sqlite_master WHERE type = ''table'' AND name = ?;');
        try
          r.Bind(1, fItems);
          hasItems := r.Step = SQLITE_ROW;
        finally
          r.Close;
        end;
        if hasItems then
          EVectorStore.RaiseUtf8('%: table % exists without a contract - refusing ' +
            'to adopt vectors of unknown meaning', [self, fItems]);
        r.Prepare(fDB.DB, FormatUtf8(
          'INSERT INTO %(key, value) VALUES (''contract'', ?);', [fMeta]));
        try
          r.Bind(1, expected);
          r.Step;
        finally
          r.Close;
        end;
      end
      else if stored <> expected then
        EVectorStore.RaiseUtf8('%: space "%" was built under another contract ' +
          '(stored %, requested %) - re-index into a new space', [self, fSpace,
          stored, expected]);
      // ext_id: optional stable external key (Upsert/Delete rows), NULL for
      // anonymous Add rows; the partial UNIQUE index maps an id to at most one row
      // the CHECK makes a wrong-sized vector impossible at the storage level too
      fDB.Execute(FormatUtf8('CREATE TABLE IF NOT EXISTS %(' +
        'id INTEGER PRIMARY KEY AUTOINCREMENT, ext_id TEXT, ' +
        'content TEXT NOT NULL, embedding BLOB NOT NULL ' +
        'CHECK (typeof(embedding) = ''blob'' AND length(embedding) = %));',
        [fItems, fBlobSize]));
      fDB.Execute(FormatUtf8('CREATE UNIQUE INDEX IF NOT EXISTS %_ext_id ' +
        'ON %(ext_id) WHERE ext_id IS NOT NULL;', [fItems, fItems]));
      fDB.Commit;
    except
      RollbackIfOpen(fDB);
      raise;
    end;
  finally
    fDB.UnLock;
  end;
end;

function TNativeVectorStore.UnitVector(const aVec: TLlmEmbedding;
  const aWhat: RawUtf8): TLlmEmbedding;
var
  i: PtrInt;
  sum, scale, x: double;
begin
  if length(aVec) <> fContract.Dim then
    EVectorStore.RaiseUtf8('%: % has % dims, contract says % ' +
      '(empty = failed embedding?)', [self, aWhat, length(aVec), fContract.Dim]);
  sum := 0;
  for i := 0 to high(aVec) do
  begin
    // test the bits, not the value: with FPU exceptions unmasked (the default
    // here) even comparing a NaN raises EInvalidOp instead of our error
    if PCardinal(@aVec[i])^ and SINGLE_EXP_MASK = SINGLE_EXP_MASK then
      EVectorStore.RaiseUtf8('%: % has a non-finite component at %',
        [self, aWhat, i]);
    // widen BEFORE squaring: FPC multiplies two singles in single precision,
    // so 1e20 would overflow and 1e-30 underflow to a false "zero vector"
    x := aVec[i];
    sum := sum + x * x;
  end;
  if sum <= 0 then
    EVectorStore.RaiseUtf8('%: % is the zero vector (no direction)', [self, aWhat]);
  scale := 1 / sqrt(sum);
  SetLength(result, length(aVec));
  for i := 0 to high(aVec) do
    result[i] := aVec[i] * scale;
end;

function TNativeVectorStore.RowIdOfKey(const aId: RawUtf8): Int64;
var
  r: TSqlRequest;
begin
  result := 0; // 0 = not found (AUTOINCREMENT ids start at 1)
  fDB.Lock; // reentrant: Upsert/Delete already hold it
  try
    r.Prepare(fDB.DB, FormatUtf8('SELECT id FROM % WHERE ext_id = ?;', [fItems]));
    try
      r.Bind(1, aId);
      if r.Step = SQLITE_ROW then
        result := r.FieldInt(0);
    finally
      r.Close;
    end;
  finally
    fDB.UnLock;
  end;
end;

function TNativeVectorStore.Add(const aText: RawUtf8;
  const aVector: TLlmEmbedding): Int64;
var
  r: TSqlRequest;
  blob: RawByteString;
begin
  blob := VectorToBlob(UnitVector(aVector, 'vector'));
  fDB.Lock;
  try
    r.Prepare(fDB.DB, FormatUtf8(
      'INSERT INTO %(content, embedding) VALUES (?, ?);', [fItems]));
    try
      r.Bind(1, aText);
      r.BindBlob(2, blob);
      r.Step;
    finally
      r.Close;
    end;
    // read under the same lock: LastInsertRowID is per connection
    result := fDB.LastInsertRowID;
  finally
    fDB.UnLock;
  end;
end;

function TNativeVectorStore.AddBatch(const aTexts: TRawUtf8DynArray;
  const aVectors: TLlmEmbeddingDynArray): integer;
var
  i: PtrInt;
  r: TSqlRequest;
  blobs: TRawByteStringDynArray;
begin
  result := 0;
  if length(aTexts) <> length(aVectors) then
    EVectorStore.RaiseUtf8('%.AddBatch: % texts but % vectors',
      [self, length(aTexts), length(aVectors)]);
  if aTexts = nil then
    exit;
  // validate + normalise EVERYTHING before the transaction opens: one bad vector
  // fails the whole batch without writing anything
  SetLength(blobs, length(aVectors));
  for i := 0 to high(aVectors) do
    blobs[i] := VectorToBlob(UnitVector(aVectors[i],
      FormatUtf8('vector #%', [i])));
  fDB.Lock;
  try
    fDB.TransactionBegin(tbImmediate); // see OpenSpace
    try
      r.Prepare(fDB.DB, FormatUtf8(
        'INSERT INTO %(content, embedding) VALUES (?, ?);', [fItems]));
      try
        for i := 0 to high(aTexts) do
        begin
          r.Bind(1, aTexts[i]);
          r.BindBlob(2, blobs[i]);
          r.Step;
          r.Reset;
        end;
      finally
        r.Close;
      end;
      fDB.Commit;
      result := length(aTexts);
    except
      RollbackIfOpen(fDB);
      raise;
    end;
  finally
    fDB.UnLock;
  end;
end;

procedure TNativeVectorStore.Upsert(const aId, aText: RawUtf8;
  const aVector: TLlmEmbedding);
var
  r: TSqlRequest;
  blob: RawByteString;
  rowid: Int64;
begin
  if aId = '' then
    EVectorStore.RaiseUtf8('%.Upsert: empty id', [self]);
  blob := VectorToBlob(UnitVector(aVector, 'vector'));
  fDB.Lock;
  try
    // IMMEDIATE: RowIdOfKey reads before the write - see OpenSpace
    fDB.TransactionBegin(tbImmediate);
    try
      rowid := RowIdOfKey(aId);
      if rowid <> 0 then
      begin
        // in place: one row holds text and vector, so the replace is atomic and
        // the id -> row mapping stays stable
        r.Prepare(fDB.DB, FormatUtf8(
          'UPDATE % SET content = ?, embedding = ? WHERE id = ?;', [fItems]));
        try
          r.Bind(1, aText);
          r.BindBlob(2, blob);
          r.Bind(3, rowid);
          r.Step;
        finally
          r.Close;
        end;
      end
      else
      begin
        r.Prepare(fDB.DB, FormatUtf8(
          'INSERT INTO %(ext_id, content, embedding) VALUES (?, ?, ?);', [fItems]));
        try
          r.Bind(1, aId);
          r.Bind(2, aText);
          r.BindBlob(3, blob);
          r.Step;
        finally
          r.Close;
        end;
      end;
      fDB.Commit;
    except
      RollbackIfOpen(fDB);
      raise;
    end;
  finally
    fDB.UnLock;
  end;
end;

procedure TNativeVectorStore.Delete(const aId: RawUtf8);
var
  r: TSqlRequest;
begin
  fDB.Lock;
  try
    // one statement on one row: atomic by itself, absent id = no-op
    r.Prepare(fDB.DB, FormatUtf8('DELETE FROM % WHERE ext_id = ?;', [fItems]));
    try
      r.Bind(1, aId);
      r.Step;
    finally
      r.Close;
    end;
  finally
    fDB.UnLock;
  end;
end;

function TNativeVectorStore.Count: Int64;
var
  r: TSqlRequest;
begin
  result := 0;
  fDB.Lock;
  try
    r.Prepare(fDB.DB, FormatUtf8('SELECT COUNT(*) FROM %;', [fItems]));
    try
      if r.Step = SQLITE_ROW then
        result := r.FieldInt(0);
    finally
      r.Close;
    end;
  finally
    fDB.UnLock;
  end;
end;

function TNativeVectorStore.Search(const aQuery: TLlmEmbedding;
  aTopK: integer): TRagHitDynArray;
begin
  result := Scan(aQuery, aTopK, '', []);
end;

const
  /// authorizer action code for a recursive CTE (not declared by mORMot)
  SQLITE_RECURSIVE_ = 33;

/// authorizer active while a key query is COMPILED: only reading is allowed
// - some statements act at prepare time already (e.g. PRAGMA busy_timeout=0
//   changes the connection while being prepared), so they must be denied
//   during compilation - checks on the compiled statement would come too late
function KeySqlAuthorizer(pUserData: pointer; code: integer;
  const zTab, zCol, zDb, zAuthContext: PUtf8Char): integer; cdecl;
begin
  case code of
    SQLITE_SELECT, SQLITE_READ, SQLITE_FUNCTION, SQLITE_RECURSIVE_:
      result := SQLITE_OK;
    SQLITE_PRAGMA:
      // FTS5 runs "PRAGMA data_version" (no argument = read only) internally
      // on every MATCH; allowing exactly that keeps full-text key queries -
      // the hybrid FTS5 + vector search - possible. Any other pragma, or
      // data_version with an argument, stays denied
      if (zCol = nil) and (zTab <> nil) and
         (StrIComp(zTab, PAnsiChar('data_version')) = 0) then // no allocation
        result := SQLITE_OK
      else
        result := SQLITE_DENY;
  else
    result := SQLITE_DENY; // PRAGMA, ATTACH, writes, transactions, DDL ...
  end;
end;

procedure TNativeVectorStore.CheckKeySql(const aKeySql: RawUtf8);
var
  stmt: TSqlite3Statement;
  tail: PUtf8Char;
  rc, cols: integer;
  ro: boolean;
  wrapped: RawUtf8;
begin
  if TrimU(aKeySql) = '' then
    EVectorStore.RaiseUtf8('%.SearchWhere: empty key query - use Search', [self]);
  // a NUL ends the text for SQLite and hides what follows from the tail check;
  // a final ';' is consumed by the first prepare but breaks the IN (...) wrap
  if PosExChar(#0, aKeySql) > 0 then
    EVectorStore.RaiseUtf8('%.SearchWhere: key query contains a NUL byte', [self]);
  if TrimU(aKeySql)[length(TrimU(aKeySql))] = ';' then
    EVectorStore.RaiseUtf8('%.SearchWhere: key query must not end with ";"', [self]);
  // compile the key query ON ITS OWN first: it must be exactly one complete,
  // read-only statement with one result column. Only then is it embedded in
  // IN (...) - this rejects a text that closes the parenthesis itself
  // ("SELECT 'a') OR 1=1 --": a syntax error standalone) and a second
  // statement after a ';' (which TSqlRequest.Prepare would silently ignore)
  stmt := 0;
  tail := nil;
  sqlite3.set_authorizer(fDB.DB, KeySqlAuthorizer, nil);
  try
    rc := sqlite3.prepare_v2(fDB.DB, pointer(aKeySql), length(aKeySql), stmt, tail);
  finally
    sqlite3.set_authorizer(fDB.DB, nil, nil);
  end;
  try
    if (rc <> SQLITE_OK) or (stmt = 0) then
      EVectorStore.RaiseUtf8('%.SearchWhere: key query does not compile: %',
        [self, sqlite3.errmsg(fDB.DB)]);
    cols := sqlite3.column_count(stmt);
    ro := sqlite3.stmt_readonly(stmt) <> 0;
  finally
    if stmt <> 0 then
      sqlite3.finalize(stmt);
  end;
  if (tail <> nil) and (TrimU(RawUtf8(tail)) <> '') then
    EVectorStore.RaiseUtf8('%.SearchWhere: key query must be ONE statement', [self]);
  if not ro then
    EVectorStore.RaiseUtf8('%.SearchWhere: key query is not read-only', [self]);
  if cols <> 1 then
    EVectorStore.RaiseUtf8('%.SearchWhere: key query returns % columns, ' +
      'expected exactly 1 (the external key)', [self, cols]);
  // finally compile it exactly as Scan will embed it: a text that is valid on
  // its own can still break the wrap (e.g. an unterminated /* comment) - report
  // that here as EVectorStore instead of a raw SQLite error in the scan
  stmt := 0;
  sqlite3.set_authorizer(fDB.DB, KeySqlAuthorizer, nil);
  try
    wrapped := FormatUtf8('SELECT 1 WHERE NULL IN (%' + #10 + ')', [aKeySql]);
    rc := sqlite3.prepare_v2(fDB.DB, pointer(wrapped), length(wrapped), stmt, tail);
    if stmt <> 0 then
      sqlite3.finalize(stmt);
  finally
    sqlite3.set_authorizer(fDB.DB, nil, nil);
  end;
  if rc <> SQLITE_OK then
    EVectorStore.RaiseUtf8('%.SearchWhere: key query cannot be used as a ' +
      'sub-select: %', [self, sqlite3.errmsg(fDB.DB)]);
end;

function TNativeVectorStore.SearchWhere(const aQuery: TLlmEmbedding;
  aTopK: integer; const aKeySql: RawUtf8;
  const aParams: array of const): TRagHitDynArray;
begin
  fDB.Lock;
  try
    CheckKeySql(aKeySql);
    result := Scan(aQuery, aTopK, aKeySql, aParams);
  finally
    fDB.UnLock;
  end;
end;

function TNativeVectorStore.Scan(const aQuery: TLlmEmbedding; aTopK: integer;
  const aKeySql: RawUtf8; const aParams: array of const): TRagHitDynArray;
var
  q: TLlmEmbedding;
  r: TSqlRequest;
  top: TTopK;
  sql: RawUtf8;
  stmt: TSqlite3Statement;
  row: pointer;
  i: integer;
  score: double;
  fpu: cardinal;
begin
  result := nil;
  q := UnitVector(aQuery, 'query');
  // non-positive TopK = no results requested (same contract as TVec0Store)
  if aTopK <= 0 then
    exit;
  top.Init(aTopK);
  sql := FormatUtf8('SELECT id, embedding FROM %', [fItems]);
  if aKeySql <> '' then
    // validated by CheckKeySql; the line break keeps a trailing "-- comment"
    // of the key query from swallowing the closing parenthesis
    sql := FormatUtf8('% WHERE ext_id IN (%' + #10 + ')', [sql, aKeySql]);
  fDB.Lock;
  try
  // ONE read transaction around the scan AND the winner fetch: otherwise another
  // connection could update or delete a winner in between, and the hit would
  // pair new text (or none) with the old score
  fDB.TransactionBegin(tbDeferred);
  try
    // the read-only authorizer stays active for the WHOLE life of the filtered
    // statement, not only its compilation: table-valued pragma functions
    // (pragma_table_info ...) prepare their PRAGMA only at the first Step, and
    // SQLite may re-prepare a statement transparently while stepping
    if aKeySql <> '' then
      sqlite3.set_authorizer(fDB.DB, KeySqlAuthorizer, nil);
    try
    r.Prepare(fDB.DB, sql);
    try
      stmt := r.Request;
      if sqlite3.stmt_readonly(stmt) = 0 then
        EVectorStore.RaiseUtf8('%.SearchWhere: key query is not read-only', [self]);
      if r.ParamCount <> length(aParams) then
        EVectorStore.RaiseUtf8('%.SearchWhere: key query has % placeholders but ' +
          '% params were given', [self, r.ParamCount, length(aParams)]);
      if length(aParams) > 0 then
        r.Bind(aParams);
      // stream the rows: the BLOB is read in place (no copy, no allocation per
      // row); the CHECK constraint guarantees its size, verified anyway because
      // a foreign writer could have bypassed it with an older schema
      // mask the FPU exceptions for the arithmetic of the scan: a foreign row
      // with Inf or signalling-NaN bytes would otherwise trap INSIDE DotRow
      // (Inf * 0 = invalid on x86_64) before the score test below could
      // report it; masked, every such row ends in a non-finite score
      fpu := SetFpuFlags(ffLibrary);
      try
      while r.Step = SQLITE_ROW do
      begin
        if sqlite3.column_bytes(stmt, 1) <> fBlobSize then
          EVectorStore.RaiseUtf8('%: row % has a corrupt embedding (% bytes, ' +
            'expected %)', [self, sqlite3.column_int64(stmt, 0),
            sqlite3.column_bytes(stmt, 1), fBlobSize]);
        row := sqlite3.column_blob(stmt, 1);
        score := DotRow(pointer(q), row, fContract.Dim);
        // the query is finite, so a non-finite score means NaN/Inf bytes in
        // the row (a foreign writer) - test the bits: comparing a NaN raises
        // EInvalidOp once the FPU flags are restored
        if PInt64(@score)^ and DOUBLE_EXP_MASK = DOUBLE_EXP_MASK then
          EVectorStore.RaiseUtf8('%: row % has a corrupt embedding (non-finite ' +
            'values)', [self, sqlite3.column_int64(stmt, 0)]);
        top.Offer(score, sqlite3.column_int64(stmt, 0));
      end;
      finally
        ResetFpuFlags(fpu);
      end;
    finally
      r.Close;
    end;
    finally
      if aKeySql <> '' then
        sqlite3.set_authorizer(fDB.DB, nil, nil);
    end;
    // fetch text + key for the winners only - the scan never touches them
    SetLength(result, top.Count);
    if top.Count > 0 then
    begin
      r.Prepare(fDB.DB, FormatUtf8(
        'SELECT content, ext_id FROM % WHERE id = ?;', [fItems]));
      try
        for i := 0 to top.Count - 1 do
        begin
          r.Bind(1, top.RowId[i]);
          if r.Step = SQLITE_ROW then
          begin
            r.FieldUtf8(0, result[i].Text);
            r.FieldUtf8(1, result[i].Key); // NULL ext_id -> '' (anonymous row)
          end;
          r.Reset;
          result[i].DocId := top.RowId[i];
          result[i].Distance := 1 - top.Score[i];
        end;
      finally
        r.Close;
      end;
    end;
    fDB.Commit; // ends the read snapshot
  except
    on E: Exception do
    begin
      RollbackIfOpen(fDB);
      // the read-only authorizer refused something only visible at Step time
      // (e.g. a pragma table-valued function): report it like CheckKeySql does
      if (E is ESqlite3Exception) and
         (ESqlite3Exception(E).ErrorCode = SQLITE_AUTH) then
        EVectorStore.RaiseUtf8('%.SearchWhere: key query uses a forbidden ' +
          'operation (only reading is allowed): %', [self, E.Message]);
      raise;
    end;
  end;
  finally
    fDB.UnLock;
  end;
end;

end.
