// LandrixAI - embedding reference generator (step 1 of docs/native-embeddings.md)
// - embeds a retrieval test corpus (corpus.jsonl + queries.jsonl) with the ORACLE
//   backend sqlite-lembed and writes the reference artefacts every later native
//   implementation is measured against:
//     reference/manifest.json   contract (model hash, runtime, tokens, metric ...),
//                               corpus hashes, stats, metrics, exact top-20 rankings
//     reference/docs.f32        doc vectors, float32 little-endian, corpus order
//     reference/queries.f32     query vectors, float32 little-endian, queries order
// - the vectors are committed so parity/quality tests run WITHOUT the lembed
//   runtime (Delphi, CI without vendor/ binaries)
// - fails loudly on contract violations: a text over the token limit (lembed
//   refuses it), a non-normalised vector, an unknown id in the relevance judgments
//
//   usage: embed-reference [datadir]   (default: tests/data/embed-de)
//   env:   SQLITE_EXT_DIR (dir with lembed0 + llama/ggml), LEMBED_MODEL (.gguf)
//   run with LD_LIBRARY_PATH=$SQLITE_EXT_DIR (Linux) - scripts/embed-reference.sh
program embed.reference;

{$I mormot.defines.inc}

{$ifdef OSWINDOWS}
  {$apptype console}
{$endif OSWINDOWS}

uses
  {$I mormot.uses.inc}
  sysutils,
  mormot.core.base,
  mormot.core.os,
  mormot.core.unicode,
  mormot.core.text,
  mormot.core.datetime,
  mormot.core.variants,
  mormot.core.json,
  mormot.crypt.secure,
  mormot.db.raw.sqlite3,
  mormot.db.raw.sqlite3.static,
  mormot.ai.llm.types,
  mormot.ai.vectorstore,
  mormot.ai.vectorstore.sqlitevec;

const
  /// the reference contract: lembed runs bge-m3 with the default llama.cpp
  // context of 512 tokens INCLUDING <s> and </s>; longer input is refused (and a
  // raised n_ctx aborts the process: encoder requires n_ubatch >= n_tokens)
  MAX_TOKENS = 512;
  /// rankings kept per query in the manifest
  TOP_N = 20;
  /// tolerance for the unit-norm check (lembed L2-normalises its output)
  NORM_EPS = 1e-4;
  MODEL_ALIAS = 'm';
  /// the contract pins the exact model file: model name and CLS pooling below are
  // only true for THIS file - a different GGUF is refused instead of being
  // labelled bge-m3 (changing the model = new contract = edit these constants)
  CONTRACT_MODEL = 'BAAI/bge-m3';
  CONTRACT_MODEL_SHA256 =
    'aa473d51f451a22f0fcf39ba3330c14bed38a385712b1113440f69df4047a173';
  CONTRACT_DIM = 1024;
  CONTRACT_POOLING = 'cls';

type
  TItem = record
    Id, Kind, Text: RawUtf8;
    Rel: TDocVariantData; // queries only: doc id -> grade
    Tokens: integer;
    Vec: TLlmEmbedding;
  end;
  TItemDynArray = array of TItem;

var
  db: TSqlDataBase;

procedure Fail(const aFmt: RawUtf8; const aArgs: array of const);
begin
  raise ESynException.CreateUtf8(aFmt, aArgs);
end;

function ScalarText(const aSql: RawUtf8): RawUtf8;
var
  r: TSqlRequest;
begin
  result := '';
  r.Prepare(db.DB, aSql);
  try
    if r.Step = SQLITE_ROW then
      r.FieldUtf8(0, result);
  finally
    r.Close;
  end;
end;

/// load a JSON-lines file (one object per non-empty line)
function LoadJsonl(const aFile: TFileName; aIsQuery: boolean): TItemDynArray;
var
  lines: TRawUtf8DynArray;
  line: RawUtf8;
  doc: TDocVariantData;
  rel: PDocVariantData;
  i, n: PtrInt;
begin
  result := nil;
  CSVToRawUtf8DynArray(pointer(StringReplaceAll(
    RawUtf8FromFile(aFile), #13, '')), lines, #10);
  SetLength(result, length(lines));
  n := 0;
  for i := 0 to high(lines) do
  begin
    line := TrimU(lines[i]);
    if line = '' then
      continue;
    doc.InitJson(line, JSON_FAST);
    if doc.Kind <> dvObject then
      Fail('%: line % is no JSON object', [aFile, i + 1]);
    result[n].Id := doc.U['id'];
    result[n].Text := doc.U['text'];
    result[n].Kind := doc.U['kind'];
    if (result[n].Id = '') or (result[n].Text = '') then
      Fail('%: line % lacks id or text', [aFile, i + 1]);
    if aIsQuery then
    begin
      // O[] would hand back a void object for a missing "rel" - be explicit
      if doc.GetAsObject('rel', rel) then
        result[n].Rel.InitCopy(variant(rel^), JSON_FAST)
      else
        Fail('%: line % (%) lacks a "rel" object', [aFile, i + 1, result[n].Id]);
    end;
    doc.Clear;
    inc(n);
  end;
  SetLength(result, n);
end;

function IndexOfId(const aItems: TItemDynArray; const aId: RawUtf8): PtrInt;
begin
  for result := 0 to high(aItems) do
    if aItems[result].Id = aId then
      exit;
  result := -1;
end;

/// reject anything that would silently distort the reference: duplicate ids,
// queries without judgments (O['rel'] yields a void object when missing), grades
// other than 1/2 (gain and ideal DCG only know these), unknown doc ids
procedure Validate(const aDocs: TItemDynArray; var aQueries: TItemDynArray);
var
  i, j, n2: PtrInt;
  g: Int64;
  d: double;
begin
  if (aDocs = nil) or (aQueries = nil) then
    Fail('empty corpus or query set', []);
  for i := 0 to high(aDocs) do
    if IndexOfId(aDocs, aDocs[i].Id) <> i then
      Fail('duplicate doc id %', [aDocs[i].Id]);
  for i := 0 to high(aQueries) do
  begin
    if IndexOfId(aQueries, aQueries[i].Id) <> i then
      Fail('duplicate query id %', [aQueries[i].Id]);
    if (aQueries[i].Rel.Kind <> dvObject) or (aQueries[i].Rel.Count = 0) then
      Fail('query %: missing or empty "rel"', [aQueries[i].Id]);
    n2 := 0;
    for j := 0 to aQueries[i].Rel.Count - 1 do
    begin
      if IndexOfId(aDocs, aQueries[i].Rel.Names[j]) < 0 then
        Fail('query %: unknown doc id %', [aQueries[i].Id, aQueries[i].Rel.Names[j]]);
      // the JSON parser keeps duplicate names, but I[] later reads the first one
      if aQueries[i].Rel.GetValueIndex(aQueries[i].Rel.Names[j]) <> j then
        Fail('query %: duplicate judgment for %', [aQueries[i].Id,
          aQueries[i].Rel.Names[j]]);
      // numbers only (VariantToInt64 would map true to 1), integral 1 or 2 -
      // JSON_FAST parses 2.0 as currency, so compare the numeric value
      if not (TVarData(aQueries[i].Rel.Values[j]).VType in [varByte, varShortInt,
         varWord, varSmallInt, varLongWord, varInteger, varInt64, varQWord,
         varCurrency, varDouble, varSingle]) or
         not VariantToDouble(aQueries[i].Rel.Values[j], d) then
        Fail('query %: grade of % is not a number', [aQueries[i].Id,
          aQueries[i].Rel.Names[j]]);
      if (d <> 1) and (d <> 2) then
        Fail('query %: grade of % must be 1 or 2', [aQueries[i].Id,
          aQueries[i].Rel.Names[j]]);
      g := trunc(d);
      // normalise to an integer so the metrics read exactly what was validated
      aQueries[i].Rel.Values[j] := g;
      if g = 2 then
        inc(n2);
    end;
    if n2 = 0 then
      Fail('query %: no grade-2 document', [aQueries[i].Id]);
  end;
end;

function TokenCount(const aText: RawUtf8): integer;
var
  r: TSqlRequest;
  json: RawUtf8;
  arr: TDocVariantData;
begin
  r.Prepare(db.DB, 'SELECT lembed_tokenize_json(?, ?);');
  try
    r.Bind(1, MODEL_ALIAS);
    r.Bind(2, aText);
    json := '';
    if r.Step = SQLITE_ROW then
      r.FieldUtf8(0, json);
  finally
    r.Close;
  end;
  arr.InitJson(json, JSON_FAST);
  if arr.Kind <> dvArray then
    Fail('lembed_tokenize_json returned %', [json]);
  result := arr.Count;
end;

function EmbedText(const aText: RawUtf8): TLlmEmbedding;
var
  r: TSqlRequest;
  blob: RawByteString;
begin
  blob := '';
  r.Prepare(db.DB, 'SELECT lembed(?, ?);');
  try
    r.Bind(1, MODEL_ALIAS);
    r.Bind(2, aText);
    if r.Step = SQLITE_ROW then
      blob := r.FieldBlob(0);
  finally
    r.Close;
  end;
  if blob = '' then
    Fail('lembed returned no vector', []);
  result := BlobToVector(blob);
end;

function Dot(const a, b: TLlmEmbedding): double;
var
  i: PtrInt;
begin
  result := 0;
  for i := 0 to high(a) do
    result := result + a[i] * b[i];
end;

/// embed all items, enforce the token limit and the unit norm
procedure EmbedAll(var aItems: TItemDynArray; const aWhat: RawUtf8;
  var aDim, aMaxTok: integer; var aTokSum: Int64;
  var aNormMin, aNormMax: double);
var
  i: PtrInt;
  nrm: double;
begin
  for i := 0 to high(aItems) do
  begin
    aItems[i].Tokens := TokenCount(aItems[i].Text);
    if aItems[i].Tokens > MAX_TOKENS then
      Fail('% %: % tokens exceed the contract limit of %',
        [aWhat, aItems[i].Id, aItems[i].Tokens, MAX_TOKENS]);
    aItems[i].Vec := EmbedText(aItems[i].Text);
    if aDim = 0 then
      aDim := length(aItems[i].Vec)
    else if length(aItems[i].Vec) <> aDim then
      Fail('% %: dimension % <> %', [aWhat, aItems[i].Id,
        length(aItems[i].Vec), aDim]);
    nrm := sqrt(Dot(aItems[i].Vec, aItems[i].Vec));
    if abs(nrm - 1) > NORM_EPS then
      Fail('% %: vector norm % is not 1 (contract: L2-normalised)',
        [aWhat, aItems[i].Id, nrm]);
    if nrm < aNormMin then
      aNormMin := nrm;
    if nrm > aNormMax then
      aNormMax := nrm;
    if aItems[i].Tokens > aMaxTok then
      aMaxTok := aItems[i].Tokens;
    inc(aTokSum, aItems[i].Tokens);
  end;
end;

function VectorsBlob(const aItems: TItemDynArray): RawByteString;
var
  i: PtrInt;
begin
  result := '';
  for i := 0 to high(aItems) do
    result := result + VectorToBlob(aItems[i].Vec);
end;

/// write the three reference artefacts together: every file is first written to
// a process-unique temp name, and only after ALL writes succeeded are they moved
// into place - a failed write leaves the old reference untouched
// - the final moves are not one atomic transaction (three files); a failure
//   there is reported loudly, and since the reference is tracked by git,
//   `git checkout -- tests/data/<corpus>/reference` restores a consistent set
procedure WriteArtefacts(const aDir: TFileName;
  const aNames: array of TFileName; const aContents: array of RawByteString);
var
  i: PtrInt;
  tmp: array of TFileName;
begin
  SetLength(tmp, length(aNames));
  for i := 0 to high(aNames) do
    tmp[i] := FormatString('%%.%.tmp', [aDir, aNames[i], GetCurrentProcessId]);
  try
    for i := 0 to high(aNames) do
      if not FileFromString(aContents[i], tmp[i]) then
        Fail('cannot write %', [tmp[i]]);
    for i := 0 to high(aNames) do
    begin
      {$ifdef OSWINDOWS}
      // Windows MoveFile does not replace an existing target (POSIX rename does)
      DeleteFile(aDir + aNames[i]);
      {$endif OSWINDOWS}
      if not RenameFile(tmp[i], aDir + aNames[i]) then
        Fail('cannot move % into place - reference incomplete, restore it with ' +
          'git checkout and regenerate', [aDir + aNames[i]]);
    end;
  finally
    for i := 0 to high(tmp) do
      if FileExists(tmp[i]) then
        DeleteFile(tmp[i]);
  end;
end;

function Log2(x: double): double;
begin
  result := ln(x) / ln(2);
end;

var
  datadir, refdir: TFileName;
  docs, queries: TItemDynArray;
  dim, maxTok, i, j, k, best, n2, hit2, firstRel, rank: integer;
  tokSum: Int64;
  normMin, normMax, s, dcg, idcg, ndcgSum, recSum, mrrSum, sc: double;
  t0, embedMs: Int64;
  scores: array of double;
  used: array of boolean;
  grades: array of integer;
  manifest, contract, corpus, stats, metrics, perQuery, q, ranking, hit: TDocVariantData;
  ext, modelPath, modelSha: RawUtf8;
begin
  if ParamCount >= 1 then
    datadir := ParamStr(1)
  else
    datadir := 'tests/data/embed-de';
  datadir := IncludeTrailingPathDelimiter(ExpandFileName(datadir));
  refdir := datadir + 'reference' + PathDelim;
  ForceDirectories(refdir);
  ext := StringToUtf8(GetEnvironmentVariable('SQLITE_EXT_DIR'));
  modelPath := StringToUtf8(GetEnvironmentVariable('LEMBED_MODEL'));
  if (ext = '') or (modelPath = '') then
  begin
    ConsoleWrite('SQLITE_EXT_DIR and LEMBED_MODEL must be set', ccLightRed);
    ExitCode := 2;
    exit;
  end;
  try
    docs := LoadJsonl(datadir + 'corpus.jsonl', false);
    queries := LoadJsonl(datadir + 'queries.jsonl', true);
    ConsoleWrite(FormatUtf8('corpus: % docs, % queries', [length(docs), length(queries)]));
    Validate(docs, queries);
    modelSha := HashFile(Utf8ToString(modelPath), hfSHA256);
    if not IdemPropNameU(modelSha, CONTRACT_MODEL_SHA256) then
      Fail('model file SHA-256 % <> contract % - a different model needs a new ' +
        'contract (edit CONTRACT_* in this program)', [modelSha, CONTRACT_MODEL_SHA256]);
    db := TSqlDataBase.Create(Utf8ToString(SQLITE_MEMORY_DATABASE_NAME), '');
    try
      EnableSqliteExtensions(db.DB);
      LoadSqliteExtension(db.DB, ext + '/lembed0' + SqliteExtSuffix);
      db.Execute(FormatUtf8('INSERT INTO temp.lembed_models(name, model) ' +
        'VALUES (''%'', lembed_model_from_file(%));',
        [MODEL_ALIAS, QuotedStr(modelPath, '''')]));
      dim := 0;
      maxTok := 0;
      tokSum := 0;
      normMin := 2;
      normMax := 0;
      t0 := GetTickCount64;
      EmbedAll(docs, 'doc', dim, maxTok, tokSum, normMin, normMax);
      embedMs := GetTickCount64 - t0;
      EmbedAll(queries, 'query', dim, maxTok, tokSum, normMin, normMax);
      if dim <> CONTRACT_DIM then
        Fail('dimension % <> contract %', [dim, CONTRACT_DIM]);
      contract.InitFast;
      contract.AddValue('model', CONTRACT_MODEL);
      contract.AddValueFromText('model_file', ExtractFileName(Utf8ToString(modelPath)));
      contract.AddValueFromText('model_sha256', modelSha);
      contract.AddValueFromText('runtime', 'sqlite-lembed ' +
        ScalarText('SELECT lembed_version();') + ' (llama.cpp)');
      contract.AddValueFromText('sqlite', ScalarText('SELECT sqlite_version();'));
      contract.AddValue('dim', dim);
      contract.AddValue('pooling', CONTRACT_POOLING);
      contract.AddValue('normalize', 'l2');
      contract.AddValue('metric', 'cosine');
      contract.AddValue('metric_note',
        'vectors are unit length: cosine = dot product, L2 ranking is identical');
      contract.AddValue('max_tokens', MAX_TOKENS);
      contract.AddValue('max_tokens_note',
        'includes <s> and </s>; the oracle refuses longer input, callers must cut before');
      contract.AddValue('query_prefix', '');
      contract.AddValue('document_prefix', '');
      contract.AddValue('add_special', true);
      contract.AddValue('parse_special', true);
      contract.AddValue('parse_special_note',
        'oracle turns <s> </s> <pad> in user text into control tokens; ' +
        'native tokenizers must not - corpus avoids these strings');
      corpus.InitFast;
      corpus.AddValue('docs', length(docs));
      corpus.AddValue('queries', length(queries));
      corpus.AddValueFromText('corpus_sha256',
        HashFile(datadir + 'corpus.jsonl', hfSHA256));
      corpus.AddValueFromText('queries_sha256',
        HashFile(datadir + 'queries.jsonl', hfSHA256));
      corpus.AddValue('vector_files', _ArrFast(['docs.f32', 'queries.f32']));
      corpus.AddValue('vector_format', 'float32 little-endian, row-major, file order');
      stats.InitFast;
      stats.AddValue('max_tokens_seen', maxTok);
      stats.AddValue('mean_tokens', tokSum / (length(docs) + length(queries)));
      stats.AddValue('norm_min', normMin);
      stats.AddValue('norm_max', normMax);
      stats.AddValue('doc_embed_ms', embedMs);
      if embedMs > 0 then
        stats.AddValue('docs_per_second', length(docs) * 1000 / embedMs);
      stats.AddValueFromText('host', OSVersionShort + ' ' + CpuInfoText);
    finally
      FreeAndNil(db);
    end;
    // exact rankings + metrics (graded nDCG@10, recall@10 of grade 2, MRR)
    SetLength(scores, length(docs));
    SetLength(used, length(docs));
    SetLength(grades, length(docs));
    perQuery.InitFast;
    ndcgSum := 0;
    recSum := 0;
    mrrSum := 0;
    for i := 0 to high(queries) do
    begin
      n2 := 0;
      for j := 0 to high(docs) do
      begin
        scores[j] := Dot(queries[i].Vec, docs[j].Vec);
        used[j] := false;
        grades[j] := queries[i].Rel.I[docs[j].Id];
        if grades[j] = 2 then
          inc(n2);
      end;
      ranking.InitArray([], JSON_FAST);
      dcg := 0;
      hit2 := 0;
      firstRel := 0;
      for k := 1 to TOP_N do
      begin
        best := -1;
        for j := 0 to high(docs) do
          if not used[j] and ((best < 0) or (scores[j] > scores[best]) or
             ((scores[j] = scores[best]) and (docs[j].Id < docs[best].Id))) then
            best := j;
        if best < 0 then
          break;
        used[best] := true;
        if k <= 10 then
        begin
          dcg := dcg + ((1 shl grades[best]) - 1) / Log2(k + 1);
          if grades[best] = 2 then
            inc(hit2);
        end;
        hit.InitArray([docs[best].Id, scores[best], grades[best]], JSON_FAST);
        ranking.AddItem(variant(hit));
        hit.Clear;
      end;
      // MRR over the FULL ranking (not only the kept top-N): rank of the best
      // grade-2 doc = 1 + number of docs ordered before it (same tie-break)
      firstRel := 0;
      for j := 0 to high(docs) do
        if grades[j] = 2 then
        begin
          rank := 1;
          for best := 0 to high(docs) do
            if (scores[best] > scores[j]) or
               ((scores[best] = scores[j]) and (docs[best].Id < docs[j].Id)) then
              inc(rank);
          if (firstRel = 0) or (rank < firstRel) then
            firstRel := rank;
        end;
      // ideal DCG: grades sorted descending
      idcg := 0;
      k := 0;
      for best := 2 downto 1 do
        for j := 0 to high(docs) do
          if (grades[j] = best) and (k < 10) then
          begin
            inc(k);
            idcg := idcg + ((1 shl best) - 1) / Log2(k + 1);
          end;
      if idcg > 0 then
        sc := dcg / idcg
      else
        sc := 0;
      ndcgSum := ndcgSum + sc;
      s := 0;
      if n2 > 0 then
        s := hit2 / MinPtrInt(n2, 10);
      recSum := recSum + s;
      if firstRel > 0 then
        mrrSum := mrrSum + 1 / firstRel;
      q.InitFast;
      q.AddValue('ndcg10', sc);
      q.AddValue('recall10', s);
      q.AddValue('first_relevant', firstRel);
      q.AddValue('top', variant(ranking));
      perQuery.AddValue(queries[i].Id, variant(q));
      q.Clear;
      ranking.Clear;
    end;
    metrics.InitFast;
    metrics.AddValue('ndcg10', ndcgSum / length(queries));
    metrics.AddValue('recall10', recSum / length(queries));
    metrics.AddValue('mrr', mrrSum / length(queries));
    manifest.InitFast;
    manifest.AddValue('format', 1);
    manifest.AddValueFromText('created_utc', NowUtcToString(true, 'T') + 'Z');
    manifest.AddValue('contract', variant(contract));
    manifest.AddValue('corpus', variant(corpus));
    manifest.AddValue('stats', variant(stats));
    manifest.AddValue('metrics', variant(metrics));
    manifest.AddValue('queries', variant(perQuery));
    WriteArtefacts(refdir, ['docs.f32', 'queries.f32', 'manifest.json'],
      [VectorsBlob(docs), VectorsBlob(queries),
       manifest.ToJson('', '', jsonHumanReadable)]);
    ConsoleWrite(FormatUtf8('dim=% maxTokens=% docs/s=% nDCG@10=% recall@10=% MRR=%',
      [dim, maxTok, stats.D['docs_per_second'], metrics.D['ndcg10'],
       metrics.D['recall10'], metrics.D['mrr']]), ccLightGreen);
    ConsoleWrite('written: ' + StringToUtf8(refdir));
  except
    on e: Exception do
    begin
      ConsoleWrite(FormatUtf8('FAIL: % %', [e.ClassType, e.Message]), ccLightRed);
      ExitCode := 1;
    end;
  end;
end.
