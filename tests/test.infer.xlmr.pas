/// tests for the XLM-R encoder of the native inference engine (phase 1a)
// - part of the mormot.ai.* extension (LandrixAI)
// - licensed under a MPL/GPL/LGPL three license - see LICENSE
// - the tiny fixture runs everywhere; the real BGE-M3 runs when the converted
//   weights exist (vendor/models/bge-m3/, or $LAI_MODEL_DIR) on a 64-bit target,
//   and says SKIP otherwise - LAI_REQUIRE_MODEL=1 turns a missing model into a
//   failure. A model that exists but does not load or match is always a failure
// - token ids come from the reference, not from our tokenizer: a forward error
//   and a tokenizer error stay apart
unit test.infer.xlmr;

interface

{$I mormot.defines.inc}

uses
  sysutils,
  mormot.core.base,
  mormot.core.os,
  mormot.core.unicode,
  mormot.core.text,
  mormot.core.variants,
  mormot.core.test,
  mormot.ai.infer.tensor,
  mormot.ai.infer.backend,
  mormot.ai.infer.backend.cpu,
  mormot.ai.infer.model.xlmr,
  mormot.ai.infer.pipeline.embed,
  test.infer.ref;

type
  /// phase 1a: the encoder forward pass against the references
  TTestInferXlmr = class(TSynTestCase)
  protected
    fCpu: TInferCpuBackend;
    fHiddenSize: PtrInt;
    fStates: array of TSingleDynArray;
    procedure Setup; override;
    procedure CleanUp; override;
    /// TOnXlmrState: copy every state, the model reuses its buffer
    procedure KeepState(aState: PtrInt; aValues: PSingleArray;
      aRows, aWidth: PtrInt);
    /// forward <aPrefix>input_ids as one sequence and compare every state and
    // the embedding
    procedure CheckSingle(aModel: TXlmrModel; aCtx: TXlmrContext;
      aRef: TInferRefSet; const aPrefix: RawUtf8; aAtol, aRtol: double);
    /// forward batch/input_ids and compare every state on the real tokens of
    // each row, and each row's embedding
    procedure CheckBatch(aModel: TXlmrModel; aCtx: TXlmrContext;
      aRef: TInferRefSet; aAtol, aRtol: double);
  published
    /// the tiny XLM-R: every state, single and batched
    procedure TinyForwardMatchesReference;
    /// a padded row gives the same result as the same sequence alone
    procedure PaddingDoesNotChangeResults;
    /// wrong ids and oversized input are refused, not read out of bounds; the
    // largest allowed width runs
    procedure ForwardRefusesBadInput;
    /// an explicit keep mask equal to ids <> pad gives the same result, and a
    // context reused from a larger call gives the same result as a fresh one
    procedure KeepMaskAndContextReuse;
    /// configs and weight files this class does not understand are refused
    procedure LoadRefusesForeignModels;
    /// BGE-M3 F32 against the PyTorch reference (needs the converted weights)
    procedure BgeM3ForwardMatchesReference;
    /// text -> tokenizer -> encoder -> CLS/L2 as one padded batch, against the
    // reference vectors; a text above 512 tokens is refused
    procedure BgeM3TextToEmbedding;
  end;


implementation

/// the folder of the converted BGE-M3 weights, '' when absent
function FindBgeM3: TFileName;
var
  dir: TFileName;
  i: integer;
begin
  result := GetEnvironmentVariable('LAI_MODEL_DIR');
  if result <> '' then
  begin
    result := IncludeTrailingPathDelimiter(result);
    exit;
  end;
  dir := ExcludeTrailingPathDelimiter(Executable.ProgramFilePath);
  for i := 1 to 6 do
  begin
    result := MakePath([dir, 'vendor', 'models', 'bge-m3'], true);
    if FileExists(result + 'model.safetensors') then
      exit;
    dir := ExtractFileDir(dir);
  end;
  result := '';
end;


{ TTestInferXlmr }

procedure TTestInferXlmr.Setup;
begin
  fCpu := TInferCpuBackend.Create;
end;

procedure TTestInferXlmr.CleanUp;
begin
  FreeAndNil(fCpu); // re-entrant, as mORMot requires
end;

procedure TTestInferXlmr.KeepState(aState: PtrInt; aValues: PSingleArray;
  aRows, aWidth: PtrInt);
var
  n: PtrInt;
begin
  if aState >= length(fStates) then
    SetLength(fStates, aState + 1);
  n := aRows * aWidth * fHiddenSize;
  SetLength(fStates[aState], n);
  MoveFast(aValues^, pointer(fStates[aState])^, n * SizeOf(single));
end;

procedure TTestInferXlmr.CheckSingle(aModel: TXlmrModel; aCtx: TXlmrContext;
  aRef: TInferRefSet; const aPrefix: RawUtf8; aAtol, aRtol: double);
var
  ids: PIntegerArray;
  ref, emb: PSingleArray;
  n, rn, k, h: PtrInt;
  cmp: TInferCompare;
  mine: TSingleDynArray;
begin
  h := aModel.Config.Hidden;
  fHiddenSize := h;
  fStates := nil;
  ids := aRef.I32(aPrefix + 'input_ids', n);
  aModel.Forward(aCtx, ids, 1, n, nil, KeepState);
  if CheckFailed(length(fStates) = aModel.Config.Layers + 1, 'every state reported') then
    exit;
  for k := 0 to aModel.Config.Layers do
  begin
    ref := aRef.F32(FormatUtf8('%hidden.%', [aPrefix, k]), rn);
    CheckEqual(rn, n * h, 'state size');
    cmp := InferCompare(pointer(fStates[k]), ref, rn, aAtol, aRtol);
    CheckUtf8(InferCompareOk(cmp), '%hidden.%: %',
      [aPrefix, k, InferCompareText(cmp)]);
  end;
  // dense vector = CLS of the last state, L2-normalised
  SetLength(mine, h);
  MoveFast(aCtx.Hidden[0], pointer(mine)^, h * SizeOf(single));
  fCpu.L2Normalize(pointer(mine), h);
  emb := aRef.F32(aPrefix + 'embedding', rn);
  cmp := InferCompare(pointer(mine), emb, h, aAtol, aRtol);
  CheckUtf8(InferCompareOk(cmp) and (cmp.Cosine >= 0.9999) and
    (Abs(InferL2Norm(pointer(mine), h) - 1) <= 1e-5), '%embedding: %',
    [aPrefix, InferCompareText(cmp)]);
end;

procedure TTestInferXlmr.CheckBatch(aModel: TXlmrModel; aCtx: TXlmrContext;
  aRef: TInferRefSet; aAtol, aRtol: double);
var
  ids: PIntegerArray;
  ref, emb: PSingleArray;
  t: PInferRefTensor;
  n, rn, rows, width, r, c, k, h, real, off: PtrInt;
  cmp: TInferCompare;
  mine: TSingleDynArray;
begin
  h := aModel.Config.Hidden;
  fHiddenSize := h;
  fStates := nil;
  t := aRef.Tensor('batch/input_ids');
  rows := t^.Shape[0];
  width := t^.Shape[1];
  ids := aRef.I32('batch/input_ids', n);
  aModel.Forward(aCtx, ids, rows, width, nil, KeepState);
  emb := aRef.F32('batch/embedding', rn);
  SetLength(mine, h);
  for r := 0 to rows - 1 do
  begin
    // padded positions are not meaningful: compare the real tokens only
    real := 0;
    for c := 0 to width - 1 do
      if ids^[r * width + c] <> aModel.Config.PadId then
        inc(real);
    if CheckFailed(real > 0, 'batch row has real tokens') then
      continue;
    off := r * width * h;
    for k := 0 to aModel.Config.Layers do
    begin
      ref := aRef.F32(FormatUtf8('batch/hidden.%', [k]), rn);
      cmp := InferCompare(@fStates[k][off], @ref^[off], real * h, aAtol, aRtol);
      CheckUtf8(InferCompareOk(cmp), 'batch row % hidden.%: %',
        [r, k, InferCompareText(cmp)]);
    end;
    MoveFast(aCtx.Hidden[off], pointer(mine)^, h * SizeOf(single));
    fCpu.L2Normalize(pointer(mine), h);
    cmp := InferCompare(pointer(mine), @emb^[r * h], h, aAtol, aRtol);
    CheckUtf8(InferCompareOk(cmp) and (cmp.Cosine >= 0.9999),
      'batch row % embedding: %', [r, InferCompareText(cmp)]);
  end;
end;

procedure TTestInferXlmr.TinyForwardMatchesReference;
var
  dir: TFileName;
  model: TXlmrModel;
  ctx: TXlmrContext;
  ref: TInferRefSet;
  i: PtrInt;
begin
  dir := FindInferRefDir('tiny-xlmr');
  if CheckFailed(dir <> '', 'tiny-xlmr found') then
    exit;
  ref := nil;
  model := nil;
  ctx := nil;
  try
    ref := TInferRefSet.CreateFromFolder(dir);
    model := TXlmrModel.Create(dir, fCpu);
    ctx := TXlmrContext.Create;
    CheckEqual(model.Config.Layers, 2, 'tiny depth');
    for i := 0 to 2 do
      CheckSingle(model, ctx, ref, FormatUtf8('single/%/', [i]), 1e-5, 1e-4);
    CheckBatch(model, ctx, ref, 1e-5, 1e-4);
  finally
    ctx.Free;
    model.Free;
    ref.Free;
  end;
end;

procedure TTestInferXlmr.PaddingDoesNotChangeResults;
var
  dir: TFileName;
  model: TXlmrModel;
  ctx: TXlmrContext;
  alone, padded: TSingleDynArray;
  ids: array[0..7] of integer;
  cmp: TInferCompare;
  h: PtrInt;
begin
  dir := FindInferRefDir('tiny-xlmr');
  if CheckFailed(dir <> '', 'tiny-xlmr found') then
    exit;
  model := TXlmrModel.Create(dir, fCpu);
  ctx := TXlmrContext.Create;
  try
    h := model.Config.Hidden;
    ids[0] := 0;
    ids[1] := 17;
    ids[2] := 33;
    ids[3] := 2;
    model.Forward(ctx, @ids, 1, 4);
    alone := copy(ctx.Hidden, 0, 4 * h);
    // the same tokens followed by four padding positions
    ids[4] := 1;
    ids[5] := 1;
    ids[6] := 1;
    ids[7] := 1;
    model.Forward(ctx, @ids, 1, 8);
    padded := copy(ctx.Hidden, 0, 4 * h);
    cmp := InferCompare(pointer(padded), pointer(alone), 4 * h, 1e-6, 1e-5);
    CheckUtf8(InferCompareOk(cmp), 'padding leaves the real tokens unchanged: %',
      [InferCompareText(cmp)]);
  finally
    ctx.Free;
    model.Free;
  end;
end;

procedure TTestInferXlmr.ForwardRefusesBadInput;
var
  dir: TFileName;
  model: TXlmrModel;
  ctx: TXlmrContext;
  ids: TIntegerDynArray;
  refused: boolean;
begin
  dir := FindInferRefDir('tiny-xlmr');
  if CheckFailed(dir <> '', 'tiny-xlmr found') then
    exit;
  model := TXlmrModel.Create(dir, fCpu);
  ctx := TXlmrContext.Create;
  try
    SetLength(ids, 3);
    ids[0] := 0;
    ids[1] := model.Config.VocabSize; // one past the end
    ids[2] := 2;
    refused := false;
    try
      model.Forward(ctx, pointer(ids), 1, 3);
    except
      on EInferError do
        refused := true;
    end;
    CheckUtf8(refused, 'id outside the vocabulary refused');
    SetLength(ids, model.Config.MaxPositions); // more tokens than positions
    FillCharFast(pointer(ids)^, length(ids) * SizeOf(integer), 0);
    refused := false;
    try
      model.Forward(ctx, pointer(ids), 1, length(ids));
    except
      on EInferError do
        refused := true;
    end;
    CheckUtf8(refused, 'more tokens than positions refused');
    // the largest width that still has a position for every token runs
    SetLength(ids, model.Config.MaxPositions - model.Config.PadId - 1);
    FillCharFast(pointer(ids)^, length(ids) * SizeOf(integer), 0);
    model.Forward(ctx, pointer(ids), 1, length(ids));
    CheckUtf8(length(ctx.Hidden) >= length(ids) * model.Config.Hidden,
      'width % accepted', [length(ids)]);
  finally
    ctx.Free;
    model.Free;
  end;
end;

procedure TTestInferXlmr.KeepMaskAndContextReuse;
var
  dir: TFileName;
  model: TXlmrModel;
  ctx, fresh: TXlmrContext;
  ids, keep: array[0..5] of integer;
  derived, explicit, reused: TSingleDynArray;
  cmp: TInferCompare;
  i, h: PtrInt;
  big: array[0..15] of integer;
begin
  dir := FindInferRefDir('tiny-xlmr');
  if CheckFailed(dir <> '', 'tiny-xlmr found') then
    exit;
  model := nil;
  ctx := nil;
  fresh := nil;
  try
    model := TXlmrModel.Create(dir, fCpu);
    ctx := TXlmrContext.Create;
    fresh := TXlmrContext.Create;
    h := model.Config.Hidden;
    ids[0] := 0;
    ids[1] := 21;
    ids[2] := 2;
    ids[3] := 1;
    ids[4] := 1;
    ids[5] := 1;
    model.Forward(fresh, @ids, 1, 6);
    derived := copy(fresh.Hidden, 0, 6 * h);
    for i := 0 to 5 do
      keep[i] := ord(ids[i] <> model.Config.PadId);
    model.Forward(fresh, @ids, 1, 6, @keep);
    explicit := copy(fresh.Hidden, 0, 6 * h);
    cmp := InferCompare(pointer(explicit), pointer(derived), 3 * h, 0, 0);
    CheckUtf8(InferCompareOk(cmp), 'explicit keep = derived keep: %',
      [InferCompareText(cmp)]);
    // masking a real token must change the result: the mask is really used
    keep[1] := 0;
    model.Forward(fresh, @ids, 1, 6, @keep);
    cmp := InferCompare(@fresh.Hidden[0], pointer(derived), h, 1e-6, 1e-6);
    CheckUtf8(not InferCompareOk(cmp), 'a masked real token changes the CLS');
    // a context grown by a larger call, then reused for a smaller one
    for i := 0 to 15 do
      big[i] := 5 + i;
    model.Forward(ctx, @big, 2, 8);
    CheckEqual(ctx.HiddenCount, 16 * h, 'hidden count of the larger call');
    model.Forward(ctx, @ids, 1, 6);
    CheckEqual(ctx.HiddenCount, 6 * h, 'hidden count follows the latest call');
    reused := copy(ctx.Hidden, 0, 6 * h);
    cmp := InferCompare(pointer(reused), pointer(derived), 3 * h, 0, 0);
    CheckUtf8(InferCompareOk(cmp), 'reused context = fresh context: %',
      [InferCompareText(cmp)]);
  finally
    fresh.Free;
    ctx.Free;
    model.Free;
  end;
end;

/// a tiny model copy in a temporary folder, with aConfig and aWeights
function TinyCopy(const aConfig: RawUtf8; const aWeights: RawByteString): TFileName;
begin
  result := IncludeTrailingPathDelimiter(TemporaryFileName);
  ForceDirectories(result);
  FileFromString(aConfig, result + 'config.json');
  FileFromString(aWeights, result + 'model.safetensors');
end;

procedure DropTinyCopy(const aFolder: TFileName);
begin
  DeleteFile(aFolder + 'config.json');
  DeleteFile(aFolder + 'model.safetensors');
  RemoveDir(aFolder);
end;

procedure TTestInferXlmr.LoadRefusesForeignModels;
var
  dir, tmp: TFileName;
  config: TDocVariantData;
  weights, header, extra: RawByteString;
  headerLen, dataLen: Int64;
  doc: TDocVariantData;
  json: RawUtf8;

  function Refuses(const aConfig: RawUtf8; const aWeights: RawByteString;
    const aReason: RawUtf8): boolean;
  var
    m: TXlmrModel;
  begin
    result := false;
    tmp := TinyCopy(aConfig, aWeights);
    try
      try
        m := TXlmrModel.Create(tmp, fCpu);
        m.Free;
      except
        on E: Exception do
          result := (E is EInferError) and
            (PosEx(aReason, StringToUtf8(E.Message)) > 0);
      end;
    finally
      DropTinyCopy(tmp);
    end;
  end;

begin
  dir := FindInferRefDir('tiny-xlmr');
  if CheckFailed(dir <> '', 'tiny-xlmr found') then
    exit;
  weights := StringFromFile(dir + 'model.safetensors');
  config.InitJsonFromFile(dir + 'config.json', JSON_FAST_FLOAT);
  CheckUtf8(not Refuses(config.ToJson, weights, 'x'), 'the fixture itself loads');
  // BERT numbers positions from 0: refused, not silently computed wrong
  config.U['model_type'] := 'bert';
  CheckUtf8(Refuses(config.ToJson, weights, 'xlm-roberta'), 'bert refused');
  config.U['model_type'] := 'xlm-roberta';
  // a missing pad_token_id would read as 0 instead of the default 1
  config.Delete('pad_token_id');
  CheckUtf8(Refuses(config.ToJson, weights, 'pad_token_id'),
    'missing pad_token_id refused');
  config.Clear;
  config.InitJsonFromFile(dir + 'config.json', JSON_FAST_FLOAT);
  config.U['hidden_act'] := 'gelu_new';
  CheckUtf8(Refuses(config.ToJson, weights, 'hidden_act'), 'tanh GELU refused');
  config.U['hidden_act'] := 'gelu';
  // a decoder has the same weight names but a causal mask
  config.B['is_decoder'] := true;
  CheckUtf8(Refuses(config.ToJson, weights, 'decoder'), 'decoder refused');
  config.B['is_decoder'] := false;
  // one more tensor than the architecture uses: a model we do not understand
  MoveFast(pointer(weights)^, headerLen, 8);
  header := copy(weights, 9, headerLen);
  dataLen := length(weights) - 8 - headerLen;
  doc.InitJson(header, JSON_FAST);
  doc.AddValue('extra.weight', _JsonFast(FormatUtf8(
    '{"dtype":"F32","shape":[1],"data_offsets":[%,%]}', [dataLen, dataLen + 4])));
  json := doc.ToJson;
  while (8 + length(json)) and 7 <> 0 do
    json := json + ' ';
  headerLen := length(json);
  SetLength(extra, 8);
  MoveFast(headerLen, pointer(extra)^, 8);
  extra := extra + json + copy(weights, 9 + length(header), dataLen) + #0#0#0#0;
  CheckUtf8(Refuses(config.ToJson, extra, 'understood'), 'unknown tensor refused');
end;

procedure TTestInferXlmr.BgeM3ForwardMatchesReference;
var
  dir, refdir: TFileName;
  model: TXlmrModel;
  ctx: TXlmrContext;
  ref: TInferRefSet;
  cases: PDocVariantData;
  i: PtrInt;
begin
  {$ifdef CPU32}
  AddConsole('SKIP BGE-M3 F32: the 2.3 GB file cannot be mapped whole in a ' +
    '32-bit process (phase 1a)');
  exit;
  {$endif CPU32}
  dir := FindBgeM3;
  if dir = '' then
  begin
    CheckUtf8(GetEnvironmentVariable('LAI_REQUIRE_MODEL') <> '1',
      'LAI_REQUIRE_MODEL=1 but no vendor/models/bge-m3/model.safetensors');
    AddConsole('SKIP BGE-M3 F32: no converted weights (tools/infer-ref, ' +
      'make_reference.py bge-m3)');
    exit;
  end;
  refdir := FindInferRefDir('bge-m3');
  if CheckFailed(refdir <> '', 'bge-m3 reference found') then
    exit;
  ref := nil;
  model := nil;
  ctx := nil;
  try
    ref := TInferRefSet.CreateFromFolder(refdir);
    model := TXlmrModel.Create(dir, fCpu);
    ctx := TXlmrContext.Create;
    CheckEqual(model.Config.Layers, 24, 'BGE-M3 depth');
    if CheckFailed(ref.Manifest^.GetAsArray('cases', cases), 'reference cases') then
      exit;
    for i := 0 to cases^.Count - 1 do
      CheckSingle(model, ctx, ref,
        FormatUtf8('case/%/', [_Safe(cases^.Values[i])^.U['id']]), 1e-4, 1e-3);
    CheckBatch(model, ctx, ref, 1e-4, 1e-3);
  finally
    ctx.Free;
    model.Free;
    ref.Free;
  end;
end;

procedure TTestInferXlmr.BgeM3TextToEmbedding;
var
  dir, refdir: TFileName;
  embedder: TInferEmbedder;
  ref: TInferRefSet;
  cases: PDocVariantData;
  texts: TRawUtf8DynArray;
  vectors: TSingleDynArray;
  expected: PSingleArray;
  cmp: TInferCompare;
  i, n, h: PtrInt;
  long: RawUtf8;
  refused: boolean;
begin
  {$ifdef CPU32}
  AddConsole('SKIP BGE-M3 text to embedding: 32-bit process (phase 1a)');
  exit;
  {$endif CPU32}
  dir := FindBgeM3;
  if (dir = '') or not FileExists(dir + 'tokenizer.json') then
  begin
    CheckUtf8(GetEnvironmentVariable('LAI_REQUIRE_MODEL') <> '1',
      'LAI_REQUIRE_MODEL=1 but no vendor/models/bge-m3');
    AddConsole('SKIP BGE-M3 text to embedding: no converted model');
    exit;
  end;
  refdir := FindInferRefDir('bge-m3');
  if CheckFailed(refdir <> '', 'bge-m3 reference found') then
    exit;
  ref := nil;
  embedder := nil;
  try
    ref := TInferRefSet.CreateFromFolder(refdir);
    embedder := TInferEmbedder.Create(dir, fCpu);
    h := embedder.Dim;
    if CheckFailed(ref.Manifest^.GetAsArray('cases', cases), 'reference cases') then
      exit;
    SetLength(texts, cases^.Count);
    for i := 0 to cases^.Count - 1 do
      texts[i] := _Safe(cases^.Values[i])^.U['text'];
    // all six texts as one right-padded batch, from text to vector
    vectors := embedder.Embed(texts);
    CheckEqual(length(vectors), cases^.Count * h, 'one vector per text');
    for i := 0 to cases^.Count - 1 do
    begin
      expected := ref.F32(FormatUtf8('case/%/embedding',
        [_Safe(cases^.Values[i])^.U['id']]), n);
      cmp := InferCompare(@vectors[i * h], expected, h, 1e-4, 1e-3);
      CheckUtf8(InferCompareOk(cmp) and (cmp.Cosine >= 0.9999),
        'text % to embedding: %', [_Safe(cases^.Values[i])^.U['id'],
        InferCompareText(cmp)]);
    end;
    // 513 tokens: refused, not cut
    long := '';
    for i := 1 to 255 do
      long := long + 'Rohr ';
    long := long + 'in';
    refused := false;
    try
      embedder.Embed([long]);
    except
      on EInferError do
        refused := true;
    end;
    CheckUtf8(refused, 'a text of 513 tokens is refused');
  finally
    embedder.Free;
    ref.Free;
  end;
end;

end.
