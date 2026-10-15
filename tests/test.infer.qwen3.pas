/// tests for the Qwen3 decoder of the native inference engine (phase 2b, 2c)
// - part of the mormot.ai.* extension (LandrixAI)
// - licensed under a MPL/GPL/LGPL three license - see LICENSE
// - 2b: every state and the logits against tests/data/infer-ref/tiny-qwen3
//   (committed, BF16 weights) and tests/data/infer-ref/qwen3 (Qwen3-0.6B, FP32)
// - 2c, the gate of phase 2: greedy continuations token-identical to
//   transformers; the KV cache gives the same logits as a full forward pass
// - the real model needs vendor/models/qwen3-0.6b (or $LAI_QWEN3_DIR) and a
//   64-bit process; it says SKIP otherwise, LAI_REQUIRE_MODEL=1 makes that a
//   failure
unit test.infer.qwen3;

interface

{$I mormot.defines.inc}

uses
  sysutils,
  mormot.core.base,
  mormot.core.os,
  mormot.core.unicode,
  mormot.core.text,
  mormot.core.variants,
  mormot.core.perf,
  mormot.core.test,
  mormot.ai.infer.tensor,
  mormot.ai.infer.backend,
  mormot.ai.infer.backend.cpu,
  mormot.ai.infer.model.qwen3,
  test.infer.ref,
  test.infer.bpe;

type
  /// phase 2b and 2c: the Qwen3 decoder against the references
  TTestInferQwen3 = class(TSynTestCase)
  protected
    fCpu: TInferCpuBackend;
    fFast: TInferCpuFastBackend;
    fNames: TRawUtf8DynArray;
    fStates: array of TSingleDynArray;
    procedure Setup; override;
    procedure CleanUp; override;
    /// TOnQwen3State: copy every state, the model reuses its buffers
    procedure KeepState(const aName: RawUtf8; aValues: PSingleArray;
      aRows, aCols: PtrInt);
    function State(const aName: RawUtf8): PSingleArray;
    /// forward <aPrefix>.ids from an empty cache and compare every kept state
    procedure CheckStates(aModel: TQwen3Model; aRef: TInferRefSet;
      const aPrefix: RawUtf8; aAtol, aRtol: double; aAllLogits: boolean);
    /// the real model folder, '' (and SKIP or a failure) when absent
    function RealModel: TFileName;
  published
    /// the tiny Qwen3 (BF16 weights): every state and all logits
    procedure TinyForwardMatchesReference;
    /// token by token through the KV cache = one forward pass, bit for bit
    procedure TinyCacheEqualsFullPass;
    /// the tiny greedy continuations are token-identical
    procedure TinyGreedyMatchesReference;
    /// wrong input and foreign configurations are refused
    procedure RefusesBadInput;
    /// 130 tokens: the cache grows twice and stays bit-identical to one pass
    procedure TinyLongContext;
    /// the context limit, Greedy edge cases, a context bound to its model
    procedure TinyLimitsAndContexts;
    /// a model with its own lm_head (tie_word_embeddings false)
    procedure TinyUntiedHead;
    /// InferArgMax: ties, NaN
    procedure ArgMaxTiesAndNan;
    /// Qwen3-0.6B: states of two prompts, last logits of all
    procedure Qwen3ForwardMatchesReference;
    /// Qwen3-0.6B: greedy continuations token-identical (the phase 2 gate)
    procedure Qwen3GreedyMatchesReference;
    /// the tiny Qwen3 with Q8_0 matrices: Exact = the scalar reference bit for
    // bit, the float path and the reference within a tolerance
    procedure TinyQ8;
    /// Qwen3-0.6B with Q8_0 matrices against the FP32 reference (phase 3):
    // logits within a tolerance, greedy equal up to the first near tie
    procedure Qwen3Q8AgainstReference;
  end;


implementation

{ TTestInferQwen3 }

procedure TTestInferQwen3.Setup;
begin
  fCpu := TInferCpuBackend.Create;
  fFast := TInferCpuFastBackend.Create;
end;

procedure TTestInferQwen3.CleanUp;
begin
  FreeAndNil(fFast); // re-entrant, as mORMot requires
  FreeAndNil(fCpu);
end;

procedure TTestInferQwen3.KeepState(const aName: RawUtf8; aValues: PSingleArray;
  aRows, aCols: PtrInt);
var
  n: PtrInt;
begin
  n := length(fNames);
  SetLength(fNames, n + 1);
  SetLength(fStates, n + 1);
  fNames[n] := aName;
  SetLength(fStates[n], aRows * aCols);
  MoveFast(aValues^, fStates[n][0], aRows * aCols * SizeOf(single));
end;

function TTestInferQwen3.State(const aName: RawUtf8): PSingleArray;
var
  i: PtrInt;
begin
  for i := 0 to high(fNames) do
    if fNames[i] = aName then
    begin
      result := pointer(fStates[i]);
      exit;
    end;
  result := nil;
end;

procedure TTestInferQwen3.CheckStates(aModel: TQwen3Model; aRef: TInferRefSet;
  const aPrefix: RawUtf8; aAtol, aRtol: double; aAllLogits: boolean);
var
  ctx: TQwen3Context;
  ids, n, i, count, vocab: PtrInt;
  idp: PIntegerArray;
  logits: TSingleDynArray;
  ref, mine: PSingleArray;
  cmp: TInferCompare;
  t: PInferRefTensor;
  name: RawUtf8;
begin
  fNames := nil;
  fStates := nil;
  idp := aRef.I32(aPrefix + '.ids', ids);
  vocab := aModel.Config.VocabSize;
  if aAllLogits then
    SetLength(logits, ids * vocab)
  else
    SetLength(logits, vocab);
  ctx := aModel.NewContext;
  try
    aModel.Forward(ctx, idp, ids, pointer(logits), aAllLogits, KeepState);
    CheckEqual(ctx.Length, ids, 'tokens cached');
  finally
    ctx.Free;
  end;
  for i := 0 to high(aRef.Tensors) do
  begin
    t := @aRef.Tensors[i];
    if copy(t^.Name, 1, length(aPrefix) + 1) <> aPrefix + '.' then
      continue;
    name := copy(t^.Name, length(aPrefix) + 2, maxInt);
    if (name = 'ids') or (name = 'greedy') or (name = 'greedy_margin') then
      continue;
    ref := aRef.F32(t^.Name, n);
    if name = 'logits' then
      mine := pointer(logits)
    else if name = 'logits_last' then
      mine := pointer(logits)
    else
      mine := State(name);
    if CheckFailed(mine <> nil, Utf8ToString(aPrefix + ' has no state ' + name)) then
      continue;
    count := n;
    cmp := InferCompare(mine, ref, count, aAtol, aRtol);
    CheckUtf8(InferCompareOk(cmp), '%.%: %', [aPrefix, name, InferCompareText(cmp)]);
    if (name = 'logits') or (name = 'logits_last') or (name = 'norm') then
      AddConsole(FormatString('%.%: %', [aPrefix, name, InferCompareText(cmp)]));
  end;
end;

procedure TTestInferQwen3.TinyForwardMatchesReference;
var
  dir: TFileName;
  ref: TInferRefSet;
  model: TQwen3Model;
  k: integer;
begin
  dir := FindInferRefDir('tiny-qwen3');
  if CheckFailed(dir <> '', 'tiny-qwen3 found') then
    exit;
  ref := TInferRefSet.CreateFromFolder(dir);
  model := nil;
  try
    model := TQwen3Model.Create(dir, fCpu);
    CheckEqual(model.Config.HeadDim, 32, 'head_dim from the configuration');
    CheckEqual(model.Config.KvHeads, 2, 'grouped-query attention');
    Check(model.Config.TiedEmbeddings, 'tied embeddings');
    for k := 0 to 2 do
      CheckStates(model, ref, FormatUtf8('case%', [k]), 2e-6, 2e-5, true);
  finally
    model.Free;
    ref.Free;
  end;
end;

procedure TTestInferQwen3.TinyCacheEqualsFullPass;
var
  dir: TFileName;
  ref: TInferRefSet;
  model: TQwen3Model;
  full, step: TQwen3Context;
  ids: PIntegerArray;
  n, i, v, vocab: PtrInt;
  all, one: TSingleDynArray;
  same: boolean;
begin
  dir := FindInferRefDir('tiny-qwen3');
  if CheckFailed(dir <> '', 'tiny-qwen3 found') then
    exit;
  ref := TInferRefSet.CreateFromFolder(dir);
  model := nil;
  full := nil;
  step := nil;
  try
    model := TQwen3Model.Create(dir, fCpu);
    vocab := model.Config.VocabSize;
    ids := ref.I32('case2.ids', n);
    full := model.NewContext;
    step := model.NewContext;
    SetLength(all, n * vocab);
    SetLength(one, vocab);
    model.Forward(full, ids, n, pointer(all), true);
    same := true;
    for i := 0 to n - 1 do
    begin
      // one token at a time: the earlier ones come from the KV cache
      model.Forward(step, @ids^[i], 1, pointer(one));
      for v := 0 to vocab - 1 do
        if PCardinal(@one[v])^ <> PCardinal(@all[i * vocab + v])^ then
          same := false;
    end;
    Check(same, 'cached logits bit-identical to the full pass');
    CheckEqual(step.Length, n, 'cache length');
    // the same context, reset: the same result again
    step.Reset;
    model.Forward(step, ids, n, pointer(one));
    same := CompareMem(pointer(one), @all[(n - 1) * vocab], vocab * SizeOf(single));
    Check(same, 'context reused after Reset');
    // prefill in two chunks
    step.Reset;
    model.Forward(step, ids, 4, nil);
    model.Forward(step, @ids^[4], n - 4, pointer(one));
    same := CompareMem(pointer(one), @all[(n - 1) * vocab], vocab * SizeOf(single));
    Check(same, 'prefill in two chunks');
  finally
    step.Free;
    full.Free;
    model.Free;
    ref.Free;
  end;
end;

/// ids as text, for messages
function IdsOf(const aIds: TIntegerDynArray): RawUtf8;
var
  i: PtrInt;
begin
  result := '';
  for i := 0 to high(aIds) do
    result := result + Int32ToUtf8(aIds[i]) + ' ';
end;

/// the greedy tokens of aPrefix against the reference: the first differing
// step, -1 when all agree; aMargin receives the reference top-2 margin there
function GreedyDiff(const aMine: TIntegerDynArray; aRef: TInferRefSet;
  const aPrefix: RawUtf8; out aMargin: double; out aDetail: RawUtf8): PtrInt;
var
  want: PIntegerArray;
  margins: PSingleArray;
  n, m, i: PtrInt;
begin
  want := aRef.I32(aPrefix + '.greedy', n);
  margins := aRef.F32(aPrefix + '.greedy_margin', m);
  aMargin := 0;
  result := -1;
  for i := 0 to n - 1 do
    if (i >= length(aMine)) or (aMine[i] <> want^[i]) then
    begin
      result := i;
      if i < m then
        aMargin := margins^[i];
      break;
    end;
  if (result < 0) and (length(aMine) <> n) then
    result := n;
  aDetail := FormatUtf8('% tokens, reference %', [length(aMine), n]);
end;

procedure TTestInferQwen3.TinyGreedyMatchesReference;
var
  dir: TFileName;
  ref: TInferRefSet;
  model: TQwen3Model;
  ctx: TQwen3Context;
  ids: PIntegerArray;
  n, k, diff: PtrInt;
  prompt, mine: TIntegerDynArray;
  margin: double;
  detail: RawUtf8;
begin
  dir := FindInferRefDir('tiny-qwen3');
  if CheckFailed(dir <> '', 'tiny-qwen3 found') then
    exit;
  ref := TInferRefSet.CreateFromFolder(dir);
  model := nil;
  try
    model := TQwen3Model.Create(dir, fCpu);
    for k := 0 to 2 do
    begin
      ids := ref.I32(FormatUtf8('case%.ids', [k]), n);
      SetLength(prompt, n);
      MoveFast(ids^, prompt[0], n * SizeOf(integer));
      ctx := model.NewContext;
      try
        mine := model.Greedy(ctx, prompt, 16, [2]);
      finally
        ctx.Free;
      end;
      diff := GreedyDiff(mine, ref, FormatUtf8('case%', [k]), margin, detail);
      CheckUtf8(diff < 0, 'case% differs at step % (margin %, %)',
        [k, diff, margin, detail]);
    end;
  finally
    model.Free;
    ref.Free;
  end;
end;

function Raises(aModel: TQwen3Model; aCtx: TQwen3Context; aId, aCount: integer): boolean;
var
  ids: array[0..1] of integer;
begin
  result := false;
  ids[0] := aId;
  ids[1] := aId;
  try
    aModel.Forward(aCtx, @ids, aCount, nil);
  except
    on EInferError do
      result := true;
  end;
end;

procedure TTestInferQwen3.RefusesBadInput;
var
  dir, tmp: TFileName;
  model: TQwen3Model;
  ctx: TQwen3Context;
  cfg: RawUtf8;
  refused: boolean;
  st, bad: RawByteString;
  hdr: variant;
  pos: PtrInt;

  function LoadRefused(const aFrom, aTo: RawUtf8): boolean;
  begin
    result := false;
    FileFromString(StringReplaceAll(cfg, aFrom, aTo), tmp + 'config.json');
    try
      TQwen3Model.Create(tmp, fCpu).Free;
    except
      on EInferError do
        result := true;
    end;
  end;

begin
  dir := FindInferRefDir('tiny-qwen3');
  if CheckFailed(dir <> '', 'tiny-qwen3 found') then
    exit;
  model := TQwen3Model.Create(dir, fCpu);
  ctx := model.NewContext;
  try
    Check(Raises(model, ctx, -1, 1), 'negative id');
    Check(Raises(model, ctx, model.Config.VocabSize, 1), 'id beyond the vocabulary');
    Check(Raises(model, ctx, 3, 0), 'no tokens');
    CheckEqual(ctx.Length, 0, 'nothing cached by refused calls');
    refused := false;
    try
      model.Forward(nil, nil, 1, nil);
    except
      on EInferError do
        refused := true;
    end;
    Check(refused, 'no context');
  finally
    ctx.Free;
    model.Free;
  end;
  // foreign configurations: a copy of the tiny model with one setting changed
  cfg := StringFromFile(dir + 'config.json');
  tmp := MakePath([GetSystemPath(spTemp), 'lai-qwen3-' + Int64ToUtf8(GetTickCount64)], true);
  ForceDirectories(tmp);
  try
    CopyFile(dir + 'model.safetensors', tmp + 'model.safetensors', false);
    Check(not LoadRefused('"x"', '"x"'), 'the copy loads');
    Check(LoadRefused('"silu"', '"gelu"'), 'other activation');
    Check(LoadRefused('"use_sliding_window": false', '"use_sliding_window": true'),
      'sliding window');
    Check(LoadRefused('"head_dim": 32', '"head_dim": 16'), 'head_dim not the weights''');
    Check(LoadRefused('"num_key_value_heads": 2', '"num_key_value_heads": 3'),
      'heads not a multiple of the KV heads');
    Check(LoadRefused('"rope_type": "default"', '"rope_type": "yarn"'), 'scaled RoPE');
    Check(LoadRefused('"qwen3"', '"qwen2"'), 'other model type');
    Check(LoadRefused('"tie_word_embeddings": true', '"tie_word_embeddings": false'),
      'untied without lm_head.weight');
    Check(LoadRefused('"head_dim": 32,', ''), 'no head_dim');
    // a NaN weight: F32 carries it along, Q8_0 has no form for it
    FileFromString(cfg, tmp + 'config.json');
    st := StringFromFile(dir + 'model.safetensors');
    hdr := _JsonFast(copy(st, 9, PInteger(st)^));
    pos := 8 + PInteger(st)^ + _Safe(hdr)^.O['model.layers.0.self_attn.q_proj.weight']^.
      A['data_offsets']^.Values[0];
    FastSetRawByteString(bad, pointer(st), length(st)); // a copy of its own
    PWord(@PByteArray(bad)^[pos])^ := $7fc0; // BF16 quiet NaN
    FileFromString(bad, tmp + 'model.safetensors');
    TQwen3Model.Create(tmp, fCpu).Free;
    refused := false;
    try
      TQwen3Model.Create(tmp, fCpu, idtQ8_0).Free;
    except
      on E: EInferError do
        refused := PosEx('NaN or infinite weight', StringToUtf8(E.Message)) > 0;
    end;
    Check(refused, 'Q8_0 refuses a NaN weight, by name');
    // finite but beyond the Q8_0 range: 9.96e6 / 127 exceeds the largest FP16
    PWord(@PByteArray(bad)^[pos])^ := $4b18;
    FileFromString(bad, tmp + 'model.safetensors');
    TQwen3Model.Create(tmp, fCpu).Free;
    refused := false;
    try
      TQwen3Model.Create(tmp, fCpu, idtQ8_0).Free;
    except
      on E: EInferError do
        refused := PosEx('beyond the Q8_0 range', StringToUtf8(E.Message)) > 0;
    end;
    Check(refused, 'Q8_0 refuses a weight beyond its range, by name');
  finally
    DirectoryDeleteAll(tmp);
    RemoveDir(tmp);
  end;
end;

/// aCount deterministic pseudo-random ids in 3..aVocab-1
function RandomIds(aCount, aVocab: PtrInt): TIntegerDynArray;
var
  i: PtrInt;
  x: cardinal;
begin
  SetLength(result, aCount);
  x := 20261015;
  for i := 0 to aCount - 1 do
  begin
    x := x * 1103515245 + 12345;
    result[i] := 3 + integer((x shr 8) mod cardinal(aVocab - 3));
  end;
end;

procedure TTestInferQwen3.TinyLongContext;
var
  dir: TFileName;
  model: TQwen3Model;
  full, step: TQwen3Context;
  ids: TIntegerDynArray;
  all, one: TSingleDynArray;
  n, i, vocab: PtrInt;
  same: boolean;
begin
  dir := FindInferRefDir('tiny-qwen3');
  if CheckFailed(dir <> '', 'tiny-qwen3 found') then
    exit;
  model := TQwen3Model.Create(dir, fCpu);
  full := model.NewContext;
  step := model.NewContext;
  try
    n := 130; // the cache starts at 64 tokens: it grows to 128, then 256
    vocab := model.Config.VocabSize;
    ids := RandomIds(n, vocab);
    SetLength(all, n * vocab);
    SetLength(one, vocab);
    model.Forward(full, pointer(ids), n, pointer(all), true);
    same := true;
    for i := 0 to n - 1 do
    begin
      model.Forward(step, @ids[i], 1, pointer(one));
      if not CompareMem(pointer(one), @all[i * vocab], vocab * SizeOf(single)) then
        same := false;
    end;
    Check(same, 'token by token over 130 positions = one pass, bit for bit');
    // a fresh context: the chunks themselves must grow the cache (64 -> 128 -> 256)
    FreeAndNil(step);
    step := model.NewContext;
    model.Forward(step, pointer(ids), 50, nil);
    model.Forward(step, @ids[50], 50, nil);
    model.Forward(step, @ids[100], n - 100, pointer(one));
    Check(CompareMem(pointer(one), @all[(n - 1) * vocab], vocab * SizeOf(single)),
      'three prefill chunks across the growth');
  finally
    step.Free;
    full.Free;
    model.Free;
  end;
end;

procedure TTestInferQwen3.TinyLimitsAndContexts;
var
  dir: TFileName;
  a, b: TQwen3Model;
  ctx, fresh: TQwen3Context;
  ids, g1, g2, g3, joined: TIntegerDynArray;
  limit, i: PtrInt;
  one: integer;
  refused: boolean;
begin
  dir := FindInferRefDir('tiny-qwen3');
  if CheckFailed(dir <> '', 'tiny-qwen3 found') then
    exit;
  a := TQwen3Model.Create(dir, fCpu);
  b := TQwen3Model.Create(dir, fCpu);
  ctx := a.NewContext;
  fresh := a.NewContext;
  try
    limit := a.Config.MaxPositions;
    // Greedy: no new tokens wanted is an error, before anything is cached
    refused := false;
    try
      a.Greedy(ctx, [5, 6], 0, []);
    except
      on EInferError do
        refused := true;
    end;
    Check(refused and (ctx.Length = 0), 'aMaxNew = 0 refused, nothing cached');
    // returned tokens enter the cache: a conversation can go on
    g1 := a.Greedy(ctx, [5, 6, 7], 4, []);
    CheckEqual(ctx.Length, 3 + 4, 'prompt and every returned token cached');
    g2 := a.Greedy(ctx, [9], 4, []);
    joined := nil;
    SetLength(joined, 3 + length(g1) + 1);
    joined[0] := 5;
    joined[1] := 6;
    joined[2] := 7;
    for i := 0 to high(g1) do
      joined[3 + i] := g1[i];
    joined[high(joined)] := 9;
    g3 := a.Greedy(fresh, joined, 4, []);
    CheckEqual(IdsOf(g2), IdsOf(g3), 'continued = fed again from scratch');
    // a context of model A is no context of model B, even of the same shape
    refused := false;
    try
      one := 5;
      b.Forward(ctx, @one, 1, nil);
    except
      on EInferError do
        refused := true;
    end;
    Check(refused, 'another model''s cache refused');
    ctx.Reset;
    one := 5;
    b.Forward(ctx, @one, 1, nil);
    CheckEqual(ctx.Length, 1, 'after Reset the other model takes it over');
    // the context limit: Greedy stops there, Forward refuses beyond it
    ctx.Reset;
    ids := RandomIds(limit - 12, b.Config.VocabSize);
    g1 := b.Greedy(ctx, ids, 100, []);
    CheckEqual(length(g1), 13, '12 positions left: 13 tokens, the last uncached');
    CheckEqual(ctx.Length, limit, 'context full');
    refused := false;
    try
      one := 5;
      b.Forward(ctx, @one, 1, nil);
    except
      on EInferError do
        refused := true;
    end;
    Check(refused and (ctx.Length = limit), 'no token beyond MaxPositions');
    // a model freed and another created, maybe at the same address: the cache
    // of the first is still not the second one's
    fresh.Reset;
    one := 5;
    a.Forward(fresh, @one, 1, nil);
    FreeAndNil(a);
    a := TQwen3Model.Create(dir, fCpu);
    refused := false;
    try
      a.Forward(fresh, @one, 1, nil);
    except
      on EInferError do
        refused := true;
    end;
    Check(refused, 'a freed model''s cache refused by a new instance');
  finally
    fresh.Free;
    ctx.Free;
    b.Free;
    a.Free;
  end;
end;

procedure TTestInferQwen3.TinyUntiedHead;
var
  dir: TFileName;
  ref: TInferRefSet;
  model: TQwen3Model;
begin
  dir := FindInferRefDir('tiny-qwen3');
  if CheckFailed(dir <> '', 'tiny-qwen3 found') then
    exit;
  dir := MakePath([dir, 'untied'], true);
  ref := TInferRefSet.CreateFromFolder(dir);
  model := nil;
  try
    model := TQwen3Model.Create(dir, fCpu);
    Check(not model.Config.TiedEmbeddings, 'untied');
    CheckStates(model, ref, 'case0', 2e-6, 2e-5, true);
  finally
    model.Free;
    ref.Free;
  end;
end;

function SingleBits(aBits: cardinal): single;
begin
  PCardinal(@result)^ := aBits;
end;

procedure TTestInferQwen3.ArgMaxTiesAndNan;
var
  v: array[0..3] of single;
begin
  v[0] := 1;
  v[1] := 3;
  v[2] := 3;
  v[3] := -1;
  CheckEqual(InferArgMax(@v, 4), 1, 'a tie goes to the lowest index');
  v[0] := SingleBits($7fc00000); // NaN
  v[1] := 1;
  v[2] := 1;
  CheckEqual(InferArgMax(@v, 3), 1, 'a NaN is not chosen');
  v[1] := SingleBits($7fc00000);
  v[2] := SingleBits($ffc00000);
  CheckEqual(InferArgMax(@v, 3), 0, 'all NaN: the first');
  v[1] := SingleBits($ff800000); // -Inf
  v[2] := SingleBits($7fc00000);
  CheckEqual(InferArgMax(@v, 3), 1, '-Inf beats a NaN');
end;

function TTestInferQwen3.RealModel: TFileName;
begin
  result := '';
  {$ifdef CPU32}
  AddConsole('SKIP Qwen3-0.6B: 2.4 GB of FP32 weights need a 64-bit process');
  exit;
  {$endif CPU32}
  result := FindQwen3Folder('model.safetensors');
  if result <> '' then
    exit;
  CheckUtf8(GetEnvironmentVariable('LAI_REQUIRE_MODEL') <> '1',
    'LAI_REQUIRE_MODEL=1 but no vendor/models/qwen3-0.6b/model.safetensors');
  AddConsole('SKIP Qwen3-0.6B: no vendor/models/qwen3-0.6b (tools/infer-ref, ' +
    'make_reference_qwen3.py)');
end;

procedure TTestInferQwen3.Qwen3ForwardMatchesReference;
var
  dir, refdir: TFileName;
  ref: TInferRefSet;
  model: TQwen3Model;
  prompts, p: PDocVariantData;
  i: PtrInt;
begin
  dir := RealModel;
  if dir = '' then
    exit;
  refdir := FindInferRefDir('qwen3');
  if CheckFailed(refdir <> '', 'qwen3 reference found') then
    exit;
  ref := TInferRefSet.CreateFromFolder(refdir);
  model := nil;
  try
    model := TQwen3Model.Create(dir, fFast);
    CheckEqual(model.Config.Layers, 28, 'Qwen3-0.6B depth');
    CheckEqual(model.Config.HeadDim, 128, 'head_dim 128, not 1024 / 16');
    prompts := ref.Manifest^.A['prompts'];
    for i := 0 to prompts^.Count - 1 do
    begin
      p := _Safe(prompts^.Values[i]);
      CheckStates(model, ref, p^.U['id'], 2e-3, 2e-3, false);
    end;
  finally
    model.Free;
    ref.Free;
  end;
end;

procedure TTestInferQwen3.TinyQ8;
var
  dir: TFileName;
  ref: TInferRefSet;
  exact: TInferCpuFastBackend;
  a, b, c: TQwen3Model;
  ca, cb, cc: TQwen3Context;
  la, lb, lc: TSingleDynArray;
  ids: PIntegerArray;
  n, m, k, v: PtrInt;
  cmp: TInferCompare;
begin
  dir := FindInferRefDir('tiny-qwen3');
  if CheckFailed(dir <> '', 'tiny-qwen3 found') then
    exit;
  ref := TInferRefSet.CreateFromFolder(dir);
  exact := TInferCpuFastBackend.Create;
  exact.Exact := true;
  a := nil;
  b := nil;
  c := nil;
  try
    a := TQwen3Model.Create(dir, fCpu, idtQ8_0);
    b := TQwen3Model.Create(dir, exact, idtQ8_0);
    c := TQwen3Model.Create(dir, fFast, idtQ8_0);
    Check(a.Weights = idtQ8_0, 'Q8_0 matrices');
    v := a.Config.VocabSize;
    for k := 0 to 2 do
    begin
      ids := ref.I32(FormatUtf8('case%.ids', [k]), n);
      SetLength(la, n * v);
      SetLength(lb, n * v);
      SetLength(lc, n * v);
      ca := a.NewContext;
      cb := b.NewContext;
      cc := c.NewContext;
      try
        a.Forward(ca, ids, n, pointer(la), true);
        b.Forward(cb, ids, n, pointer(lb), true);
        c.Forward(cc, ids, n, pointer(lc), true);
      finally
        ca.Free;
        cb.Free;
        cc.Free;
      end;
      Check(CompareMem(pointer(la), pointer(lb), n * v * SizeOf(single)),
        'Exact on all cores = the scalar reference, bit for bit');
      cmp := InferCompare(pointer(lc), pointer(la), n * v, 2e-2, 1e-2);
      CheckUtf8(InferCompareOk(cmp), 'case % float path: %', [k, InferCompareText(cmp)]);
      cmp := InferCompare(pointer(la), ref.F32(FormatUtf8('case%.logits', [k]), m),
        n * v, 0.05, 0.05);
      CheckUtf8(InferCompareOk(cmp) and (cmp.Cosine > 0.999), 'case % Q8_0 vs FP32: %',
        [k, InferCompareText(cmp)]);
    end;
    // the untied head: its own matrix quantized, not the embedding
    FreeAndNil(c);
    FreeAndNil(b);
    FreeAndNil(a);
    // against the stored logits of the untied reference: an F32 model of the
    // same code would share a mistake in the head
    dir := MakePath([dir, 'untied'], true);
    ref.Free;
    ref := TInferRefSet.CreateFromFolder(dir);
    a := TQwen3Model.Create(dir, fCpu, idtQ8_0);
    Check(not a.Config.TiedEmbeddings, 'untied');
    ids := ref.I32('case0.ids', n);
    v := a.Config.VocabSize;
    SetLength(la, n * v);
    ca := a.NewContext;
    try
      a.Forward(ca, ids, n, pointer(la), true);
    finally
      ca.Free;
    end;
    cmp := InferCompare(pointer(la), ref.F32('case0.logits', m), n * v, 0.05, 0.05);
    CheckUtf8(InferCompareOk(cmp) and (cmp.Cosine > 0.999),
      'untied Q8_0 vs the FP32 reference: %', [InferCompareText(cmp)]);
  finally
    c.Free;
    b.Free;
    a.Free;
    exact.Free;
    ref.Free;
  end;
end;

procedure TTestInferQwen3.Qwen3Q8AgainstReference;
var
  dir, refdir: TFileName;
  ref: TInferRefSet;
  model: TQwen3Model;
  ctx: TQwen3Context;
  prompts, p: PDocVariantData;
  ids, want: PIntegerArray;
  margins: PSingleArray;
  i, n, m, diff: PtrInt;
  prompt, mine: TIntegerDynArray;
  logits: TSingleDynArray;
  cmp: TInferCompare;
  margin: double;
  timer: TPrecisionTimer;
begin
  dir := RealModel;
  if dir = '' then
    exit;
  refdir := FindInferRefDir('qwen3');
  if CheckFailed(refdir <> '', 'qwen3 reference found') then
    exit;
  ref := TInferRefSet.CreateFromFolder(refdir);
  model := nil;
  try
    timer.Start;
    model := TQwen3Model.Create(dir, fFast, idtQ8_0);
    AddConsole(FormatString('Q8_0 model quantized in %', [timer.Stop]));
    prompts := ref.Manifest^.A['prompts'];
    for i := 0 to prompts^.Count - 1 do
    begin
      p := _Safe(prompts^.Values[i]);
      ids := ref.I32(p^.U['id'] + '.ids', n);
      SetLength(prompt, n);
      MoveFast(ids^, prompt[0], n * SizeOf(integer));
      SetLength(logits, model.Config.VocabSize);
      ctx := model.NewContext;
      try
        model.Forward(ctx, pointer(prompt), n, pointer(logits));
      finally
        ctx.Free;
      end;
      cmp := InferCompare(pointer(logits), ref.F32(p^.U['id'] + '.logits_last', m),
        model.Config.VocabSize, 1, 0);
      want := ref.I32(p^.U['id'] + '.greedy', m);
      margins := ref.F32(p^.U['id'] + '.greedy_margin', m);
      ctx := model.NewContext;
      try
        mine := model.Greedy(ctx, prompt, m, [151645, 151643]);
      finally
        ctx.Free;
      end;
      diff := 0;
      while (diff < m) and (diff < length(mine)) and (mine[diff] = want^[diff]) do
        inc(diff);
      margin := 0;
      if diff < m then
        margin := margins^[diff];
      if (diff < m) and (diff < length(mine)) then
        AddConsole(FormatString('% Q8_0: logits cos % maxabs %, greedy equal for ' +
          '% of % tokens, reference margin there % (Q8_0 took %, reference %)',
          [p^.U['id'], cmp.Cosine, cmp.MaxAbs, diff, m, margin, mine[diff],
           want^[diff]]))
      else if diff < m then
        AddConsole(FormatString('% Q8_0: logits cos % maxabs %, stopped after % ' +
          'of % tokens, reference margin there %', [p^.U['id'], cmp.Cosine,
          cmp.MaxAbs, diff, m, margin]))
      else
        AddConsole(FormatString('% Q8_0: logits cos % maxabs %, greedy equal for ' +
          'all % tokens', [p^.U['id'], cmp.Cosine, cmp.MaxAbs, m]));
      // the gate of Q8_0, measured 2026-10-15 on the 193 tokens of chat_tool,
      // the most sensitive prompt: cosine 0.9935 and max error 1.72 in exact
      // (double) arithmetic - the same bits on aarch64 and x86_64 - and 0.9965 /
      // 1.36 with NEON, 0.9868 / 2.11 with AVX2; float against exact Q8_0
      // alone: cosine 0.998 to 0.999, up to 0.95 apart on both platforms (likely
      // because a changed rounding flips single int8 activations of the next
      // layer - not measured inside).
      // Greedy leaves the FP32 tokens only where the reference itself was less
      // than 0.28 sure. A broken quantizer fails all three by far (shifted rows:
      // cosine 0.23, max error 24)
      CheckUtf8(cmp.Cosine >= 0.98, '% Q8_0 logits cosine %', [p^.U['id'], cmp.Cosine]);
      CheckUtf8(cmp.MaxAbs <= 3, '% Q8_0 logits max error %', [p^.U['id'], cmp.MaxAbs]);
      CheckUtf8((diff >= m) or (margin < 0.5),
        '% Q8_0 greedy leaves the reference at step % where it was % sure',
        [p^.U['id'], diff, margin]);
    end;
  finally
    model.Free;
    ref.Free;
  end;
end;

procedure TTestInferQwen3.Qwen3GreedyMatchesReference;
var
  dir, refdir: TFileName;
  ref: TInferRefSet;
  model: TQwen3Model;
  ctx: TQwen3Context;
  prompts, p: PDocVariantData;
  ids: PIntegerArray;
  i, n, diff: PtrInt;
  prompt, mine: TIntegerDynArray;
  margin: double;
  detail: RawUtf8;
  timer: TPrecisionTimer;
begin
  dir := RealModel;
  if dir = '' then
    exit;
  refdir := FindInferRefDir('qwen3');
  if CheckFailed(refdir <> '', 'qwen3 reference found') then
    exit;
  ref := TInferRefSet.CreateFromFolder(refdir);
  model := nil;
  try
    model := TQwen3Model.Create(dir, fFast);
    prompts := ref.Manifest^.A['prompts'];
    for i := 0 to prompts^.Count - 1 do
    begin
      p := _Safe(prompts^.Values[i]);
      ids := ref.I32(p^.U['id'] + '.ids', n);
      SetLength(prompt, n);
      MoveFast(ids^, prompt[0], n * SizeOf(integer));
      ctx := model.NewContext;
      try
        timer.Start;
        mine := model.Greedy(ctx, prompt, 48, [151645, 151643]);
        timer.Stop;
      finally
        ctx.Free;
      end;
      diff := GreedyDiff(mine, ref, p^.U['id'], margin, detail);
      CheckUtf8(diff < 0, '% differs at step % (reference margin %, %)',
        [p^.U['id'], diff, margin, detail]);
      AddConsole(FormatString('% greedy: % + % tokens in %, identical',
        [p^.U['id'], n, length(mine), timer.Time]));
    end;
  finally
    model.Free;
    ref.Free;
  end;
end;

end.
