// LandrixAI - speed of the native Qwen3 decoder (phase 3 of
// docs/native-inference.md)
// - feeds a prompt of a given length in one pass (prefill), then generates
//   tokens one by one through the KV cache (decode), greedy, and reports
//   tokens/s of both, the share of the matrix products (backend Linear) and
//   the peak memory
//
//   usage: llm-bench [model folder] [prompt tokens=200] [new tokens=32]
//                    [threads=0 (default) | ref | exact] [q8 (default) | f32]
//   - ref: the scalar reference backend; exact: TInferCpuFastBackend with
//     Exact = true; otherwise TInferCpuFastBackend in its float mode
//   e.g.   llm-bench vendor/models/qwen3-0.6b 200 32
program llm.bench;

{$I mormot.defines.inc}

{$ifdef OSWINDOWS}
  {$apptype console}
{$endif OSWINDOWS}

uses
  {$I mormot.uses.inc}
  {$ifdef OSWINDOWS}
  windows,
  {$endif OSWINDOWS}
  sysutils,
  mormot.core.base,
  mormot.core.os,
  mormot.core.unicode,
  mormot.core.text,
  mormot.core.perf,
  mormot.ai.infer.tensor,
  mormot.ai.infer.backend,
  mormot.ai.infer.backend.cpu,
  mormot.ai.infer.kernels,
  mormot.ai.infer.tokenizer.bpe,
  mormot.ai.infer.model.qwen3,
  mormot.ai.infer.sampler;

type
  /// forwards to an inner backend and sums the time of Linear and Add
  TTimedBackend = class(TInferBackend)
  protected
    fInner: TInferBackend;
  public
    LinearUs, AddUs: Int64;
    LinearCalls: Int64;
    constructor Create(aInner: TInferBackend);
    function Supports(aOp: TInferOp; aWeights: TInferDType): boolean; override;
    procedure Linear(x: PSingleArray; aRows, aIn: PtrInt; const w: TInferWeight;
      aOut: PtrInt; bias: PSingleArray; y: PSingleArray); override;
    procedure Add(x, y: PSingleArray; aCount: PtrInt); override;
    procedure LayerNorm(x: PSingleArray; aRows, aCols: PtrInt;
      gamma, beta: PSingleArray; aEps: double; y: PSingleArray); override;
    procedure Gelu(x: PSingleArray; aCount: PtrInt); override;
    procedure SoftmaxMasked(x: PSingleArray; aRows, aCols: PtrInt;
      aKeep: PIntegerArray); override;
    procedure Attention(q, k, v: PSingleArray; aSeq, aHeads, aHeadDim: PtrInt;
      aKeep: PIntegerArray; aScratch, y: PSingleArray); override;
    procedure L2Normalize(x: PSingleArray; aCount: PtrInt); override;
    procedure CausalAttention(q: PSingleArray; aRows, aHeads, aHeadDim: PtrInt;
      keys, values: PSingleArray; aPast, aKvHeads: PtrInt; y: PSingleArray); override;
    procedure Reset;
  end;

constructor TTimedBackend.Create(aInner: TInferBackend);
begin
  inherited Create;
  fInner := aInner;
end;

function TTimedBackend.Supports(aOp: TInferOp; aWeights: TInferDType): boolean;
begin
  result := fInner.Supports(aOp, aWeights);
end;

procedure TTimedBackend.Linear(x: PSingleArray; aRows, aIn: PtrInt;
  const w: TInferWeight; aOut: PtrInt; bias: PSingleArray; y: PSingleArray);
var
  t0, t1: Int64;
begin
  QueryPerformanceMicroSeconds(t0);
  fInner.Linear(x, aRows, aIn, w, aOut, bias, y);
  QueryPerformanceMicroSeconds(t1);
  inc(LinearUs, t1 - t0);
  inc(LinearCalls);
end;

procedure TTimedBackend.Add(x, y: PSingleArray; aCount: PtrInt);
var
  t0, t1: Int64;
begin
  QueryPerformanceMicroSeconds(t0);
  fInner.Add(x, y, aCount);
  QueryPerformanceMicroSeconds(t1);
  inc(AddUs, t1 - t0);
end;

procedure TTimedBackend.LayerNorm(x: PSingleArray; aRows, aCols: PtrInt;
  gamma, beta: PSingleArray; aEps: double; y: PSingleArray);
begin
  fInner.LayerNorm(x, aRows, aCols, gamma, beta, aEps, y);
end;

procedure TTimedBackend.Gelu(x: PSingleArray; aCount: PtrInt);
begin
  fInner.Gelu(x, aCount);
end;

procedure TTimedBackend.SoftmaxMasked(x: PSingleArray; aRows, aCols: PtrInt;
  aKeep: PIntegerArray);
begin
  fInner.SoftmaxMasked(x, aRows, aCols, aKeep);
end;

procedure TTimedBackend.Attention(q, k, v: PSingleArray; aSeq, aHeads,
  aHeadDim: PtrInt; aKeep: PIntegerArray; aScratch, y: PSingleArray);
begin
  fInner.Attention(q, k, v, aSeq, aHeads, aHeadDim, aKeep, aScratch, y);
end;

procedure TTimedBackend.L2Normalize(x: PSingleArray; aCount: PtrInt);
begin
  fInner.L2Normalize(x, aCount);
end;

procedure TTimedBackend.CausalAttention(q: PSingleArray; aRows, aHeads,
  aHeadDim: PtrInt; keys, values: PSingleArray; aPast, aKvHeads: PtrInt;
  y: PSingleArray);
begin
  fInner.CausalAttention(q, aRows, aHeads, aHeadDim, keys, values, aPast, aKvHeads, y);
end;

procedure TTimedBackend.Reset;
begin
  LinearUs := 0;
  AddUs := 0;
  LinearCalls := 0;
end;

/// peak resident memory in MB, -1 if unknown
function PeakRssMB: integer;
{$ifdef OSLINUX}
var
  s: RawUtf8;
  p: PUtf8Char;
{$endif OSLINUX}
{$ifdef OSWINDOWS}
var
  info: TMemoryInfo;
{$endif OSWINDOWS}
begin
  result := -1;
  {$ifdef OSLINUX}
  s := StringFromFileNoSize('/proc/self/status');
  p := StrPosI('VMHWM:', pointer(s));
  if p <> nil then
    result := GetCardinal(PUtf8Char(GotoNextNotSpace(p + 6))) div 1024;
  {$endif OSLINUX}
  {$ifdef OSWINDOWS}
  if GetMemoryInfo(info, true) then
    result := info.allocreserved shr 20;
  {$endif OSWINDOWS}
end;

function Pct(aPart, aTotal: Int64): double;
begin
  if aTotal <= 0 then
    result := 0
  else
    result := aPart * 100 / aTotal;
end;

const
  /// a German paragraph, repeated until the prompt is long enough
  PROMPT_TEXT = 'Beim Verlegen eines Kupferrohrs in der Wand wird zuerst der ' +
    'Verlauf angezeichnet, dann wird ein Schlitz gefraest, das Rohr gebogen, ' +
    'eingesetzt, befestigt und vor dem Verputzen auf Dichtheit geprueft. ';

var
  folder: TFileName;
  promptLen, newTokens, threads, i, n: PtrInt;
  inner: TInferBackend;
  timed: TTimedBackend;
  tok: TBpeTokenizer;
  model: TQwen3Model;
  ctx: TQwen3Context;
  text: RawUtf8;
  ids, all: TIntegerDynArray;
  logits: TSingleDynArray;
  id: integer;
  t0, t1, prefillUs, decodeUs: Int64;
  mode: RawUtf8;
  weights: TInferDType;
begin
  folder := 'vendor/models/qwen3-0.6b';
  if ParamCount >= 1 then
    folder := ParamStr(1);
  promptLen := 200;
  if ParamCount >= 2 then
    promptLen := StrToIntDef(ParamStr(2), 200);
  newTokens := 32;
  if ParamCount >= 3 then
    newTokens := StrToIntDef(ParamStr(3), 32);
  if (promptLen < 1) or (newTokens < 1) then
  begin
    ConsoleWrite('prompt tokens and new tokens: at least 1 each');
    ExitCode := 1;
    exit;
  end;
  mode := '';
  if ParamCount >= 4 then
    mode := StringToUtf8(ParamStr(4));
  if mode = 'ref' then
    inner := TInferCpuBackend.Create
  else
  begin
    threads := 0;
    if mode <> 'exact' then
      threads := Utf8ToInteger(mode, 0);
    inner := TInferCpuFastBackend.Create(threads);
    TInferCpuFastBackend(inner).Exact := mode = 'exact';
  end;
  weights := idtQ8_0;
  if (ParamCount >= 5) and (ParamStr(5) = 'f32') then
    weights := idtF32;
  timed := TTimedBackend.Create(inner);
  tok := nil;
  model := nil;
  try
    if inner is TInferCpuFastBackend then
      ConsoleWrite('backend: fast, % threads, Q8_0 kernel %, exact %',
        [TInferCpuFastBackend(inner).Threads, InferQ8KernelName,
         TInferCpuFastBackend(inner).Exact])
    else
      ConsoleWrite('backend: scalar reference');
    QueryPerformanceMicroSeconds(t0);
    tok := TBpeTokenizer.Create(IncludeTrailingPathDelimiter(folder) + 'tokenizer.json');
    model := TQwen3Model.Create(folder, timed, weights);
    QueryPerformanceMicroSeconds(t1);
    ConsoleWrite('loaded in % ms: % layers, hidden %, vocab %, weights %, peak % MB',
      [(t1 - t0) div 1000, model.Config.Layers, model.Config.Hidden,
       model.Config.VocabSize, INFER_DTYPE_NAME[weights], PeakRssMB]);
    // a prompt of exactly promptLen ids
    text := '';
    repeat
      text := text + PROMPT_TEXT;
      all := tok.Encode(text);
    until length(all) >= promptLen;
    ids := copy(all, 0, promptLen);
    SetLength(logits, model.Config.VocabSize);
    ctx := model.NewContext;
    try
      timed.Reset;
      QueryPerformanceMicroSeconds(t0);
      model.Forward(ctx, pointer(ids), length(ids), pointer(logits));
      QueryPerformanceMicroSeconds(t1);
      prefillUs := t1 - t0;
      ConsoleWrite('prefill: % tokens in % ms = % tokens/s; linear % pct, ' +
        'add % pct, rest (attention, norms, rope) % pct',
        [promptLen, prefillUs div 1000, promptLen * 1000000 / prefillUs,
         Pct(timed.LinearUs, prefillUs), Pct(timed.AddUs, prefillUs),
         Pct(prefillUs - timed.LinearUs - timed.AddUs, prefillUs)]);
      timed.Reset;
      n := 0;
      QueryPerformanceMicroSeconds(t0);
      for i := 1 to newTokens do
      begin
        id := InferArgMax(pointer(logits), model.Config.VocabSize);
        model.Forward(ctx, @id, 1, pointer(logits));
        inc(n);
      end;
      QueryPerformanceMicroSeconds(t1);
      decodeUs := t1 - t0;
      ConsoleWrite('decode: % tokens in % ms = % tokens/s (% ms/token); ' +
        'linear % pct, add % pct, rest % pct',
        [n, decodeUs div 1000, n * 1000000 / decodeUs, decodeUs / 1000 / n,
         Pct(timed.LinearUs, decodeUs), Pct(timed.AddUs, decodeUs),
         Pct(decodeUs - timed.LinearUs - timed.AddUs, decodeUs)]);
      ConsoleWrite('context: % tokens, peak % MB', [ctx.Length, PeakRssMB]);
    finally
      ctx.Free;
    end;
  finally
    model.Free;
    tok.Free;
    timed.Free;
    inner.Free;
  end;
end.
