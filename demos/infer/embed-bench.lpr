// LandrixAI - throughput of the native embedder (phase 1c of
// docs/native-inference.md)
// - embeds the embed-de corpus (documents, then queries) one text at a time, as
//   the lembed oracle does when its reference was made, and reports texts/s,
//   tokens/s, the time per backend operation, the peak memory and the cosine
//   against the lembed vectors of tests/data/embed-de/reference
//
//   usage: embed-bench [gguf] [texts=all] [threads=0 (all cores) | ref]
//   - ref: the scalar reference backend, otherwise TInferCpuFastBackend
//   e.g.   embed-bench vendor/models/bge-m3-q8_0.gguf 40 1
program embed.bench;

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
  mormot.core.buffers,
  mormot.core.variants,
  mormot.core.perf,
  mormot.ai.infer.tensor,
  mormot.ai.infer.backend,
  mormot.ai.infer.backend.cpu,
  mormot.ai.infer.kernels,
  mormot.ai.infer.pipeline.embed;

type
  /// forwards every operation to an inner backend and sums its time per op
  TTimedBackend = class(TInferBackend)
  protected
    fInner: TInferBackend;
    fTime: array[TInferOp] of Int64;
    fCalls: array[TInferOp] of Int64;
    fStart: Int64;
    procedure Enter;
    procedure Leave(aOp: TInferOp);
  public
    constructor Create(aInner: TInferBackend);
    function Supports(aOp: TInferOp; aWeights: TInferDType): boolean; override;
    procedure Linear(x: PSingleArray; aRows, aIn: PtrInt; const w: TInferWeight;
      aOut: PtrInt; bias: PSingleArray; y: PSingleArray); override;
    procedure LayerNorm(x: PSingleArray; aRows, aCols: PtrInt;
      gamma, beta: PSingleArray; aEps: double; y: PSingleArray); override;
    procedure Gelu(x: PSingleArray; aCount: PtrInt); override;
    procedure Add(x, y: PSingleArray; aCount: PtrInt); override;
    procedure SoftmaxMasked(x: PSingleArray; aRows, aCols: PtrInt;
      aKeep: PIntegerArray); override;
    procedure Attention(q, k, v: PSingleArray; aSeq, aHeads, aHeadDim: PtrInt;
      aKeep: PIntegerArray; aScratch, y: PSingleArray); override;
    procedure L2Normalize(x: PSingleArray; aCount: PtrInt); override;
    function Report(aTotal: Int64): RawUtf8;
  end;

const
  OP_NAME: array[TInferOp] of RawUtf8 = (
    'linear', 'layernorm', 'gelu', 'add', 'softmax', 'attention', 'l2');

function NotZero(v: Int64): Int64;
begin
  if v < 1 then
    result := 1
  else
    result := v;
end;

constructor TTimedBackend.Create(aInner: TInferBackend);
begin
  inherited Create;
  fInner := aInner;
end;

procedure TTimedBackend.Enter;
begin
  QueryPerformanceMicroSeconds(fStart);
end;

procedure TTimedBackend.Leave(aOp: TInferOp);
var
  now: Int64;
begin
  QueryPerformanceMicroSeconds(now);
  inc(fTime[aOp], now - fStart);
  inc(fCalls[aOp]);
end;

function TTimedBackend.Supports(aOp: TInferOp; aWeights: TInferDType): boolean;
begin
  result := fInner.Supports(aOp, aWeights);
end;

procedure TTimedBackend.Linear(x: PSingleArray; aRows, aIn: PtrInt;
  const w: TInferWeight; aOut: PtrInt; bias: PSingleArray; y: PSingleArray);
begin
  Enter;
  fInner.Linear(x, aRows, aIn, w, aOut, bias, y);
  Leave(ioLinear);
end;

procedure TTimedBackend.LayerNorm(x: PSingleArray; aRows, aCols: PtrInt;
  gamma, beta: PSingleArray; aEps: double; y: PSingleArray);
begin
  Enter;
  fInner.LayerNorm(x, aRows, aCols, gamma, beta, aEps, y);
  Leave(ioLayerNorm);
end;

procedure TTimedBackend.Gelu(x: PSingleArray; aCount: PtrInt);
begin
  Enter;
  fInner.Gelu(x, aCount);
  Leave(ioGelu);
end;

procedure TTimedBackend.Add(x, y: PSingleArray; aCount: PtrInt);
begin
  Enter;
  fInner.Add(x, y, aCount);
  Leave(ioAdd);
end;

procedure TTimedBackend.SoftmaxMasked(x: PSingleArray; aRows, aCols: PtrInt;
  aKeep: PIntegerArray);
begin
  Enter;
  fInner.SoftmaxMasked(x, aRows, aCols, aKeep);
  Leave(ioSoftmaxMasked);
end;

procedure TTimedBackend.Attention(q, k, v: PSingleArray; aSeq, aHeads,
  aHeadDim: PtrInt; aKeep: PIntegerArray; aScratch, y: PSingleArray);
begin
  Enter;
  fInner.Attention(q, k, v, aSeq, aHeads, aHeadDim, aKeep, aScratch, y);
  Leave(ioAttention);
end;

procedure TTimedBackend.L2Normalize(x: PSingleArray; aCount: PtrInt);
begin
  Enter;
  fInner.L2Normalize(x, aCount);
  Leave(ioL2Normalize);
end;

function TTimedBackend.Report(aTotal: Int64): RawUtf8;
var
  op: TInferOp;
  sum: Int64;
begin
  result := '';
  sum := 0;
  for op := low(op) to high(op) do
  begin
    inc(sum, fTime[op]);
    if fCalls[op] > 0 then
      result := FormatUtf8('%  %: % ms (% %%)'#10, [result, OP_NAME[op],
        fTime[op] div 1000, (fTime[op] * 1000 div NotZero(aTotal)) / 10]);
  end;
  result := FormatUtf8('%  rest (tokenizer, embeddings, copies): % ms'#10,
    [result, (aTotal - sum) div 1000]);
end;

/// peak resident memory in MB, -1 if unknown
function PeakRssMB: integer;
{$ifdef OSLINUX}
var
  s: RawUtf8;
  p: PUtf8Char;
{$endif OSLINUX}
begin
  result := -1;
  {$ifdef OSLINUX}
  s := StringFromFileNoSize('/proc/self/status');
  p := StrPosI('VMHWM:', pointer(s));
  if p <> nil then
    result := GetCardinal(PUtf8Char(GotoNextNotSpace(p + 6))) div 1024;
  {$endif OSLINUX}
end;

/// <dir>/<aPath> for the first of the program folder and its parents holding it
function FindUp(const aPath: TFileName): TFileName;
var
  dir: TFileName;
  i: integer;
begin
  dir := ExcludeTrailingPathDelimiter(Executable.ProgramFilePath);
  for i := 1 to 6 do
  begin
    result := MakePath([dir, aPath]);
    if FileExists(result) then
      exit;
    dir := ExtractFileDir(dir);
  end;
  result := '';
end;

var
  gguf, corpus, queries, refDocs, refQueries: TFileName;
  texts, lines: TRawUtf8DynArray;
  ref, v: TSingleDynArray;
  raw: RawUtf8;
  row: TDocVariantData;
  cpu: TInferCpuBackend;
  timed: TTimedBackend;
  emb: TInferEmbedder;
  i, n, limit, tokens, h, below: PtrInt;
  t0, t1: Int64;
  cos, minCos, a: double;
  j: PtrInt;
begin
  gguf := '';
  if ParamCount >= 1 then
    gguf := ParamStr(1);
  if gguf = '' then
    gguf := FindUp(MakePath(['vendor', 'models', 'bge-m3-q8_0.gguf']));
  limit := MaxInt;
  if (ParamCount >= 2) and (ParamStr(2) <> 'all') then
    limit := GetInteger(pointer(StringToUtf8(ParamStr(2))));
  corpus := FindUp(MakePath(['tests', 'data', 'embed-de', 'corpus.jsonl']));
  queries := FindUp(MakePath(['tests', 'data', 'embed-de', 'queries.jsonl']));
  refDocs := FindUp(MakePath(['tests', 'data', 'embed-de', 'reference', 'docs.f32']));
  refQueries := FindUp(MakePath(['tests', 'data', 'embed-de', 'reference', 'queries.f32']));
  if (gguf = '') or (corpus = '') then
  begin
    ConsoleWrite('usage: embed-bench [gguf] [texts=all] - model or corpus not found');
    ExitCode := 2;
    exit;
  end;
  // documents then queries, in file order: the order of the lembed vectors
  texts := nil;
  ref := nil;
  for i := 0 to 1 do
  begin
    if i = 0 then
      raw := StringFromFile(corpus)
    else
      raw := StringFromFile(queries);
    lines := nil; // CsvToRawUtf8DynArray appends
    CsvToRawUtf8DynArray(pointer(raw), lines, #10);
    for j := 0 to high(lines) do
      if TrimU(lines[j]) <> '' then
      begin
        row.Clear;
        row.InitJson(lines[j], JSON_FAST);
        AddRawUtf8(texts, row.U['text']);
      end;
  end;
  raw := StringFromFile(refDocs) + StringFromFile(refQueries);
  if length(raw) = length(texts) * 1024 * 4 then
  begin
    SetLength(ref, length(texts) * 1024);
    MoveFast(pointer(raw)^, pointer(ref)^, length(raw));
  end;
  n := MinPtrInt(limit, length(texts));
  if (ParamCount >= 3) and (ParamStr(3) = 'ref') then
  begin
    cpu := TInferCpuBackend.Create;
    ConsoleWrite('backend: reference (scalar, one thread)');
  end
  else
  begin
    j := 0;
    if ParamCount >= 3 then
      j := GetInteger(pointer(StringToUtf8(ParamStr(3))));
    cpu := TInferCpuFastBackend.Create(j);
    ConsoleWrite('backend: fast, % threads, Q8_0 kernel %',
      [TInferCpuFastBackend(cpu).Threads, InferQ8KernelName]);
  end;
  timed := TTimedBackend.Create(cpu);
  QueryPerformanceMicroSeconds(t0);
  emb := TInferEmbedder.CreateFromGguf(gguf, timed);
  try
    QueryPerformanceMicroSeconds(t1);
    ConsoleWrite('loaded % in % ms', [ExtractFileName(gguf), (t1 - t0) div 1000]);
    h := emb.Dim;
    tokens := 0;
    minCos := 1;
    below := 0;
    QueryPerformanceMicroSeconds(t0);
    for i := 0 to n - 1 do
    begin
      inc(tokens, length(emb.Tokenizer.Encode(texts[i])));
      v := emb.Embed([texts[i]]);
      if ref <> nil then
      begin
        cos := 0;
        for j := 0 to h - 1 do
        begin
          a := v[j];
          cos := cos + a * ref[i * h + j];
        end;
        if cos < minCos then
          minCos := cos;
        if cos < 0.999 then
          inc(below);
      end;
    end;
    QueryPerformanceMicroSeconds(t1);
    ConsoleWrite('% texts, % tokens (mean %) in % ms: % texts/s, % tokens/s',
      [n, tokens, tokens / MaxPtrInt(n, 1), (t1 - t0) div 1000,
       n * 1e6 / NotZero(t1 - t0), tokens * 1e6 / NotZero(t1 - t0)]);
    if ref <> nil then
      ConsoleWrite('cosine against lembed: min %, % below 0.999', [minCos, below]);
    ConsoleWrite('peak RSS: % MB', [PeakRssMB]);
    ConsoleWrite(timed.Report(t1 - t0));
  finally
    emb.Free;
    timed.Free;
    cpu.Free;
  end;
end.
