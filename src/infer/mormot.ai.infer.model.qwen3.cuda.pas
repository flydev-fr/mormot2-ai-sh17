/// native inference engine - the Qwen3 forward pass on a CUDA device
// - part of the mormot.ai.* extension (LandrixAI)
// - licensed under a MPL/GPL/LGPL three license - see LICENSE
// - backlog B3, slice C4: TQwen3Cuda runs the layers of a TQwen3Model on the
//   device of a TInferCudaBackend - RMSNorm, the projections, rotary positions,
//   causal attention over a KV cache that lives on the device, SwiGLU, the
//   residuals, the final norm and the head - as kernels enqueued on one
//   stream per chunk of tokens: the embeddings and rotary factors go up, the
//   logits come down, and the host waits once (the download)
// - the arithmetic is float throughout (the weights dequantized as
//   InferRowToF32 does): logits match the CPU within a tolerance, not in bits
// - the KV cache holds half floats by default (KvF16), as on Vulkan
// - the model's weights are uploaded once, at Create; the model and the
//   backend must outlive the TQwen3Cuda and its contexts
unit mormot.ai.infer.model.qwen3.cuda;

{
  *****************************************************************************

    - TQwen3CuContext: a sequence and its KV cache on the device
    - TQwen3Cuda: Forward and Greedy of a TQwen3Model on the device

  *****************************************************************************
}

interface

{$I mormot.defines.inc}

uses
  sysutils,
  mormot.core.base,
  mormot.core.os,
  mormot.core.unicode,
  mormot.core.text,
  mormot.ai.infer.tensor,
  mormot.ai.infer.backend,
  mormot.ai.infer.cuda,
  mormot.ai.infer.backend.cuda,
  mormot.ai.infer.model.qwen3;

type
  TQwen3Cuda = class;

  /// one device buffer per layer
  TInferCuBufferDynArray = array of TInferCuBuffer;

  /// one sequence: its token ids and its KV cache on the device
  // - a TQwen3Context of TQwen3Cuda: Length, Common, Truncate, Reset and the
  // ids at every position as there (Capacity: of the device cache); the CPU
  // arrays stay empty
  TQwen3CuContext = class(TQwen3Context)
  protected
    fRows: PtrInt;
    fDevKeys, fDevValues: TInferCuBufferDynArray; // per layer [capacity, kvWidth]
    fKvF16: boolean; // the cache holds half floats (TQwen3Cuda.KvF16)
    // device scratch for up to fRows token rows
    fDevX, fDevH, fDevQ, fDevK, fDevV, fDevAttn, fDevTmp, fDevGate, fDevUp,
      fDevCos: TInferCuBuffer;
    fPart, fLogits: TInferCuBuffer; // the key ranges of the attention, logits
    // the inputs of a chunk on the host: embeddings, then cos, then sin
    fIn: TSingleDynArray;
    // the decode step as a graph (TQwen3Cuda.UseGraphs), on fCu
    fCu: TInferCuda;
    fGraph: TCuHandle;
    procedure FreeScratch;
    procedure FreeCache;
  public
    destructor Destroy; override;
    /// true when the cache holds half floats (TQwen3Cuda.KvF16 when it was
    // allocated)
    property KvF16: boolean
      read fKvF16;
  end;

  /// the forward pass of a TQwen3Model on the device of a TInferCudaBackend
  // - a TQwen3Device: Model.Device := this instance makes Model.NewContext
  // (and so a TNativeLlmClient) decode here
  TQwen3Cuda = class(TQwen3Device)
  protected
    fBackend: TInferCudaBackend;
    fCu: TInferCuda;
    fOps, fBasic: TInferCuModule;
    fNorm, fAddNorm, fSwiGlu, fQkRope, fAttention, fMerge, fAdd: TCuHandle;
    fQkRope16, fAttention16: TCuHandle; // for a KV cache of half floats
    fEpoch: integer; // the backend's WeightEpoch of the uploaded weights
    fSplitKeys: integer;
    fUseGraphs, fKvF16: boolean;
    procedure UploadWeights;
    function Vector(v: PSingleArray; aCount: PtrInt): TCuDevicePtr;
    function ChunkRows(aPast, aCount: PtrInt; aAllLogits: boolean): PtrInt;
    procedure Prepare(aCtx: TQwen3CuContext; aTokens, aRows, aOutRows: PtrInt);
    procedure Chunk(aCtx: TQwen3CuContext; aIds: PIntegerArray; aCount: PtrInt;
      aLogits: PSingleArray; aAllLogits: boolean);
    procedure EnqueueNorm(x: TInferCuBuffer; aRows, aCols: PtrInt;
      w: PSingleArray; y: TInferCuBuffer);
    procedure EnqueueAddNorm(x, t: TInferCuBuffer; aRows, aCols: PtrInt;
      w: PSingleArray; y: TInferCuBuffer);
  public
    /// upload every weight of aModel to the device of aBackend; raises
    // EInferError when a matrix type has no kernel (F32, Q8_0, Q4_K, Q6_K do)
    // - aModel and aBackend are not owned and must outlive this instance
    constructor Create(aModel: TQwen3Model; aBackend: TInferCudaBackend);
    destructor Destroy; override;
    /// a context for this model on this device (owned by the caller)
    function NewContext: TQwen3CuContext;
    /// NewContext, for TQwen3Model.NewContext
    function NewDeviceContext: TQwen3Context; override;
    /// TQwen3Model.Forward on the device: append aCount tokens to aCtx (a
    // context of this instance) and compute them; aLogits: nil, the last
    // token's [VocabSize] or, when aAllLogits, [aCount, VocabSize]
    procedure Forward(aCtx: TQwen3Context; aIds: PIntegerArray; aCount: PtrInt;
      aLogits: PSingleArray; aAllLogits: boolean = false); override;
    /// keep the cached tokens of aCtx that start aIds, at most aCount - 1
    function Reuse(aCtx: TQwen3Context; aIds: PIntegerArray;
      aCount: PtrInt): PtrInt; override;
    /// TQwen3Model.Greedy on the device
    function Greedy(aCtx: TQwen3Context; const aPrompt: array of integer;
      aMaxNew: PtrInt; const aStop: array of integer): TIntegerDynArray;
    property Backend: TInferCudaBackend
      read fBackend;
    /// the attention cuts the keys of a row into ranges of about SplitKeys
    // when there are few blocks (decoding), merged afterwards - 0: never
    // (the tests compare both)
    property SplitKeys: integer
      read fSplitKeys write fSplitKeys;
    /// a decode step (one token) is recorded as a CUDA graph and launched at
    // once instead of some 370 kernel launches - default true; the graph of
    // the previous step takes the new parameters (TInferCuda.CaptureEnd)
    property UseGraphs: boolean
      read fUseGraphs write fUseGraphs;
    /// the KV cache in half floats (default) instead of floats: half the
    // memory, and half the bytes the attention reads per decoded token -
    // the keys and values rounded to 11 significant bits, the logits then
    // within a looser tolerance of the CPU's
    // - a context takes it when its cache is allocated, and again on its
    // next Forward while it is empty (after Reset)
    property KvF16: boolean
      read fKvF16 write fKvF16;
  end;


implementation

const
  /// token rows per chunk at most
  CHUNK_MAX = 256;
  /// the logits of every row of a chunk (rows * vocabulary floats): the
  // chunk rows are halved down to this size
  LOGITS_MAX = 256 shl 20;
  /// the queries of one attention block at most (MAXQ in tools/cuda/ops.cu):
  // the heads of one KV head times rows, which share every key and value read
  ATTENTION_QUERIES = 8;
  /// the attention cuts the keys of a row into ranges of about SPLIT_KEYS
  // when there are few blocks - decoding one token - at most SPLIT_GROUPS
  // blocks in all (then attnmerge combines them); as on Vulkan
  SPLIT_KEYS = 64;
  SPLIT_GROUPS = 2048;
  /// the threads of a block of tools/cuda/ops.cu
  OPS_BLOCK = 128;

/// a device buffer of at least aBytes, reallocated (content lost) when smaller
procedure NeedDevice(aCu: TInferCuda; var aBuffer: TInferCuBuffer; aBytes: Int64);
begin
  if (aBuffer <> nil) and (aBuffer.Size >= aBytes) then
    exit;
  FreeAndNil(aBuffer);
  aBuffer := TInferCuBuffer.Create(aCu, aBytes);
end;


{ TQwen3CuContext }

destructor TQwen3CuContext.Destroy;
begin
  if fCu <> nil then
    fCu.GraphFree(fGraph);
  FreeScratch;
  FreeCache;
  inherited Destroy;
end;

procedure TQwen3CuContext.FreeScratch;
begin
  FreeAndNil(fDevX);
  FreeAndNil(fDevH);
  FreeAndNil(fDevQ);
  FreeAndNil(fDevK);
  FreeAndNil(fDevV);
  FreeAndNil(fDevAttn);
  FreeAndNil(fDevTmp);
  FreeAndNil(fDevGate);
  FreeAndNil(fDevUp);
  FreeAndNil(fDevCos);
  fRows := 0;
end;

procedure TQwen3CuContext.FreeCache;
var
  i: PtrInt;
begin
  for i := 0 to high(fDevKeys) do
  begin
    FreeAndNil(fDevKeys[i]);
    FreeAndNil(fDevValues[i]);
  end;
  fDevKeys := nil;
  fDevValues := nil;
  FreeAndNil(fPart);
  FreeAndNil(fLogits);
  fCapacity := 0;
end;


{ TQwen3Cuda }

constructor TQwen3Cuda.Create(aModel: TQwen3Model; aBackend: TInferCudaBackend);
begin
  inherited Create;
  if (aModel = nil) or (aBackend = nil) then
    EInferError.RaiseUtf8('%.Create: no model or no backend', [self]);
  // qkrope keeps one normalized head in shared memory, the attention reads
  // the keys as float4
  if (aModel.Config.HeadDim > 256) or (aModel.Config.HeadDim and 3 <> 0) then
    EInferError.RaiseUtf8('%: head dimension % not supported', [self,
      aModel.Config.HeadDim]);
  fModel := aModel;
  fBackend := aBackend;
  fCu := aBackend.Cuda;
  fOps := TInferCuModule.Create(fCu, @PTX_OPS);
  fBasic := TInferCuModule.Create(fCu, @PTX_BASIC);
  fNorm := fOps.Kernel('rmsnorm');
  fAddNorm := fOps.Kernel('addnorm');
  fSwiGlu := fOps.Kernel('swiglu');
  fQkRope := fOps.Kernel('qkrope');
  fAttention := fOps.Kernel('attention');
  fQkRope16 := fOps.Kernel('qkrope16');
  fAttention16 := fOps.Kernel('attention16');
  fMerge := fOps.Kernel('attnmerge');
  fSplitKeys := SPLIT_KEYS;
  fUseGraphs := true;
  fKvF16 := true;
  fAdd := fBasic.Kernel('add');
  fBackend.GpuLock;
  try
    UploadWeights;
  finally
    fBackend.GpuUnLock;
  end;
end;

destructor TQwen3Cuda.Destroy;
begin
  // the model no longer hands out contexts of this device
  if (fModel <> nil) and (fModel.Device = self) then
    fModel.Device := nil;
  fBasic.Free;
  fOps.Free;
  inherited Destroy;
end;

function TQwen3Cuda.Vector(v: PSingleArray; aCount: PtrInt): TCuDevicePtr;
begin
  result := fBackend.DeviceWeight(v, idtF32, Int64(aCount) * 4).Ptr;
end;

procedure TQwen3Cuda.UploadWeights;
var
  c: TQwen3Config;
  l, qw, kvw: PtrInt;
  ly: ^TQwen3Layer;

  procedure Matrix(const w: TInferWeight; aRows, aCols: PtrInt);
  begin
    if not fBackend.GpuLinear(w.DType, aCols) then
      EInferError.RaiseUtf8('%: no kernel for a % matrix of % columns',
        [self, INFER_DTYPE_NAME[w.DType], aCols]);
    fBackend.DeviceWeight(w.Data, w.DType, Int64(aRows) *
      (aCols div INFER_DTYPE_BLOCK[w.DType]) * INFER_DTYPE_SIZE[w.DType]);
  end;

begin
  // everything up front: the first token would otherwise carry the transfer
  // - the caller holds GpuLock
  c := fModel.Config;
  qw := c.Heads * c.HeadDim;
  kvw := c.KvHeads * c.HeadDim;
  for l := 0 to c.Layers - 1 do
  begin
    ly := @fModel.Layers[l];
    Matrix(ly^.Q, qw, c.Hidden);
    Matrix(ly^.K, kvw, c.Hidden);
    Matrix(ly^.V, kvw, c.Hidden);
    Matrix(ly^.O, c.Hidden, qw);
    Matrix(ly^.Gate, c.Intermediate, c.Hidden);
    Matrix(ly^.Up, c.Intermediate, c.Hidden);
    Matrix(ly^.Down, c.Hidden, c.Intermediate);
    Vector(ly^.InputNorm, c.Hidden);
    Vector(ly^.PostNorm, c.Hidden);
    Vector(ly^.QNorm, c.HeadDim);
    Vector(ly^.KNorm, c.HeadDim);
  end;
  Vector(fModel.FinalNorm, c.Hidden);
  Matrix(fModel.Head, c.VocabSize, c.Hidden);
  fEpoch := fBackend.WeightEpoch;
end;

function TQwen3Cuda.NewContext: TQwen3CuContext;
begin
  result := TQwen3CuContext.Create;
  result.fDevice := self;
  result.fCu := fCu;
end;

function TQwen3Cuda.NewDeviceContext: TQwen3Context;
begin
  result := NewContext;
end;

function TQwen3Cuda.ChunkRows(aPast, aCount: PtrInt; aAllLogits: boolean): PtrInt;
var
  c: TQwen3Config;
begin
  c := fModel.Config;
  result := aCount;
  if result > CHUNK_MAX then
    result := CHUNK_MAX;
  // the logits of every row of the chunk in one buffer
  if aAllLogits then
    while (result > 1) and (Int64(result) * c.VocabSize * 4 > LOGITS_MAX) do
      result := result shr 1;
end;

procedure TQwen3Cuda.Prepare(aCtx: TQwen3CuContext; aTokens, aRows,
  aOutRows: PtrInt);
var
  c: TQwen3Config;
  cap, kvw, l, half, es: PtrInt;
  keys, values: TInferCuBufferDynArray;
begin
  c := fModel.Config;
  kvw := c.KvHeads * c.HeadDim;
  half := c.HeadDim shr 1;
  // an empty context takes the cache format of KvF16
  if (aCtx.fLength = 0) and (aCtx.fCapacity > 0) and (aCtx.fKvF16 <> fKvF16) then
    aCtx.FreeCache;
  if aCtx.fCapacity = 0 then
    aCtx.fKvF16 := fKvF16;
  es := 4; // bytes per cached value
  if aCtx.fKvF16 then
    es := 2;
  // the KV cache: doubled, a quarter beyond the tokens wanted (as on the CPU),
  // the cached tokens copied into the new buffers
  if aTokens > aCtx.fCapacity then
  begin
    cap := aCtx.fCapacity * 2;
    if cap < aTokens + aTokens shr 2 then
      cap := aTokens + aTokens shr 2;
    if cap < 64 then
      cap := 64;
    if cap > c.MaxPositions then
      cap := c.MaxPositions;
    SetLength(keys, c.Layers);
    SetLength(values, c.Layers);
    try
      for l := 0 to c.Layers - 1 do
      begin
        keys[l] := TInferCuBuffer.Create(fCu, Int64(cap) * kvw * es);
        values[l] := TInferCuBuffer.Create(fCu, Int64(cap) * kvw * es);
        if aCtx.fLength > 0 then
        begin
          aCtx.fDevKeys[l].CopyTo(0, keys[l], 0, Int64(aCtx.fLength) * kvw * es);
          aCtx.fDevValues[l].CopyTo(0, values[l], 0, Int64(aCtx.fLength) * kvw * es);
        end;
      end;
    except
      for l := 0 to high(keys) do
      begin
        keys[l].Free;
        values[l].Free;
      end;
      raise;
    end;
    for l := 0 to high(aCtx.fDevKeys) do
    begin
      aCtx.fDevKeys[l].Free;
      aCtx.fDevValues[l].Free;
    end;
    aCtx.fDevKeys := keys;
    aCtx.fDevValues := values;
    SetLength(aCtx.fIds, cap);
    aCtx.fCapacity := cap;
  end;
  // the scratch of a chunk
  if aRows > aCtx.fRows then
  begin
    aCtx.FreeScratch;
    NeedDevice(fCu, aCtx.fDevX, Int64(aRows) * c.Hidden * 4);
    NeedDevice(fCu, aCtx.fDevH, Int64(aRows) * c.Hidden * 4);
    NeedDevice(fCu, aCtx.fDevTmp, Int64(aRows) * c.Hidden * 4);
    NeedDevice(fCu, aCtx.fDevQ, Int64(aRows) * c.Heads * c.HeadDim * 4);
    NeedDevice(fCu, aCtx.fDevAttn, Int64(aRows) * c.Heads * c.HeadDim * 4);
    NeedDevice(fCu, aCtx.fDevK, Int64(aRows) * kvw * 4);
    NeedDevice(fCu, aCtx.fDevV, Int64(aRows) * kvw * 4);
    NeedDevice(fCu, aCtx.fDevGate, Int64(aRows) * c.Intermediate * 4);
    NeedDevice(fCu, aCtx.fDevUp, Int64(aRows) * c.Intermediate * 4);
    NeedDevice(fCu, aCtx.fDevCos, Int64(aRows) * half * 2 * 4); // cos, then sin
    SetLength(aCtx.fIn, aRows * (c.Hidden + 2 * half));
    aCtx.fRows := aRows;
  end;
  if aOutRows > 0 then
    NeedDevice(fCu, aCtx.fLogits, Int64(aOutRows) * c.VocabSize * 4);
end;

procedure TQwen3Cuda.EnqueueNorm(x: TInferCuBuffer; aRows, aCols: PtrInt;
  w: PSingleArray; y: TInferCuBuffer);
var
  px, pw, py: TCuDevicePtr;
  cols: cardinal;
  eps: single;
begin
  px := x.Ptr;
  pw := Vector(w, aCols);
  py := y.Ptr;
  cols := aCols;
  eps := fModel.Config.RmsEps;
  fCu.Enqueue(fNorm, [@px, @pw, @py, @cols, @eps], aRows, OPS_BLOCK);
end;

procedure TQwen3Cuda.EnqueueAddNorm(x, t: TInferCuBuffer; aRows, aCols: PtrInt;
  w: PSingleArray; y: TInferCuBuffer);
var
  px, pt, pw, py: TCuDevicePtr;
  cols: cardinal;
  eps: single;
begin
  px := x.Ptr;
  pt := t.Ptr;
  pw := Vector(w, aCols);
  py := y.Ptr;
  cols := aCols;
  eps := fModel.Config.RmsEps;
  fCu.Enqueue(fAddNorm, [@px, @pt, @pw, @py, @cols, @eps], aRows, OPS_BLOCK);
end;

procedure TQwen3Cuda.Chunk(aCtx: TQwen3CuContext; aIds: PIntegerArray;
  aCount: PtrInt; aLogits: PSingleArray; aAllLogits: boolean);
var
  c: TQwen3Config;
  n, h, f, qw, kvw, half, past, l, outRows, hb, rb, blocks, splits, most: PtrInt;
  ly: ^TQwen3Layer;
  p: PSingleArray;
  pq, pk, pv, pqn, pkn, pcs, pkc, pvc, pa, ppart, px, pt: TCuDevicePtr;
  rows, heads, kvHeads, headDim, ppast, nsplits, hblock, rblock, count: cardinal;
  eps: single;
  capture: boolean;
  qkrope, attention: TCuHandle;
begin
  c := fModel.Config;
  n := aCount;
  h := c.Hidden;
  f := c.Intermediate;
  qw := c.Heads * c.HeadDim;
  kvw := c.KvHeads * c.HeadDim;
  half := c.HeadDim shr 1;
  past := aCtx.fLength;
  outRows := 0;
  if aLogits <> nil then
    if aAllLogits then
      outRows := n
    else
      outRows := 1;
  Prepare(aCtx, past + n, n, outRows);
  // the kernels of the cache format of aCtx (set by Prepare)
  qkrope := fQkRope;
  attention := fAttention;
  if aCtx.fKvF16 then
  begin
    qkrope := fQkRope16;
    attention := fAttention16;
  end;
  // the attention blocks: hb heads of one KV head (hb divides the group)
  // times rb rows, the keys of a row cut into splits ranges unless there are
  // enough blocks without
  hb := c.Heads div c.KvHeads;
  while (hb > ATTENTION_QUERIES) or (c.Heads div c.KvHeads mod hb <> 0) do
    dec(hb);
  rb := ATTENTION_QUERIES div hb;
  blocks := (n + rb - 1) div rb;
  splits := 1;
  if fSplitKeys > 0 then
    splits := (past + n + fSplitKeys - 1) div fSplitKeys;
  most := SPLIT_GROUPS div ((c.Heads div hb) * blocks);
  if splits > most then
    splits := most;
  if splits < 1 then
    splits := 1;
  ppart := 0;
  if splits > 1 then
  begin
    NeedDevice(fCu, aCtx.fPart, Int64(c.Heads) * n * splits * (c.HeadDim + 2) * 4);
    ppart := aCtx.fPart.Ptr;
  end;
  // the inputs: embeddings, then the rotary factors (cos, then sin)
  p := pointer(aCtx.fIn);
  fModel.Embed(aIds, n, p);
  fModel.Rotary(past, n, @p^[n * h], @p^[n * h + n * half]);
  aCtx.fDevX.Upload(0, p, Int64(n) * h * 4);
  aCtx.fDevCos.Upload(0, @p^[n * h], Int64(n) * half * 2 * 4);
  rows := n;
  heads := c.Heads;
  kvHeads := c.KvHeads;
  headDim := c.HeadDim;
  ppast := past;
  nsplits := splits;
  hblock := hb;
  rblock := rb;
  eps := c.RmsEps;
  pq := aCtx.fDevQ.Ptr;
  pk := aCtx.fDevK.Ptr;
  pv := aCtx.fDevV.Ptr;
  pcs := aCtx.fDevCos.Ptr;
  pa := aCtx.fDevAttn.Ptr;
  // a single token: the whole step recorded as a graph, launched at once
  capture := fUseGraphs and (n = 1);
  if capture then
    fCu.CaptureBegin;
  try
    // the input norm of layer 0; every later one comes with the residual of the
    // layer before (addnorm)
    EnqueueNorm(aCtx.fDevX, n, h, fModel.Layers[0].InputNorm, aCtx.fDevH);
    for l := 0 to c.Layers - 1 do
    begin
      ly := @fModel.Layers[l];
      // attention block: the projections, per head the norms, rotary positions
      // and the store of k and v into the cache, the attention, the output
      fBackend.EnqueueLinear(aCtx.fDevH, n, h, ly^.Q, qw, aCtx.fDevQ);
      fBackend.EnqueueLinear(aCtx.fDevH, n, h, ly^.K, kvw, aCtx.fDevK);
      fBackend.EnqueueLinear(aCtx.fDevH, n, h, ly^.V, kvw, aCtx.fDevV);
      pqn := Vector(ly^.QNorm, c.HeadDim);
      pkn := Vector(ly^.KNorm, c.HeadDim);
      pkc := aCtx.fDevKeys[l].Ptr;
      pvc := aCtx.fDevValues[l].Ptr;
      fCu.Enqueue(qkrope, [@pq, @pk, @pv, @pqn, @pkn, @pcs, @pkc, @pvc, @rows,
        @heads, @kvHeads, @headDim, @ppast, @eps], c.Heads + c.KvHeads, OPS_BLOCK, n);
      fCu.Enqueue(attention, [@pq, @pkc, @pvc, @pa, @ppart, @rows, @heads,
        @headDim, @kvHeads, @ppast, @nsplits, @hblock, @rblock], c.Heads div hb,
        OPS_BLOCK, blocks, splits);
      if splits > 1 then
        fCu.Enqueue(fMerge, [@ppart, @pa, @rows, @heads, @headDim, @nsplits],
          c.Heads, OPS_BLOCK, n);
      fBackend.EnqueueLinear(aCtx.fDevAttn, n, qw, ly^.O, h, aCtx.fDevTmp);
      // residual and the feed-forward pre-norm, gate and up, SwiGLU, down
      EnqueueAddNorm(aCtx.fDevX, aCtx.fDevTmp, n, h, ly^.PostNorm, aCtx.fDevH);
      fBackend.EnqueueLinear(aCtx.fDevH, n, h, ly^.Gate, f, aCtx.fDevGate);
      fBackend.EnqueueLinear(aCtx.fDevH, n, h, ly^.Up, f, aCtx.fDevUp);
      px := aCtx.fDevGate.Ptr;
      pt := aCtx.fDevUp.Ptr;
      count := n * f;
      fCu.Enqueue(fSwiGlu, [@px, @pt, @count], (count + OPS_BLOCK - 1) div OPS_BLOCK,
        OPS_BLOCK);
      fBackend.EnqueueLinear(aCtx.fDevGate, n, f, ly^.Down, h, aCtx.fDevTmp);
      // the residual, and the input norm of the next layer
      if l < c.Layers - 1 then
        EnqueueAddNorm(aCtx.fDevX, aCtx.fDevTmp, n, h, fModel.Layers[l + 1].InputNorm,
          aCtx.fDevH)
      else
      begin
        px := aCtx.fDevX.Ptr;
        pt := aCtx.fDevTmp.Ptr;
        count := n * h;
        fCu.Enqueue(fAdd, [@px, @pt, @px, @count],
          (count + OPS_BLOCK - 1) div OPS_BLOCK, OPS_BLOCK);
      end;
    end;
    if outRows > 0 then
    begin
      if outRows = 1 then
      begin
        // the last row only: to the start of the scratch, its norm, the head
        aCtx.fDevX.CopyTo(Int64(n - 1) * h * 4, aCtx.fDevTmp, 0, Int64(h) * 4);
        EnqueueNorm(aCtx.fDevTmp, 1, h, fModel.FinalNorm, aCtx.fDevH);
      end
      else
        EnqueueNorm(aCtx.fDevX, n, h, fModel.FinalNorm, aCtx.fDevH);
      fBackend.EnqueueLinear(aCtx.fDevH, outRows, h, fModel.Head, c.VocabSize,
        aCtx.fLogits);
    end;
    if capture then
      fCu.CaptureEnd(aCtx.fGraph);
  except
    fCu.CaptureAbort;
    raise;
  end;
  // waits for every kernel before it (one stream)
  if outRows > 0 then
    aCtx.fLogits.Download(0, aLogits, Int64(outRows) * c.VocabSize * 4)
  else
    fCu.Synchronize;
  MoveFast(aIds^[0], aCtx.fIds[past], n * SizeOf(integer));
  aCtx.fLength := past + n;
end;

procedure TQwen3Cuda.Forward(aCtx: TQwen3Context; aIds: PIntegerArray;
  aCount: PtrInt; aLogits: PSingleArray; aAllLogits: boolean);
var
  i, n, done, vocab: PtrInt;
  last: boolean;
  ctx: TQwen3CuContext;
begin
  if (aCtx = nil) or (aCtx.Device <> self) then
    EInferError.RaiseUtf8('%.Forward: no context of this instance', [self]);
  ctx := TQwen3CuContext(aCtx); // only NewContext gives this device
  if aCount <= 0 then
    EInferError.RaiseUtf8('%.Forward: no tokens', [self]);
  if aCount > fModel.Config.MaxPositions - ctx.fLength then
    EInferError.RaiseUtf8('%.Forward: % + % tokens exceed the % positions',
      [self, ctx.fLength, aCount, fModel.Config.MaxPositions]);
  vocab := fModel.Config.VocabSize;
  for i := 0 to aCount - 1 do
    if (aIds^[i] < 0) or (aIds^[i] >= vocab) then
      EInferError.RaiseUtf8('%.Forward: token id % outside the vocabulary',
        [self, aIds^[i]]);
  fBackend.GpuLock;
  try
    // the backend forgot the weights (ClearWeights): up again
    if fEpoch <> fBackend.WeightEpoch then
      UploadWeights;
    done := 0;
    while done < aCount do
    begin
      n := ChunkRows(ctx.fLength, aCount - done, aAllLogits and (aLogits <> nil));
      last := done + n = aCount;
      if aAllLogits and (aLogits <> nil) then
        Chunk(ctx, @aIds^[done], n, @aLogits^[done * vocab], true)
      else if last then
        Chunk(ctx, @aIds^[done], n, aLogits, false)
      else
        Chunk(ctx, @aIds^[done], n, nil, false);
      inc(done, n);
    end;
  finally
    fBackend.GpuUnLock;
  end;
end;

function TQwen3Cuda.Reuse(aCtx: TQwen3Context; aIds: PIntegerArray;
  aCount: PtrInt): PtrInt;
begin
  if (aCtx = nil) or (aCtx.Device <> self) then
    EInferError.RaiseUtf8('%.Reuse: no context of this instance', [self]);
  result := 0;
  if aCount > 0 then
    result := aCtx.Common(aIds, aCount - 1);
  aCtx.Truncate(result);
end;

function TQwen3Cuda.Greedy(aCtx: TQwen3Context; const aPrompt: array of integer;
  aMaxNew: PtrInt; const aStop: array of integer): TIntegerDynArray;
begin
  if (aCtx = nil) or (aCtx.Device <> self) then
    EInferError.RaiseUtf8('%.Greedy: no context of this instance', [self]);
  // the model hands a context of this device to Forward here
  result := fModel.Greedy(aCtx, aPrompt, aMaxNew, aStop);
end;

end.
