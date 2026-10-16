/// native inference engine - the Qwen3 forward pass on a Vulkan device
// - part of the mormot.ai.* extension (LandrixAI)
// - licensed under a MPL/GPL/LGPL three license - see LICENSE
// - phases 4c/4d of docs/native-inference.md: TQwen3Vulkan runs the layers of a
//   TQwen3Model on the device of a TInferVulkanBackend - RMSNorm, the
//   projections, rotary positions, causal attention over a KV cache that lives
//   on the device, SwiGLU, the residuals, the final norm and the head - as one
//   recorded batch per chunk of tokens: the embeddings and rotary factors go
//   up, the logits come down, and the host waits once
// - the arithmetic is float throughout (the weights dequantized as
//   InferRowToF32 does): logits match the CPU within a tolerance, not in bits
// - the model's weights are uploaded once, at Create; the model and the
//   backend must outlive the TQwen3Vulkan and its contexts
unit mormot.ai.infer.model.qwen3.vulkan;

{
  *****************************************************************************

    - TQwen3VkContext: a sequence and its KV cache on the device
    - TQwen3Vulkan: Forward and Greedy of a TQwen3Model on the device

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
  mormot.ai.infer.vulkan,
  mormot.ai.infer.backend.vulkan,
  mormot.ai.infer.model.qwen3,
  mormot.ai.infer.sampler; // InferArgMax

type
  TQwen3Vulkan = class;

  /// one device buffer per layer
  TInferVkBufferDynArray = array of TInferVkBuffer;

  /// one sequence: its token ids and its KV cache on the device
  TQwen3VkContext = class
  protected
    fOwner: TQwen3Vulkan;
    fLength, fCapacity, fRows: PtrInt;
    fIds: TIntegerDynArray;
    fKeys, fValues: TInferVkBufferDynArray; // per layer [capacity, kvWidth]
    // device scratch for up to fRows token rows
    fX, fH, fQ, fK, fV, fAttn, fTmp, fGate, fUp, fCos, fSin: TInferVkBuffer;
    fScores, fLogits: TInferVkBuffer;
    // host-visible: the inputs of a batch, and the logits it returns
    fIn, fOut: TInferVkBuffer;
    procedure FreeScratch;
    procedure FreeCache;
  public
    destructor Destroy; override;
    /// forget the cached tokens - the buffers stay allocated
    procedure Reset;
    /// keep the first aLength cached tokens (0 <= aLength <= Length)
    procedure Truncate(aLength: PtrInt);
    /// how many of the first tokens of aIds[0..aCount-1] the cache holds
    function Common(aIds: PIntegerArray; aCount: PtrInt): PtrInt;
    /// tokens in the KV cache = position of the next token
    property Length: PtrInt
      read fLength;
  end;

  /// the forward pass of a TQwen3Model on the device of a TInferVulkanBackend
  TQwen3Vulkan = class
  protected
    fModel: TQwen3Model;
    fBackend: TInferVulkanBackend;
    fVk: TInferVulkan;
    fNorm, fRope, fSwiGlu, fAttention, fAdd: TInferVkPipeline;
    fEpoch: integer; // the backend's WeightEpoch of the uploaded weights
    procedure UploadWeights;
    function Vector(v: PSingleArray; aCount: PtrInt): TInferVkBuffer;
    function ChunkRows(aPast, aCount: PtrInt; aAllLogits: boolean): PtrInt;
    procedure Prepare(aCtx: TQwen3VkContext; aTokens, aRows, aOutRows: PtrInt);
    procedure Chunk(aCtx: TQwen3VkContext; aIds: PIntegerArray; aCount: PtrInt;
      aLogits: PSingleArray; aAllLogits: boolean);
    procedure RecordNorm(x: TInferVkBuffer; aRows, aCols: PtrInt;
      w: PSingleArray; y: TInferVkBuffer);
    procedure RecordElementwise(aPipeline: TInferVkPipeline;
      const aBuffers: array of TInferVkBuffer; aCount: PtrInt);
  public
    /// upload every weight of aModel to the device of aBackend; raises
    // EInferError when a matrix type has no shader (F32, Q8_0, Q4_K, Q6_K do)
    // - aModel and aBackend are not owned and must outlive this instance
    constructor Create(aModel: TQwen3Model; aBackend: TInferVulkanBackend);
    destructor Destroy; override;
    /// a context for this model on this device (owned by the caller)
    function NewContext: TQwen3VkContext;
    /// TQwen3Model.Forward on the device: append aCount tokens to aCtx and
    // compute them; aLogits: nil, the last token's [VocabSize] or, when
    // aAllLogits, [aCount, VocabSize]
    procedure Forward(aCtx: TQwen3VkContext; aIds: PIntegerArray; aCount: PtrInt;
      aLogits: PSingleArray; aAllLogits: boolean = false);
    /// keep the cached tokens of aCtx that start aIds, at most aCount - 1
    function Reuse(aCtx: TQwen3VkContext; aIds: PIntegerArray; aCount: PtrInt): PtrInt;
    /// TQwen3Model.Greedy on the device
    function Greedy(aCtx: TQwen3VkContext; const aPrompt: array of integer;
      aMaxNew: PtrInt; const aStop: array of integer): TIntegerDynArray;
    property Model: TQwen3Model
      read fModel;
    property Backend: TInferVulkanBackend
      read fBackend;
  end;


implementation

type
  TNormPush = record
    Rows, Cols: cardinal;
    Eps: single;
  end;

  TRopePush = record
    Rows, Heads, HeadDim: cardinal;
  end;

  TAttentionPush = record
    Rows, Heads, HeadDim, KvHeads, Past: cardinal;
  end;

const
  /// token rows per recorded batch at most: every dispatch dimension stays
  // below 65535 work groups (rows * Intermediate / 128 for SwiGLU)
  CHUNK_MAX = 256;
  /// the attention scores of one chunk at most (heads * rows * context floats)
  SCORES_MAX = 256 shl 20;

/// a device buffer of at least aBytes, reallocated (content lost) when smaller
procedure NeedDevice(aVk: TInferVulkan; var aBuffer: TInferVkBuffer;
  aBytes: Int64);
begin
  if (aBuffer <> nil) and (aBuffer.Size >= aBytes) then
    exit;
  FreeAndNil(aBuffer);
  aBuffer := TInferVkBuffer.Create(aVk, aBytes, true);
end;


{ TQwen3VkContext }

destructor TQwen3VkContext.Destroy;
begin
  FreeScratch;
  FreeCache;
  inherited Destroy;
end;

procedure TQwen3VkContext.FreeScratch;
begin
  FreeAndNil(fX);
  FreeAndNil(fH);
  FreeAndNil(fQ);
  FreeAndNil(fK);
  FreeAndNil(fV);
  FreeAndNil(fAttn);
  FreeAndNil(fTmp);
  FreeAndNil(fGate);
  FreeAndNil(fUp);
  FreeAndNil(fCos);
  FreeAndNil(fSin);
  FreeAndNil(fIn);
  fRows := 0;
end;

procedure TQwen3VkContext.FreeCache;
var
  i: PtrInt;
begin
  for i := 0 to high(fKeys) do
  begin
    FreeAndNil(fKeys[i]);
    FreeAndNil(fValues[i]);
  end;
  fKeys := nil;
  fValues := nil;
  FreeAndNil(fScores);
  FreeAndNil(fLogits);
  FreeAndNil(fOut);
  fCapacity := 0;
end;

procedure TQwen3VkContext.Reset;
begin
  fLength := 0;
end;

procedure TQwen3VkContext.Truncate(aLength: PtrInt);
begin
  if (aLength < 0) or (aLength > fLength) then
    EInferError.RaiseUtf8('%.Truncate(%) of % tokens', [self, aLength, fLength]);
  fLength := aLength;
end;

function TQwen3VkContext.Common(aIds: PIntegerArray; aCount: PtrInt): PtrInt;
begin
  result := 0;
  if aCount > fLength then
    aCount := fLength;
  while (result < aCount) and (fIds[result] = aIds^[result]) do
    inc(result);
end;


{ TQwen3Vulkan }

constructor TQwen3Vulkan.Create(aModel: TQwen3Model; aBackend: TInferVulkanBackend);
begin
  inherited Create;
  if (aModel = nil) or (aBackend = nil) then
    EInferError.RaiseUtf8('%.Create: no model or no backend', [self]);
  fModel := aModel;
  fBackend := aBackend;
  fVk := aBackend.Vulkan;
  fNorm := TInferVkPipeline.Create(fVk, @SPV_RMSNORM, length(SPV_RMSNORM), 3,
    SizeOf(TNormPush));
  fRope := TInferVkPipeline.Create(fVk, @SPV_ROPE, length(SPV_ROPE), 3,
    SizeOf(TRopePush));
  fSwiGlu := TInferVkPipeline.Create(fVk, @SPV_SWIGLU, length(SPV_SWIGLU), 2, 4);
  fAttention := TInferVkPipeline.Create(fVk, @SPV_ATTENTION, length(SPV_ATTENTION),
    5, SizeOf(TAttentionPush));
  fAdd := TInferVkPipeline.Create(fVk, @SPV_ADD, length(SPV_ADD), 3, 4);
  fBackend.GpuLock;
  try
    UploadWeights;
  finally
    fBackend.GpuUnLock;
  end;
end;

destructor TQwen3Vulkan.Destroy;
begin
  fAdd.Free;
  fAttention.Free;
  fSwiGlu.Free;
  fRope.Free;
  fNorm.Free;
  inherited Destroy;
end;

function TQwen3Vulkan.Vector(v: PSingleArray; aCount: PtrInt): TInferVkBuffer;
begin
  result := fBackend.DeviceWeight(v, idtF32, Int64(aCount) * 4);
end;

procedure TQwen3Vulkan.UploadWeights;
var
  c: TQwen3Config;
  l, qw, kvw: PtrInt;
  ly: ^TQwen3Layer;

  procedure Matrix(const w: TInferWeight; aRows, aCols: PtrInt);
  var
    bytes: Int64;
  begin
    if not fBackend.GpuLinear(w.DType, aCols) then
      EInferError.RaiseUtf8('%: no shader for a % matrix of % columns',
        [self, INFER_DTYPE_NAME[w.DType], aCols]);
    bytes := Int64(aRows) * (aCols div INFER_DTYPE_BLOCK[w.DType]) *
      INFER_DTYPE_SIZE[w.DType];
    // as allocated: whole 4-byte words (TInferVkBuffer)
    if (bytes + 3) and not Int64(3) > fVk.MaxStorageRange then
      EInferError.RaiseUtf8('%: a matrix of % bytes, the device binds at most %',
        [self, bytes, fVk.MaxStorageRange]);
    fBackend.DeviceWeight(w.Data, w.DType, bytes);
  end;

begin
  // everything up front: a batch cannot upload (and the first token would
  // otherwise carry the whole transfer)
  // the caller holds GpuLock
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

function TQwen3Vulkan.NewContext: TQwen3VkContext;
begin
  result := TQwen3VkContext.Create;
  result.fOwner := self;
end;

function TQwen3Vulkan.ChunkRows(aPast, aCount: PtrInt; aAllLogits: boolean): PtrInt;
var
  limit, widest: Int64;
  c: TQwen3Config;
begin
  c := fModel.Config;
  result := aCount;
  if result > CHUNK_MAX then
    result := CHUNK_MAX;
  // an element-wise dispatch (one invocation per value, 128 per work group)
  // has at most 65535 work groups
  widest := c.Intermediate;
  if c.Hidden > widest then
    widest := c.Hidden;
  if Int64(c.Heads) * (c.HeadDim shr 1) > widest then
    widest := Int64(c.Heads) * (c.HeadDim shr 1);
  if Int64(result) * widest > Int64(65535) * 128 then
    result := (Int64(65535) * 128) div widest;
  // the per-head norms take one work group per row and head
  if Int64(result) * c.Heads > 65535 then
    result := 65535 div c.Heads;
  // the logits of every row of the chunk in one buffer
  if aAllLogits then
    while (result > 1) and (Int64(result) * c.VocabSize * 4 > fVk.MaxStorageRange) do
      result := result shr 1;
  limit := SCORES_MAX;
  if limit > fVk.MaxStorageRange then
    limit := fVk.MaxStorageRange;
  // the scores of every head and row over the whole context must fit
  while (result > 1) and
        (Int64(fModel.Config.Heads) * result * (aPast + result) * 4 > limit) do
    result := result shr 1;
  if Int64(fModel.Config.Heads) * result * (aPast + result) * 4 > limit then
    EInferError.RaiseUtf8('%: a context of % tokens is too long for this device',
      [self, aPast + result]);
end;

procedure TQwen3Vulkan.Prepare(aCtx: TQwen3VkContext; aTokens, aRows,
  aOutRows: PtrInt);
var
  c: TQwen3Config;
  cap, kvw, l, half: PtrInt;
  keys, values: TInferVkBufferDynArray;
begin
  c := fModel.Config;
  kvw := c.KvHeads * c.HeadDim;
  half := c.HeadDim shr 1;
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
    // no more than one layer's buffer the device can bind
    if Int64(cap) * kvw * 4 > fVk.MaxStorageRange then
      cap := fVk.MaxStorageRange div (Int64(kvw) * 4);
    if cap < aTokens then
      EInferError.RaiseUtf8('%: a KV cache of % tokens is too large for this device',
        [self, aTokens]);
    SetLength(keys, c.Layers);
    SetLength(values, c.Layers);
    try
      for l := 0 to c.Layers - 1 do
      begin
        keys[l] := TInferVkBuffer.Create(fVk, Int64(cap) * kvw * 4, true);
        values[l] := TInferVkBuffer.Create(fVk, Int64(cap) * kvw * 4, true);
      end;
      if aCtx.fLength > 0 then
      begin
        fVk.BatchBegin;
        try
          for l := 0 to c.Layers - 1 do
          begin
            fVk.BatchCopy(aCtx.fKeys[l], 0, keys[l], 0, Int64(aCtx.fLength) * kvw * 4);
            fVk.BatchCopy(aCtx.fValues[l], 0, values[l], 0,
              Int64(aCtx.fLength) * kvw * 4);
          end;
        except
          fVk.BatchAbort;
          raise;
        end;
        fVk.BatchSubmit;
      end;
    except
      for l := 0 to high(keys) do
      begin
        keys[l].Free;
        values[l].Free;
      end;
      raise;
    end;
    for l := 0 to high(aCtx.fKeys) do
    begin
      aCtx.fKeys[l].Free;
      aCtx.fValues[l].Free;
    end;
    aCtx.fKeys := keys;
    aCtx.fValues := values;
    SetLength(aCtx.fIds, cap);
    aCtx.fCapacity := cap;
  end;
  // the scratch of a chunk
  if aRows > aCtx.fRows then
  begin
    aCtx.FreeScratch;
    NeedDevice(fVk, aCtx.fX, Int64(aRows) * c.Hidden * 4);
    NeedDevice(fVk, aCtx.fH, Int64(aRows) * c.Hidden * 4);
    NeedDevice(fVk, aCtx.fTmp, Int64(aRows) * c.Hidden * 4);
    NeedDevice(fVk, aCtx.fQ, Int64(aRows) * c.Heads * c.HeadDim * 4);
    NeedDevice(fVk, aCtx.fAttn, Int64(aRows) * c.Heads * c.HeadDim * 4);
    NeedDevice(fVk, aCtx.fK, Int64(aRows) * kvw * 4);
    NeedDevice(fVk, aCtx.fV, Int64(aRows) * kvw * 4);
    NeedDevice(fVk, aCtx.fGate, Int64(aRows) * c.Intermediate * 4);
    NeedDevice(fVk, aCtx.fUp, Int64(aRows) * c.Intermediate * 4);
    NeedDevice(fVk, aCtx.fCos, Int64(aRows) * half * 4);
    NeedDevice(fVk, aCtx.fSin, Int64(aRows) * half * 4);
    // host-visible: embeddings, then cos, then sin
    aCtx.fIn := TInferVkBuffer.Create(fVk, Int64(aRows) * (c.Hidden + 2 * half) * 4);
    aCtx.fRows := aRows;
  end;
  NeedDevice(fVk, aCtx.fScores, Int64(c.Heads) * aRows * aTokens * 4);
  if aOutRows > 0 then
  begin
    NeedDevice(fVk, aCtx.fLogits, Int64(aOutRows) * c.VocabSize * 4);
    if (aCtx.fOut = nil) or (aCtx.fOut.Size < Int64(aOutRows) * c.VocabSize * 4) then
    begin
      FreeAndNil(aCtx.fOut);
      aCtx.fOut := TInferVkBuffer.Create(fVk, Int64(aOutRows) * c.VocabSize * 4);
    end;
  end;
end;

procedure TQwen3Vulkan.RecordNorm(x: TInferVkBuffer; aRows, aCols: PtrInt;
  w: PSingleArray; y: TInferVkBuffer);
var
  push: TNormPush;
begin
  push.Rows := aRows;
  push.Cols := aCols;
  push.Eps := fModel.Config.RmsEps;
  fVk.BatchDispatch(fNorm, [x, Vector(w, aCols), y], @push, aRows);
end;

procedure TQwen3Vulkan.RecordElementwise(aPipeline: TInferVkPipeline;
  const aBuffers: array of TInferVkBuffer; aCount: PtrInt);
var
  n: cardinal;
begin
  n := aCount;
  fVk.BatchDispatch(aPipeline, aBuffers, @n, (aCount + 127) div 128);
end;

procedure TQwen3Vulkan.Chunk(aCtx: TQwen3VkContext; aIds: PIntegerArray;
  aCount: PtrInt; aLogits: PSingleArray; aAllLogits: boolean);
var
  c: TQwen3Config;
  n, h, f, qw, kvw, half, past, l, outRows: PtrInt;
  ly: ^TQwen3Layer;
  p: PSingleArray;
  rope: TRopePush;
  att: TAttentionPush;
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
  // the inputs, written into the host-visible buffer before the batch
  p := aCtx.fIn.Mapped;
  fModel.Embed(aIds, n, p);
  fModel.Rotary(past, n, @p^[n * h], @p^[n * h + n * half]);
  fVk.BatchBegin;
  try
    fVk.BatchCopy(aCtx.fIn, 0, aCtx.fX, 0, Int64(n) * h * 4);
    fVk.BatchCopy(aCtx.fIn, Int64(n) * h * 4, aCtx.fCos, 0, Int64(n) * half * 4);
    fVk.BatchCopy(aCtx.fIn, Int64(n) * (h + half) * 4, aCtx.fSin, 0,
      Int64(n) * half * 4);
    rope.Rows := n;
    rope.HeadDim := c.HeadDim;
    att.Rows := n;
    att.Heads := c.Heads;
    att.HeadDim := c.HeadDim;
    att.KvHeads := c.KvHeads;
    att.Past := past;
    for l := 0 to c.Layers - 1 do
    begin
      ly := @fModel.Layers[l];
      // attention block: pre-norm, projections, per-head norms, rotary positions
      RecordNorm(aCtx.fX, n, h, ly^.InputNorm, aCtx.fH);
      fBackend.RecordLinear(aCtx.fH, n, h, ly^.Q, qw, aCtx.fQ);
      fBackend.RecordLinear(aCtx.fH, n, h, ly^.K, kvw, aCtx.fK);
      fBackend.RecordLinear(aCtx.fH, n, h, ly^.V, kvw, aCtx.fV);
      RecordNorm(aCtx.fQ, n * c.Heads, c.HeadDim, ly^.QNorm, aCtx.fQ);
      RecordNorm(aCtx.fK, n * c.KvHeads, c.HeadDim, ly^.KNorm, aCtx.fK);
      rope.Heads := c.Heads;
      fVk.BatchDispatch(fRope, [aCtx.fQ, aCtx.fCos, aCtx.fSin], @rope,
        (n * c.Heads * half + 127) div 128);
      rope.Heads := c.KvHeads;
      fVk.BatchDispatch(fRope, [aCtx.fK, aCtx.fCos, aCtx.fSin], @rope,
        (n * c.KvHeads * half + 127) div 128);
      fVk.BatchCopy(aCtx.fK, 0, aCtx.fKeys[l], Int64(past) * kvw * 4,
        Int64(n) * kvw * 4);
      fVk.BatchCopy(aCtx.fV, 0, aCtx.fValues[l], Int64(past) * kvw * 4,
        Int64(n) * kvw * 4);
      fVk.BatchDispatch(fAttention, [aCtx.fQ, aCtx.fKeys[l], aCtx.fValues[l],
        aCtx.fAttn, aCtx.fScores], @att, c.Heads, n);
      fBackend.RecordLinear(aCtx.fAttn, n, qw, ly^.O, h, aCtx.fTmp);
      RecordElementwise(fAdd, [aCtx.fX, aCtx.fTmp, aCtx.fX], n * h);
      // feed-forward block: pre-norm, SwiGLU, residual
      RecordNorm(aCtx.fX, n, h, ly^.PostNorm, aCtx.fH);
      fBackend.RecordLinear(aCtx.fH, n, h, ly^.Gate, f, aCtx.fGate);
      fBackend.RecordLinear(aCtx.fH, n, h, ly^.Up, f, aCtx.fUp);
      RecordElementwise(fSwiGlu, [aCtx.fGate, aCtx.fUp], n * f);
      fBackend.RecordLinear(aCtx.fGate, n, f, ly^.Down, h, aCtx.fTmp);
      RecordElementwise(fAdd, [aCtx.fX, aCtx.fTmp, aCtx.fX], n * h);
    end;
    if outRows > 0 then
    begin
      if outRows = 1 then
      begin
        // the last row only: to the start of the scratch, its norm, the head
        fVk.BatchCopy(aCtx.fX, Int64(n - 1) * h * 4, aCtx.fTmp, 0, Int64(h) * 4);
        RecordNorm(aCtx.fTmp, 1, h, fModel.FinalNorm, aCtx.fH);
      end
      else
        RecordNorm(aCtx.fX, n, h, fModel.FinalNorm, aCtx.fH);
      fBackend.RecordLinear(aCtx.fH, outRows, h, fModel.Head, c.VocabSize,
        aCtx.fLogits);
      fVk.BatchCopy(aCtx.fLogits, 0, aCtx.fOut, 0,
        Int64(outRows) * c.VocabSize * 4);
    end;
  except
    fVk.BatchAbort;
    raise;
  end;
  fVk.BatchSubmit;
  if outRows > 0 then
    MoveFast(aCtx.fOut.Mapped^, aLogits^, outRows * c.VocabSize * SizeOf(single));
  MoveFast(aIds^[0], aCtx.fIds[past], n * SizeOf(integer));
  aCtx.fLength := past + n;
end;

procedure TQwen3Vulkan.Forward(aCtx: TQwen3VkContext; aIds: PIntegerArray;
  aCount: PtrInt; aLogits: PSingleArray; aAllLogits: boolean);
var
  i, n, done, vocab: PtrInt;
  last: boolean;
begin
  if (aCtx = nil) or (aCtx.fOwner <> self) then
    EInferError.RaiseUtf8('%.Forward: no context of this instance', [self]);
  if aCount <= 0 then
    EInferError.RaiseUtf8('%.Forward: no tokens', [self]);
  if aCount > fModel.Config.MaxPositions - aCtx.fLength then
    EInferError.RaiseUtf8('%.Forward: % + % tokens exceed the % positions',
      [self, aCtx.fLength, aCount, fModel.Config.MaxPositions]);
  vocab := fModel.Config.VocabSize;
  for i := 0 to aCount - 1 do
    if (aIds^[i] < 0) or (aIds^[i] >= vocab) then
      EInferError.RaiseUtf8('%.Forward: token id % outside the vocabulary',
        [self, aIds^[i]]);
  fBackend.GpuLock;
  try
    // the backend forgot the weights (ClearWeights): up again, before a batch
    if fEpoch <> fBackend.WeightEpoch then
      UploadWeights;
    done := 0;
    while done < aCount do
    begin
      n := ChunkRows(aCtx.fLength, aCount - done, aAllLogits and (aLogits <> nil));
      last := done + n = aCount;
      if aAllLogits and (aLogits <> nil) then
        Chunk(aCtx, @aIds^[done], n, @aLogits^[done * vocab], true)
      else if last then
        Chunk(aCtx, @aIds^[done], n, aLogits, false)
      else
        Chunk(aCtx, @aIds^[done], n, nil, false);
      inc(done, n);
    end;
  finally
    fBackend.GpuUnLock;
  end;
end;

function TQwen3Vulkan.Reuse(aCtx: TQwen3VkContext; aIds: PIntegerArray;
  aCount: PtrInt): PtrInt;
begin
  if (aCtx = nil) or (aCtx.fOwner <> self) then
    EInferError.RaiseUtf8('%.Reuse: no context of this instance', [self]);
  result := 0;
  if aCount > 0 then
    result := aCtx.Common(aIds, aCount - 1);
  aCtx.Truncate(result);
end;

function TQwen3Vulkan.Greedy(aCtx: TQwen3VkContext; const aPrompt: array of integer;
  aMaxNew: PtrInt; const aStop: array of integer): TIntegerDynArray;
var
  logits: TSingleDynArray;
  n, i, next: PtrInt;
  id: integer;
  stop: boolean;
begin
  result := nil;
  if length(aPrompt) = 0 then
    EInferError.RaiseUtf8('%.Greedy: empty prompt', [self]);
  if aMaxNew <= 0 then
    EInferError.RaiseUtf8('%.Greedy: % new tokens', [self, aMaxNew]);
  SetLength(logits, fModel.Config.VocabSize);
  Forward(aCtx, @aPrompt[0], length(aPrompt), pointer(logits));
  SetLength(result, aMaxNew);
  n := 0;
  repeat
    next := InferArgMax(pointer(logits), fModel.Config.VocabSize);
    result[n] := next;
    inc(n);
    stop := n = aMaxNew;
    for i := 0 to high(aStop) do
      if aStop[i] = next then
        stop := true;
    // as TQwen3Model.Greedy: every returned token enters the cache, so a
    // conversation can go on; only a full context leaves the last one out
    if aCtx.Length >= fModel.Config.MaxPositions then
      break;
    id := next;
    if stop then
      Forward(aCtx, @id, 1, nil)
    else
      Forward(aCtx, @id, 1, pointer(logits));
  until stop;
  SetLength(result, n);
end;

end.
