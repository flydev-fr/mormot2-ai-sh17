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
  mormot.ai.infer.model.qwen3;

type
  TQwen3Vulkan = class;

  /// one device buffer per layer
  TInferVkBufferDynArray = array of TInferVkBuffer;

  /// one sequence: its token ids and its KV cache on the device
  // - a TQwen3Context of TQwen3Vulkan: Length, Common, Truncate, Reset and the
  // ids at every position as there (Capacity: of the device cache); the CPU
  // arrays stay empty
  TQwen3VkContext = class(TQwen3Context)
  protected
    fRows: PtrInt;
    fDevKeys, fDevValues: TInferVkBufferDynArray; // per layer [capacity, kvWidth]
    // device scratch for up to fRows token rows
    fDevX, fDevH, fDevQ, fDevK, fDevV, fDevAttn, fDevTmp, fDevGate, fDevUp, fDevCos: TInferVkBuffer;
    fPart, fLogits: TInferVkBuffer;
    // host-visible: the inputs of a batch, and the logits it returns
    fIn, fOut: TInferVkBuffer;
    procedure FreeScratch;
    procedure FreeCache;
  public
    destructor Destroy; override;
  end;

  /// the forward pass of a TQwen3Model on the device of a TInferVulkanBackend
  // - a TQwen3Device: Model.Device := this instance makes Model.NewContext
  // (and so a TNativeLlmClient) decode here
  TQwen3Vulkan = class(TQwen3Device)
  protected
    fBackend: TInferVulkanBackend;
    fVk: TInferVulkan;
    fNorm, fSwiGlu, fAttention, fMerge, fAdd, fQkRope, fAddNorm,
      fFlash: TInferVkPipeline;
    fUseFlash: boolean;
    fEpoch: integer; // the backend's WeightEpoch of the uploaded weights
    fSplitKeys: integer;
    procedure UploadWeights;
    function Vector(v: PSingleArray; aCount: PtrInt): TInferVkBuffer;
    function ChunkRows(aCount: PtrInt; aAllLogits: boolean): PtrInt;
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
    property Backend: TInferVulkanBackend
      read fBackend;
    /// the attention cuts the keys of a row into ranges of about this many
    // keys while there are few work groups (decoding); 0 = never - for tests
    // and tuning, the default (64) suits a discrete GPU
    property SplitKeys: integer
      read fSplitKeys write fSplitKeys;
    /// the attention of FLASH_ROWS token rows or more (prefill) through the
    // tiled flash.comp instead of attention.comp - possible with a head
    // dimension up to 128, work groups of 256 and 27 KB of shared memory, by
    // default on an Apple GPU (VendorId $106B) only: there Qwen3-0.6B Q8_0
    // prefills 4000 tokens at 708 instead of 381 tokens/s, while an RTX 5090
    // drops from 9880 to 5840 with it (backlog B8)
    property UseFlash: boolean
      read fUseFlash write fUseFlash;
    /// true when the device runs flash.comp (UseFlash can take it)
    function HasFlash: boolean;
  end;


implementation

type
  TNormPush = record
    Rows, Cols: cardinal;
    Eps: single;
  end;

  TAttentionPush = record
    Rows, Heads, HeadDim, KvHeads, Past, Splits, HeadBlock, RowBlock: cardinal;
  end;

  TMergePush = record
    Rows, Heads, HeadDim, Splits: cardinal;
  end;

  TQkRopePush = record
    Rows, Heads, KvHeads, HeadDim, Past: cardinal;
    Eps: single;
  end;

const
  /// token rows per recorded batch at most: every dispatch dimension stays
  // below 65535 work groups (rows * Intermediate / 128 for SwiGLU)
  CHUNK_MAX = 256;
  /// the queries of one attention.comp work group at most (MAXQ there): the
  // heads of one KV head times rows, which share every key and value read
  ATTENTION_QUERIES = 8;
  /// attention.comp cuts the keys of a row into ranges of about SPLIT_KEYS
  // when there are few work groups - decoding one token - at most
  // SPLIT_GROUPS work groups in all (then attnmerge.comp combines them)
  SPLIT_KEYS = 64;
  SPLIT_GROUPS = 2048;
  /// the queries of one flash.comp work group at most (QN there)
  FLASH_QUERIES = 32;
  /// from this many token rows on the attention runs flash.comp (UseFlash)
  FLASH_ROWS = 16;
  /// the shared memory of flash.comp in bytes: queries, a tile of keys or
  // values, the scores and three floats per query
  FLASH_SHARED = 32 * 32 * 16 + 16 * 32 * 16 + 32 * 16 * 4 + 3 * 32 * 4;

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
  FreeAndNil(fIn);
  fRows := 0;
end;

procedure TQwen3VkContext.FreeCache;
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
  FreeAndNil(fOut);
  fCapacity := 0;
end;


{ TQwen3Vulkan }

constructor TQwen3Vulkan.Create(aModel: TQwen3Model; aBackend: TInferVulkanBackend);
begin
  inherited Create;
  if (aModel = nil) or (aBackend = nil) then
    EInferError.RaiseUtf8('%.Create: no model or no backend', [self]);
  // qkrope.comp keeps one normalized head in shared memory
  // and attention.comp reads it as vec4
  if (aModel.Config.HeadDim > 256) or (aModel.Config.HeadDim and 3 <> 0) then
    EInferError.RaiseUtf8('%: head dimension % not supported', [self,
      aModel.Config.HeadDim]);
  fModel := aModel;
  fBackend := aBackend;
  fSplitKeys := SPLIT_KEYS;
  fVk := aBackend.Vulkan;
  fNorm := TInferVkPipeline.Create(fVk, @SPV_RMSNORM, length(SPV_RMSNORM), 3,
    SizeOf(TNormPush));
  fSwiGlu := TInferVkPipeline.Create(fVk, @SPV_SWIGLU, length(SPV_SWIGLU), 2, 4);
  fAttention := TInferVkPipeline.Create(fVk, @SPV_ATTENTION, length(SPV_ATTENTION),
    5, SizeOf(TAttentionPush));
  fMerge := TInferVkPipeline.Create(fVk, @SPV_ATTNMERGE, length(SPV_ATTNMERGE), 2,
    SizeOf(TMergePush));
  if (aModel.Config.HeadDim <= 128) and (fVk.MaxInvocations >= 256) and
     (fVk.MaxGroupSizeX >= 256) and (fVk.MaxSharedMemory >= FLASH_SHARED) then
  begin
    fFlash := TInferVkPipeline.Create(fVk, @SPV_FLASH, length(SPV_FLASH), 5,
      SizeOf(TAttentionPush));
    fFlash.Name := 'flash';
    fUseFlash := fVk.VendorId = $106B; // Apple
  end;
  fAdd := TInferVkPipeline.Create(fVk, @SPV_ADD, length(SPV_ADD), 3, 4);
  fQkRope := TInferVkPipeline.Create(fVk, @SPV_QKROPE, length(SPV_QKROPE), 8,
    SizeOf(TQkRopePush));
  fAddNorm := TInferVkPipeline.Create(fVk, @SPV_ADDNORM, length(SPV_ADDNORM), 4,
    SizeOf(TNormPush));
  fQkRope.Name := 'qkrope';
  fAddNorm.Name := 'addnorm';
  fNorm.Name := 'rmsnorm';
  fSwiGlu.Name := 'swiglu';
  fAttention.Name := 'attention';
  fMerge.Name := 'attnmerge';
  fAdd.Name := 'add';
  fBackend.GpuLock;
  try
    UploadWeights;
  finally
    fBackend.GpuUnLock;
  end;
end;

function TQwen3Vulkan.HasFlash: boolean;
begin
  result := fFlash <> nil;
end;

destructor TQwen3Vulkan.Destroy;
begin
  // the model no longer hands out contexts of this device
  if (fModel <> nil) and (fModel.Device = self) then
    fModel.Device := nil;
  fAddNorm.Free;
  fQkRope.Free;
  fAdd.Free;
  fFlash.Free;
  fMerge.Free;
  fAttention.Free;
  fSwiGlu.Free;
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
  result.fDevice := self;
end;

function TQwen3Vulkan.NewDeviceContext: TQwen3Context;
begin
  result := NewContext;
end;

function TQwen3Vulkan.ChunkRows(aCount: PtrInt; aAllLogits: boolean): PtrInt;
var
  widest: Int64;
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
            fVk.BatchCopy(aCtx.fDevKeys[l], 0, keys[l], 0, Int64(aCtx.fLength) * kvw * 4);
            fVk.BatchCopy(aCtx.fDevValues[l], 0, values[l], 0,
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
    NeedDevice(fVk, aCtx.fDevX, Int64(aRows) * c.Hidden * 4);
    NeedDevice(fVk, aCtx.fDevH, Int64(aRows) * c.Hidden * 4);
    NeedDevice(fVk, aCtx.fDevTmp, Int64(aRows) * c.Hidden * 4);
    NeedDevice(fVk, aCtx.fDevQ, Int64(aRows) * c.Heads * c.HeadDim * 4);
    NeedDevice(fVk, aCtx.fDevAttn, Int64(aRows) * c.Heads * c.HeadDim * 4);
    NeedDevice(fVk, aCtx.fDevK, Int64(aRows) * kvw * 4);
    NeedDevice(fVk, aCtx.fDevV, Int64(aRows) * kvw * 4);
    NeedDevice(fVk, aCtx.fDevGate, Int64(aRows) * c.Intermediate * 4);
    NeedDevice(fVk, aCtx.fDevUp, Int64(aRows) * c.Intermediate * 4);
    NeedDevice(fVk, aCtx.fDevCos, Int64(aRows) * half * 2 * 4); // cos, then sin
    // host-visible: embeddings, then cos, then sin
    aCtx.fIn := TInferVkBuffer.Create(fVk, Int64(aRows) * (c.Hidden + 2 * half) * 4);
    aCtx.fRows := aRows;
  end;
  if aOutRows > 0 then
  begin
    NeedDevice(fVk, aCtx.fLogits, Int64(aOutRows) * c.VocabSize * 4);
    if (aCtx.fOut = nil) or (aCtx.fOut.Size < Int64(aOutRows) * c.VocabSize * 4) then
    begin
      FreeAndNil(aCtx.fOut);
      aCtx.fOut := TInferVkBuffer.CreateReadback(fVk, Int64(aOutRows) * c.VocabSize * 4);
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
  n, h, f, qw, kvw, half, past, l, outRows, splits, most, hb, rb, blocks,
    queries: PtrInt;
  attention: TInferVkPipeline;
  ly: ^TQwen3Layer;
  p: PSingleArray;
  qk: TQkRopePush;
  norm: TNormPush;
  att: TAttentionPush;
  merge: TMergePush;
  part: TInferVkBuffer;
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
  // the attention work groups: hb heads of one KV head (hb divides the
  // group) times rb rows, the keys of a row cut into splits ranges unless
  // there are enough work groups without
  // - flash.comp for FLASH_ROWS rows or more
  attention := fAttention;
  queries := ATTENTION_QUERIES;
  if fUseFlash and (fFlash <> nil) and (n >= FLASH_ROWS) then
  begin
    attention := fFlash;
    queries := FLASH_QUERIES;
  end;
  hb := c.Heads div c.KvHeads;
  while (hb > queries) or (c.Heads div c.KvHeads mod hb <> 0) do
    dec(hb);
  rb := queries div hb;
  blocks := (n + rb - 1) div rb;
  splits := 1;
  if fSplitKeys > 0 then
    splits := (past + n + fSplitKeys - 1) div fSplitKeys;
  most := SPLIT_GROUPS div ((c.Heads div hb) * blocks);
  if splits > most then
    splits := most;
  if splits < 1 then
    splits := 1;
  if splits > 1 then
  begin
    NeedDevice(fVk, aCtx.fPart, Int64(c.Heads) * n * splits * (c.HeadDim + 2) * 4);
    part := aCtx.fPart;
  end
  else
    part := aCtx.fDevAttn; // bound, but not accessed by attention.comp
  // the inputs, written into the host-visible buffer before the batch
  p := aCtx.fIn.Mapped;
  fModel.Embed(aIds, n, p);
  fModel.Rotary(past, n, @p^[n * h], @p^[n * h + n * half]);
  fVk.BatchBegin;
  try
    fVk.BatchCopy(aCtx.fIn, 0, aCtx.fDevX, 0, Int64(n) * h * 4);
    fVk.BatchCopy(aCtx.fIn, Int64(n) * h * 4, aCtx.fDevCos, 0, Int64(n) * half * 2 * 4);
    qk.Rows := n;
    qk.Heads := c.Heads;
    qk.KvHeads := c.KvHeads;
    qk.HeadDim := c.HeadDim;
    qk.Past := past;
    qk.Eps := c.RmsEps;
    norm.Rows := n;
    norm.Cols := h;
    norm.Eps := c.RmsEps;
    att.Rows := n;
    att.Heads := c.Heads;
    att.HeadDim := c.HeadDim;
    att.KvHeads := c.KvHeads;
    att.Past := past;
    att.Splits := splits;
    att.HeadBlock := hb;
    att.RowBlock := rb;
    merge.Rows := n;
    merge.Heads := c.Heads;
    merge.HeadDim := c.HeadDim;
    merge.Splits := splits;
    // the input norm of layer 0; every later one comes with the residual of
    // the layer before (addnorm)
    RecordNorm(aCtx.fDevX, n, h, fModel.Layers[0].InputNorm, aCtx.fDevH);
    for l := 0 to c.Layers - 1 do
    begin
      ly := @fModel.Layers[l];
      // attention block: the three projections side by side (no barrier
      // between them: they only read fDevH), then per head the norms, rotary
      // positions and the store of k and v into the cache
      fBackend.RecordLinear(aCtx.fDevH, n, h, ly^.Q, qw, aCtx.fDevQ, false);
      fBackend.RecordLinear(aCtx.fDevH, n, h, ly^.K, kvw, aCtx.fDevK, false);
      fBackend.RecordLinear(aCtx.fDevH, n, h, ly^.V, kvw, aCtx.fDevV);
      fVk.BatchDispatch(fQkRope, [aCtx.fDevQ, aCtx.fDevK, aCtx.fDevV,
        Vector(ly^.QNorm, c.HeadDim), Vector(ly^.KNorm, c.HeadDim), aCtx.fDevCos,
        aCtx.fDevKeys[l], aCtx.fDevValues[l]], @qk, c.Heads + c.KvHeads, n);
      fVk.BatchDispatch(attention, [aCtx.fDevQ, aCtx.fDevKeys[l], aCtx.fDevValues[l],
        aCtx.fDevAttn, part], @att, c.Heads div hb, blocks, splits);
      if splits > 1 then
        fVk.BatchDispatch(fMerge, [part, aCtx.fDevAttn], @merge, c.Heads, n);
      fBackend.RecordLinear(aCtx.fDevAttn, n, qw, ly^.O, h, aCtx.fDevTmp);
      // residual and the feed-forward pre-norm, gate and up side by side,
      // SwiGLU, down
      fVk.BatchDispatch(fAddNorm, [aCtx.fDevX, aCtx.fDevTmp, Vector(ly^.PostNorm, h),
        aCtx.fDevH], @norm, n);
      fBackend.RecordLinear(aCtx.fDevH, n, h, ly^.Gate, f, aCtx.fDevGate, false);
      fBackend.RecordLinear(aCtx.fDevH, n, h, ly^.Up, f, aCtx.fDevUp);
      RecordElementwise(fSwiGlu, [aCtx.fDevGate, aCtx.fDevUp], n * f);
      fBackend.RecordLinear(aCtx.fDevGate, n, f, ly^.Down, h, aCtx.fDevTmp);
      // the residual, and the input norm of the next layer
      if l < c.Layers - 1 then
        fVk.BatchDispatch(fAddNorm, [aCtx.fDevX, aCtx.fDevTmp,
          Vector(fModel.Layers[l + 1].InputNorm, h), aCtx.fDevH], @norm, n)
      else
        RecordElementwise(fAdd, [aCtx.fDevX, aCtx.fDevTmp, aCtx.fDevX], n * h);
      // the device starts on the first layer, then gets every fourth while
      // the host records the rest (BatchFlush: at most 7 per batch)
      if (l = 0) or (l and 3 = 3) then
        fVk.BatchFlush;
    end;
    if outRows > 0 then
    begin
      if outRows = 1 then
      begin
        // the last row only: to the start of the scratch, its norm, the head
        fVk.BatchCopy(aCtx.fDevX, Int64(n - 1) * h * 4, aCtx.fDevTmp, 0, Int64(h) * 4);
        RecordNorm(aCtx.fDevTmp, 1, h, fModel.FinalNorm, aCtx.fDevH);
      end
      else
        RecordNorm(aCtx.fDevX, n, h, fModel.FinalNorm, aCtx.fDevH);
      fBackend.RecordLinear(aCtx.fDevH, outRows, h, fModel.Head, c.VocabSize,
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

procedure TQwen3Vulkan.Forward(aCtx: TQwen3Context; aIds: PIntegerArray;
  aCount: PtrInt; aLogits: PSingleArray; aAllLogits: boolean);
var
  i, n, done, vocab: PtrInt;
  last: boolean;
  ctx: TQwen3VkContext;
begin
  if (aCtx = nil) or (aCtx.Device <> self) then
    EInferError.RaiseUtf8('%.Forward: no context of this instance', [self]);
  ctx := TQwen3VkContext(aCtx); // only NewContext gives this device
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
    // the backend forgot the weights (ClearWeights): up again, before a batch
    if fEpoch <> fBackend.WeightEpoch then
      UploadWeights;
    done := 0;
    while done < aCount do
    begin
      n := ChunkRows(aCount - done, aAllLogits and (aLogits <> nil));
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

function TQwen3Vulkan.Reuse(aCtx: TQwen3Context; aIds: PIntegerArray;
  aCount: PtrInt): PtrInt;
begin
  if (aCtx = nil) or (aCtx.Device <> self) then
    EInferError.RaiseUtf8('%.Reuse: no context of this instance', [self]);
  result := 0;
  if aCount > 0 then
    result := aCtx.Common(aIds, aCount - 1);
  aCtx.Truncate(result);
end;

function TQwen3Vulkan.Greedy(aCtx: TQwen3Context; const aPrompt: array of integer;
  aMaxNew: PtrInt; const aStop: array of integer): TIntegerDynArray;
begin
  if (aCtx = nil) or (aCtx.Device <> self) then
    EInferError.RaiseUtf8('%.Greedy: no context of this instance', [self]);
  // the model hands a context of this device to Forward here
  result := fModel.Greedy(aCtx, aPrompt, aMaxNew, aStop);
end;

end.
