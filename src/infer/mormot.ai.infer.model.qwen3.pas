/// native inference engine - Qwen3 decoder (causal language model)
// - part of the mormot.ai.* extension (LandrixAI)
// - licensed under a MPL/GPL/LGPL three license - see LICENSE
// - clean-room from the published architecture (looked up in Hugging Face's
//   modeling_qwen3.py, Apache-2.0, no code taken; llama2.c, MIT, for the shape
//   of a decoder loop): token embedding, then per layer pre-norm self-attention
//   (RMSNorm, Q/K/V projections, RMSNorm of every Q and K head, rotary
//   positions, grouped-query causal attention, output projection) and a
//   pre-norm SwiGLU feed-forward block, both with a residual; a final RMSNorm
//   and the language-model head (tied to the embedding or its own matrix)
// - head_dim comes from the configuration: Qwen3-0.6B has 16 heads of 128 for
//   a hidden size of 1024, not 1024 / 16
// - weights: F32 is used in place (memory-mapped), BF16 widened to F32 once at
//   load (exact: BF16 is the upper half of an F32) - the FP32 path of phase 2;
//   or every matrix quantized to Q8_0 at load, byte for byte as ggml does it
//   (phase 3: a quarter of the memory traffic, the SIMD kernels of the
//   backend); the embedding rows are then read from the file per token
// - arithmetic: matrix products through the backend; norms, rotary positions
//   and attention in double, stored as single. Rotary angles follow the float32
//   steps of the reference (inverse frequency and angle rounded to single)
// - the model is immutable after loading and may serve several threads; every
//   sequence brings its own TQwen3Context with its KV cache and scratch buffers
unit mormot.ai.infer.model.qwen3;

{
  *****************************************************************************

    - TQwen3Config: the decoder hyper-parameters, from config.json
    - TQwen3Context: KV cache and scratch buffers of one sequence
    - TQwen3Model: weights validated at load, Forward over new tokens, Greedy

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
  mormot.core.variants,
  mormot.ai.infer.tensor,
  mormot.ai.infer.loader.safetensors,
  mormot.ai.infer.backend,
  mormot.ai.infer.backend.cpu,
  mormot.ai.infer.sampler; // InferArgMax

type
  /// decoder hyper-parameters
  TQwen3Config = record
    VocabSize: PtrInt;
    Hidden: PtrInt;
    Layers: PtrInt;
    Heads: PtrInt;
    KvHeads: PtrInt;
    HeadDim: PtrInt;
    Intermediate: PtrInt;
    MaxPositions: PtrInt;
    RmsEps: double;
    RopeTheta: double;
    TiedEmbeddings: boolean;
  end;

  /// the weights of one decoder layer - F32 views
  TQwen3Layer = record
    Q, K, V, O, Gate, Up, Down: TInferWeight;
    QNorm, KNorm, InputNorm, PostNorm: PSingleArray;
  end;
  TQwen3LayerDynArray = array of TQwen3Layer;

  /// called with intermediate states of a Forward, for diagnosis and tests
  // - aName: 'embed' (input of layer 0), 'layerNN' (output of layer NN),
  // 'layer00_attn' / 'layer00_mlp' (attention and MLP output of layer 0, before
  // the residual), 'norm' (final RMSNorm); aValues is [aRows, aCols] and only
  // valid during the call
  TOnQwen3State = procedure(const aName: RawUtf8; aValues: PSingleArray;
    aRows, aCols: PtrInt) of object;

  /// the KV cache and scratch buffers of one sequence
  // - one per concurrent sequence; may be reused for another sequence after
  // Reset, and for another model of the same shape
  TQwen3Context = class
  protected
    fLength, fCapacity: PtrInt;
    fKvWidth, fLayers: PtrInt;
    fOwner: integer; // serial of the model whose keys and values are cached
    fKeys, fValues: array of TSingleDynArray; // per layer [capacity, kvWidth]
    fX, fH, fQ, fK, fV, fAttn, fTmp, fGate, fUp: TSingleDynArray;
  public
    /// forget the cached tokens - the buffers stay allocated
    procedure Reset;
    /// tokens in the KV cache = position of the next token
    property Length: PtrInt
      read fLength;
  end;

  /// a Qwen3 causal language model over a safetensors file
  TQwen3Model = class
  protected
    fSerial: integer; // unique per instance: an address may be reused after Free
    fConfig: TQwen3Config;
    fFile: TSafeTensorsFile;
    fBackend: TInferBackend;
    fOwned: array of TSingleDynArray; // BF16 tensors widened at load
    fOwnedQ8: array of array of TInferBlockQ8_0; // matrices quantized at load
    fWeights: TInferDType;
    fEmbedSrc: PInferTensor; // the embedding as stored: F32 or BF16
    fEmbed, fHead: TInferWeight;
    fNorm: PSingleArray;
    fLayer: TQwen3LayerDynArray;
    fInvFreq: TSingleDynArray;
    procedure LoadConfig(const aConfigFile: TFileName);
    procedure LoadWeights;
    function Checked(const aName: RawUtf8; const aDims: array of Int64): PInferTensor;
    function Weight(const aName: RawUtf8; const aDims: array of Int64): PSingleArray;
    function Matrix(const aName: RawUtf8; aRows, aCols: Int64): TInferWeight;
    procedure Grow(aCtx: TQwen3Context; aTokens, aNew: PtrInt);
    procedure RmsNorm(x: PSingleArray; aRows, aCols: PtrInt; w: PSingleArray;
      y: PSingleArray);
    procedure Rope(x: PSingleArray; aRows, aHeads, aFirstPos: PtrInt);
    procedure Attention(aCtx: TQwen3Context; aLayer, aRows: PtrInt);
  public
    /// load <aFolder>config.json and <aFolder>model.safetensors
    // - raises EInferError when a setting is unsupported or a weight is missing
    // or shaped otherwise; aBackend is not owned
    // - aWeights: idtF32 (the reference arithmetic of phase 2) or idtQ8_0 (every
    // matrix quantized at load; needs the hidden, attention and feed-forward
    // widths to be multiples of 32)
    constructor Create(const aFolder: TFileName; aBackend: TInferBackend;
      aWeights: TInferDType = idtF32);
    /// unmap the weights
    destructor Destroy; override;
    /// a context sized for this model (KV cache grows on demand)
    function NewContext: TQwen3Context;
    /// append aCount tokens to the sequence of aCtx and compute them
    // - their positions are aCtx.Length .. aCtx.Length + aCount - 1; their keys
    // and values enter the KV cache
    // - aLogits: nil computes no logits; else it receives the logits of the
    // last token ([VocabSize]), or of every new token when aAllLogits
    // ([aCount, VocabSize])
    procedure Forward(aCtx: TQwen3Context; aIds: PIntegerArray; aCount: PtrInt;
      aLogits: PSingleArray; aAllLogits: boolean = false;
      const aOnState: TOnQwen3State = nil);
    /// greedy continuation: the prompt is fed after aCtx's cached tokens, then
    // the highest logit is taken until one of aStop or aMaxNew tokens
    // - returns the new ids, the stop token included when one was produced;
    // ties go to the lowest id (like torch.argmax)
    // - afterwards aCtx holds the prompt and every returned token - except the
    // last one when the context reached MaxPositions
    function Greedy(aCtx: TQwen3Context; const aPrompt: array of integer;
      aMaxNew: PtrInt; const aStop: array of integer): TIntegerDynArray;
    /// the hyper-parameters
    property Config: TQwen3Config
      read fConfig;
    /// the backend the model computes with (not owned)
    property Backend: TInferBackend
      read fBackend;
    /// the type of the matrices: idtF32 or idtQ8_0
    property Weights: TInferDType
      read fWeights;
  end;



implementation


/// aValue rounded to the nearest single - a cast of a double is no conversion
// on every compiler, an assignment is
function RoundToSingle(aValue: double): single;
begin
  result := aValue;
end;


{ TQwen3Context }

procedure TQwen3Context.Reset;
begin
  fLength := 0;
end;


{ TQwen3Model }

const
  REQUIRED_KEYS: array[0..9] of RawUtf8 = (
    'vocab_size', 'hidden_size', 'num_hidden_layers', 'num_attention_heads',
    'num_key_value_heads', 'head_dim', 'intermediate_size',
    'max_position_embeddings', 'rms_norm_eps', 'hidden_act');

var
  /// the last model serial handed out (0 = none, an empty context's owner)
  Qwen3ModelSerial: integer;

constructor TQwen3Model.Create(const aFolder: TFileName; aBackend: TInferBackend;
  aWeights: TInferDType);
begin
  inherited Create;
  fSerial := InterlockedIncrement(Qwen3ModelSerial);
  if aBackend = nil then
    EInferError.RaiseUtf8('%.Create: no backend', [self]);
  if not (aWeights in [idtF32, idtQ8_0]) then
    EInferError.RaiseUtf8('%.Create: weights as % not supported',
      [self, INFER_DTYPE_NAME[aWeights]]);
  fBackend := aBackend;
  fWeights := aWeights;
  if not fBackend.Supports(ioLinear, aWeights) then
    EInferError.RaiseUtf8('%: the backend has no % Linear',
      [self, INFER_DTYPE_NAME[aWeights]]);
  if not fBackend.Supports(ioCausalAttention, idtF32) then
    EInferError.RaiseUtf8('%: the backend has no causal attention', [self]);
  LoadConfig(IncludeTrailingPathDelimiter(aFolder) + 'config.json');
  if (aWeights = idtQ8_0) and
     ((fConfig.Hidden and 31 <> 0) or (fConfig.Intermediate and 31 <> 0) or
      ((fConfig.Heads * fConfig.HeadDim) and 31 <> 0)) then
    EInferError.RaiseUtf8('%: Q8_0 needs widths that are multiples of 32', [self]);
  fFile := TSafeTensorsFile.Create(IncludeTrailingPathDelimiter(aFolder) +
    'model.safetensors');
  LoadWeights;
end;

destructor TQwen3Model.Destroy;
begin
  fFile.Free;
  inherited Destroy;
end;

procedure TQwen3Model.LoadConfig(const aConfigFile: TFileName);
var
  doc: TDocVariantData;
  rope, types: PDocVariantData;
  i: PtrInt;
begin
  if not doc.InitJsonFromFile(aConfigFile, JSON_FAST_FLOAT) then
    EInferError.RaiseUtf8('%: cannot read %', [self, aConfigFile]);
  if doc.U['model_type'] <> 'qwen3' then
    EInferError.RaiseUtf8('%: model_type "%" is not qwen3', [self, doc.U['model_type']]);
  // a missing key would read as 0: refused, also head_dim (no hidden / heads
  // fallback - that is exactly the value Qwen3 does not have)
  for i := 0 to high(REQUIRED_KEYS) do
    if doc.GetValueIndex(REQUIRED_KEYS[i]) < 0 then
      EInferError.RaiseUtf8('%: % has no "%"', [self, aConfigFile, REQUIRED_KEYS[i]]);
  if doc.U['hidden_act'] <> 'silu' then
    EInferError.RaiseUtf8('%: hidden_act "%" not supported', [self, doc.U['hidden_act']]);
  if doc.B['attention_bias'] then
    EInferError.RaiseUtf8('%: attention_bias not supported', [self]);
  if doc.B['use_sliding_window'] then
    EInferError.RaiseUtf8('%: sliding window attention not supported', [self]);
  if doc.GetAsArray('layer_types', types) then
    for i := 0 to types^.Count - 1 do
      if VariantToUtf8(types^.Values[i]) <> 'full_attention' then
        EInferError.RaiseUtf8('%: layer type "%" not supported',
          [self, VariantToUtf8(types^.Values[i])]);
  // rope_theta at the top (transformers 4) or in rope_parameters (5), and
  // plain RoPE only: a scaling would change every angle
  if doc.GetAsObject('rope_parameters', rope) then
  begin
    if (rope^.U['rope_type'] <> '') and (rope^.U['rope_type'] <> 'default') then
      EInferError.RaiseUtf8('%: rope_type "%" not supported', [self, rope^.U['rope_type']]);
    fConfig.RopeTheta := rope^.D['rope_theta'];
  end
  else
    fConfig.RopeTheta := doc.D['rope_theta'];
  if doc.GetAsObject('rope_scaling', rope) and (rope^.Count > 0) then
    EInferError.RaiseUtf8('%: rope_scaling not supported', [self]);
  fConfig.VocabSize := doc.I['vocab_size'];
  fConfig.Hidden := doc.I['hidden_size'];
  fConfig.Layers := doc.I['num_hidden_layers'];
  fConfig.Heads := doc.I['num_attention_heads'];
  fConfig.KvHeads := doc.I['num_key_value_heads'];
  fConfig.HeadDim := doc.I['head_dim'];
  fConfig.Intermediate := doc.I['intermediate_size'];
  fConfig.MaxPositions := doc.I['max_position_embeddings'];
  fConfig.RmsEps := doc.D['rms_norm_eps'];
  fConfig.TiedEmbeddings := doc.B['tie_word_embeddings'];
  // every dimension bounded on its own before any product: a product of
  // unbounded values could wrap around and pass
  if (fConfig.VocabSize <= 0) or (fConfig.VocabSize > 1 shl 22) or
     (fConfig.Hidden <= 0) or (fConfig.Hidden > 1 shl 16) or
     (fConfig.Layers <= 0) or (fConfig.Layers > 1024) or
     (fConfig.Heads <= 0) or (fConfig.Heads > 1 shl 12) or
     (fConfig.KvHeads <= 0) or (fConfig.KvHeads > fConfig.Heads) or
     (fConfig.Heads mod fConfig.KvHeads <> 0) or
     (fConfig.HeadDim <= 0) or (fConfig.HeadDim > 1024) or odd(fConfig.HeadDim) or
     (fConfig.Heads * fConfig.HeadDim > 1 shl 16) or
     (fConfig.Intermediate <= 0) or (fConfig.Intermediate > 1 shl 18) or
     (fConfig.MaxPositions <= 0) or (fConfig.MaxPositions > 1 shl 24) or
     not (fConfig.RmsEps > 0) or (fConfig.RmsEps > 1) or
     not (fConfig.RopeTheta > 1) or (fConfig.RopeTheta > 1e12) then
    EInferError.RaiseUtf8('%: % has an invalid or unsupported shape', [self, aConfigFile]);
  // inverse rotary frequencies as the reference forms them in float32:
  // 1 / theta ^ (2i / d), each step rounded to single
  SetLength(fInvFreq, fConfig.HeadDim shr 1);
  for i := 0 to high(fInvFreq) do
    fInvFreq[i] := 1 / RoundToSingle(Exp(Ln(fConfig.RopeTheta) *
      RoundToSingle((2 * i) / fConfig.HeadDim)));
end;

function TQwen3Model.Checked(const aName: RawUtf8;
  const aDims: array of Int64): PInferTensor;
var
  i: PtrInt;
  ok: boolean;
  want: TInt64DynArray;
  t: PInferTensor;
begin
  t := fFile.Tensor(aName);
  ok := length(t^.Shape) = length(aDims);
  if ok then
    for i := 0 to high(aDims) do
      if t^.Shape[i] <> aDims[i] then
        ok := false;
  if not ok then
  begin
    SetLength(want, length(aDims));
    for i := 0 to high(aDims) do
      want[i] := aDims[i];
    EInferError.RaiseUtf8('%: % is shaped %, expected %',
      [self, aName, InferShapeText(t^.Shape), InferShapeText(want)]);
  end;
  if not (t^.DType in [idtF32, idtBF16]) then
    EInferError.RaiseUtf8('%: % is %, only F32 and BF16 are supported',
      [self, aName, INFER_DTYPE_NAME[t^.DType]]);
  result := t;
end;

function TQwen3Model.Weight(const aName: RawUtf8;
  const aDims: array of Int64): PSingleArray;
var
  t: PInferTensor;
  i, n: PtrInt;
  src: PWordArray;
  dst: PCardinalArray;
begin
  result := nil; // every other type raises in Checked
  t := Checked(aName, aDims);
  case t^.DType of
    idtF32:
      result := t^.Data;
    idtBF16:
      begin
        // BF16 is the upper half of an F32: widening is exact
        if t^.Count > High(PtrInt) div 4 then
          EInferError.RaiseUtf8('%: % too large', [self, aName]);
        n := length(fOwned);
        SetLength(fOwned, n + 1);
        SetLength(fOwned[n], t^.Count);
        src := t^.Data;
        dst := pointer(fOwned[n]);
        for i := 0 to t^.Count - 1 do
          dst^[i] := cardinal(src^[i]) shl 16;
        result := pointer(fOwned[n]);
      end;
  end;
end;

/// row aRow of an F32 or BF16 matrix of aCols columns, as singles in aDest
procedure WidenRow(t: PInferTensor; aRow, aCols: PtrInt; aDest: PSingleArray);
var
  i: PtrInt;
  src: PWordArray;
begin
  if t^.DType = idtF32 then
    MoveFast(PSingleArray(t^.Data)^[aRow * aCols], aDest^, aCols * SizeOf(single))
  else
  begin
    // BF16 is the upper half of an F32: widening is exact
    src := @PWordArray(t^.Data)^[aRow * aCols];
    for i := 0 to aCols - 1 do
      PCardinalArray(aDest)^[i] := cardinal(src^[i]) shl 16;
  end;
end;

function TQwen3Model.Matrix(const aName: RawUtf8; aRows, aCols: Int64): TInferWeight;
var
  t: PInferTensor;
  row: TSingleDynArray;
  nb, r, n, c: PtrInt;
begin
  if fWeights = idtF32 then
  begin
    result := InferWeight(Weight(aName, [aRows, aCols]), idtF32);
    exit;
  end;
  // Q8_0: row by row through single precision, as ggml quantizes
  t := Checked(aName, [aRows, aCols]);
  nb := aCols shr 5;
  n := length(fOwnedQ8);
  SetLength(fOwnedQ8, n + 1);
  SetLength(fOwnedQ8[n], aRows * nb);
  SetLength(row, aCols);
  for r := 0 to aRows - 1 do
  begin
    WidenRow(t, r, aCols, pointer(row));
    // a NaN or an infinity has no Q8_0 form: refused, by its bits (no FPU)
    for c := 0 to aCols - 1 do
      if PCardinalArray(row)^[c] and $7f800000 = $7f800000 then
        EInferError.RaiseUtf8('%: % has a NaN or infinite weight at row % column %',
          [self, aName, r, c]);
    InferQuantizeRowQ8_0(pointer(row), aCols, @fOwnedQ8[n][r * nb]);
    // a finite weight beyond 127 * 65504 still gets the FP16 scale +Inf (ggml
    // writes it too): every product with it would be Inf or NaN
    for c := r * nb to r * nb + nb - 1 do
      if fOwnedQ8[n][c].d and $7fff >= $7c00 then
        EInferError.RaiseUtf8('%: % has a weight beyond the Q8_0 range in row %',
          [self, aName, r]);
  end;
  result := InferWeight(pointer(fOwnedQ8[n]), idtQ8_0);
end;

procedure TQwen3Model.LoadWeights;
var
  i, used: PtrInt;
  h, f, qw, kvw: Int64;
  p: RawUtf8;
  l: ^TQwen3Layer;
  head, emb: PInferTensor;
begin
  h := fConfig.Hidden;
  f := fConfig.Intermediate;
  qw := fConfig.Heads * fConfig.HeadDim;
  kvw := fConfig.KvHeads * fConfig.HeadDim;
  fEmbedSrc := Checked('model.embed_tokens.weight', [fConfig.VocabSize, h]);
  if fWeights = idtF32 then
    fEmbed := InferWeight(Weight('model.embed_tokens.weight', [fConfig.VocabSize, h]),
      idtF32);
  fNorm := Weight('model.norm.weight', [h]);
  used := 2;
  head := fFile.Find('lm_head.weight');
  if fConfig.TiedEmbeddings then
  begin
    // a tied head may still be stored: only when it is the very same matrix
    if head <> nil then
    begin
      emb := fFile.Tensor('model.embed_tokens.weight');
      if (head^.DType <> emb^.DType) or (head^.Bytes <> emb^.Bytes) or
         not CompareMem(head^.Data, emb^.Data, head^.Bytes) then
        EInferError.RaiseUtf8('%: tie_word_embeddings but lm_head.weight differs ' +
          'from the embedding', [self]);
      inc(used);
    end;
    if fWeights = idtF32 then
      fHead := fEmbed
    else
      fHead := Matrix('model.embed_tokens.weight', fConfig.VocabSize, h);
  end
  else
  begin
    fHead := Matrix('lm_head.weight', fConfig.VocabSize, h);
    inc(used);
  end;
  SetLength(fLayer, fConfig.Layers);
  for i := 0 to fConfig.Layers - 1 do
  begin
    p := FormatUtf8('model.layers.%.', [i]);
    l := @fLayer[i];
    l^.Q := Matrix(p + 'self_attn.q_proj.weight', qw, h);
    l^.K := Matrix(p + 'self_attn.k_proj.weight', kvw, h);
    l^.V := Matrix(p + 'self_attn.v_proj.weight', kvw, h);
    l^.O := Matrix(p + 'self_attn.o_proj.weight', h, qw);
    l^.QNorm := Weight(p + 'self_attn.q_norm.weight', [fConfig.HeadDim]);
    l^.KNorm := Weight(p + 'self_attn.k_norm.weight', [fConfig.HeadDim]);
    l^.Gate := Matrix(p + 'mlp.gate_proj.weight', f, h);
    l^.Up := Matrix(p + 'mlp.up_proj.weight', f, h);
    l^.Down := Matrix(p + 'mlp.down_proj.weight', h, f);
    l^.InputNorm := Weight(p + 'input_layernorm.weight', [h]);
    l^.PostNorm := Weight(p + 'post_attention_layernorm.weight', [h]);
    inc(used, 11);
  end;
  // an unknown tensor means a model we do not understand
  if used <> length(fFile.Tensors) then
    EInferError.RaiseUtf8('%: % tensors in the file, % understood',
      [self, length(fFile.Tensors), used]);
end;

function TQwen3Model.NewContext: TQwen3Context;
begin
  result := TQwen3Context.Create;
end;

procedure Need(var aBuffer: TSingleDynArray; aCount: PtrInt); overload;
begin
  if length(aBuffer) < aCount then
    SetLength(aBuffer, aCount);
end;

procedure TQwen3Model.Grow(aCtx: TQwen3Context; aTokens, aNew: PtrInt);
var
  i, cap, kvw: PtrInt;
begin
  // the cache belongs to one model: a context holding another model's
  // sequence is refused (same shape or not), an empty one is taken over
  kvw := fConfig.KvHeads * fConfig.HeadDim;
  if aCtx.fOwner <> fSerial then
  begin
    if aCtx.fLength <> 0 then
      EInferError.RaiseUtf8('%.Forward: context holds a sequence of another model',
        [self]);
    aCtx.fOwner := fSerial;
  end;
  if (aCtx.fKvWidth <> kvw) or (aCtx.fLayers <> fConfig.Layers) then
  begin
    aCtx.fKvWidth := kvw;
    aCtx.fLayers := fConfig.Layers;
    aCtx.fCapacity := 0;
    SetLength(aCtx.fKeys, fConfig.Layers);
    SetLength(aCtx.fValues, fConfig.Layers);
  end;
  if aTokens > aCtx.fCapacity then
  begin
    cap := aCtx.fCapacity * 2;
    if cap < aTokens then
      cap := aTokens;
    if cap < 64 then
      cap := 64;
    if cap > fConfig.MaxPositions then
      cap := fConfig.MaxPositions;
    // the byte size must fit PtrInt (a 32-bit process): SetLength would wrap
    if cap > (High(PtrInt) div SizeOf(single)) div kvw then
      EInferError.RaiseUtf8('%.Forward: a KV cache of % tokens is too large for ' +
        'this process', [self, cap]);
    for i := 0 to fConfig.Layers - 1 do
    begin
      SetLength(aCtx.fKeys[i], cap * kvw);
      SetLength(aCtx.fValues[i], cap * kvw);
    end;
    aCtx.fCapacity := cap;
  end;
  Need(aCtx.fX, aNew * fConfig.Hidden);
  Need(aCtx.fH, aNew * fConfig.Hidden);
  Need(aCtx.fTmp, aNew * fConfig.Hidden);
  Need(aCtx.fQ, aNew * fConfig.Heads * fConfig.HeadDim);
  Need(aCtx.fAttn, aNew * fConfig.Heads * fConfig.HeadDim);
  Need(aCtx.fK, aNew * kvw);
  Need(aCtx.fV, aNew * kvw);
  Need(aCtx.fGate, aNew * fConfig.Intermediate);
  Need(aCtx.fUp, aNew * fConfig.Intermediate);
end;

procedure TQwen3Model.RmsNorm(x: PSingleArray; aRows, aCols: PtrInt;
  w: PSingleArray; y: PSingleArray);
var
  r, c: PtrInt;
  sum, a, inv: double;
  xr, yr: PSingleArray;
  n: single;
begin
  for r := 0 to aRows - 1 do
  begin
    xr := @x^[r * aCols];
    yr := @y^[r * aCols];
    sum := 0;
    for c := 0 to aCols - 1 do
    begin
      a := xr^[c];
      sum := sum + a * a;
    end;
    inv := 1 / Sqrt(sum / aCols + fConfig.RmsEps);
    for c := 0 to aCols - 1 do
    begin
      // the reference rounds the normalized value to single before the weight
      n := xr^[c] * inv;
      yr^[c] := w^[c] * n;
    end;
  end;
end;

procedure TQwen3Model.Rope(x: PSingleArray; aRows, aHeads, aFirstPos: PtrInt);
var
  r, h, i, half, d: PtrInt;
  angle: single;
  c, s, a, b: double;
  v: PSingleArray;
begin
  d := fConfig.HeadDim;
  half := d shr 1;
  for r := 0 to aRows - 1 do
    for i := 0 to half - 1 do
    begin
      // the angle in single, as the reference's float32 product
      angle := fInvFreq[i] * (aFirstPos + r);
      c := RoundToSingle(Cos(angle));
      s := RoundToSingle(Sin(angle));
      for h := 0 to aHeads - 1 do
      begin
        v := @x^[(r * aHeads + h) * d];
        a := v^[i];
        b := v^[i + half];
        v^[i] := a * c - b * s;
        v^[i + half] := b * c + a * s;
      end;
    end;
end;

procedure TQwen3Model.Attention(aCtx: TQwen3Context; aLayer, aRows: PtrInt);
begin
  // the reference backend computes as phase 2 did (double); a fast one splits
  // by head and row block
  fBackend.CausalAttention(pointer(aCtx.fQ), aRows, fConfig.Heads, fConfig.HeadDim,
    pointer(aCtx.fKeys[aLayer]), pointer(aCtx.fValues[aLayer]), aCtx.fLength,
    fConfig.KvHeads, pointer(aCtx.fAttn));
end;

/// 'layerNN' - two digits at least, like the reference ('layer%02d')
function LayerName(aLayer: PtrInt): RawUtf8;
begin
  if aLayer < 10 then
    result := FormatUtf8('layer0%', [aLayer])
  else
    result := FormatUtf8('layer%', [aLayer]);
end;

procedure TQwen3Model.Forward(aCtx: TQwen3Context; aIds: PIntegerArray;
  aCount: PtrInt; aLogits: PSingleArray; aAllLogits: boolean;
  const aOnState: TOnQwen3State);
var
  i, l, n, h, f, qw, kvw, past, id: PtrInt;
  ly: ^TQwen3Layer;
  a, b: double;
begin
  if aCtx = nil then
    EInferError.RaiseUtf8('%.Forward: no context', [self]);
  if aCount <= 0 then
    EInferError.RaiseUtf8('%.Forward: no tokens', [self]);
  past := aCtx.fLength;
  if aCount > fConfig.MaxPositions - past then
    EInferError.RaiseUtf8('%.Forward: % + % tokens exceed the % positions',
      [self, past, aCount, fConfig.MaxPositions]);
  n := aCount;
  h := fConfig.Hidden;
  f := fConfig.Intermediate;
  qw := fConfig.Heads * fConfig.HeadDim;
  kvw := fConfig.KvHeads * fConfig.HeadDim;
  if n > (High(PtrInt) div SizeOf(single)) div MaxPtrInt(f, MaxPtrInt(h, qw)) then
    EInferError.RaiseUtf8('%.Forward: % tokens too many at once', [self, n]);
  for i := 0 to n - 1 do
  begin
    id := aIds^[i];
    if (id < 0) or (id >= fConfig.VocabSize) then
      EInferError.RaiseUtf8('%.Forward: token id % outside the vocabulary', [self, id]);
  end;
  Grow(aCtx, past + n, n);
  // embeddings
  for i := 0 to n - 1 do
    WidenRow(fEmbedSrc, aIds^[i], h, @aCtx.fX[i * h]);
  if Assigned(aOnState) then
    aOnState('embed', pointer(aCtx.fX), n, h);
  for l := 0 to fConfig.Layers - 1 do
  begin
    ly := @fLayer[l];
    // attention block: pre-norm, projections, per-head norms, rotary positions
    RmsNorm(pointer(aCtx.fX), n, h, ly^.InputNorm, pointer(aCtx.fH));
    fBackend.Linear(pointer(aCtx.fH), n, h, ly^.Q, qw, nil, pointer(aCtx.fQ));
    fBackend.Linear(pointer(aCtx.fH), n, h, ly^.K, kvw, nil, pointer(aCtx.fK));
    fBackend.Linear(pointer(aCtx.fH), n, h, ly^.V, kvw, nil, pointer(aCtx.fV));
    RmsNorm(pointer(aCtx.fQ), n * fConfig.Heads, fConfig.HeadDim, ly^.QNorm,
      pointer(aCtx.fQ));
    RmsNorm(pointer(aCtx.fK), n * fConfig.KvHeads, fConfig.HeadDim, ly^.KNorm,
      pointer(aCtx.fK));
    Rope(pointer(aCtx.fQ), n, fConfig.Heads, past);
    Rope(pointer(aCtx.fK), n, fConfig.KvHeads, past);
    MoveFast(pointer(aCtx.fK)^, aCtx.fKeys[l][past * kvw], n * kvw * SizeOf(single));
    MoveFast(pointer(aCtx.fV)^, aCtx.fValues[l][past * kvw], n * kvw * SizeOf(single));
    Attention(aCtx, l, n);
    fBackend.Linear(pointer(aCtx.fAttn), n, qw, ly^.O, h, nil, pointer(aCtx.fTmp));
    if Assigned(aOnState) and (l = 0) then
      aOnState('layer00_attn', pointer(aCtx.fTmp), n, h);
    fBackend.Add(pointer(aCtx.fX), pointer(aCtx.fTmp), n * h);
    // feed-forward block: pre-norm, SwiGLU, residual
    RmsNorm(pointer(aCtx.fX), n, h, ly^.PostNorm, pointer(aCtx.fH));
    fBackend.Linear(pointer(aCtx.fH), n, h, ly^.Gate, f, nil, pointer(aCtx.fGate));
    fBackend.Linear(pointer(aCtx.fH), n, h, ly^.Up, f, nil, pointer(aCtx.fUp));
    for i := 0 to n * f - 1 do
    begin
      // silu(g) * u, each step rounded to single like the reference
      a := aCtx.fGate[i];
      b := RoundToSingle(a / (1 + Exp(-a)));
      aCtx.fGate[i] := b * aCtx.fUp[i];
    end;
    fBackend.Linear(pointer(aCtx.fGate), n, f, ly^.Down, h, nil, pointer(aCtx.fTmp));
    if Assigned(aOnState) and (l = 0) then
      aOnState('layer00_mlp', pointer(aCtx.fTmp), n, h);
    fBackend.Add(pointer(aCtx.fX), pointer(aCtx.fTmp), n * h);
    if Assigned(aOnState) then
      aOnState(LayerName(l), pointer(aCtx.fX), n, h);
  end;
  aCtx.fLength := past + n;
  if (aLogits = nil) and not Assigned(aOnState) then
    exit;
  RmsNorm(pointer(aCtx.fX), n, h, fNorm, pointer(aCtx.fH));
  if Assigned(aOnState) then
    aOnState('norm', pointer(aCtx.fH), n, h);
  if aLogits = nil then
    exit;
  if aAllLogits then
    fBackend.Linear(pointer(aCtx.fH), n, h, fHead, fConfig.VocabSize, nil, aLogits)
  else
    fBackend.Linear(@aCtx.fH[(n - 1) * h], 1, h, fHead, fConfig.VocabSize, nil,
      aLogits);
end;

function TQwen3Model.Greedy(aCtx: TQwen3Context; const aPrompt: array of integer;
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
  SetLength(logits, fConfig.VocabSize);
  Forward(aCtx, @aPrompt[0], length(aPrompt), pointer(logits));
  SetLength(result, aMaxNew);
  n := 0;
  repeat
    next := InferArgMax(pointer(logits), fConfig.VocabSize);
    result[n] := next;
    inc(n);
    stop := n = aMaxNew;
    for i := 0 to high(aStop) do
      if aStop[i] = next then
        stop := true;
    // every returned token enters the cache, so a conversation can go on;
    // only a full context leaves the last one out
    if aCtx.fLength >= fConfig.MaxPositions then
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
