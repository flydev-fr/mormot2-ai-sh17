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
//   or every matrix quantized to Q8_0 (or Q4_0) at load, byte for byte as a
//   GGUF writer does it (phase 3: a quarter of the memory traffic, the SIMD
//   kernels of the backend); the embedding rows are then read per token
// - or a llama.cpp GGUF file of the qwen3 architecture (CreateFromGguf): its
//   metadata for the settings, its Q8_0 / Q4_0 matrices used in place
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
  mormot.ai.infer.loader.gguf,
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
  // - remembers the id at every cached position: a prompt that starts with
  // the same tokens keeps their keys and values (Common, then Truncate) - the
  // cache of position i depends on the tokens 0..i only
  TQwen3Context = class
  protected
    fLength, fCapacity: PtrInt;
    fKvWidth, fLayers: PtrInt;
    fOwner: integer; // serial of the model whose keys and values are cached
    fKeys, fValues: array of TSingleDynArray; // per layer [capacity, kvWidth]
    fIds: TIntegerDynArray; // [capacity] the token at every cached position
    fX, fH, fQ, fK, fV, fAttn, fTmp, fGate, fUp: TSingleDynArray;
    fCos, fSin: TSingleDynArray; // [new tokens, HeadDim / 2] rotary factors
    function GetId(aPosition: PtrInt): integer;
  public
    /// forget the cached tokens - the buffers stay allocated
    procedure Reset;
    /// keep the first aLength cached tokens, forget the others
    // - raises EInferError unless 0 <= aLength <= Length
    procedure Truncate(aLength: PtrInt);
    /// how many of the first tokens of aIds[0..aCount-1] the cache holds:
    // the length of the common prefix
    function Common(aIds: PIntegerArray; aCount: PtrInt): PtrInt;
    /// tokens in the KV cache = position of the next token
    property Length: PtrInt
      read fLength;
    /// the token cached at aPosition (0 .. Length - 1)
    property Id[aPosition: PtrInt]: integer
      read GetId;
  end;

  /// a Qwen3 causal language model over a safetensors or a GGUF file
  TQwen3Model = class
  protected
    fSerial: integer; // unique per instance: an address may be reused after Free
    fConfig: TQwen3Config;
    fFile: TSafeTensorsFile; // Create: a Hugging Face folder
    fGguf: TGgufFile; // CreateFromGguf: one llama.cpp file
    fBackend: TInferBackend;
    fOwned: array of TSingleDynArray; // BF16 tensors widened at load
    fOwnedQ8: array of array of TInferBlockQ8_0; // matrices quantized at load
    fOwnedQ4: array of array of TInferBlockQ4_0;
    fWeights: TInferDType;
    fEmbedSrc: PInferTensor; // the embedding as stored: read per token
    fEmbed, fHead: TInferWeight;
    fNorm: PSingleArray;
    fLayer: TQwen3LayerDynArray;
    fInvFreq: TSingleDynArray;
    procedure Init(aBackend: TInferBackend);
    procedure LoadConfig(const aConfigFile: TFileName);
    procedure LoadGgufConfig;
    procedure CheckConfig(const aSource: TFileName);
    procedure LoadWeights;
    /// a tensor by its Hugging Face name - in a GGUF file by llama.cpp's name
    function Find(const aName: RawUtf8): PInferTensor;
    function TensorCount: PtrInt;
    function Checked(const aName: RawUtf8; const aDims: array of Int64;
      aBlocks: boolean = false): PInferTensor;
    function Weight(const aName: RawUtf8; const aDims: array of Int64): PSingleArray;
    function Matrix(const aName: RawUtf8; aRows, aCols: Int64;
      aType: TInferDType): TInferWeight;
    procedure Grow(aCtx: TQwen3Context; aTokens, aNew: PtrInt);
    procedure RopeFactors(aCtx: TQwen3Context; aRows, aFirstPos: PtrInt);
    procedure Attention(aCtx: TQwen3Context; aLayer, aRows: PtrInt);
  public
    /// load <aFolder>config.json and <aFolder>model.safetensors
    // - raises EInferError when a setting is unsupported or a weight is missing
    // or shaped otherwise; aBackend is not owned
    // - aWeights: idtF32 (the reference arithmetic of phase 2), idtQ8_0 (every
    // matrix quantized at load) or idtQ4_0 (the layers in Q4_0, the head in
    // Q8_0); both need the hidden, attention and feed-forward widths to be
    // multiples of 32
    constructor Create(const aFolder: TFileName; aBackend: TInferBackend;
      aWeights: TInferDType = idtF32);
    /// load a llama.cpp GGUF file of the qwen3 architecture: the settings
    // from its metadata, every matrix as stored (Q8_0 and Q4_0 used in place,
    // F16 and BF16 widened to F32), a head tied to token_embd when the file
    // has no output.weight
    // - raises EInferError for another architecture, a tensor of another type
    // or shape, a scale that is not finite, or a tensor not understood
    constructor CreateFromGguf(const aFileName: TFileName; aBackend: TInferBackend);
    /// unmap the weights
    destructor Destroy; override;
    /// a context sized for this model (KV cache grows on demand)
    function NewContext: TQwen3Context;
    /// append aCount tokens to the sequence of aCtx and compute them
    // - their positions are aCtx.Length .. aCtx.Length + aCount - 1; their ids,
    // keys and values enter the KV cache - Length only grows once every layer
    // is computed, so an exception in the layers leaves the tokens it held
    // - aLogits: nil computes no logits; else it receives the logits of the
    // last token ([VocabSize]), or of every new token when aAllLogits
    // ([aCount, VocabSize])
    procedure Forward(aCtx: TQwen3Context; aIds: PIntegerArray; aCount: PtrInt;
      aLogits: PSingleArray; aAllLogits: boolean = false;
      const aOnState: TOnQwen3State = nil);
    /// keep the cached tokens of aCtx that start aIds[0..aCount-1], forget the
    // others, and return how many stayed: the rest of aIds is to be computed
    // - at most aCount - 1 stay, so a Forward of the rest has a token to give
    // logits for; none when aCtx holds a sequence of another model
    function Reuse(aCtx: TQwen3Context; aIds: PIntegerArray; aCount: PtrInt): PtrInt;
    /// greedy continuation: the prompt is fed after aCtx's cached tokens, then
    // the highest logit is taken until one of aStop or aMaxNew tokens
    // - returns the new ids, the stop token included when one was produced;
    // ties go to the lowest id (like torch.argmax)
    // - afterwards aCtx holds the prompt and every returned token - except the
    // last one when the context reached MaxPositions
    function Greedy(aCtx: TQwen3Context; const aPrompt: array of integer;
      aMaxNew: PtrInt; const aStop: array of integer): TIntegerDynArray;
    /// the F32 embeddings of aIds[0..aCount-1] into aDest ([aCount, Hidden]),
    // as Forward takes them (for another executor, e.g. the GPU)
    // - raises EInferError for an id outside the vocabulary
    procedure Embed(aIds: PIntegerArray; aCount: PtrInt; aDest: PSingleArray);
    /// the rotary factors of aRows positions from aFirstPos into aCos and aSin
    // ([aRows, HeadDim / 2] each), as Forward computes them
    procedure Rotary(aFirstPos, aRows: PtrInt; aCos, aSin: PSingleArray);
    /// the weights of the final RMSNorm, as used
    property FinalNorm: PSingleArray
      read fNorm;
    /// the hyper-parameters
    property Config: TQwen3Config
      read fConfig;
    /// the backend the model computes with (not owned)
    property Backend: TInferBackend
      read fBackend;
    /// the type of the layer matrices: idtF32, idtQ8_0 or idtQ4_0 (for a GGUF
    // file: the type of the first attention matrix)
    property Weights: TInferDType
      read fWeights;
    /// the weights of every layer, as used - for diagnosis and tests
    property Layers: TQwen3LayerDynArray
      read fLayer;
    /// the language-model head, as used
    property Head: TInferWeight
      read fHead;
    /// the GGUF file of CreateFromGguf (nil after Create): its metadata holds
    // the tokenizer too
    property Gguf: TGgufFile
      read fGguf;
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

procedure TQwen3Context.Truncate(aLength: PtrInt);
begin
  if (aLength < 0) or (aLength > fLength) then
    EInferError.RaiseUtf8('%.Truncate(%) of % tokens', [self, aLength, fLength]);
  fLength := aLength;
end;

function TQwen3Context.Common(aIds: PIntegerArray; aCount: PtrInt): PtrInt;
begin
  result := 0;
  if aCount > fLength then
    aCount := fLength;
  while (result < aCount) and (fIds[result] = aIds^[result]) do
    inc(result);
end;

function TQwen3Context.GetId(aPosition: PtrInt): integer;
begin
  if (aPosition < 0) or (aPosition >= fLength) then
    EInferError.RaiseUtf8('%.Id[%] of % tokens', [self, aPosition, fLength]);
  result := fIds[aPosition];
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

procedure TQwen3Model.Init(aBackend: TInferBackend);
begin
  fSerial := InterlockedIncrement(Qwen3ModelSerial);
  if aBackend = nil then
    EInferError.RaiseUtf8('%.Create: no backend', [self]);
  fBackend := aBackend;
  if not fBackend.Supports(ioCausalAttention, idtF32) or
     not fBackend.Supports(ioRmsNorm, idtF32) or
     not fBackend.Supports(ioSwiGlu, idtF32) or
     not fBackend.Supports(ioRope, idtF32) then
    EInferError.RaiseUtf8('%: the backend lacks a decoder operation', [self]);
end;

constructor TQwen3Model.Create(const aFolder: TFileName; aBackend: TInferBackend;
  aWeights: TInferDType);
begin
  inherited Create;
  Init(aBackend);
  if not (aWeights in [idtF32, idtQ8_0, idtQ4_0]) then
    EInferError.RaiseUtf8('%.Create: weights as % not supported',
      [self, INFER_DTYPE_NAME[aWeights]]);
  fWeights := aWeights;
  if not fBackend.Supports(ioLinear, aWeights) then
    EInferError.RaiseUtf8('%: the backend has no % Linear',
      [self, INFER_DTYPE_NAME[aWeights]]);
  LoadConfig(IncludeTrailingPathDelimiter(aFolder) + 'config.json');
  if (aWeights <> idtF32) and
     ((fConfig.Hidden and 31 <> 0) or (fConfig.Intermediate and 31 <> 0) or
      ((fConfig.Heads * fConfig.HeadDim) and 31 <> 0)) then
    EInferError.RaiseUtf8('%: % needs widths that are multiples of 32',
      [self, INFER_DTYPE_NAME[aWeights]]);
  fFile := TSafeTensorsFile.Create(IncludeTrailingPathDelimiter(aFolder) +
    'model.safetensors');
  LoadWeights;
end;

constructor TQwen3Model.CreateFromGguf(const aFileName: TFileName;
  aBackend: TInferBackend);
begin
  inherited Create;
  Init(aBackend);
  fGguf := TGgufFile.Create(aFileName);
  LoadGgufConfig;
  LoadWeights;
  fWeights := fLayer[0].Q.DType;
end;

destructor TQwen3Model.Destroy;
begin
  fFile.Free;
  fGguf.Free;
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
  CheckConfig(aConfigFile);
end;

procedure TQwen3Model.LoadGgufConfig;
var
  emb: PInferTensor;

  // a positive count, bounded before it becomes a PtrInt: a value beyond 32
  // bits must not wrap into a valid one in a 32-bit process
  function Count(const aKey: RawUtf8): PtrInt;
  var
    v: Int64;
  begin
    v := fGguf.Int(aKey);
    if (v <= 0) or (v > 1 shl 30) then
      EInferError.RaiseUtf8('%: % = % out of range', [self, aKey, v]);
    result := v;
  end;

begin
  if fGguf.Str('general.architecture') <> 'qwen3' then
    EInferError.RaiseUtf8('%: architecture "%" is not qwen3',
      [self, fGguf.Str('general.architecture')]);
  // a scaled RoPE or a sliding window would change the arithmetic: refused
  if fGguf.Has('qwen3.rope.scaling.type') and
     (fGguf.Str('qwen3.rope.scaling.type') <> 'none') then
    EInferError.RaiseUtf8('%: rope scaling "%" not supported',
      [self, fGguf.Str('qwen3.rope.scaling.type')]);
  if fGguf.Has('qwen3.attention.sliding_window') then
    EInferError.RaiseUtf8('%: sliding window attention not supported', [self]);
  emb := fGguf.Tensor('token_embd.weight');
  if length(emb^.Shape) <> 2 then
    EInferError.RaiseUtf8('%: token_embd.weight is shaped %',
      [self, InferShapeText(emb^.Shape)]);
  if (emb^.Shape[0] <= 0) or (emb^.Shape[0] > 1 shl 22) then
    EInferError.RaiseUtf8('%: a vocabulary of % tokens', [self, emb^.Shape[0]]);
  fConfig.VocabSize := emb^.Shape[0];
  fConfig.Hidden := Count('qwen3.embedding_length');
  fConfig.Layers := Count('qwen3.block_count');
  fConfig.Heads := Count('qwen3.attention.head_count');
  fConfig.KvHeads := Count('qwen3.attention.head_count_kv');
  fConfig.HeadDim := Count('qwen3.attention.key_length');
  // RoPE over a part of each head only would be another arithmetic
  if fGguf.Has('qwen3.rope.dimension_count') and
     (Count('qwen3.rope.dimension_count') <> fConfig.HeadDim) then
    EInferError.RaiseUtf8('%: RoPE over % of % dimensions not supported',
      [self, Count('qwen3.rope.dimension_count'), fConfig.HeadDim]);
  if fGguf.Has('qwen3.attention.value_length') and
     (Count('qwen3.attention.value_length') <> fConfig.HeadDim) then
    EInferError.RaiseUtf8('%: key and value heads of different sizes', [self]);
  fConfig.Intermediate := Count('qwen3.feed_forward_length');
  fConfig.MaxPositions := Count('qwen3.context_length');
  fConfig.RmsEps := fGguf.Float('qwen3.attention.layer_norm_rms_epsilon');
  fConfig.RopeTheta := fGguf.Float('qwen3.rope.freq_base');
  // llama.cpp ties the head to the embedding by leaving output.weight out
  fConfig.TiedEmbeddings := fGguf.FindTensor('output.weight') = nil;
  CheckConfig(fGguf.FileName);
end;

procedure TQwen3Model.CheckConfig(const aSource: TFileName);
var
  i: PtrInt;
begin
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
    EInferError.RaiseUtf8('%: % has an invalid or unsupported shape', [self, aSource]);
  // inverse rotary frequencies as the reference forms them in float32:
  // 1 / theta ^ (2i / d), each step rounded to single
  SetLength(fInvFreq, fConfig.HeadDim shr 1);
  for i := 0 to high(fInvFreq) do
    fInvFreq[i] := 1 / RoundToSingle(Exp(Ln(fConfig.RopeTheta) *
      RoundToSingle((2 * i) / fConfig.HeadDim)));
end;

const
  /// Hugging Face names and llama.cpp's for the tensors of one layer
  LAYER_NAMES: array[0..10, 0..1] of RawUtf8 = (
    ('self_attn.q_proj.weight', 'attn_q.weight'),
    ('self_attn.k_proj.weight', 'attn_k.weight'),
    ('self_attn.v_proj.weight', 'attn_v.weight'),
    ('self_attn.o_proj.weight', 'attn_output.weight'),
    ('self_attn.q_norm.weight', 'attn_q_norm.weight'),
    ('self_attn.k_norm.weight', 'attn_k_norm.weight'),
    ('mlp.gate_proj.weight', 'ffn_gate.weight'),
    ('mlp.up_proj.weight', 'ffn_up.weight'),
    ('mlp.down_proj.weight', 'ffn_down.weight'),
    ('input_layernorm.weight', 'attn_norm.weight'),
    ('post_attention_layernorm.weight', 'ffn_norm.weight'));

/// llama.cpp's name of a Hugging Face tensor name, '' if there is none
function GgufName(const aName: RawUtf8): RawUtf8;
var
  p, i: PtrInt;
  layer, rest: RawUtf8;
begin
  result := '';
  if aName = 'model.embed_tokens.weight' then
    result := 'token_embd.weight'
  else if aName = 'model.norm.weight' then
    result := 'output_norm.weight'
  else if aName = 'lm_head.weight' then
    result := 'output.weight'
  else if IdemPChar(pointer(aName), 'MODEL.LAYERS.') then
  begin
    rest := copy(aName, 14, maxInt);
    p := PosEx('.', rest);
    if p <= 1 then
      exit;
    layer := copy(rest, 1, p - 1);
    rest := copy(rest, p + 1, maxInt);
    for i := 0 to high(LAYER_NAMES) do
      if rest = LAYER_NAMES[i, 0] then
      begin
        result := 'blk.' + layer + '.' + LAYER_NAMES[i, 1];
        exit;
      end;
  end;
end;

function TQwen3Model.Find(const aName: RawUtf8): PInferTensor;
begin
  if fGguf <> nil then
  begin
    result := nil;
    if GgufName(aName) <> '' then
      result := fGguf.FindTensor(GgufName(aName));
  end
  else
    result := fFile.Find(aName);
end;

function TQwen3Model.TensorCount: PtrInt;
begin
  if fGguf <> nil then
    result := length(fGguf.Tensors)
  else
    result := length(fFile.Tensors);
end;

function TQwen3Model.Checked(const aName: RawUtf8;
  const aDims: array of Int64; aBlocks: boolean): PInferTensor;
var
  i: PtrInt;
  ok: boolean;
  want: TInt64DynArray;
  t: PInferTensor;
begin
  t := Find(aName);
  if t = nil then
  begin
    if fGguf = nil then
      fFile.Tensor(aName); // raises with the file's message
    EInferError.RaiseUtf8('%: no tensor % (%) in %',
      [self, GgufName(aName), aName, fGguf.FileName]);
  end;
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
  // F32 and BF16 from safetensors; a GGUF file adds F16, and Q8_0 and Q4_0
  // for a matrix (aBlocks)
  if not ((t^.DType in [idtF32, idtBF16]) or
          ((fGguf <> nil) and ((t^.DType = idtF16) or
           (aBlocks and (t^.DType in [idtQ8_0, idtQ4_0, idtQ4_K, idtQ6_K]))))) then
    EInferError.RaiseUtf8('%: % is %, not supported here',
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
  t := Checked(aName, aDims);
  if t^.DType = idtF32 then
  begin
    result := t^.Data;
    exit;
  end;
  // BF16 is the upper half of an F32, F16 a subset of it: widening is exact
  if t^.Count > High(PtrInt) div 4 then
    EInferError.RaiseUtf8('%: % too large', [self, aName]);
  n := length(fOwned);
  SetLength(fOwned, n + 1);
  SetLength(fOwned[n], t^.Count);
  src := t^.Data;
  dst := pointer(fOwned[n]);
  if t^.DType = idtBF16 then
    for i := 0 to t^.Count - 1 do
      dst^[i] := cardinal(src^[i]) shl 16
  else
    for i := 0 to t^.Count - 1 do
      dst^[i] := InferHalfToSingleBits(src^[i]);
  result := pointer(fOwned[n]);
end;

/// row aRow of a [rows, aCols] matrix as stored, as singles in aDest - exact
// for every type (BF16 and F16 widened, Q8_0 and Q4_0 dequantized)
procedure WidenRow(t: PInferTensor; aRow, aCols: PtrInt; aDest: PSingleArray);
var
  i: PtrInt;
  src: PWordArray;
begin
  case t^.DType of
    idtBF16:
      begin
        src := @PWordArray(t^.Data)^[aRow * aCols];
        for i := 0 to aCols - 1 do
          PCardinalArray(aDest)^[i] := cardinal(src^[i]) shl 16;
      end;
    idtF16:
      begin
        src := @PWordArray(t^.Data)^[aRow * aCols];
        for i := 0 to aCols - 1 do
          PCardinalArray(aDest)^[i] := InferHalfToSingleBits(src^[i]);
      end;
  else
    InferRowToF32(t^.Data, t^.DType, aRow, aCols, aDest); // F32, Q8_0, Q4_0
  end;
end;

/// raise unless every FP16 scale of the blocks of t is finite
procedure CheckScales(t: PInferTensor; const aName: RawUtf8; aModel: TObject);
var
  b, size, first, second: PtrInt;
  p: PByteArray;
begin
  size := INFER_DTYPE_SIZE[t^.DType];
  // where the half-float scales sit: d first (Q8_0, Q4_0), d and dmin first
  // (Q4_K), d last (Q6_K)
  first := 0;
  second := -1;
  case t^.DType of
    idtQ4_K:
      second := 2;
    idtQ6_K:
      first := SizeOf(TInferBlockQ6_K) - 2;
  end;
  p := t^.Data;
  for b := 0 to t^.Count div INFER_DTYPE_BLOCK[t^.DType] - 1 do
  begin
    if (PWord(@p^[first])^ and $7fff >= $7c00) or
       ((second >= 0) and (PWord(@p^[second])^ and $7fff >= $7c00)) then
      EInferError.RaiseUtf8('%: % has a scale that is no finite number in block %',
        [aModel, aName, b]);
    p := @p^[size];
  end;
end;

function TQwen3Model.Matrix(const aName: RawUtf8; aRows, aCols: Int64;
  aType: TInferDType): TInferWeight;
var
  t: PInferTensor;
  row: TSingleDynArray;
  nb, r, n, c: PtrInt;
  d: word;
begin
  if fGguf <> nil then
  begin
    // a GGUF file decides: blocks in place, F32 in place, F16 / BF16 widened
    t := Checked(aName, [aRows, aCols], true);
    case t^.DType of
      idtQ8_0, idtQ4_0, idtQ4_K, idtQ6_K:
        begin
          CheckScales(t, aName, self);
          result := InferWeight(t^.Data, t^.DType);
        end;
    else
      result := InferWeight(Weight(aName, [aRows, aCols]), idtF32);
    end;
    if not fBackend.Supports(ioLinear, result.DType) then
      EInferError.RaiseUtf8('%: the backend has no % Linear for %',
        [self, INFER_DTYPE_NAME[result.DType], aName]);
    exit;
  end;
  if aType = idtF32 then
  begin
    result := InferWeight(Weight(aName, [aRows, aCols]), idtF32);
    exit;
  end;
  // Q8_0 or Q4_0: row by row through single precision, as ggml quantizes
  t := Checked(aName, [aRows, aCols]);
  nb := aCols shr 5;
  if aType = idtQ8_0 then
  begin
    n := length(fOwnedQ8);
    SetLength(fOwnedQ8, n + 1);
    SetLength(fOwnedQ8[n], aRows * nb);
    result := InferWeight(pointer(fOwnedQ8[n]), idtQ8_0);
  end
  else
  begin
    n := length(fOwnedQ4);
    SetLength(fOwnedQ4, n + 1);
    SetLength(fOwnedQ4[n], aRows * nb);
    result := InferWeight(pointer(fOwnedQ4[n]), idtQ4_0);
  end;
  SetLength(row, aCols);
  for r := 0 to aRows - 1 do
  begin
    WidenRow(t, r, aCols, pointer(row));
    // a NaN or an infinity has no quantized form: refused, by its bits (no FPU)
    for c := 0 to aCols - 1 do
      if PCardinalArray(row)^[c] and $7f800000 = $7f800000 then
        EInferError.RaiseUtf8('%: % has a NaN or infinite weight at row % column %',
          [self, aName, r, c]);
    if aType = idtQ8_0 then
      InferQuantizeRowQ8_0Ref(pointer(row), aCols, @fOwnedQ8[n][r * nb])
    else
      InferQuantizeRowQ4_0(pointer(row), aCols, @fOwnedQ4[n][r * nb]);
    // a finite weight beyond the range still gets the FP16 scale +Inf (ggml
    // writes it too): every product with it would be Inf or NaN
    for c := r * nb to r * nb + nb - 1 do
    begin
      if aType = idtQ8_0 then
        d := fOwnedQ8[n][c].d
      else
        d := fOwnedQ4[n][c].d;
      if d and $7fff >= $7c00 then
        EInferError.RaiseUtf8('%: % has a weight beyond the % range in row %',
          [self, aName, INFER_DTYPE_NAME[aType], r]);
    end;
  end;
end;

procedure TQwen3Model.LoadWeights;
var
  i, used: PtrInt;
  h, f, qw, kvw: Int64;
  p: RawUtf8;
  l: ^TQwen3Layer;
  head, emb: PInferTensor;
  headType: TInferDType;
begin
  // with Q4_0 the head stays Q8_0, as llama.cpp keeps the output matrix finer
  // than the layers: it decides every token (a GGUF file brings its own)
  headType := fWeights;
  if headType = idtQ4_0 then
    headType := idtQ8_0;
  h := fConfig.Hidden;
  f := fConfig.Intermediate;
  qw := fConfig.Heads * fConfig.HeadDim;
  kvw := fConfig.KvHeads * fConfig.HeadDim;
  fEmbedSrc := Checked('model.embed_tokens.weight', [fConfig.VocabSize, h],
    fGguf <> nil);
  // read per token: its scales checked here, also when the head is untied
  if fEmbedSrc^.DType in [idtQ8_0, idtQ4_0, idtQ4_K, idtQ6_K] then
    CheckScales(fEmbedSrc, 'model.embed_tokens.weight', self);
  if (fGguf = nil) and (fWeights = idtF32) then
    fEmbed := InferWeight(Weight('model.embed_tokens.weight', [fConfig.VocabSize, h]),
      idtF32);
  fNorm := Weight('model.norm.weight', [h]);
  used := 2;
  head := Find('lm_head.weight');
  if fConfig.TiedEmbeddings then
  begin
    // a tied head may still be stored: only when it is the very same matrix
    if head <> nil then
    begin
      emb := Find('model.embed_tokens.weight');
      if (head^.DType <> emb^.DType) or (head^.Bytes <> emb^.Bytes) or
         not CompareMem(head^.Data, emb^.Data, head^.Bytes) then
        EInferError.RaiseUtf8('%: tie_word_embeddings but lm_head.weight differs ' +
          'from the embedding', [self]);
      inc(used);
    end;
    if (fGguf = nil) and (fWeights = idtF32) then
      fHead := fEmbed
    else
      fHead := Matrix('model.embed_tokens.weight', fConfig.VocabSize, h, headType);
  end
  else
  begin
    fHead := Matrix('lm_head.weight', fConfig.VocabSize, h, headType);
    inc(used);
  end;
  SetLength(fLayer, fConfig.Layers);
  for i := 0 to fConfig.Layers - 1 do
  begin
    p := FormatUtf8('model.layers.%.', [i]);
    l := @fLayer[i];
    l^.Q := Matrix(p + 'self_attn.q_proj.weight', qw, h, fWeights);
    l^.K := Matrix(p + 'self_attn.k_proj.weight', kvw, h, fWeights);
    l^.V := Matrix(p + 'self_attn.v_proj.weight', kvw, h, fWeights);
    l^.O := Matrix(p + 'self_attn.o_proj.weight', h, qw, fWeights);
    l^.QNorm := Weight(p + 'self_attn.q_norm.weight', [fConfig.HeadDim]);
    l^.KNorm := Weight(p + 'self_attn.k_norm.weight', [fConfig.HeadDim]);
    l^.Gate := Matrix(p + 'mlp.gate_proj.weight', f, h, fWeights);
    l^.Up := Matrix(p + 'mlp.up_proj.weight', f, h, fWeights);
    l^.Down := Matrix(p + 'mlp.down_proj.weight', h, f, fWeights);
    l^.InputNorm := Weight(p + 'input_layernorm.weight', [h]);
    l^.PostNorm := Weight(p + 'post_attention_layernorm.weight', [h]);
    inc(used, 11);
  end;
  // an unknown tensor means a model we do not understand
  if used <> TensorCount then
    EInferError.RaiseUtf8('%: % tensors in the file, % understood',
      [self, TensorCount, used]);
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
    // the shape is only taken once its arrays are there: a failed SetLength
    // leaves a context that tries again
    aCtx.fCapacity := 0;
    SetLength(aCtx.fKeys, fConfig.Layers);
    SetLength(aCtx.fValues, fConfig.Layers);
    aCtx.fKvWidth := kvw;
    aCtx.fLayers := fConfig.Layers;
  end;
  if aTokens > aCtx.fCapacity then
  begin
    // doubled, and a quarter beyond the tokens wanted: a prefill is followed
    // by decoding, which must not copy the whole cache at its first token
    cap := aCtx.fCapacity * 2;
    if cap < aTokens + aTokens shr 2 then
      cap := aTokens + aTokens shr 2;
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
    SetLength(aCtx.fIds, cap);
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
  Need(aCtx.fCos, aNew * (fConfig.HeadDim shr 1));
  Need(aCtx.fSin, aNew * (fConfig.HeadDim shr 1));
end;

procedure TQwen3Model.RopeFactors(aCtx: TQwen3Context; aRows, aFirstPos: PtrInt);
begin
  // once per Forward: every layer and both Q and K use the same factors
  Rotary(aFirstPos, aRows, pointer(aCtx.fCos), pointer(aCtx.fSin));
end;

procedure TQwen3Model.Rotary(aFirstPos, aRows: PtrInt; aCos, aSin: PSingleArray);
var
  r, i, half: PtrInt;
  angle: single;
begin
  half := fConfig.HeadDim shr 1;
  for r := 0 to aRows - 1 do
    for i := 0 to half - 1 do
    begin
      // the angle in single, as the reference's float32 product
      angle := fInvFreq[i] * (aFirstPos + r);
      aCos^[r * half + i] := Cos(angle);
      aSin^[r * half + i] := Sin(angle);
    end;
end;

procedure TQwen3Model.Embed(aIds: PIntegerArray; aCount: PtrInt; aDest: PSingleArray);
var
  i, id: PtrInt;
begin
  for i := 0 to aCount - 1 do
  begin
    id := aIds^[i];
    if (id < 0) or (id >= fConfig.VocabSize) then
      EInferError.RaiseUtf8('%.Embed: token id % outside the vocabulary', [self, id]);
    WidenRow(fEmbedSrc, id, fConfig.Hidden, @aDest^[i * fConfig.Hidden]);
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
  RopeFactors(aCtx, n, past);
  // embeddings
  for i := 0 to n - 1 do
    WidenRow(fEmbedSrc, aIds^[i], h, @aCtx.fX[i * h]);
  if Assigned(aOnState) then
    aOnState('embed', pointer(aCtx.fX), n, h);
  for l := 0 to fConfig.Layers - 1 do
  begin
    ly := @fLayer[l];
    // attention block: pre-norm, projections, per-head norms, rotary positions
    fBackend.RmsNorm(pointer(aCtx.fX), n, h, ly^.InputNorm, fConfig.RmsEps,
      pointer(aCtx.fH));
    fBackend.Linear(pointer(aCtx.fH), n, h, ly^.Q, qw, nil, pointer(aCtx.fQ));
    fBackend.Linear(pointer(aCtx.fH), n, h, ly^.K, kvw, nil, pointer(aCtx.fK));
    fBackend.Linear(pointer(aCtx.fH), n, h, ly^.V, kvw, nil, pointer(aCtx.fV));
    fBackend.RmsNorm(pointer(aCtx.fQ), n * fConfig.Heads, fConfig.HeadDim,
      ly^.QNorm, fConfig.RmsEps, pointer(aCtx.fQ));
    fBackend.RmsNorm(pointer(aCtx.fK), n * fConfig.KvHeads, fConfig.HeadDim,
      ly^.KNorm, fConfig.RmsEps, pointer(aCtx.fK));
    fBackend.Rope(pointer(aCtx.fQ), n, fConfig.Heads, fConfig.HeadDim,
      pointer(aCtx.fCos), pointer(aCtx.fSin));
    fBackend.Rope(pointer(aCtx.fK), n, fConfig.KvHeads, fConfig.HeadDim,
      pointer(aCtx.fCos), pointer(aCtx.fSin));
    MoveFast(pointer(aCtx.fK)^, aCtx.fKeys[l][past * kvw], n * kvw * SizeOf(single));
    MoveFast(pointer(aCtx.fV)^, aCtx.fValues[l][past * kvw], n * kvw * SizeOf(single));
    Attention(aCtx, l, n);
    fBackend.Linear(pointer(aCtx.fAttn), n, qw, ly^.O, h, nil, pointer(aCtx.fTmp));
    if Assigned(aOnState) and (l = 0) then
      aOnState('layer00_attn', pointer(aCtx.fTmp), n, h);
    fBackend.Add(pointer(aCtx.fX), pointer(aCtx.fTmp), n * h);
    // feed-forward block: pre-norm, SwiGLU, residual
    fBackend.RmsNorm(pointer(aCtx.fX), n, h, ly^.PostNorm, fConfig.RmsEps,
      pointer(aCtx.fH));
    fBackend.Linear(pointer(aCtx.fH), n, h, ly^.Gate, f, nil, pointer(aCtx.fGate));
    fBackend.Linear(pointer(aCtx.fH), n, h, ly^.Up, f, nil, pointer(aCtx.fUp));
    fBackend.SwiGlu(pointer(aCtx.fGate), pointer(aCtx.fUp), n * f);
    fBackend.Linear(pointer(aCtx.fGate), n, f, ly^.Down, h, nil, pointer(aCtx.fTmp));
    if Assigned(aOnState) and (l = 0) then
      aOnState('layer00_mlp', pointer(aCtx.fTmp), n, h);
    fBackend.Add(pointer(aCtx.fX), pointer(aCtx.fTmp), n * h);
    if Assigned(aOnState) then
      aOnState(LayerName(l), pointer(aCtx.fX), n, h);
  end;
  MoveFast(aIds^[0], aCtx.fIds[past], n * SizeOf(integer));
  aCtx.fLength := past + n;
  if (aLogits = nil) and not Assigned(aOnState) then
    exit;
  fBackend.RmsNorm(pointer(aCtx.fX), n, h, fNorm, fConfig.RmsEps, pointer(aCtx.fH));
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

function TQwen3Model.Reuse(aCtx: TQwen3Context; aIds: PIntegerArray;
  aCount: PtrInt): PtrInt;
begin
  if aCtx = nil then
    EInferError.RaiseUtf8('%.Reuse: no context', [self]);
  result := 0;
  if (aCtx.fOwner = fSerial) and (aCount > 0) then
    result := aCtx.Common(aIds, aCount - 1);
  aCtx.Truncate(result);
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
