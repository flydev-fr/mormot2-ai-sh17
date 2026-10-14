/// native inference engine - XLM-RoBERTa encoder (BGE-M3)
// - part of the mormot.ai.* extension (LandrixAI)
// - licensed under a MPL/GPL/LGPL three license - see LICENSE
// - clean-room from the published architecture (looked up in Hugging Face's
//   modeling_xlm_roberta.py and candle, no code taken): embeddings (word +
//   position + token type 0, LayerNorm), then per layer post-norm self-attention
//   and a GELU feed-forward block; dropout is an identity at inference
// - XLM-R positions: real tokens count from pad + 1, padding sits at pad - they
//   derive from the token ids, not from the attention mask
// - the model is immutable after loading and may serve several threads; every
//   call brings its own TXlmrContext with the scratch buffers
// - BERT is refused on purpose: it numbers positions from 0, not from pad + 1,
//   and would load fine but compute wrong vectors
unit mormot.ai.infer.model.xlmr;

{
  *****************************************************************************

    - TXlmrConfig: the encoder hyper-parameters, read from config.json
    - TXlmrModel: weights validated at load, batched forward pass
    - TXlmrContext: per-call scratch buffers, reusable between calls

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
  mormot.ai.infer.backend;

type
  /// encoder hyper-parameters
  TXlmrConfig = record
    VocabSize: PtrInt;
    Hidden: PtrInt;
    Layers: PtrInt;
    Heads: PtrInt;
    HeadDim: PtrInt;
    Intermediate: PtrInt;
    MaxPositions: PtrInt;
    TypeVocab: PtrInt;
    PadId: PtrInt;
    LayerNormEps: double;
  end;

  /// the weights of one encoder layer - views into the mapped file
  TXlmrLayer = record
    Q, QBias, K, KBias, V, VBias: PSingleArray;
    AttnOut, AttnOutBias, AttnNorm, AttnNormBias: PSingleArray;
    Inter, InterBias, Output, OutputBias, OutNorm, OutNormBias: PSingleArray;
  end;
  TXlmrLayerDynArray = array of TXlmrLayer;

  /// called with every hidden state of a forward pass, for diagnosis and tests
  // - aState 0 is the embedding output, 1..Layers the output of each layer;
  // aValues is [aRows * aWidth, Hidden] and only valid during the call
  TOnXlmrState = procedure(aState: PtrInt; aValues: PSingleArray;
    aRows, aWidth: PtrInt) of object;

  /// per-call scratch buffers of a forward pass - one per concurrent call
  // - may be reused for any model and any batch shape: every buffer is checked
  // against what the next call needs, never against what an earlier call had
  TXlmrContext = class
  protected
    fHiddenCount: PtrInt;
    fHidden, fQ, fK, fV, fCtx, fTmp, fInter, fScores: TSingleDynArray;
    fKeep, fPos: TIntegerDynArray;
  public
    /// the last hidden state of the latest Forward: [rows * width, hidden]
    // - the array may be longer (capacity kept from a larger call): only the
    // first HiddenCount values belong to the latest Forward
    property Hidden: TSingleDynArray
      read fHidden;
    /// rows * width * hidden of the latest Forward
    property HiddenCount: PtrInt
      read fHiddenCount;
  end;

  /// an XLM-RoBERTa encoder over a safetensors file
  TXlmrModel = class
  protected
    fConfig: TXlmrConfig;
    fFile: TSafeTensorsFile;
    fBackend: TInferBackend;
    fWord, fPosition, fTokenType, fEmbNorm, fEmbNormBias: PSingleArray;
    fLayer: TXlmrLayerDynArray;
    procedure LoadConfig(const aConfigFile: TFileName);
    procedure LoadWeights;
    procedure Prepare(aCtx: TXlmrContext; aRows, aWidth: PtrInt);
  public
    /// load <aFolder>config.json and <aFolder>model.safetensors
    // - raises EInferError when a setting is unsupported or a weight is missing
    // or shaped otherwise; aBackend is not owned
    constructor Create(const aFolder: TFileName; aBackend: TInferBackend);
    /// unmap the weights
    destructor Destroy; override;
    /// encode aRows sequences of aWidth token ids each (right-padded with PadId)
    // - aIds is [aRows, aWidth]; aKeep[r * aWidth + c] = 0 marks padding, nil
    // derives it from aIds <> PadId
    // - the last hidden state is left in aCtx.Hidden; aOnState sees every state
    procedure Forward(aCtx: TXlmrContext; aIds: PIntegerArray;
      aRows, aWidth: PtrInt; aKeep: PIntegerArray = nil;
      const aOnState: TOnXlmrState = nil);
    /// the hyper-parameters
    property Config: TXlmrConfig
      read fConfig;
    /// the backend the model computes with (not owned)
    property Backend: TInferBackend
      read fBackend;
  end;


implementation

{ TXlmrModel }

constructor TXlmrModel.Create(const aFolder: TFileName; aBackend: TInferBackend);
var
  folder: TFileName;
begin
  inherited Create;
  if aBackend = nil then
    EInferError.RaiseUtf8('%.Create: no backend', [self]);
  fBackend := aBackend;
  folder := IncludeTrailingPathDelimiter(aFolder);
  LoadConfig(folder + 'config.json');
  fFile := TSafeTensorsFile.Create(folder + 'model.safetensors');
  LoadWeights;
end;

destructor TXlmrModel.Destroy;
begin
  fFile.Free;
  inherited Destroy;
end;

const
  REQUIRED_KEYS: array[0..9] of RawUtf8 = (
    'vocab_size', 'hidden_size', 'num_hidden_layers', 'num_attention_heads',
    'intermediate_size', 'max_position_embeddings', 'type_vocab_size',
    'pad_token_id', 'layer_norm_eps', 'hidden_act');

procedure TXlmrModel.LoadConfig(const aConfigFile: TFileName);
var
  doc: TDocVariantData;
  posType: RawUtf8;
  i: PtrInt;
begin
  if not doc.InitJsonFromFile(aConfigFile, JSON_FAST_FLOAT) then
    EInferError.RaiseUtf8('%: cannot read %', [self, aConfigFile]);
  if doc.U['model_type'] <> 'xlm-roberta' then
    EInferError.RaiseUtf8('%: model_type "%" is not xlm-roberta',
      [self, doc.U['model_type']]);
  // a missing key would read as 0 - e.g. pad_token_id, whose default is 1
  for i := 0 to high(REQUIRED_KEYS) do
    if doc.GetValueIndex(REQUIRED_KEYS[i]) < 0 then
      EInferError.RaiseUtf8('%: % has no "%"', [self, aConfigFile, REQUIRED_KEYS[i]]);
  if doc.U['hidden_act'] <> 'gelu' then
    EInferError.RaiseUtf8('%: hidden_act "%" not supported (exact gelu only)',
      [self, doc.U['hidden_act']]);
  // a decoder (causal mask, cross-attention) has the same weight names but
  // another computation: refused instead of run bidirectionally
  if doc.B['is_decoder'] or doc.B['add_cross_attention'] then
    EInferError.RaiseUtf8('%: decoder configurations are not supported', [self]);
  posType := doc.U['position_embedding_type'];
  if (posType <> '') and (posType <> 'absolute') then
    EInferError.RaiseUtf8('%: position_embedding_type "%" not supported',
      [self, posType]);
  fConfig.VocabSize := doc.I['vocab_size'];
  fConfig.Hidden := doc.I['hidden_size'];
  fConfig.Layers := doc.I['num_hidden_layers'];
  fConfig.Heads := doc.I['num_attention_heads'];
  fConfig.Intermediate := doc.I['intermediate_size'];
  fConfig.MaxPositions := doc.I['max_position_embeddings'];
  fConfig.TypeVocab := doc.I['type_vocab_size'];
  fConfig.PadId := doc.I['pad_token_id'];
  fConfig.LayerNormEps := doc.D['layer_norm_eps'];
  if (fConfig.VocabSize <= 0) or (fConfig.Hidden <= 0) or (fConfig.Layers <= 0) or
     (fConfig.Heads <= 0) or (fConfig.Intermediate <= 0) or
     (fConfig.MaxPositions <= fConfig.PadId + 1) or (fConfig.TypeVocab <= 0) or
     (fConfig.PadId < 0) or (fConfig.LayerNormEps <= 0) or
     (fConfig.Hidden mod fConfig.Heads <> 0) then
    EInferError.RaiseUtf8('%: inconsistent hyper-parameters in %', [self, aConfigFile]);
  fConfig.HeadDim := fConfig.Hidden div fConfig.Heads;
end;

procedure TXlmrModel.LoadWeights;
var
  i: PtrInt;
  h, f: Int64;
  p: RawUtf8;
  l: ^TXlmrLayer;
  used: integer;
begin
  h := fConfig.Hidden;
  f := fConfig.Intermediate;
  fWord := fFile.F32('embeddings.word_embeddings.weight', [fConfig.VocabSize, h]);
  fPosition := fFile.F32('embeddings.position_embeddings.weight',
    [fConfig.MaxPositions, h]);
  fTokenType := fFile.F32('embeddings.token_type_embeddings.weight',
    [fConfig.TypeVocab, h]);
  fEmbNorm := fFile.F32('embeddings.LayerNorm.weight', [h]);
  fEmbNormBias := fFile.F32('embeddings.LayerNorm.bias', [h]);
  used := 5;
  SetLength(fLayer, fConfig.Layers);
  for i := 0 to fConfig.Layers - 1 do
  begin
    p := FormatUtf8('encoder.layer.%.', [i]);
    l := @fLayer[i];
    l^.Q := fFile.F32(p + 'attention.self.query.weight', [h, h]);
    l^.QBias := fFile.F32(p + 'attention.self.query.bias', [h]);
    l^.K := fFile.F32(p + 'attention.self.key.weight', [h, h]);
    l^.KBias := fFile.F32(p + 'attention.self.key.bias', [h]);
    l^.V := fFile.F32(p + 'attention.self.value.weight', [h, h]);
    l^.VBias := fFile.F32(p + 'attention.self.value.bias', [h]);
    l^.AttnOut := fFile.F32(p + 'attention.output.dense.weight', [h, h]);
    l^.AttnOutBias := fFile.F32(p + 'attention.output.dense.bias', [h]);
    l^.AttnNorm := fFile.F32(p + 'attention.output.LayerNorm.weight', [h]);
    l^.AttnNormBias := fFile.F32(p + 'attention.output.LayerNorm.bias', [h]);
    l^.Inter := fFile.F32(p + 'intermediate.dense.weight', [f, h]);
    l^.InterBias := fFile.F32(p + 'intermediate.dense.bias', [f]);
    l^.Output := fFile.F32(p + 'output.dense.weight', [h, f]);
    l^.OutputBias := fFile.F32(p + 'output.dense.bias', [h]);
    l^.OutNorm := fFile.F32(p + 'output.LayerNorm.weight', [h]);
    l^.OutNormBias := fFile.F32(p + 'output.LayerNorm.bias', [h]);
    inc(used, 16);
  end;
  // anything else must be the (unused) pooler: an unknown tensor means a model
  // we do not understand, not something to ignore silently
  for i := 0 to high(fFile.Tensors) do
    if IdemPChar(pointer(fFile.Tensors[i].Name), 'POOLER.') then
      inc(used);
  if used <> length(fFile.Tensors) then
    EInferError.RaiseUtf8('%: % tensors in the file, % understood',
      [self, length(fFile.Tensors), used]);
end;

procedure Need(var aBuffer: TSingleDynArray; aCount: PtrInt); overload;
begin
  if length(aBuffer) < aCount then
    SetLength(aBuffer, aCount);
end;

procedure Need(var aBuffer: TIntegerDynArray; aCount: PtrInt); overload;
begin
  if length(aBuffer) < aCount then
    SetLength(aBuffer, aCount);
end;

procedure TXlmrModel.Prepare(aCtx: TXlmrContext; aRows, aWidth: PtrInt);
var
  n, nh: PtrInt;
begin
  // sized from THIS model and THIS batch: a context grown by another model or
  // shape is never trusted
  n := aRows * aWidth;
  nh := n * fConfig.Hidden;
  Need(aCtx.fHidden, nh);
  Need(aCtx.fQ, nh);
  Need(aCtx.fK, nh);
  Need(aCtx.fV, nh);
  Need(aCtx.fCtx, nh);
  Need(aCtx.fTmp, nh);
  Need(aCtx.fInter, n * fConfig.Intermediate);
  Need(aCtx.fKeep, n);
  Need(aCtx.fPos, n);
  Need(aCtx.fScores, aWidth * aWidth);
  aCtx.fHiddenCount := nh;
end;

procedure TXlmrModel.Forward(aCtx: TXlmrContext; aIds: PIntegerArray;
  aRows, aWidth: PtrInt; aKeep: PIntegerArray; const aOnState: TOnXlmrState);
var
  r, c, i, n, h, f, id, seen, off: PtrInt;
  l: ^TXlmrLayer;
  dst, wordRow, posRow: PSingleArray;
begin
  if aCtx = nil then
    EInferError.RaiseUtf8('%.Forward: no context', [self]);
  if (aRows <= 0) or (aWidth <= 0) then
    EInferError.RaiseUtf8('%.Forward: empty batch', [self]);
  if aWidth > fConfig.MaxPositions - fConfig.PadId - 1 then
    EInferError.RaiseUtf8('%.Forward: % tokens exceed the % positions',
      [self, aWidth, fConfig.MaxPositions - fConfig.PadId - 1]);
  // every buffer is tokens * max(hidden, intermediate) singles, the attention
  // scratch width * width: refuse a batch whose size would overflow PtrInt (a
  // real limit on 32-bit targets)
  if (aRows > MaxInt div aWidth) or
     (aWidth > (High(PtrInt) div SizeOf(single)) div aWidth) or
     (aRows * aWidth > (High(PtrInt) div SizeOf(single)) div
       MaxPtrInt(fConfig.Intermediate, fConfig.Hidden)) then
    EInferError.RaiseUtf8('%.Forward: batch of % x % tokens too large',
      [self, aRows, aWidth]);
  Prepare(aCtx, aRows, aWidth);
  n := aRows * aWidth;
  h := fConfig.Hidden;
  f := fConfig.Intermediate;
  // keep mask and XLM-R positions: per row, real tokens count from pad + 1
  for r := 0 to aRows - 1 do
  begin
    seen := 0;
    for c := 0 to aWidth - 1 do
    begin
      i := r * aWidth + c;
      id := aIds^[i];
      if (id < 0) or (id >= fConfig.VocabSize) then
        EInferError.RaiseUtf8('%.Forward: token id % outside the vocabulary',
          [self, id]);
      if aKeep <> nil then
        aCtx.fKeep[i] := ord(aKeep^[i] <> 0)
      else
        aCtx.fKeep[i] := ord(id <> fConfig.PadId);
      if id = fConfig.PadId then
        aCtx.fPos[i] := fConfig.PadId
      else
      begin
        inc(seen);
        aCtx.fPos[i] := seen + fConfig.PadId;
      end;
    end;
  end;
  // embeddings: word + position + token type 0, then LayerNorm
  for i := 0 to n - 1 do
  begin
    dst := @aCtx.fHidden[i * h];
    wordRow := @fWord^[aIds^[i] * h];
    posRow := @fPosition^[aCtx.fPos[i] * h];
    for c := 0 to h - 1 do
      dst^[c] := wordRow^[c] + posRow^[c] + fTokenType^[c];
  end;
  fBackend.LayerNorm(pointer(aCtx.fHidden), n, h, fEmbNorm, fEmbNormBias,
    fConfig.LayerNormEps, pointer(aCtx.fHidden));
  if Assigned(aOnState) then
    aOnState(0, pointer(aCtx.fHidden), aRows, aWidth);
  for i := 0 to fConfig.Layers - 1 do
  begin
    l := @fLayer[i];
    // self-attention, each sequence on its own
    fBackend.Linear(pointer(aCtx.fHidden), n, h, l^.Q, h, l^.QBias, pointer(aCtx.fQ));
    fBackend.Linear(pointer(aCtx.fHidden), n, h, l^.K, h, l^.KBias, pointer(aCtx.fK));
    fBackend.Linear(pointer(aCtx.fHidden), n, h, l^.V, h, l^.VBias, pointer(aCtx.fV));
    for r := 0 to aRows - 1 do
    begin
      off := r * aWidth * h;
      fBackend.Attention(@aCtx.fQ[off], @aCtx.fK[off], @aCtx.fV[off], aWidth,
        fConfig.Heads, fConfig.HeadDim, @aCtx.fKeep[r * aWidth],
        pointer(aCtx.fScores), @aCtx.fCtx[off]);
    end;
    // output projection, residual, post-norm
    fBackend.Linear(pointer(aCtx.fCtx), n, h, l^.AttnOut, h, l^.AttnOutBias,
      pointer(aCtx.fTmp));
    fBackend.Add(pointer(aCtx.fTmp), pointer(aCtx.fHidden), n * h);
    fBackend.LayerNorm(pointer(aCtx.fTmp), n, h, l^.AttnNorm, l^.AttnNormBias,
      fConfig.LayerNormEps, pointer(aCtx.fTmp));
    // feed-forward, residual, post-norm
    fBackend.Linear(pointer(aCtx.fTmp), n, h, l^.Inter, f, l^.InterBias,
      pointer(aCtx.fInter));
    fBackend.Gelu(pointer(aCtx.fInter), n * f);
    fBackend.Linear(pointer(aCtx.fInter), n, f, l^.Output, h, l^.OutputBias,
      pointer(aCtx.fHidden));
    fBackend.Add(pointer(aCtx.fHidden), pointer(aCtx.fTmp), n * h);
    fBackend.LayerNorm(pointer(aCtx.fHidden), n, h, l^.OutNorm, l^.OutNormBias,
      fConfig.LayerNormEps, pointer(aCtx.fHidden));
    if Assigned(aOnState) then
      aOnState(i + 1, pointer(aCtx.fHidden), aRows, aWidth);
  end;
end;

end.
