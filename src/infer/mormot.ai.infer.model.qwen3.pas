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
// - the family of mormot.ai.infer.decoder: this unit reads the settings of
//   config.json or of the GGUF metadata, maps the tensor names and loads the
//   weights; the forward pass, the KV cache and the device hand-over are
//   TDecoderModel's
// - head_dim comes from the configuration: Qwen3-0.6B has 16 heads of 128 for
//   a hidden size of 1024, not 1024 / 16
// - or a llama.cpp GGUF file of the qwen3 architecture (CreateFromGguf): its
//   metadata for the settings, its Q8_0 / Q4_0 / Q4_K / Q6_K matrices used in
//   place
unit mormot.ai.infer.model.qwen3;

{
  *****************************************************************************

    - TQwen3Model: the qwen3 family of TDecoderModel
    - TQwen3Config, TQwen3Context, TQwen3Device...: the decoder types under
      their former names

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
  mormot.ai.infer.loader.gguf,
  mormot.ai.infer.backend,
  mormot.ai.infer.decoder;

type
  /// the decoder types under their former names
  TQwen3Config = TDecoderConfig;
  TQwen3Layer = TDecoderLayer;
  TQwen3LayerDynArray = TDecoderLayerDynArray;
  TOnQwen3State = TOnDecoderState;
  TQwen3Context = TDecoderContext;
  TQwen3Device = TDecoderDevice;

  /// a Qwen3 causal language model over a safetensors folder or a GGUF file
  // - Create (config.json and model.safetensors) and CreateFromGguf as
  // TDecoderModel's; model_type and architecture qwen3
  TQwen3Model = class(TDecoderModel)
  protected
    procedure LoadConfig(const aConfigFile: TFileName); override;
    procedure LoadGgufConfig; override;
    procedure LoadWeights; override;
    function GgufName(const aName: RawUtf8): RawUtf8; override;
  public
    /// 'qwen3'
    class function ModelType: RawUtf8; override;
    /// 'qwen3'
    class function Architecture: RawUtf8; override;
  end;


implementation


{ TQwen3Model }

class function TQwen3Model.ModelType: RawUtf8;
begin
  result := 'qwen3';
end;

class function TQwen3Model.Architecture: RawUtf8;
begin
  result := 'qwen3';
end;

const
  REQUIRED_KEYS: array[0..9] of RawUtf8 = (
    'vocab_size', 'hidden_size', 'num_hidden_layers', 'num_attention_heads',
    'num_key_value_heads', 'head_dim', 'intermediate_size',
    'max_position_embeddings', 'rms_norm_eps', 'hidden_act');

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

function TQwen3Model.GgufName(const aName: RawUtf8): RawUtf8;
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

procedure TQwen3Model.LoadWeights;
var
  i, used: PtrInt;
  h, f, qw, kvw: Int64;
  p: RawUtf8;
  l: ^TDecoderLayer;
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
    CheckBlockScales(fEmbedSrc, 'model.embed_tokens.weight');
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


initialization
  RegisterDecoderModel(TQwen3Model);

end.
