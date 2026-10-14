/// LandrixAI - local IEmbedder on the native inference engine (no foreign library)
// - part of the mormot.ai.* extension (LandrixAI)
// - licensed under a MPL/GPL/LGPL three license - see LICENSE
// - implements IEmbedder with the engine of src/infer (docs/native-inference.md):
//   one GGUF file holds tokenizer and weights - the same bge-m3-q8_0.gguf the
//   lembed oracle uses, with the same vectors within the reference contract v1
//   (cosine >= 0.999 against lembed, nDCG@10 0.8989 against 0.8963)
// - computes on TInferCpuFastBackend: two thirds of the cores, the SIMD kernels
//   of the CPU, float arithmetic like ggml (within a tolerance of the double
//   reference; cosine against lembed >= 0.9994 on the embed-de corpus)
// - process-scope service like TLembedEmbedder: SharedNativeEmbedder() keeps one
//   instance per logical model name, so N stores share one mapped model
// - thread-safe without a lock: the model is immutable, every Embed brings its
//   own buffers; parallel callers compute on their own thread while one of them
//   holds the thread pool
// - a text above 512 tokens raises, it is never cut (same contract as lembed)
unit mormot.ai.embed.native;

{
  *****************************************************************************

    - TNativeEmbedder: IEmbedder over a GGUF model run by the native engine
    - SharedNativeEmbedder: one process-wide embedder per model name

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
  mormot.ai.llm.types,
  mormot.ai.embeddings,
  mormot.ai.infer.backend.cpu,
  mormot.ai.infer.pipeline.embed;

type
  /// a local IEmbedder computed by the native inference engine
  TNativeEmbedder = class(TInterfacedObject, IEmbedder)
  protected
    fBackend: TInferCpuFastBackend;
    fEmbedder: TInferEmbedder;
    fModel: RawUtf8;
  public
    /// load the GGUF file aModelPath (tokenizer and weights) under the logical
    // name aModelName; aThreads as TInferCpuFastBackend.Create (0 = default)
    // - prefer SharedNativeEmbedder(), so a model is mapped once per process
    constructor Create(const aModelPath, aModelName: RawUtf8;
      aThreads: integer = 0); reintroduce;
    destructor Destroy; override;
    function Embed(const aText: RawUtf8): TLlmEmbedding;
    function EmbedBatch(const aTexts: TRawUtf8DynArray): TLlmEmbeddingDynArray;
    function Model: RawUtf8;
    /// the engine behind, e.g. for its MaxTokens or Tokenizer
    property Embedder: TInferEmbedder
      read fEmbedder;
  end;

/// the process-wide shared native embedder for a logical model name
// - the first call for aModelName loads the model, later calls with the same
// name return the SAME IEmbedder; aModelPath and aThreads only count on the
// first call (the name is the model's identity, as for SharedLembedEmbedder)
function SharedNativeEmbedder(const aModelPath, aModelName: RawUtf8;
  aThreads: integer = 0): IEmbedder;


implementation

{ TNativeEmbedder }

constructor TNativeEmbedder.Create(const aModelPath, aModelName: RawUtf8;
  aThreads: integer);
begin
  inherited Create;
  fModel := aModelName;
  fBackend := TInferCpuFastBackend.Create(aThreads);
  fEmbedder := TInferEmbedder.CreateFromGguf(Utf8ToString(aModelPath), fBackend);
end;

destructor TNativeEmbedder.Destroy;
begin
  fEmbedder.Free;
  fBackend.Free;
  inherited Destroy;
end;

function TNativeEmbedder.Embed(const aText: RawUtf8): TLlmEmbedding;
begin
  result := fEmbedder.Embed([aText]);
end;

function TNativeEmbedder.EmbedBatch(
  const aTexts: TRawUtf8DynArray): TLlmEmbeddingDynArray;
var
  i: PtrInt;
begin
  // one text at a time: no padding to the longest text of a batch, and every
  // vector the same as from Embed()
  SetLength(result, length(aTexts));
  for i := 0 to high(aTexts) do
    result[i] := fEmbedder.Embed([aTexts[i]]);
end;

function TNativeEmbedder.Model: RawUtf8;
begin
  result := fModel;
end;


{ process-wide singleton registry (keyed by logical model name) }

type
  TNativeCacheEntry = record
    Name: RawUtf8;
    Emb: IEmbedder;
  end;

var
  gNativeLock: TOSLightLock;
  gNative: array of TNativeCacheEntry;

function SharedNativeEmbedder(const aModelPath, aModelName: RawUtf8;
  aThreads: integer): IEmbedder;
var
  i: PtrInt;
begin
  result := nil;
  gNativeLock.Lock;
  try
    for i := 0 to high(gNative) do
      if gNative[i].Name = aModelName then
      begin
        result := gNative[i].Emb;
        exit;
      end;
    result := TNativeEmbedder.Create(aModelPath, aModelName, aThreads);
    SetLength(gNative, length(gNative) + 1);
    gNative[high(gNative)].Name := aModelName;
    gNative[high(gNative)].Emb := result;
  finally
    gNativeLock.UnLock;
  end;
end;

initialization
  gNativeLock.Init;

finalization
  gNative := nil; // release the cached embedders (unmaps the models)
  gNativeLock.Done;

end.
