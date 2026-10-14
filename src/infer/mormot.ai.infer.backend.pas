/// native inference engine - the backend contract
// - part of the mormot.ai.* extension (LandrixAI)
// - licensed under a MPL/GPL/LGPL three license - see LICENSE
// - one non-generic abstract class with typed operations on F32 buffers; a
//   backend reports per operation whether it supports it, and the model falls
//   back to the CPU reference only for "not supported", never after a failure
// - every operation works on caller-owned memory: a backend allocates nothing
//   per call, so the hot loops stay allocation-free
unit mormot.ai.infer.backend;

{
  *****************************************************************************

    - TInferOp: the operations a model may ask for
    - TInferBackend: abstract backend with the typed operations

  *****************************************************************************
}

interface

{$I mormot.defines.inc}

uses
  sysutils,
  mormot.core.base,
  mormot.ai.infer.tensor;

type
  /// the operations of the backend contract
  TInferOp = (
    ioLinear,
    ioLayerNorm,
    ioGelu,
    ioAdd,
    ioSoftmaxMasked,
    ioAttention,
    ioL2Normalize);

  /// a weight matrix [out, in] as stored in the file: its first byte and type
  // - F32 holds out * in singles, Q8_0 out rows of in div 32 blocks
  TInferWeight = record
    Data: pointer;
    DType: TInferDType;
  end;

  /// abstract backend: typed operations on caller-owned F32 buffers
  // - layout is always row-major; "rows" are tokens, "cols" are features
  TInferBackend = class
  public
    /// true when this backend implements aOp for weights of type aWeights
    function Supports(aOp: TInferOp; aWeights: TInferDType): boolean; virtual; abstract;
    /// y[rows, out] = x[rows, in] * w[out, in]^T + bias[out]
    // - w is stored as nn.Linear stores it: one row per output feature, of the
    // type w.DType (see Supports); a Q8_0 row needs aIn to be a multiple of 32
    // - bias may be nil; y must not overlap x
    procedure Linear(x: PSingleArray; aRows, aIn: PtrInt; const w: TInferWeight;
      aOut: PtrInt; bias: PSingleArray; y: PSingleArray); virtual; abstract;
    /// y[rows, cols] = (x - mean) / sqrt(var + eps) * gamma + beta, per row
    // - population variance; y may be x
    procedure LayerNorm(x: PSingleArray; aRows, aCols: PtrInt;
      gamma, beta: PSingleArray; aEps: double; y: PSingleArray); virtual; abstract;
    /// x := 0.5 x (1 + erf(x / sqrt 2)) in place - the exact GELU
    procedure Gelu(x: PSingleArray; aCount: PtrInt); virtual; abstract;
    /// x := x + y elementwise
    procedure Add(x, y: PSingleArray; aCount: PtrInt); virtual; abstract;
    /// softmax over each row of x[rows, cols], in place; aKeep[col] = 0 drops a
    // column: its result is exactly 0 and it does not enter the sum
    // - aKeep may be nil (keep all); a row with no kept column becomes all 0
    procedure SoftmaxMasked(x: PSingleArray; aRows, aCols: PtrInt;
      aKeep: PIntegerArray); virtual; abstract;
    /// multi-head self-attention of one sequence
    // - q, k, v and y are [aSeq, aHeads * aHeadDim]; head h uses the columns
    // h*aHeadDim .. h*aHeadDim + aHeadDim - 1
    // - aKeep[key] = 0 masks a key position (padding); aScratch holds at least
    // aSeq * aSeq singles
    procedure Attention(q, k, v: PSingleArray; aSeq, aHeads, aHeadDim: PtrInt;
      aKeep: PIntegerArray; aScratch, y: PSingleArray); virtual; abstract;
    /// x := x / ||x|| in place; a zero vector stays zero
    procedure L2Normalize(x: PSingleArray; aCount: PtrInt); virtual; abstract;
  end;

/// a TInferWeight from its parts
function InferWeight(aData: pointer; aType: TInferDType): TInferWeight;


implementation

function InferWeight(aData: pointer; aType: TInferDType): TInferWeight;
begin
  result.Data := aData;
  result.DType := aType;
end;

end.
