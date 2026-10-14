/// native inference engine - tensor views, element types and shapes
// - part of the mormot.ai.* extension (LandrixAI)
// - licensed under a MPL/GPL/LGPL three license - see LICENSE
// - a TInferTensor is a VIEW: it describes elements that live elsewhere (a
//   memory-mapped weights file, a scratch buffer) and never owns them
// - plan and phases: docs/native-inference.md
unit mormot.ai.infer.tensor;

{
  *****************************************************************************

    - EInferError: the exception of the whole engine
    - TInferDType: element types and their sizes
    - TInferTensor: a named, typed, shaped view on memory owned by someone else
    - InferShapeCount / InferShapeText / InferShapeIs: shape helpers

  *****************************************************************************
}

interface

{$I mormot.defines.inc}

uses
  sysutils,
  mormot.core.base,
  mormot.core.text;

type
  /// raised by the inference engine on invalid input, files or calls
  EInferError = class(ESynException);

  /// element type of a tensor
  // - names follow safetensors: F32, F16, BF16, I32, I64, U8, I8
  TInferDType = (
    idtUnknown,
    idtF32,
    idtF16,
    idtBF16,
    idtI32,
    idtI64,
    idtU8,
    idtI8);

  /// a named, typed, shaped view on elements owned by someone else
  TInferTensor = record
    /// the tensor name, e.g. 'encoder.layer.0.attention.self.query.weight'
    Name: RawUtf8;
    /// element type
    DType: TInferDType;
    /// dimensions, row-major (C order)
    Shape: TInt64DynArray;
    /// number of elements = product of Shape
    Count: Int64;
    /// size in bytes = Count * element size
    Bytes: Int64;
    /// first element - owned by the file or buffer the view was taken from
    Data: pointer;
  end;
  PInferTensor = ^TInferTensor;
  TInferTensorDynArray = array of TInferTensor;

const
  /// size in bytes of one element of each type (0 = unknown)
  INFER_DTYPE_SIZE: array[TInferDType] of byte = (
    0, 4, 2, 2, 4, 8, 1, 1);

  /// the safetensors name of each type
  INFER_DTYPE_NAME: array[TInferDType] of RawUtf8 = (
    '', 'F32', 'F16', 'BF16', 'I32', 'I64', 'U8', 'I8');

/// the element type of a safetensors dtype name, idtUnknown if not supported
function InferDTypeFromName(const aName: RawUtf8): TInferDType;

/// product of aShape, refused (false) when a dimension is negative or the product
// would exceed aLimit - checked before multiplying, so it can never overflow
// - a zero dimension makes the product 0 whatever the other dimensions are
function InferShapeCount(const aShape: TInt64DynArray; aLimit: Int64;
  out aCount: Int64): boolean;

/// '[24,1024]' - for messages
function InferShapeText(const aShape: TInt64DynArray): RawUtf8;

/// true when aTensor has exactly the dimensions aDims
function InferShapeIs(const aTensor: TInferTensor;
  const aDims: array of Int64): boolean;


implementation

function InferDTypeFromName(const aName: RawUtf8): TInferDType;
var
  t: TInferDType;
begin
  for t := succ(low(t)) to high(t) do
    if INFER_DTYPE_NAME[t] = aName then
    begin
      result := t;
      exit;
    end;
  result := idtUnknown;
end;

function InferShapeCount(const aShape: TInt64DynArray; aLimit: Int64;
  out aCount: Int64): boolean;
var
  i: PtrInt;
  d: Int64;
begin
  result := false;
  aCount := 1;
  for i := 0 to high(aShape) do
    if aShape[i] < 0 then
      exit
    else if aShape[i] = 0 then
      aCount := 0; // an empty tensor is valid whatever its other dimensions
  if aCount = 0 then
  begin
    result := true;
    exit;
  end;
  for i := 0 to high(aShape) do
  begin
    d := aShape[i];
    if (d > aLimit) or (aCount > aLimit div d) then
      exit;
    aCount := aCount * d;
  end;
  result := true;
end;

function InferShapeText(const aShape: TInt64DynArray): RawUtf8;
var
  i: PtrInt;
begin
  result := '[';
  for i := 0 to high(aShape) do
  begin
    if i > 0 then
      result := result + ',';
    result := result + Int64ToUtf8(aShape[i]);
  end;
  result := result + ']';
end;

function InferShapeIs(const aTensor: TInferTensor;
  const aDims: array of Int64): boolean;
var
  i: PtrInt;
begin
  result := false;
  if length(aTensor.Shape) <> length(aDims) then
    exit;
  for i := 0 to high(aDims) do
    if aTensor.Shape[i] <> aDims[i] then
      exit;
  result := true;
end;

end.
