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
  // - idtQ8_0 and idtQ4_0 are block types of GGUF files: 32 elements share one
  // FP16 scale (see TInferBlockQ8_0, TInferBlockQ4_0) - only GGUF files hold
  // them, safetensors never
  TInferDType = (
    idtUnknown,
    idtF32,
    idtF16,
    idtBF16,
    idtI32,
    idtI64,
    idtU8,
    idtI8,
    idtQ8_0,
    idtQ4_0);

  /// one Q8_0 block as ggml stores it: value[i] = d * qs[i]
  // - d is an IEEE half float (raw bits), 34 bytes per 32 elements
  TInferBlockQ8_0 = packed record
    d: word;
    qs: array[0..31] of shortint;
  end;
  PInferBlockQ8_0 = ^TInferBlockQ8_0;
  TInferBlockQ8_0Array = array[0..(MaxInt div SizeOf(TInferBlockQ8_0)) - 1] of
    TInferBlockQ8_0;
  PInferBlockQ8_0Array = ^TInferBlockQ8_0Array;

  /// one Q4_0 block as ggml stores it: value[i] = d * (q[i] - 8), q in 0..15
  // - d is an IEEE half float (raw bits); qs[j] holds q[j] in its low and
  // q[j + 16] in its high nibble - 18 bytes per 32 elements
  TInferBlockQ4_0 = packed record
    d: word;
    qs: array[0..15] of byte;
  end;
  PInferBlockQ4_0 = ^TInferBlockQ4_0;
  TInferBlockQ4_0Array = array[0..(MaxInt div SizeOf(TInferBlockQ4_0)) - 1] of
    TInferBlockQ4_0;
  PInferBlockQ4_0Array = ^TInferBlockQ4_0Array;

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
    /// size in bytes = Count div block * block size
    Bytes: Int64;
    /// first element - owned by the file or buffer the view was taken from
    Data: pointer;
  end;
  PInferTensor = ^TInferTensor;
  TInferTensorDynArray = array of TInferTensor;

const
  /// size in bytes of one block of each type (0 = unknown)
  INFER_DTYPE_SIZE: array[TInferDType] of byte = (
    0, 4, 2, 2, 4, 8, 1, 1, 34, 18);

  /// elements per block of each type: 1 for plain element types
  INFER_DTYPE_BLOCK: array[TInferDType] of byte = (
    0, 1, 1, 1, 1, 1, 1, 1, 32, 32);

  /// the name of each type (safetensors names, ggml's for block types)
  INFER_DTYPE_NAME: array[TInferDType] of RawUtf8 = (
    '', 'F32', 'F16', 'BF16', 'I32', 'I64', 'U8', 'I8', 'Q8_0', 'Q4_0');

/// the element type of a safetensors dtype name, idtUnknown if not supported
// - block types are no safetensors types and never returned
function InferDTypeFromName(const aName: RawUtf8): TInferDType;

/// bytes of aCount elements of aType - false if aCount is no whole number of
// blocks (a block type) or the type is unknown
function InferDTypeBytes(aType: TInferDType; aCount: Int64; out aBytes: Int64): boolean;

/// an IEEE 754 half float (raw bits) as single - exact for every value,
// subnormals, infinities and NaN included
function InferHalfToSingle(aHalf: word): single;

/// the bits of InferHalfToSingle(aHalf) - without passing a NaN through the
// FPU: Delphi 7 returns a single on the x87 stack, and loading a signaling NaN
// there raises with the default exception mask
function InferHalfToSingleBits(aHalf: word): cardinal;

/// a single as IEEE 754 half float (raw bits), rounded to nearest even like
// the F16C / NEON conversion ggml uses; overflow gives infinity
function InferSingleToHalf(aValue: single): word;

/// row aRow of a [rows, aCols] matrix of type aType, as F32 into aDst
// - F32 is copied, Q8_0 and Q4_0 dequantized (d * q, d * (q - 8): exact in
// single); aCols must be a multiple of the block size
procedure InferRowToF32(aData: pointer; aType: TInferDType; aRow, aCols: PtrInt;
  aDst: PSingleArray);

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
    if (INFER_DTYPE_BLOCK[t] = 1) and (INFER_DTYPE_NAME[t] = aName) then
    begin
      result := t;
      exit;
    end;
  result := idtUnknown;
end;

function InferDTypeBytes(aType: TInferDType; aCount: Int64; out aBytes: Int64): boolean;
begin
  result := false;
  aBytes := 0;
  if (aType = idtUnknown) or (aCount < 0) or
     (aCount mod INFER_DTYPE_BLOCK[aType] <> 0) then
    exit;
  // callers bound aCount by the file size: the product cannot overflow
  aBytes := (aCount div INFER_DTYPE_BLOCK[aType]) * INFER_DTYPE_SIZE[aType];
  result := true;
end;

function InferHalfToSingle(aHalf: word): single;
begin
  PCardinal(@result)^ := InferHalfToSingleBits(aHalf);
end;

function InferHalfToSingleBits(aHalf: word): cardinal;
var
  exp, mant, bits: cardinal;
begin
  exp := (aHalf shr 10) and $1f;
  mant := aHalf and $3ff;
  if exp = 0 then
    if mant = 0 then
      bits := 0
    else
    begin
      // subnormal half: normalise into a single exponent
      exp := 127 - 15 + 1;
      while mant and $400 = 0 do
      begin
        mant := mant shl 1;
        dec(exp);
      end;
      bits := (exp shl 23) or ((mant and $3ff) shl 13);
    end
  else if exp = $1f then
    bits := $7f800000 or (mant shl 13) // infinity or NaN
  else
    bits := ((exp + 127 - 15) shl 23) or (mant shl 13);
  result := bits or (cardinal(aHalf and $8000) shl 16);
end;

function InferSingleToHalf(aValue: single): word;
var
  bits, sign, mant, rest, half: cardinal;
  exp: integer;
begin
  bits := PCardinal(@aValue)^;
  sign := (bits shr 16) and $8000;
  exp := integer((bits shr 23) and $ff);
  mant := bits and $7fffff;
  if exp = $ff then
  begin
    // infinity, or NaN kept quiet with its top payload bits
    if mant = 0 then
      result := sign or $7c00
    else
      result := sign or $7e00 or (mant shr 13);
    exit;
  end;
  exp := exp - 127 + 15;
  if exp >= $1f then
  begin
    result := sign or $7c00; // too large: infinity
    exit;
  end;
  if exp <= 0 then
  begin
    // subnormal half (or zero): shift the implicit bit in, round to even
    if exp < -10 then
    begin
      result := sign; // below half the smallest subnormal
      exit;
    end;
    mant := mant or $800000;
    rest := 14 - exp; // bits dropped: 13 + (1 - exp)
    half := mant shr rest;
    mant := mant and ((cardinal(1) shl rest) - 1);
    if (mant > cardinal(1) shl (rest - 1)) or
       ((mant = cardinal(1) shl (rest - 1)) and (half and 1 <> 0)) then
      inc(half);
    result := sign or half;
    exit;
  end;
  half := (cardinal(exp) shl 10) or (mant shr 13);
  rest := mant and $1fff;
  // round to nearest even; a carry into the exponent is the correct result,
  // up to infinity
  if (rest > $1000) or ((rest = $1000) and (half and 1 <> 0)) then
    inc(half);
  result := sign or half;
end;

procedure InferRowToF32(aData: pointer; aType: TInferDType; aRow, aCols: PtrInt;
  aDst: PSingleArray);
var
  b, j, nb: PtrInt;
  blk: PInferBlockQ8_0;
  b4: PInferBlockQ4_0;
  d: single;
begin
  case aType of
    idtF32:
      MoveFast(PSingleArray(aData)^[aRow * aCols], aDst^[0], aCols * SizeOf(single));
    idtQ8_0:
      begin
        nb := aCols div 32;
        blk := @PInferBlockQ8_0Array(aData)^[aRow * nb];
        for b := 0 to nb - 1 do
        begin
          d := InferHalfToSingle(blk^.d);
          for j := 0 to 31 do
            aDst^[b * 32 + j] := d * blk^.qs[j];
          inc(blk);
        end;
      end;
    idtQ4_0:
      begin
        nb := aCols div 32;
        b4 := @PInferBlockQ4_0Array(aData)^[aRow * nb];
        for b := 0 to nb - 1 do
        begin
          d := InferHalfToSingle(b4^.d);
          for j := 0 to 15 do
          begin
            aDst^[b * 32 + j] := d * (integer(b4^.qs[j] and 15) - 8);
            aDst^[b * 32 + j + 16] := d * (integer(b4^.qs[j] shr 4) - 8);
          end;
          inc(b4);
        end;
      end;
  else
    EInferError.RaiseUtf8('InferRowToF32: % not supported', [INFER_DTYPE_NAME[aType]]);
  end;
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
