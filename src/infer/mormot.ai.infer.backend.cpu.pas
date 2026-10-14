/// native inference engine - the scalar CPU reference backend
// - part of the mormot.ai.* extension (LandrixAI)
// - licensed under a MPL/GPL/LGPL three license - see LICENSE
// - plain Pascal, no SIMD, no threads: it runs on every compiler and target down
//   to Delphi 7 / Win32 and is the yardstick for every optimised path
// - products and sums are computed in double (each factor widened before the
//   multiplication), results are stored as single: x87 keeps intermediates in
//   extended precision, so the last bit of a double may differ from SSE and NEON,
//   far below any stored single; the reference trades speed for the smallest
//   rounding error
// - Q8_0 weights are multiplied as ggml does (the arithmetic of the lembed oracle,
//   measured to matter: dequantized weights with F32 activations miss the cosine
//   gate on one corpus text): each activation row is quantized to Q8_0 blocks
//   (d = max|x| / 127 stored as FP16, q = x * (1 / d) rounded to nearest even
//   like the NEON and AVX paths of ggml-cpu - its plain C reference rounds ties
//   away from zero, the CPU kernels lembed runs do not; the AVX path also forms
//   the inverse as 127 / amax instead of 1 / d, which can move a near-tie by
//   one step: byte-equal to the aarch64 oracle, not to every ggml build), the
//   block products are exact integer sums, scaled by both FP16 scales and summed
//   in double
// - erf is our own (no Math unit): Taylor series below 2.5, the continued fraction
//   of erfc above; negative GELU inputs use erfc, so the tail beyond 2.5 never
//   cancels (below 2.5, erfc = 1 - erf keeps an absolute error of ~1e-15)
unit mormot.ai.infer.backend.cpu;

{
  *****************************************************************************

    - InferErf / InferErfc: double-precision error function
    - InferQuantizeRowQ8_0: activations to Q8_0 blocks, as ggml does
    - TInferCpuBackend: scalar reference implementation of TInferBackend

  *****************************************************************************
}

interface

{$I mormot.defines.inc}

uses
  sysutils,
  mormot.core.base,
  mormot.ai.infer.tensor,
  mormot.ai.infer.backend;

/// the error function: absolute error below 2e-15 over the whole range (the
// series near 2.5 sums terms of ~200 to ~1, so not "a few ulp" there)
function InferErf(x: double): double;

/// the complementary error function 1 - erf(x)
// - from 2.5 on by its continued fraction (relative error ~1e-15, no
// cancellation); below 2.5 as 1 - erf(x) (absolute error ~1e-15)
function InferErfc(x: double): double;

/// quantize aCount (a multiple of 32) singles into aCount div 32 Q8_0 blocks
// - ggml-cpu's quantize_row_q8_0: d = max|x| / 127 (kept as FP16), q = the
// single product x * (1 / d) rounded to nearest, ties to even
procedure InferQuantizeRowQ8_0(x: PSingleArray; aCount: PtrInt;
  aBlocks: PInferBlockQ8_0Array);

type
  /// scalar reference backend: every operation; Linear takes F32 or Q8_0
  TInferCpuBackend = class(TInferBackend)
  public
    function Supports(aOp: TInferOp; aWeights: TInferDType): boolean; override;
    procedure Linear(x: PSingleArray; aRows, aIn: PtrInt; const w: TInferWeight;
      aOut: PtrInt; bias: PSingleArray; y: PSingleArray); override;
    procedure LayerNorm(x: PSingleArray; aRows, aCols: PtrInt;
      gamma, beta: PSingleArray; aEps: double; y: PSingleArray); override;
    procedure Gelu(x: PSingleArray; aCount: PtrInt); override;
    procedure Add(x, y: PSingleArray; aCount: PtrInt); override;
    procedure SoftmaxMasked(x: PSingleArray; aRows, aCols: PtrInt;
      aKeep: PIntegerArray); override;
    procedure Attention(q, k, v: PSingleArray; aSeq, aHeads, aHeadDim: PtrInt;
      aKeep: PIntegerArray; aScratch, y: PSingleArray); override;
    procedure L2Normalize(x: PSingleArray; aCount: PtrInt); override;
  end;


implementation

const
  TWO_DIV_SQRTPI = 1.1283791670955125739; // 2 / sqrt(pi)
  INV_SQRTPI = 0.56418958354775628695;    // 1 / sqrt(pi)
  INV_SQRT2 = 0.70710678118654752440;     // 1 / sqrt(2)

/// erf by its Maclaurin series - used for |x| < 2.5 only
function ErfSeries(x: double): double;
var
  term, sum, x2: double;
  n: integer;
begin
  if x = 0 then
  begin
    result := 0; // the loop below would never meet its relative stop
    exit;
  end;
  // erf(x) = 2/sqrt(pi) * sum (-1)^n x^(2n+1) / (n! (2n+1))
  x2 := x * x;
  term := x;
  sum := x;
  n := 0;
  repeat
    inc(n);
    term := -term * x2 / n;
    sum := sum + term / (2 * n + 1);
  until (Abs(term) <= 1e-17 * Abs(sum)) or (n > 200);
  result := TWO_DIV_SQRTPI * sum;
end;

/// erfc by its continued fraction - used for x >= 2.5 only
function ErfcFraction(x: double): double;
var
  f: double;
  k: integer;
begin
  // erfc(x) = exp(-x^2)/sqrt(pi) / (x + (1/2)/(x + 1/(x + (3/2)/(x + 2/(x + ...)))))
  // evaluated backwards; 120 terms are far beyond double precision at x >= 2.5
  f := x;
  for k := 120 downto 1 do
    f := x + (k * 0.5) / f;
  result := Exp(-x * x) * INV_SQRTPI / f;
end;

function InferErf(x: double): double;
begin
  if Abs(x) < 2.5 then
    result := ErfSeries(x)
  else if x > 0 then
    result := 1 - ErfcFraction(x)
  else
    result := ErfcFraction(-x) - 1;
end;

function InferErfc(x: double): double;
begin
  if x >= 2.5 then
    result := ErfcFraction(x)
  else if x <= -2.5 then
    result := 2 - ErfcFraction(-x)
  else
    result := 1 - ErfSeries(x);
end;


{ TInferCpuBackend }

procedure InferQuantizeRowQ8_0(x: PSingleArray; aCount: PtrInt;
  aBlocks: PInferBlockQ8_0Array);
var
  b, j: PtrInt;
  amax, v, d, id, t: single;
  f: integer;
  xb: PSingleArray;
begin
  for b := 0 to (aCount shr 5) - 1 do
  begin
    xb := @x^[b * 32];
    amax := 0;
    for j := 0 to 31 do
    begin
      // a NaN or an infinity has no Q8_0 form: refused by its exponent bits
      // (a comparison would skip a NaN), never truncated
      if PCardinal(@xb^[j])^ and $7f800000 = $7f800000 then
        EInferError.RaiseUtf8('InferQuantizeRowQ8_0: non-finite activation at %',
          [b * 32 + j]);
      v := Abs(xb^[j]);
      if v > amax then
        amax := v;
    end;
    // single results of one division each: correctly rounded on every FPU
    // (an x87 or double intermediate rounds twice, harmlessly for / of singles)
    d := amax / 127;
    aBlocks^[b].d := InferSingleToHalf(d);
    // d = 0, or so small that 1 / d overflows a single: its FP16 scale is 0,
    // the block adds nothing to any dot product - all q = 0 (ggml: 0 for
    // d = 0, undefined for the overflow)
    if d < 2.9387359e-39 then // 1 / 3.4028235e38: beyond, 1 / d is no single
    begin
      FillCharFast(aBlocks^[b].qs, SizeOf(aBlocks^[b].qs), 0);
      continue;
    end;
    id := 1 / d;
    for j := 0 to 31 do
    begin
      // one correctly rounded single product, as the SIMD kernels form it
      t := xb^[j] * id;
      // nearest, ties to even - spelled out: Round depends on the FPU mode
      f := Trunc(t);
      v := t - f; // exact: f and t are close singles
      if v > 0.5 then
        inc(f)
      else if v < -0.5 then
        dec(f)
      else if (v = 0.5) and odd(f) then
        inc(f)
      else if (v = -0.5) and odd(f) then
        dec(f);
      aBlocks^[b].qs[j] := f;
    end;
  end;
end;

function TInferCpuBackend.Supports(aOp: TInferOp; aWeights: TInferDType): boolean;
begin
  if aOp = ioLinear then
    result := aWeights in [idtF32, idtQ8_0]
  else
    result := aWeights = idtF32;
end;

procedure LinearQ8_0(x: PSingleArray; aRows, aIn: PtrInt; w: PInferBlockQ8_0Array;
  aOut: PtrInt; bias: PSingleArray; y: PSingleArray);
var
  r, o, b, j, nb: PtrInt;
  sum: integer;
  acc, dx, dw: double;
  xq: array of TInferBlockQ8_0;
  wb, xb: PInferBlockQ8_0;
  dxs: array of double;
begin
  nb := aIn shr 5;
  SetLength(xq, nb);
  SetLength(dxs, nb);
  for r := 0 to aRows - 1 do
  begin
    InferQuantizeRowQ8_0(@x^[r * aIn], aIn, pointer(xq));
    for b := 0 to nb - 1 do
      dxs[b] := InferHalfToSingle(xq[b].d);
    for o := 0 to aOut - 1 do
    begin
      wb := @w^[o * nb];
      if bias <> nil then
        acc := bias^[o]
      else
        acc := 0;
      for b := 0 to nb - 1 do
      begin
        xb := @xq[b];
        sum := 0; // at most 32 * 127 * 128: no overflow
        for j := 0 to 31 do
          inc(sum, integer(wb^.qs[j]) * xb^.qs[j]);
        dw := InferHalfToSingle(wb^.d);
        dx := dxs[b];
        acc := acc + sum * (dw * dx);
        inc(wb);
      end;
      y^[r * aOut + o] := acc;
    end;
  end;
end;

procedure TInferCpuBackend.Linear(x: PSingleArray; aRows, aIn: PtrInt;
  const w: TInferWeight; aOut: PtrInt; bias: PSingleArray; y: PSingleArray);
var
  r, o, i: PtrInt;
  acc, a: double;
  xr, wo: PSingleArray;
begin
  case w.DType of
    idtF32:
      ;
    idtQ8_0:
      begin
        if aIn and 31 <> 0 then
          EInferError.RaiseUtf8('%.Linear: % inputs are no whole Q8_0 blocks',
            [self, aIn]);
        LinearQ8_0(x, aRows, aIn, w.Data, aOut, bias, y);
        exit;
      end;
  else
    EInferError.RaiseUtf8('%.Linear: % weights not supported',
      [self, INFER_DTYPE_NAME[w.DType]]);
  end;
  for r := 0 to aRows - 1 do
  begin
    xr := @x^[r * aIn];
    for o := 0 to aOut - 1 do
    begin
      wo := @PSingleArray(w.Data)^[o * aIn];
      if bias <> nil then
        acc := bias^[o]
      else
        acc := 0;
      for i := 0 to aIn - 1 do
      begin
        a := xr^[i]; // widen before multiplying: same result on every FPU
        acc := acc + a * wo^[i];
      end;
      y^[r * aOut + o] := acc;
    end;
  end;
end;

procedure TInferCpuBackend.LayerNorm(x: PSingleArray; aRows, aCols: PtrInt;
  gamma, beta: PSingleArray; aEps: double; y: PSingleArray);
var
  r, c: PtrInt;
  mean, variance, d, inv: double;
  xr, yr: PSingleArray;
begin
  for r := 0 to aRows - 1 do
  begin
    xr := @x^[r * aCols];
    yr := @y^[r * aCols];
    // two passes: the one-pass formula loses digits on nearly constant rows
    mean := 0;
    for c := 0 to aCols - 1 do
      mean := mean + xr^[c];
    mean := mean / aCols;
    variance := 0;
    for c := 0 to aCols - 1 do
    begin
      d := xr^[c] - mean;
      variance := variance + d * d;
    end;
    variance := variance / aCols;
    inv := 1 / Sqrt(variance + aEps);
    for c := 0 to aCols - 1 do
      yr^[c] := (xr^[c] - mean) * inv * gamma^[c] + beta^[c];
  end;
end;

procedure TInferCpuBackend.Gelu(x: PSingleArray; aCount: PtrInt);
var
  i: PtrInt;
  v, z: double;
begin
  for i := 0 to aCount - 1 do
  begin
    v := x^[i];
    z := v * INV_SQRT2;
    if z >= 0 then
      x^[i] := 0.5 * v * (1 + InferErf(z))
    else
      x^[i] := 0.5 * v * InferErfc(-z); // 1 + erf(z) = erfc(-z), no cancellation
  end;
end;

procedure TInferCpuBackend.Add(x, y: PSingleArray; aCount: PtrInt);
var
  i: PtrInt;
begin
  for i := 0 to aCount - 1 do
    x^[i] := x^[i] + y^[i];
end;

procedure TInferCpuBackend.SoftmaxMasked(x: PSingleArray; aRows, aCols: PtrInt;
  aKeep: PIntegerArray);
var
  r, c: PtrInt;
  m, sum, e: double;
  xr: PSingleArray;
  any: boolean;
begin
  for r := 0 to aRows - 1 do
  begin
    xr := @x^[r * aCols];
    any := false;
    m := 0;
    for c := 0 to aCols - 1 do
      if (aKeep = nil) or (aKeep^[c] <> 0) then
        if not any or (xr^[c] > m) then
        begin
          m := xr^[c];
          any := true;
        end;
    if not any then
    begin
      FillCharFast(xr^, aCols * SizeOf(single), 0);
      continue;
    end;
    // subtract the maximum: exp() of a large logit would overflow
    sum := 0;
    for c := 0 to aCols - 1 do
      if (aKeep = nil) or (aKeep^[c] <> 0) then
      begin
        e := Exp(xr^[c] - m);
        xr^[c] := e;
        sum := sum + e;
      end
      else
        xr^[c] := 0;
    sum := 1 / sum;
    for c := 0 to aCols - 1 do
      xr^[c] := xr^[c] * sum;
  end;
end;

procedure TInferCpuBackend.Attention(q, k, v: PSingleArray;
  aSeq, aHeads, aHeadDim: PtrInt; aKeep: PIntegerArray; aScratch, y: PSingleArray);
var
  h, i, j, d, width, off: PtrInt;
  scale, acc, a: double;
  qi, kj, p: PSingleArray;
begin
  width := aHeads * aHeadDim;
  scale := 1 / Sqrt(aHeadDim);
  for h := 0 to aHeads - 1 do
  begin
    off := h * aHeadDim;
    // scores[i, j] = q_i . k_j / sqrt(d)
    for i := 0 to aSeq - 1 do
    begin
      qi := @q^[i * width + off];
      for j := 0 to aSeq - 1 do
      begin
        kj := @k^[j * width + off];
        acc := 0;
        for d := 0 to aHeadDim - 1 do
        begin
          a := qi^[d];
          acc := acc + a * kj^[d];
        end;
        aScratch^[i * aSeq + j] := acc * scale;
      end;
    end;
    SoftmaxMasked(aScratch, aSeq, aSeq, aKeep);
    // y_i = sum_j p[i, j] v_j
    for i := 0 to aSeq - 1 do
    begin
      p := @aScratch^[i * aSeq];
      for d := 0 to aHeadDim - 1 do
      begin
        acc := 0;
        for j := 0 to aSeq - 1 do
        begin
          a := p^[j];
          acc := acc + a * v^[j * width + off + d];
        end;
        y^[i * width + off + d] := acc;
      end;
    end;
  end;
end;

procedure TInferCpuBackend.L2Normalize(x: PSingleArray; aCount: PtrInt);
var
  i: PtrInt;
  sum, v: double;
begin
  sum := 0;
  for i := 0 to aCount - 1 do
  begin
    v := x^[i];
    sum := sum + v * v;
  end;
  if sum = 0 then
    exit;
  sum := 1 / Sqrt(sum);
  for i := 0 to aCount - 1 do
    x^[i] := x^[i] * sum;
end;

end.
